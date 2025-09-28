import copy
import torch
from torch import nn
from torch import distributions as torchd
import networks
import tools
#import self_attention_net as self_atten
to_np = lambda x: x.detach().cpu().numpy()
import torch.nn.functional as F

class RewardEMA:
    """running mean and std"""

    def __init__(self, device, alpha=1e-2):
        self.device = device
        self.alpha = alpha
        self.range = torch.tensor([0.05, 0.95]).to(device)

    def __call__(self, x, ema_vals):
        flat_x = torch.flatten(x.detach())
        x_quantile = torch.quantile(input=flat_x, q=self.range)
        # this should be in-place operation
        ema_vals[:] = self.alpha * x_quantile + (1 - self.alpha) * ema_vals
        scale = torch.clip(ema_vals[1] - ema_vals[0], min=1.0)
        offset = ema_vals[0]
        return offset.detach(), scale.detach()


class WorldModel(nn.Module):
    def __init__(self, obs_space, act_space, step, config):
        super(WorldModel, self).__init__()
        self._step = step
        self._use_amp = True if config.precision == 16 else False

        self._config = config
        shapes = {k: tuple(v.shape) for k, v in obs_space.spaces.items()}
        self.static_encoder = networks.MultiEncoder(shapes, **config.encoder) 
        self.dynamic_encoder = networks.MultiEncoder(shapes, **config.dynamic_encoder)                                   
        self.static_size = self.static_encoder.outdim
 
        self.modulator_size = self.dynamic_encoder.outdim
        self.dynamics = networks.RSSM(
            config.dyn_stoch,
            config.dyn_deter,
            config.dyn_hidden,
            config.dyn_rec_depth,
            config.dyn_discrete,
            config.modulator_dyn_stoch,
            config.modulator_dyn_discrete,
            self.static_size,
            config.act,
            config.norm,
            config.dyn_mean_act,
            config.dyn_std_act,
            config.dyn_min_std,
            config.unimix_ratio,
            config.initial,
            config.num_actions,
            self.modulator_size,
            config.device,
        )
        self.heads = nn.ModuleDict()
        if config.dyn_discrete:
            feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter + config.modulator_dyn_discrete * config.modulator_dyn_stoch
        else:
            feat_size = config.dyn_stoch + config.dyn_deter
        self.heads["decoder"] = networks.MultiDecoder(
            feat_size , shapes, **config.decoder
        )
 
        self.heads["reward"] = networks.MLP(
            feat_size,
            (255,) if config.reward_head["dist"] == "symlog_disc" else (),
            config.reward_head["layers"],
            config.units,
            config.act,
            config.norm,
            dist=config.reward_head["dist"],
            outscale=config.reward_head["outscale"],
            device=config.device,
            name="Reward",
        )
        self.heads["cont"] = networks.MLP(
            feat_size,
            (),
            config.cont_head["layers"],
            config.units,
            config.act,
            config.norm,
            dist="binary",
            outscale=config.cont_head["outscale"],
            device=config.device,
            name="Cont",
        )
        for name in config.grad_heads:
            assert name in self.heads, name
        self._model_opt = tools.Optimizer(
            "model",
            self.parameters(),
            config.model_lr,
            config.opt_eps,
            config.grad_clip,
            config.weight_decay,
            opt=config.opt,
            use_amp=self._use_amp,
        )
        print(
            f"Optimizer model_opt has {sum(param.numel() for param in self.parameters())} variables."
        )
        # other losses are scaled by 1.0.
        self._scales = dict(
            reward=config.reward_head["loss_scale"],
            cont=config.cont_head["loss_scale"],
        )

    def _train(self, data):
        data = self.preprocess(data)

        with tools.RequiresGrad(self):
            with torch.cuda.amp.autocast(self._use_amp):
                mask_image = self.extract_dynamic_mask(data["image"])
    
                data_wo_first = self.change_data_wo_first(data)
                data = data_wo_first
                mask_data = self.change_image(data,mask_image)
                embed = self.static_encoder(data)
                mod_embed = self.dynamic_encoder(mask_data)

                post, prior = self.dynamics.observe(
                    embed, mod_embed, data["action"], data["is_first"]
                ) 
                kl_free = self._config.kl_free
                d_scale = torch.tensor(1.0) 
                dyn_scale = torch.tensor(0.5) 
                rep_scale =  torch.tensor(0.1) 
                image_scale = torch.tensor(1.0)
                reward_scale = torch.tensor(1.0)
                cont_scale = torch.tensor(1.0)
                dyn_img_scale = torch.tensor(1.0)
                dyn_kl_scale = torch.tensor(1.0)
                dif_div_scale = torch.tensor(1.0)
                dif_img_rec_scale = torch.tensor(1.0)
                kl_loss, kl_value, dyn_loss, rep_loss, dyn_mod_loss, rep_mod_loss = self.dynamics.kl_loss(
                    post,prior,kl_free,dyn_scale,rep_scale,dyn_kl_scale
                )
                assert kl_loss.shape == embed.shape[:2], kl_loss.shape
                preds = {}
                for name, head in self.heads.items():
                    grad_head = name in self._config.grad_heads
                    feat = self.dynamics.get_feat(post)
                    feat = feat if grad_head else feat.detach()
                    if name == "decoder":
                        feat = self.dynamics.get_feat(post)
                        feat = feat if grad_head else feat.detach()
                        pred = head(feat)
 
                    else:
                        feat = self.dynamics.get_feat(post)
                        pred = head(feat)
                    if type(pred) is dict:
                        preds.update(pred)
                    else:
                        preds[name] = pred
                losses = {}
                model_loss = 0.
                for name, pred in preds.items():
                    loss = -pred.log_prob(data[name])
                    assert loss.shape == embed.shape[:2], (name, loss.shape)
                    losses[name] = loss
                    if name == "image":
                        model_loss += torch.mean(losses[name]) *image_scale 
                    elif name == "reward":
                        model_loss += torch.mean(losses[name]) *reward_scale
                    elif name == "cont":
                        model_loss += torch.mean(losses[name]) *cont_scale  
                pred_image = preds["image"].mode()
                mask_pred_image = self.extract_dynamic_mask(pred_image)
                dif_fiv_kl_loss = self.diff_div_reg(pred_image,data["image"]) *dif_div_scale 

 
                dif_img_rec_loss = torch.mean(((mask_pred_image[:,0:2,:] -mask_image[:,0:2,:] )**2).sum())*dif_img_rec_scale

 
                model_loss = model_loss + torch.mean( kl_loss) *d_scale \
                                    + dif_fiv_kl_loss#+ dyn_fea_loss #+recon_mask_loss*dyn_img_scale#+dif_img_rec_loss
            metrics = self._model_opt(model_loss, self.parameters())

        metrics.update({f"{name}_loss": to_np(loss) for name, loss in losses.items()})
        metrics["kl_free"] = kl_free
        metrics["d_scale"] = to_np(d_scale)
        metrics["image_scale"] = to_np(image_scale)
        metrics["reward_scale"] = to_np(reward_scale)
        metrics["cont_scale"] = to_np(cont_scale)
        metrics["dyn_kl_scale"] = to_np(dyn_kl_scale) 
        metrics["dyn_img_scale"] = to_np(dyn_img_scale)
        metrics["dif_img_rec_scale"] = to_np(dif_img_rec_scale)

        metrics["dif_div_scale"] = to_np(dif_div_scale)
        metrics["dyn_loss"] = to_np(dyn_loss)
        metrics["rep_loss"] = to_np(rep_loss)
        #metrics["dyn_fea_loss"] = to_np(dyn_fea_loss)
        metrics["dyn_mod_loss"] = to_np(dyn_mod_loss)
        metrics["rep_mod_loss"] = to_np(rep_mod_loss)
       # metrics["mask_img_rec_loss"] = to_np(recon_mask_loss)
        metrics["dif_div_kl_loss"] = to_np(dif_fiv_kl_loss)
        metrics["dif_img_rec_loss"] = to_np(dif_img_rec_loss)
        metrics["kl"] = to_np(torch.mean(kl_value))
        with torch.cuda.amp.autocast(self._use_amp):
            metrics["prior_ent"] = to_np(
                torch.mean(self.dynamics.get_dist(prior).entropy())
            )
            metrics["post_ent"] = to_np(
                torch.mean(self.dynamics.get_dist(post).entropy())
            )
            context = dict(
                embed=embed,
                feat=self.dynamics.get_feat(post),
                kl=kl_value,
                postent=self.dynamics.get_dist(post).entropy(),
            )
        post = {k: v.detach() for k, v in post.items()} 
        return post, context, metrics 

    # this function is called during both rollout and training
    def preprocess(self, obs):
        obs = obs.copy()
 
        
        obs["image"] = torch.Tensor(obs["image"]) / 255.0

        if "discount" in obs:
            obs["discount"] *= self._config.discount
            # (batch_size, batch_length) -> (batch_size, batch_length, 1)
            obs["discount"] = torch.Tensor(obs["discount"]).unsqueeze(-1)
        # 'is_first' is necesarry to initialize hidden state at training
        assert "is_first" in obs
        # 'is_terminal' is necesarry to train cont_head
        assert "is_terminal" in obs
        obs["cont"] = torch.Tensor(1.0 - obs["is_terminal"]).unsqueeze(-1)
        obs = {k: torch.Tensor(v).to(self._config.device) for k, v in obs.items()}
        return obs
    def preprocess_last_obs(self, last_obs):
        last_obs = last_obs.copy()
        last_obs["image"] = torch.Tensor(last_obs["image"]).to(self._config.device) / 255.0 
        #last_obs["cont"] = torch.Tensor(1.0 - last_obs["is_terminal"]).unsqueeze(-1)
        last_obs = {k: torch.Tensor(v).to(self._config.device) for k, v in last_obs.items()} 
        return last_obs
 

    def initial_mask_image(self, obs, last_obs):
        assert obs["image"].dim() == 4
        assert last_obs["image"].dim() == 4
        last_image = last_obs["image"].unsqueeze(0)
        current_image = obs["image"].unsqueeze(0)
 
        image = torch.cat([last_image,current_image],dim=1)
        initial_image = self.extract_dynamic_mask(image)[0]
        initial_image = initial_image[0,:].unsqueeze(0)
        return initial_image
 
 

    def diff_div_reg(self, pred_y, batch_y, tau=0.001, eps=1e-12):
        assert batch_y.dim() == 5  
        B, T,H, W,C = batch_y.shape
  
        if T <= 2:  return 0
        gap_pred_y = (pred_y[:, 1:] - pred_y[:, :-1]).reshape(B, T-1, -1) 
        gap_batch_y = (batch_y[:, 1:] - batch_y[:, :-1]).reshape(B, T-1, -1) 
        softmax_gap_p = F.softmax(gap_pred_y / tau, -1)
        softmax_gap_b = F.softmax(gap_batch_y / tau, -1)
        loss_gap = softmax_gap_p * \
            torch.log(softmax_gap_p / (softmax_gap_b + eps) + eps)
        return loss_gap.mean()
    
    def extract_dynamic_mask(self, x, dilation_size = 5):
        assert x.dim() == 5
        error = 1e-3
        dilation_kernel = torch.ones((1, 1, dilation_size, dilation_size), dtype=torch.float32).to(self._config.device)
        dilation_rate = 1
        diff = torch.abs(x[:, 1:] - x[:, :-1]) 
        dynamic_mask = (diff > error).float().to(self._config.device)

        B, T, H, W, C = dynamic_mask.shape
        dynamic_mask = dynamic_mask.permute(0, 1, 4, 2, 3).reshape(B * T * C, 1, H, W)
        dilated_mask = F.conv2d(dynamic_mask, dilation_kernel, padding=(dilation_size // 2), dilation= dilation_rate)
    
        dilated_mask = (dilated_mask > 0).float() 
        dilated_mask = dilated_mask.reshape(B, T, C, H, W).permute(0, 1, 3, 4, 2)
    
        return dilated_mask*x[:, 1:]
 
 
    def change_image(self, data, atten_image):
        change_image = data.copy()
        assert data["image"].shape == atten_image.shape
        change_image["image"] = atten_image
        return change_image

    def change_data_wo_first(self, data):
        change_data = data.copy()
        for name, tensor in change_data.items():
            try:
                change_data[name] = tensor[:, 1:]
            except IndexError:
                print(f"{name}: Tensor has fewer dimensions, cannot slice [:, 1:]")
        return change_data

 
    def video_pred(self, data):
        data = self.preprocess(data)

        mask_image = self.extract_dynamic_mask(data["image"]) 
        data_wo_first = self.change_data_wo_first(data)
        data = data_wo_first
        mask_data = self.change_image(data,mask_image)
        embed = self.static_encoder(data)
        mod_embed = self.dynamic_encoder(mask_data)
 
        states, _ = self.dynamics.observe(
            embed[:6, :5], mod_embed[:6, :5], data["action"][:6, :5], data["is_first"][:6, :5]
        )
     
        input_recon = self.dynamics.get_feat(states) 
        recon = self.heads["decoder"](input_recon)["image"].mode()[
            :6
        ]
 
        reward_post = self.heads["reward"](self.dynamics.get_feat(states)).mode()[:6] 
        init = {k: v[:, -1] for k, v in states.items()}
        prior = self.dynamics.imagine_with_action(data["action"][:6, 5:], init)
 
        
        input_openl  = self.dynamics.get_feat(prior) 
 
        openl = self.heads["decoder"](input_openl)["image"].mode()
 
        reward_prior = self.heads["reward"](self.dynamics.get_feat(prior)).mode()
        # observed image is given until 5 steps
        model = torch.cat([recon[:, :5], openl], 1) 
        truth = data["image"][:6]
        model = model
        error = (model - truth + 1.0) / 2.0 

        return torch.cat([truth, model, error, mask_image[:6] ], 2)


class ImagBehavior(nn.Module):
    def __init__(self, config, world_model):
        super(ImagBehavior, self).__init__()
        self._use_amp = True if config.precision == 16 else False
        self._config = config
        self._world_model = world_model
        if config.dyn_discrete:
            feat_size = config.dyn_deter + config.modulator_dyn_stoch * config.modulator_dyn_discrete + config.dyn_stoch * config.dyn_discrete
        else:
            feat_size = config.dyn_stoch + config.dyn_deter
        self.actor = networks.MLP(
            feat_size,
            (config.num_actions,),
            config.actor["layers"],
            config.units,
            config.act,
            config.norm,
            config.actor["dist"],
            config.actor["std"],
            config.actor["min_std"],
            config.actor["max_std"],
            absmax=1.0,
            temp=config.actor["temp"],
            unimix_ratio=config.actor["unimix_ratio"],
            outscale=config.actor["outscale"],
            name="Actor",
        )
        self.value = networks.MLP(
            feat_size,
            (255,) if config.critic["dist"] == "symlog_disc" else (),
            config.critic["layers"],
            config.units,
            config.act,
            config.norm,
            config.critic["dist"],
            outscale=config.critic["outscale"],
            device=config.device,
            name="Value",
        )
        if config.critic["slow_target"]:
            self._slow_value = copy.deepcopy(self.value)
            self._updates = 0
        kw = dict(wd=config.weight_decay, opt=config.opt, use_amp=self._use_amp)
        self._actor_opt = tools.Optimizer(
            "actor",
            self.actor.parameters(),
            config.actor["lr"],
            config.actor["eps"],
            config.actor["grad_clip"],
            **kw,
        )
        print(
            f"Optimizer actor_opt has {sum(param.numel() for param in self.actor.parameters())} variables."
        )
        self._value_opt = tools.Optimizer(
            "value",
            self.value.parameters(),
            config.critic["lr"],
            config.critic["eps"],
            config.critic["grad_clip"],
            **kw,
        )
        print(
            f"Optimizer value_opt has {sum(param.numel() for param in self.value.parameters())} variables."
        )
        if self._config.reward_EMA:
            # register ema_vals to nn.Module for enabling torch.save and torch.load
            self.register_buffer("ema_vals", torch.zeros((2,)).to(self._config.device))
            self.reward_ema = RewardEMA(device=self._config.device)

    def _train(
        self,
        start,
        objective,
    ):
        self._update_slow_target()
        metrics = {}

        with tools.RequiresGrad(self.actor):
            with torch.cuda.amp.autocast(self._use_amp):
                imag_feat, imag_state, imag_action = self._imagine(
                    start, self.actor,self._config.imag_horizon
                )
                #imag_feat_inp = self._world_model.concat_with_modulator(imag_feat,static_modulator)
                imag_feat_inp = imag_feat
                reward = objective(imag_feat_inp, imag_state, imag_action)
                actor_ent = self.actor(imag_feat_inp).entropy()
                state_ent = self._world_model.dynamics.get_dist(imag_state).entropy()
                # this target is not scaled by ema or sym_log.
                target, weights, base = self._compute_target(
                    imag_feat_inp, imag_state, reward
                )
                actor_loss, mets = self._compute_actor_loss(
                    imag_feat_inp,
                    imag_action,
                    target,
                    weights,
                    base,
                )
                actor_loss -= self._config.actor["entropy"] * actor_ent[:-1, ..., None]
                actor_loss = torch.mean(actor_loss)
                metrics.update(mets)
                value_input = imag_feat_inp

        with tools.RequiresGrad(self.value):
            with torch.cuda.amp.autocast(self._use_amp):
                value = self.value(value_input[:-1].detach())
                target = torch.stack(target, dim=1)
                # (time, batch, 1), (time, batch, 1) -> (time, batch)
                value_loss = -value.log_prob(target.detach())
                slow_target = self._slow_value(value_input[:-1].detach())
                if self._config.critic["slow_target"]:
                    value_loss -= value.log_prob(slow_target.mode().detach())
                # (time, batch, 1), (time, batch, 1) -> (1,)
                value_loss = torch.mean(weights[:-1] * value_loss[:, :, None])
        metrics.update(tools.tensorstats(value.mode(), "value"))
        metrics.update(tools.tensorstats(target, "target"))
        metrics.update(tools.tensorstats(reward, "imag_reward"))
        if self._config.actor["dist"] in ["onehot"]:
            metrics.update(
                tools.tensorstats(
                    torch.argmax(imag_action, dim=-1).float(), "imag_action"
                )
            )
        else:
            metrics.update(tools.tensorstats(imag_action, "imag_action"))
        metrics["actor_entropy"] = to_np(torch.mean(actor_ent))
        with tools.RequiresGrad(self):
            metrics.update(self._actor_opt(actor_loss, self.actor.parameters()))
            metrics.update(self._value_opt(value_loss, self.value.parameters()))
        return imag_feat, imag_state, imag_action, weights, metrics

    def _imagine(self, start, policy,horizon):
        dynamics = self._world_model.dynamics
        flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
        start = {k: flatten(v) for k, v in start.items()}

        def step(prev, _):
            state, _, _ = prev
            feat = dynamics.get_feat(state)
            inp = feat.detach() 
            input = inp
            #input = self._world_model.concat_with_modulator(inp,static_modulator)
            action = policy(input).sample()
            succ = dynamics.img_step(state, action)
            return succ, feat, action

        succ, feats, actions = tools.static_scan(
            step, [torch.arange(horizon)], (start, None, None)
        )
        states = {k: torch.cat([start[k][None], v[:-1]], 0) for k, v in succ.items()}

        return feats, states, actions

    def _compute_target(self, imag_feat, imag_state, reward):
        if "cont" in self._world_model.heads:
            inp = self._world_model.dynamics.get_feat(imag_state)
            #inp = self._world_model.concat_with_modulator(self._world_model.dynamics.get_feat(imag_state),static_modulator)
            discount = self._config.discount * self._world_model.heads["cont"](inp).mean
        else:
            discount = self._config.discount * torch.ones_like(reward)
        value = self.value(imag_feat).mode()
        target = tools.lambda_return(
            reward[1:],
            value[:-1],
            discount[1:],
            bootstrap=value[-1],
            lambda_=self._config.discount_lambda,
            axis=0,
        )
        weights = torch.cumprod(
            torch.cat([torch.ones_like(discount[:1]), discount[:-1]], 0), 0
        ).detach()
        return target, weights, value[:-1]

    def _compute_actor_loss(
        self,
        imag_feat,
        imag_action,
        target,
        weights,
        base,
    ):
        metrics = {}
        inp = imag_feat.detach()
        policy = self.actor(inp)
        # Q-val for actor is not transformed using symlog
        target = torch.stack(target, dim=1)
        if self._config.reward_EMA:
            offset, scale = self.reward_ema(target, self.ema_vals)
            normed_target = (target - offset) / scale
            normed_base = (base - offset) / scale
            adv = normed_target - normed_base
            metrics.update(tools.tensorstats(normed_target, "normed_target"))
            metrics["EMA_005"] = to_np(self.ema_vals[0])
            metrics["EMA_095"] = to_np(self.ema_vals[1])

        if self._config.imag_gradient == "dynamics":
            actor_target = adv
        elif self._config.imag_gradient == "reinforce":
            actor_target = (
                policy.log_prob(imag_action)[:-1][:, :, None]
                * (target - self.value(imag_feat[:-1]).mode()).detach()
            )
        elif self._config.imag_gradient == "both":
            actor_target = (
                policy.log_prob(imag_action)[:-1][:, :, None]
                * (target - self.value(imag_feat[:-1]).mode()).detach()
            )
            mix = self._config.imag_gradient_mix
            actor_target = mix * target + (1 - mix) * actor_target
            metrics["imag_gradient_mix"] = mix
        else:
            raise NotImplementedError(self._config.imag_gradient)
        actor_loss = -weights[:-1] * actor_target
        return actor_loss, metrics

    def _update_slow_target(self):
        if self._config.critic["slow_target"]:
            if self._updates % self._config.critic["slow_target_update"] == 0:
                mix = self._config.critic["slow_target_fraction"]
                for s, d in zip(self.value.parameters(), self._slow_value.parameters()):
                    d.data = mix * s.data + (1 - mix) * d.data
            self._updates += 1