#!/bin/bash
tasks=( "alien" ) 
seeds=(0)  
for i in "${!seeds[@]}"; do
    seed=${seeds[$i]}
    gpu_id=$((seed))
    echo "Seed: $seed, GPU: cuda:$gpu_id"
 
    python3 dymodreamer/dreamer.py --configs atari100k \
    --seed $seed --logdir dymodreamer/${tasks}/seed_${seed}_cuda_${gpu_id} \
    --device cuda:${gpu_id} --task atari_${tasks}  &
done 
wait