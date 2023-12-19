#!/bin/bash

for ((i=25; i<=200; i+=25)); do
    echo "Outer loop value: $i"
    #for strategy in "Free"; do 
    for strategy in "DQN" "Fixed" "Random" "Free"; do
         #echo "/scratch/bk2g18/experiment_19112023_${strategy}_${i}_highway"
         sbatch create_process_results_job.slurm /scratch/bk2g18/experiment_08122023_${strategy}_${i}_highway_50 experiment_08122023_${strategy}_${i}_highway_50.json
    done
done
