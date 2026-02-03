#!/bin/bash 

# Make sure to run from the script directory
cd "$(dirname "$0")"

sbatch appworld/vllm/experiment_a_1.sbatch
sleep 1
sbatch appworld/vllm/experiment_a_2.sbatch
sleep 1


sbatch officebench/vllm/experiment_b_1.sbatch
sleep 1
sbatch officebench/vllm/experiment_b_2.sbatch
sleep 1


sbatch smolagents/vllm/experiment_c_1.sbatch
sleep 1
sbatch smolagents/vllm/experiment_c_2.sbatch

echo "All jobs scheduled!"
