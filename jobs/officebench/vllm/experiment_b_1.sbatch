#!/bin/bash
#SBATCH -p lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH -o output_%x_%j.out
#SBATCH --job-name=experiment_b_1

LLM="Qwen/Qwen3-8B"
BENCHMARK="officebench"
DATA_SPLIT="train"
TAG_NAME="baseline"
VLLM_USE_V1=1
PROVIDER="vllm"
VLLM_PORT="8000"
TENSOR_PARALLEL=1

cd "$(dirname "$0")" || exit

nvidia-smi
enroot create /dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2/acon/acon_v3.sqsh

# Launch ACON evals
enroot start \
    --root \
    -e HF_TOKEN_PATH=/dss/dsshome1/09/ge56heh2/.cache/huggingface/token \
    -e HF_HOME=/dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2/.cache/hf/misc \
    -e HF_DATASETS_CACHE=/dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2/.cache/datasets \
    -e HF_DATASETS_TRUST_REMOTE_CODE=True \
    -e VLLM_USE_V1="${VLLM_USE_V1}" \
    -e LLM="${LLM}" \
    -e PROVIDER="${PROVIDER}" \
    -e VLLM_PORT="${VLLM_PORT}" \
    -e TENSOR_PARALLEL="${TENSOR_PARALLEL}" \
    -e BENCHMARK="${BENCHMARK}" \
    -e DATA_SPLIT="${DATA_SPLIT}" \
    -e SLURM_JOB_NAME="${SLURM_JOB_NAME}" \
    -e TAG_NAME="${TAG_NAME}" \
    --mount /dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2/ \
    --mount /dss/dsshome1/09/ge56heh2/ \
    acon_v3 /bin/bash /dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2/acon/scripts/officebench_launcher.sh
