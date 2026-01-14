#!/bin/bash

# Default values
MODEL_NAME=""
HOST="0.0.0.0"
PORT=8000
GPU_MEMORY_UTILIZATION=0.9
TENSOR_PARALLEL_SIZE=1
MAX_MODEL_LEN=""
DTYPE="auto"
TRUST_REMOTE_CODE=""
QUANTIZATION=""

# Help function
show_help() {
    cat << EOF
Usage: ${0##*/} --model MODEL_NAME [OPTIONS]

Required:
    --model MODEL              Model name or path (e.g., meta-llama/Llama-2-7b-hf)

Optional:
    --host HOST                Host address (default: 0.0.0.0)
    --port PORT                Port number (default: 8000)
    --gpu-memory GPU_MEM       GPU memory utilization 0.0-1.0 (default: 0.9)
    --tensor-parallel SIZE     Tensor parallel size (default: 1)
    --max-model-len LEN        Maximum model length
    --dtype DTYPE              Data type: auto, float16, bfloat16, float32 (default: auto)
    --trust-remote-code        Trust remote code
    --quantization METHOD      Quantization method: awq, gptq, squeezellm
    --help                     Show this help message

Example:
    ${0##*/} --model meta-llama/Llama-2-7b-hf --port 8000 --tensor-parallel 2
EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_NAME="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --gpu-memory)
            GPU_MEMORY_UTILIZATION="$2"
            shift 2
            ;;
        --tensor-parallel)
            TENSOR_PARALLEL_SIZE="$2"
            shift 2
            ;;
        --max-model-len)
            MAX_MODEL_LEN="$2"
            shift 2
            ;;
        --dtype)
            DTYPE="$2"
            shift 2
            ;;
        --trust-remote-code)
            TRUST_REMOTE_CODE="--trust-remote-code"
            shift
            ;;
        --quantization)
            QUANTIZATION="$2"
            shift 2
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Check if model name is provided
if [ -z "$MODEL_NAME" ]; then
    echo "Error: --model is required"
    show_help
    exit 1
fi

# Build the command
CMD="python -m vllm.entrypoints.openai.api_server \
    --model ${MODEL_NAME} \
    --host ${HOST} \
    --port ${PORT} \
    --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION} \
    --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} \
    --dtype ${DTYPE}"

# Add optional parameters
if [ -n "$MAX_MODEL_LEN" ]; then
    CMD="${CMD} --max-model-len ${MAX_MODEL_LEN}"
fi

if [ -n "$TRUST_REMOTE_CODE" ]; then
    CMD="${CMD} ${TRUST_REMOTE_CODE}"
fi

if [ -n "$QUANTIZATION" ]; then
    CMD="${CMD} --quantization ${QUANTIZATION}"
fi

# Print the command
echo "Starting vLLM server with command:"
echo "${CMD}"
echo ""

# Run the command
eval ${CMD}