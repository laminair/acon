#!/bin/bash

# Function to cleanup vllm
cleanup() {
    echo "Cleaning up vllm process..."
    if [ ! -z "$VLLM_PID" ] && kill -0 $VLLM_PID 2>/dev/null; then
        echo "Terminating vllm (PID: $VLLM_PID)..."
        kill $VLLM_PID

        # Wait up to 10 seconds for graceful shutdown
        for i in {1..10}; do
            if ! kill -0 $VLLM_PID 2>/dev/null; then
                echo "vllm terminated gracefully"
                return
            fi
            sleep 1
        done

        # Force kill if still running
        if kill -0 $VLLM_PID 2>/dev/null; then
            echo "Force killing vllm..."
            kill -9 $VLLM_PID
        fi
    fi
}

# Set up trap to catch interrupts and exit
trap cleanup EXIT INT TERM

cd /dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2/acon/experiments/"${BENCHMARK}"/

# Start vllm in background
echo "Starting vllm server..."
uv run vllm serve \
    --model "${LLM}" \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 1 \
    --dtype auto \
    --max-model-len 131072 &

VLLM_PID=$!
echo "vllm started with PID: $VLLM_PID"

# Wait for vllm to be ready (health check)
echo "Waiting for vllm server to be ready..."
MAX_WAIT=120
WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "vllm server is ready!"
        break
    fi
    sleep 2
    WAITED=$((WAITED + 2))
    echo "Waiting... ($WAITED/$MAX_WAIT seconds)"
done

if [ $WAITED -ge $MAX_WAIT ]; then
    echo "ERROR: vllm server failed to start within $MAX_WAIT seconds"
    exit 1
fi

# Run your main script
echo "Running ACON experiments..."
uv run run_all.py \
    --split "${DATA_SPLIT}" \
    --model_name "${LLM}" \
    --tag baseline \
    --base-config /dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2/acon/experiments/"${BENCHMARK}"/configs/base_config.yaml \
    --co_config_path /dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2/acon/configs/"${BENCHMARK}"/"${PROVIDER}"/"${SLURM_JOB_NAME}".yaml \
    --debug

# Capture exit code
EXIT_CODE=$?

echo "Experiments completed with exit code: $EXIT_CODE"

# Cleanup happens automatically via trap
exit $EXIT_CODE