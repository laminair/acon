#!/bin/bash

# Function to cleanup processes
# shellcheck disable=SC2329
cleanup() {
    echo "Cleaning up processes..."

    # Cleanup vllm if running
    if [ ! -z "$VLLM_PID" ] && kill -0 $VLLM_PID 2>/dev/null; then
        echo "Terminating vllm (PID: $VLLM_PID)..."
        kill $VLLM_PID

        # Wait up to 10 seconds for graceful shutdown
        for i in {1..10}; do
            if ! kill -0 $VLLM_PID 2>/dev/null; then
                echo "vllm terminated gracefully"
                break
            fi
            sleep 1
        done

        # Force kill if still running
        if kill -0 $VLLM_PID 2>/dev/null; then
            echo "Force killing vllm..."
            kill -9 $VLLM_PID
        fi
    fi

    # Cleanup retriever server if running
    if [ ! -z "$RETRIEVER_PID" ] && kill -0 $RETRIEVER_PID 2>/dev/null; then
        echo "Terminating retriever server (PID: $RETRIEVER_PID)..."
        kill $RETRIEVER_PID

        # Wait up to 10 seconds for graceful shutdown
        for i in {1..10}; do
            if ! kill -0 $RETRIEVER_PID 2>/dev/null; then
                echo "Retriever server terminated gracefully"
                break
            fi
            sleep 1
        done

        # Force kill if still running
        if kill -0 $RETRIEVER_PID 2>/dev/null; then
            echo "Force killing retriever server..."
            kill -9 $RETRIEVER_PID
        fi
    fi
}

# Set up trap to catch interrupts and exit
trap cleanup EXIT INT TERM

cd /dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2/acon/experiments/"${BENCHMARK}"/ || exit

# Start retriever server in background
echo "Starting retriever server..."
uv run /dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2/acon/search/retriever_server.py \
    --index_path /dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2/acon/search/database/wikipedia/bm25 &
RETRIEVER_PID=$!
echo "Retriever server started with PID: $RETRIEVER_PID"

# Wait for retriever server to be ready
echo "Waiting for retriever server to be ready..."
MAX_WAIT=60
WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
    if curl -s http://localhost:8005 > /dev/null 2>&1; then
        echo "Retriever server is ready!"
        break
    fi
    sleep 2
    WAITED=$((WAITED + 2))
    echo "Waiting... ($WAITED/$MAX_WAIT seconds)"
done

if [ $WAITED -ge $MAX_WAIT ]; then
    echo "ERROR: Retriever server failed to start within $MAX_WAIT seconds"
    exit 1
fi

# Start vllm in background
echo "Starting vllm server..."
uv run vllm serve \
    --model "${LLM}" \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 1 \
    --dtype auto \
    --max-model-len 38912 &
VLLM_PID=$!
echo "vllm started with PID: $VLLM_PID"

# Wait for vllm to be ready (health check)
echo "Waiting for vllm server to be ready..."
MAX_WAIT=90
WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
    if curl -s http://localhost:8000 > /dev/null 2>&1; then
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

# Run smolagents experiments
echo "Running smolagents experiments..."
uv run run.py \
    --split "${DATA_SPLIT}" \
    --model_name "${LLM}" \
    --tag "${TAG_NAME}" \
    --appworld-config /dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2/acon/experiments/"${BENCHMARK}"/configs/base_config.yaml \
    --global-config /dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2/acon/configs/global_config.yaml \
    --co_config_path /dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2/acon/configs/"${BENCHMARK}"/"${PROVIDER}"/"${SLURM_JOB_NAME}".yaml

# Capture exit code
EXIT_CODE=$?

echo "Experiments completed with exit code: $EXIT_CODE"

# Cleanup happens automatically via trap
exit $EXIT_CODE
