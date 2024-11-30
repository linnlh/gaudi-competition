#!/bin/sh

port=${PORT:-8000}

function wait_for_server() {
    local port=$1
    local timeout=${2:-300}
    local start_time=$(date +%s)
    local health_url="http://127.0.0.1:$port/health"

    while true; do
        if curl --output /dev/null --silent --head --fail "$health_url"; then
            echo "Server is up and running on port $port"
            break
        else
            echo "Waiting for server to start..."
        fi
        sleep 5
        local current_time=$(date +%s)
        if (( current_time - start_time > timeout )); then
            echo "Timeout: Server did not start within $timeout seconds"
            exit 1
        fi
    done
}

function main() {
    vllm serve /data/chatglm3-6b \
        --port $port \
        --enable-lora \
        --lora-module advgen=/data/chatglm3-6b-lora \
        --dtype=bfloat16 \
        --trust-remote-code
    # wait_for_server $port
}

main $@