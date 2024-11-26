#!/bin/sh

port=8000

function usage() {
    echo "TODO: filling the usage"
}

function parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --model | -m)
                model=$2
                shift 2
                ;;
            --port | -p)
                port=$2
                shift 2
                ;;
            --help)
                usage
                shift
                ;;
            *)
                echo "Invalid option: $1"
                usage
                exit 1
                ;;
        esac
    done
}

function wait_for_server() {
    local port=$1
    local timeout=${2:-30}
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
    parse_args $@
    # vllm serve $model
    wait_for_server $port
}

main $@