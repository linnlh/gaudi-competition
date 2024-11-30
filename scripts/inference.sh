#!/bin/bash

python inference/inference.py \
    --model-dir=/data/chatglm3-6b \
    --datasets=AdvertiseGen/dev.json \
    --request-rate 3 \
    --client vllm \
    --profile