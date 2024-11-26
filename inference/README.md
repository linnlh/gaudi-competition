# 推理脚本

## 运行
```bash
python inference.py \
    --model-dir=/xxx/models \
    --prompt="请从1数到10：1，2" \
    --num-prompts 30 \
    --request-rate 10 \
    --client dummy \
    --profile
```