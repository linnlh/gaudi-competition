# 推理脚本

## Run vllm server

```bash
export VLLM_SKIP_WARMUP="true"
vllm serve /data/models/ --enable-lora --lora-module advgen=/root/codes/finetune-glm3/output --dtype=bfloat16 --trust_remote_code
```

## Test with curl

```bash
curl http://localhost:8000/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
  "model": "/data/models/",
  "messages": [
    {"role": "system", "content": "你是广告文案大师，请你根据以下内容帮我生成广告："},
    {"role": "user", "content": "类型#上衣*材质#牛仔布*颜色#白色*风格#简约*图案#刺绣*衣样式#外套*衣款式#破洞"}
  ],
  "max_tokens": 150,
  "temperature": 0.7
}'
```

## 运行
```bash
python inference/inference.py \
    --model-dir=/data/models/ \
    --datasets=AdvertiseGen/dev.json \
    --request-rate 3 \
    --client vllm \
    --profile
```

## request-rate = 2
```
==================== 推理性能测试结果 ====================
推理总耗时 (s)：                               577.91    
请求成功数量：                                  1068      
总的输入 Token 数量：                           42209     
总的生成 Token 数量：                           283104    
请求吞吐率 (req/s):                           1.85      
新生成 Token 吞吐率 (tok/s):                   489.88    
总 Token 吞吐率 (tok/s):                     562.91    
--------------------首包延迟（TTFT）--------------------
首包延迟平均值 (ms):                            121.38    
首包延迟中位数 (ms):                            29.40     
首包延迟 P50  (ms):                          22.31     
首包延迟 P90  (ms):                          22.53     
首包延迟 P99  (ms):                          22.56     
-------------------端到端延迟（E2EL）--------------------
端到端延迟平均值 (ms):                           3958.17   
端到端延迟中位数 (ms):                           1702.87   
端到端延迟 P50  (ms):                         80.89     
端到端延迟 P90  (ms):                         83.07     
端到端延迟 P99  (ms):                         83.71     
==================================================
```

## request-rate = 5
```
==================== 推理性能测试结果 ====================
推理总耗时 (s)：                               232.09    
请求成功数量：                                  1038      
总的输入 Token 数量：                           40982     
总的生成 Token 数量：                           252943    
请求吞吐率 (req/s):                           4.47      
新生成 Token 吞吐率 (tok/s):                   1089.84   
总 Token 吞吐率 (tok/s):                     1266.42   
--------------------首包延迟（TTFT）--------------------
首包延迟平均值 (ms):                            757.40    
首包延迟中位数 (ms):                            30.93     
首包延迟 P50  (ms):                          22.30     
首包延迟 P90  (ms):                          22.52     
首包延迟 P99  (ms):                          22.53     
-------------------端到端延迟（E2EL）--------------------
端到端延迟平均值 (ms):                           5131.57   
端到端延迟中位数 (ms):                           1985.33   
端到端延迟 P50  (ms):                         82.05     
端到端延迟 P90  (ms):                         88.92     
端到端延迟 P99  (ms):                         89.39     
==================================================
```