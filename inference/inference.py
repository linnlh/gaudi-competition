import asyncio
import argparse
import client
import datasets
import numpy as np
import schemas
import time
import transformers

from typing import AsyncGenerator, List, Tuple, Type


async def get_request(
    inputs: List[Tuple[str, int, int]],
    request_rate: float,
    burstiness: float = 1.0,
) -> AsyncGenerator[Tuple[str, int, int], None]:
    """以指定请求速率异步的获取请求数据

    参数:
        inputs (List[Tuple[str, int, int]]): 输入请求列表
        request_rate (float): 请求速率
        burstiness (float, optional): 请求突发性，默认值为 1.0

    返回:
        AsyncGenerator[Tuple[str, int, int]]: 异步生成器、生成请求元祖

    生成:
        Iterator[AsyncGenerator[Tuple[str, int, int]]]: 生成请求元祖
    """

    assert burstiness > 0, f"突发值只能设为正值，burstiness：{burstiness}"

    theta = 1.0 / (request_rate * burstiness)
    inputs = iter(inputs)
    for request in inputs:
        yield request

        # 请求速率无穷大则无需等待
        if request_rate == float("inf"):
            continue

        # 使用 Gamma 分布来模拟请求间隔，如果 burstiness = 1，则请求
        # 之前没有关联性
        interval = np.random.gamma(shape=burstiness, scale=theta)
        await asyncio.sleep(interval)


def calculate_metric(
    inputs: List[Tuple[str, int, int]],
    outputs: List[schemas.RequestOutput],
    dur_s: float,
    tokenizer: transformers.PreTrainedTokenizer,
    percentiles: List[float] = [0.5, 0.9, 0.99],
) -> schemas.BenchmarkMetrics:
    """计算性能指标

    参数:
        inputs (List[Tuple[str, int, int]]): 输入请求列表
        outputs (List[schemas.RequestOutput]): 输出结果列表
        dur_s (float): 总耗时
        toktokenizer (transformers.PreTrainedTokenizer): 分词器
        percentiles: (List[float]): 计算百分位

    返回:
        schemas.BenchmarkMetrics: 性能指标
    """

    actual_output_lens: List[int] = []
    total_input = 0
    completed = 0
    ttfts: List[float] = []
    e2els: List[float] = []
    for idx, output in enumerate(outputs):
        if output.success:
            output_len = len(
                tokenizer(output.generated_text, add_special_tokens=False).input_ids
            )
            actual_output_lens.append(output_len)
            total_input += inputs[idx][1]

            ttfts.append(output.ttft)
            e2els.append(output.latency)
            completed += 1
        else:
            actual_output_lens.append(0)

    metric = schemas.BenchmarkMetrics(
        completed=completed,
        total_input=total_input,
        total_output=sum(actual_output_lens),
        request_throughput=completed / dur_s,
        output_throughput=sum(actual_output_lens) / dur_s,
        total_token_throughput=(total_input + sum(actual_output_lens)) / dur_s,
        mean_ttft_ms=np.mean(ttfts or 0) * 1000,
        std_ttft_ms=np.std(ttfts or 0) * 1000,
        median_ttft_ms=np.median(ttfts or 0) * 1000,
        percentiles_ttft_ms=[
            (p, np.percentile(ttfts or 0, p) * 1000) for p in percentiles
        ],
        mean_e2el_ms=np.mean(e2els or 0) * 1000,
        std_e2el_ms=np.std(e2els or 0) * 1000,
        median_e2el_ms=np.median(e2els or 0) * 1000,
        percentiles_e2el_ms=[
            (p, np.percentile(e2els or 0, p) * 1000) for p in percentiles
        ],
    )

    return metric


async def inference(
    model: str,
    tokenizer: transformers.PreTrainedTokenizer,
    client: Type[client.ClientBase],
    inputs: List[Tuple[str, int, int]],
    base_url: str,
    request_rate: float,
    burstiness: float,
    profile: bool,
) -> List[schemas.RequestOutput]:
    """异步推理函数"""

    inference_start_time = time.perf_counter()
    tasks = []
    async for request in get_request(inputs, request_rate, burstiness):
        prompt, prompt_len, max_tokens = request
        inp = schemas.RequestInput(
            prompt=prompt,
            api_url=base_url,
            prompt_len=prompt_len,
            max_tokens=max_tokens,
            model=model,
        )
        task = asyncio.create_task(client.completions_async(inp))
        tasks.append(task)
    outputs = await asyncio.gather(*tasks)
    inference_duration = time.perf_counter() - inference_start_time

    if profile:
        metrics = calculate_metric(
            inputs, outputs, inference_duration, tokenizer, percentiles=[0.5, 0.9, 0.99]
        )

        def process_one_metric(
            metric_attr_name: str, metric_name: str, metric_header: str
        ):
            print("{s:{c}^{n}}".format(s=metric_header, n=50, c="-"))
            print(
                "{:<40} {:<10.2f}".format(
                    f"{metric_name}平均值 (ms):",
                    getattr(metrics, f"mean_{metric_attr_name}_ms"),
                )
            )
            print(
                "{:<40} {:<10.2f}".format(
                    f"{metric_name}中位数 (ms):",
                    getattr(metrics, f"median_{metric_attr_name}_ms"),
                )
            )
            for p, value in getattr(metrics, f"percentiles_{metric_attr_name}_ms"):
                p_word = int(p * 100)
                print(
                    "{:<40} {:<10.2f}".format(f"{metric_name} P{p_word}  (ms):", value)
                )

        print("{s:{c}^{n}}".format(s=" 推理性能测试结果 ", n=50, c="="))
        print("{:<40} {:<10.2f}".format("推理总耗时 (s)：", inference_duration))
        print("{:<40} {:<10}".format("请求成功数量：", metrics.completed))
        print("{:<40} {:<10}".format("总的输入 Token 数量：", metrics.total_input))
        print("{:<40} {:<10}".format("总的生成 Token 数量：", metrics.total_output))
        print(
            "{:<40} {:<10.2f}".format("请求吞吐率 (req/s):", metrics.request_throughput)
        )
        print(
            "{:<40} {:<10.2f}".format(
                "新生成 Token 吞吐率 (tok/s):", metrics.output_throughput
            )
        )
        print(
            "{:<40} {:<10.2f}".format(
                "总 Token 吞吐率 (tok/s):", metrics.total_token_throughput
            )
        )
        process_one_metric("ttft", "首包延迟", "首包延迟（TTFT）")
        process_one_metric("e2el", "端到端延迟", "端到端延迟（E2EL）")
        print("=" * 50)
    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="local path of model"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="the host of model service"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="the port of model service"
    )
    parser.add_argument(
        "--client",
        type=str,
        default="vllm",
        help="the type of client"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="dummy",
        help="the type of datasets"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="max tokens of generated text"
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=1024,
        help="request prompt numbers"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=False,
        help="sample prompt, set when dataset's type is dummy",
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=1.0,
        help="request rate"
    )
    parser.add_argument(
        "--burstiness",
        type=float,
        default=1.0,
        help="burstiness of requests"
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="whether to profile"
    )

    args = parser.parse_args()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_dir, trust_remote_code=True
    )

    dataset_cls = datasets.DatasetRegistry.get_dataset(args.datasets)
    dataset = dataset_cls(**vars(args))
    inputs = dataset.generate(tokenizer)

    base_url = f"http://{args.host}:{args.port}"
    client_cls = client.ClientRegistry.get_client(args.client)
    asyncio.run(
        inference(
            args.model_dir,
            tokenizer,
            client_cls,
            inputs,
            base_url,
            request_rate=args.request_rate,
            burstiness=args.burstiness,
            profile=args.profile,
        )
    )


if __name__ == "__main__":
    main()
