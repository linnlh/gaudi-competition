import abc
import aiohttp
import json
import schemas
import sys
import time
import traceback

from typing import Callable, Dict, Type

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=30)

class ClientBase(abc.ABC):
    """大模型客户端基类，定义异步完成请求的抽象方法

    参数:
        abc (abc.ABC): 抽象基类，用于定义接口
    """

    @classmethod
    @abc.abstractmethod
    async def completions_async(
        cls,
        request: schemas.RequestInput
    ) -> schemas.RequestOutput:
        """异步完成请求的抽象方法。

        参数:
            request (schemas.RequestInput): 请求输入对象。

        返回:
            schemas.RequestOutput: 请求输出对象。
        """
        raise NotImplementedError


class ClientRegistry:
    """客户端注册表类"""

    _registry: Dict[str, Type[ClientBase]] = {}

    @classmethod
    def register(cls, name: str) -> Callable:
        """注册客户端"""

        def decorator(dataset_cls: Type[ClientBase]) -> Type[ClientBase]:
            cls._registry[name] = dataset_cls
            return dataset_cls

        return decorator

    @classmethod
    def get_client(cls, name: str) -> Type[ClientBase]:
        """根据名称获取服务端"""

        if name not in cls._registry:
            raise ValueError(f"未注册的服务端：'\{name}'")
        return cls._registry[name]

    
@ClientRegistry.register("vllm")
class VllmClient(ClientBase):
    """
    Vllm客户端类，继承自ClientBase，实现了异步完成请求的方法
    """

    @classmethod
    async def completions_async(
        cls,
        request: schemas.RequestInput
    ) -> schemas.RequestOutput:
        """异步完成请求的方法

        参数:
            request (schemas.RequestInput): 请求输入对象，包含 API URL、
            模型名称、提示文本等信息

        返回:
            schemas.RequestOutput: 请求输出对象，包含生成的文本、延迟时间、成功状态等
        """
        api_url = f"{request.api_url}/v1/completions"
        print("api url: ", api_url)
        async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
            payload = {
                "model": request.model,
                "prompt": request.prompt,
                "temperature": 0.0,
                "max_tokens": request.max_tokens,
                "stream": True,
            }

            output = schemas.RequestOutput()
            output.prompt_len = request.prompt_len

            generated_text = ""
            ttft = 0.0
            st = time.perf_counter()
            most_recent_timestamp = st
            try:
                async with session.post(url=api_url, json=payload) as resp:
                    if resp.status == 200:
                        first_chunk_received = False
                        async for chunk_bytes in resp.content:
                            chunk_bytes = chunk_bytes.strip()
                            if not chunk_bytes:
                                continue
                            chunk = chunk_bytes.decode("utf-8").removeprefix("data: ")
                            if chunk == "[DONE]":
                                latency = time.perf_counter() - st
                            else:
                                data = json.loads(chunk)

                            if data["choices"][0]["text"]:
                                timestamp = time.perf_counter()
                                # First token
                                if not first_chunk_received:
                                    first_chunk_received = True
                                    ttft = time.perf_counter() - st
                                    output.ttft = ttft
                                # Decoding phase
                                else:
                                    output.itl.append(timestamp - most_recent_timestamp)
                                    most_recent_timestamp = timestamp
                                    generated_text += data["choices"][0]["text"]
                        if first_chunk_received:
                            output.success = True
                        else:
                            output.success = False
                            output.error = (
                                "Never received a valid chunk to calculate TTFT."
                                "This response will be marked as failed!"
                            )
                        output.generated_text = generated_text
                        output.latency = latency
                    else:
                        output.error = resp.reason or ""
                        output.success = False
            except Exception:
                output.success = False
                exc_info = sys.exc_info()
                output.error = "".join(traceback.format_exception(*exc_info))

            return output


@ClientRegistry.register("dummy")
class DummyClient(ClientBase):
    """客户端模拟类"""

    @classmethod
    async def completions_async(
        cls,
        request: schemas.RequestInput
    ) -> schemas.RequestOutput:
        """
        异步完成请求的方法，模拟生成文本的过程。

        参数:
            request (schemas.RequestInput): 请求输入对象，包含API URL、模型名称、提示文本等信息。

        返回:
            schemas.RequestOutput: 请求输出对象，包含生成的文本、延迟时间、成功状态等信息。
        """

        output = schemas.RequestOutput()
        output.prompt_len = request.prompt_len

        # 模拟生成文本的过程
        generated_text = "Dummy"
        ttft = 0.1
        latency = 0.2

        # 设置输出对象的属性
        output.generated_text = generated_text
        output.ttft = ttft
        output.latency = latency
        output.success = True

        return output
        