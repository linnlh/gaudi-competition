import abc
import numpy as np
import json

from typing import List, Tuple, Dict, Type, Callable, Any
from transformers import PreTrainedTokenizer, AutoTokenizer


class DatasetBase(abc.ABC):
    """抽象基类，用于定义推理数据集的基本行为"""

    @abc.abstractmethod
    def generate(self, tokenizer: PreTrainedTokenizer) -> List[Tuple[str, int, int]]:
        """生成数据集

        抛出:
            NotImplementedError: 子类必须实现此方法

        返回值:
            List[Tuple[str, int, int]]: 生成的数据集，
            每个元素为一个元祖，包含：提示文本、输入长度和输出长度
        """
        raise NotImplementedError


class DatasetRegistry:
    """数据集注册表类"""

    _registry: Dict[str, Type[DatasetBase]] = {}

    @classmethod
    def register(cls, name: str) -> Callable:
        """注册数据集类"""

        def decorator(dataset_cls: Type[DatasetBase]) -> Type[DatasetBase]:
            cls._registry[name] = dataset_cls
            return dataset_cls

        return decorator

    @classmethod
    def get_dataset(cls, name: str) -> Type[DatasetBase]:
        """根据名称获取数据集"""

        if name in cls._registry:
            return cls._registry[name]

        return cls._registry["json"]


@DatasetRegistry.register("random")
class RandomDataset(DatasetBase):
    """随机数据集类"""

    def __init__(
        self,
        max_tokens: int = 1024,
        num_prompts: int = 1024,
        range_ratio: float = 0,
        **kwargs: Dict[str, Any]
    ) -> None:
        """初始化随机数据集

        参数:
            max_tokens (int): 最大 Token 数
            num_prompts (int): 生成的数据集数量
            range_ratio (float): 输入输出长度范围比例
        """

        self.max_tokens = max_tokens
        self.num_prompts = num_prompts
        self.range_ratio = range_ratio

    def generate(self, tokenizer: PreTrainedTokenizer) -> List[Tuple[str, int, int]]:
        """生成随机数据集"""

        start, end = int(self.max_tokens * self.range_ratio), self.max_tokens
        input_lens = np.random.randint(start, end, size=self.num_prompts)
        output_lens = np.random.randint(start, end, size=self.num_prompts)
        offsets = np.random.randint(0, tokenizer.vocab_size, size=self.num_prompts)

        inputs = []
        for index in range(self.num_prompts):
            prompt = tokenizer.decode(
                [
                    (offsets[index] + j) % tokenizer.vocab_size
                    for j in range(input_lens[index])
                ]
            )
            inputs.append((prompt, input_lens[index], output_lens[index]))
        return inputs


@DatasetRegistry.register("dummy")
class DummyDataset(DatasetBase):
    """数据集模拟类"""

    def __init__(
        self, max_tokens: int = 1024, num_prompts: int = 1024, **kwargs: Dict[str, Any]
    ) -> None:
        """初始化虚拟数据集"""

        self.prompt = kwargs["prompt"]
        self.max_tokens = max_tokens
        self.num_prompts = num_prompts

    def generate(self, tokenizer: PreTrainedTokenizer) -> List[Tuple[str, int, int]]:
        """生成随机数据集"""

        inputs = []
        prompt_len = len(tokenizer.encode(self.prompt))
        for _ in range(self.num_prompts):
            inputs.append((self.prompt, prompt_len, self.max_tokens))
        return inputs


@DatasetRegistry.register("json")
class JsonDataset(DatasetBase):
    """Json 文件数据集类"""

    def __init__(
        self,
        dataset: str,
        max_tokens: int = 1024
    ) -> None:
        """初始化虚拟数据集"""

        self.dataset = dataset
        self.max_tokens = max_tokens

    def generate(self, tokenizer: PreTrainedTokenizer) -> List[Tuple[str, int, int]]:
        """生成随机数据集"""
        
        inputs = []
        with open(self.dataset, 'r') as file:
            lines = file.readlines()
            for line in lines:
                data = json.loads(line.strip())
                prompt = data.get("content", None)
                if prompt:
                    prompt_len = len(tokenizer.encode(prompt))
                    inputs.append((prompt, prompt_len, self.max_tokens))
        return inputs


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("models", trust_remote_code=True)
    tp = "AdvertiseGen/dev.json"
    dataset_cls = DatasetRegistry.get_dataset(tp)
    dataset = dataset_cls(tp)
    requests = dataset.generate(tokenizer)
    for request in requests[:10]:
        print(request)
