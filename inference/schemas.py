from typing import List, Tuple
from dataclasses import dataclass, field

@dataclass
class RequestInput:
    """请求输入类"""

    prompt: str
    api_url: str
    prompt_len: int
    max_tokens: int
    model: str


@dataclass
class RequestOutput:
    """请求输出类"""

    generated_text: str = ""
    success: bool = False
    latency: float = 0.0
    ttft: float = 0.0
    itl: List[float] = field(default_factory=list)
    prompt_len: int = 0
    error: str = ""

@dataclass
class BenchmarkMetrics:
    """基准测试指标类"""

    completed: int
    total_input: int
    total_output: int
    request_throughput: float
    output_throughput: float
    total_token_throughput: float
    mean_ttft_ms: float
    median_ttft_ms: float
    std_ttft_ms: float
    percentiles_ttft_ms: List[Tuple[float, float]]
    mean_e2el_ms: float
    median_e2el_ms: float
    std_e2el_ms: float
    percentiles_e2el_ms: List[Tuple[float, float]]