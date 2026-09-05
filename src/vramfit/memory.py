"""Pure single-GPU weight and full-attention KV budget calculations."""

from dataclasses import dataclass
from fractions import Fraction
from math import ceil, isfinite

from vramfit.models import DecoderSpec, ParameterEstimate

GIB = 1024 ** 3
DTYPE_BYTES = {"float32": 4, "float16": 2, "bfloat16": 2}
RUNTIME_WARNING = (
    "Known memory includes learned weights and KV cache only. Activations, CUDA "
    "context, kernels, allocator/workspace costs, cache block rounding and runtime "
    "buffers are unmodelled. Fitting this budget is not a runtime guarantee."
)


@dataclass(frozen=True)
class Workload:
    prompt_length: int
    max_output_length: int
    target_concurrency: int = 1
    dtype: str = "float16"

    def __post_init__(self) -> None:
        for name in ("prompt_length", "max_output_length", "target_concurrency"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if not isinstance(self.dtype, str) or self.dtype not in DTYPE_BYTES:
            raise ValueError("dtype must be float32, float16 or bfloat16.")

    @property
    def tokens_per_request(self) -> int:
        return self.prompt_length + self.max_output_length


@dataclass(frozen=True)
class GPUCapacity:
    vram_gib: float
    headroom_percent: float = 10

    def __post_init__(self) -> None:
        for name in ("vram_gib", "headroom_percent"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not isfinite(value):
                raise ValueError(f"{name} must be a finite number.")
        if self.vram_gib <= 0:
            raise ValueError("vram_gib must be positive physical GPU capacity.")
        if not 0 <= self.headroom_percent < 100:
            raise ValueError("headroom_percent must be at least 0 and below 100.")


@dataclass(frozen=True)
class MemoryEstimate:
    physical_vram_bytes: int
    reserved_headroom_bytes: int
    usable_vram_bytes: int
    weight_bytes: int | None
    kv_bytes_per_token: int
    kv_bytes_per_request: int
    workload_kv_bytes: int
    known_memory_bytes: int | None
    known_headroom_bytes: int | None
    weights_fit: bool | None
    workload_fits: bool | None
    theoretical_max_concurrency: int | None
    assumptions: tuple[str, ...]
    warnings: tuple[str, ...]


def calculate_memory(
    spec: DecoderSpec, parameters: ParameterEstimate,
    workload: Workload, gpu: GPUCapacity,
) -> MemoryEstimate:
    """Budget known components, preserving unknown counts instead of guessing.

    Decimal input spelling is converted to exact fractions. Physical capacity
    rounds down and the reserve rounds up to bytes, avoiding an optimistic fit
    caused by floating-point rounding at a boundary.
    """
    if workload.tokens_per_request > spec.max_context_length:
        raise ValueError(
            f"Prompt plus maximum output ({workload.tokens_per_request} tokens) "
            f"exceeds the configured context limit ({spec.max_context_length} tokens)."
        )
    element_bytes = DTYPE_BYTES[workload.dtype]
    physical = int(Fraction(str(gpu.vram_gib)) * GIB)
    reserved = ceil(physical * Fraction(str(gpu.headroom_percent)) / 100)
    usable = physical - reserved
    kv_per_token = 2 * spec.num_layers * spec.num_key_value_heads * spec.head_dim * element_bytes
    kv_per_request = kv_per_token * workload.tokens_per_request
    kv_workload = kv_per_request * workload.target_concurrency
    count = parameters.learned_parameter_count
    weights = None if count is None else count * element_bytes
    known = None if weights is None else weights + kv_workload
    remaining = None if known is None else usable - known
    max_concurrency = None if weights is None else max(0, (usable - weights) // kv_per_request)
    assumptions = spec.assumptions + parameters.assumptions + (
        f"Weights and KV cache both use {workload.dtype} ({element_bytes} bytes per element).",
        "Each concurrent request retains prompt plus maximum output tokens in a full-attention KV cache.",
        "Theoretical concurrency is a memory upper bound, not a throughput or latency promise.",
    )
    return MemoryEstimate(
        physical_vram_bytes=physical, reserved_headroom_bytes=reserved,
        usable_vram_bytes=usable, weight_bytes=weights,
        kv_bytes_per_token=kv_per_token, kv_bytes_per_request=kv_per_request,
        workload_kv_bytes=kv_workload, known_memory_bytes=known,
        known_headroom_bytes=remaining,
        weights_fit=None if weights is None else weights <= usable,
        workload_fits=None if known is None else known <= usable,
        theoretical_max_concurrency=max_concurrency,
        assumptions=assumptions, warnings=parameters.warnings + (RUNTIME_WARNING,),
    )
