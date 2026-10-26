"""Hardware catalog.

Every number is either taken from a public spec sheet (listed in ``sources``) or
marked as an assumption (listed in ``assumptions``). Peak compute is *dense*
(no structured sparsity). Link bandwidth is per direction, per device, on the
scale-up fabric (NVLink for GPUs, chip-to-chip links for LPUs).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

# Fallback order when a device has no native support for a dtype: the op runs
# at the next wider precision's rate.
_DTYPE_FALLBACK = {"fp4": "fp8", "fp8": "bf16", "int8": "bf16", "bf16": "fp16", "fp16": "bf16"}


@dataclass(frozen=True)
class HardwareSpec:
    name: str
    kind: str  # "gpu" or "lpu"
    peak_flops: dict[str, float]  # dense FLOP/s (or OP/s) per dtype
    mem_bw: float  # bytes/s of the memory that holds weights (HBM, or on-die SRAM for LPU)
    mem_capacity: float  # bytes
    link_bw: float  # bytes/s per direction on the scale-up fabric
    link_latency: float = 2e-6  # seconds per message on the scale-up fabric
    launch_overhead: float = 3e-6  # seconds of fixed cost per kernel / instruction group
    eff_compute: float = 0.70  # achievable fraction of peak compute in a large GEMM
    eff_memory: float = 0.85  # achievable fraction of peak memory bandwidth
    tile_m: int = 64  # GEMM tile sizes: compute is spent on padded tiles
    tile_n: int = 128
    usable_memory: float = 0.90  # fraction of capacity the runtime can use
    sources: tuple[str, ...] = field(default=())
    assumptions: tuple[str, ...] = field(default=())

    def peak(self, dtype: str) -> float:
        seen = set()
        d = dtype
        while d not in self.peak_flops:
            if d in seen or d not in _DTYPE_FALLBACK:
                raise KeyError(f"{self.name} has no peak for {dtype}")
            seen.add(d)
            d = _DTYPE_FALLBACK[d]
        return self.peak_flops[d]

    def ridge_point(self, dtype: str) -> float:
        """Arithmetic intensity (FLOP/byte) where the roofline turns flat."""
        return self.peak(dtype) / self.mem_bw

    def with_overrides(self, **kwargs) -> HardwareSpec:
        return replace(self, **kwargs)


T = 1e12
P = 1e15
GB = 1e9
TBs = 1e12

CATALOG: dict[str, HardwareSpec] = {
    "t4": HardwareSpec(
        name="NVIDIA T4",
        kind="gpu",
        peak_flops={"fp16": 65 * T, "int8": 130 * T, "fp32": 8.1 * T},
        mem_bw=320 * GB,
        mem_capacity=16 * GB,
        link_bw=16 * GB,  # PCIe Gen3 x16, no NVLink
        link_latency=10e-6,
        launch_overhead=5e-6,
        sources=("NVIDIA T4 Tensor Core GPU datasheet: 65 FP16 TFLOPS (mixed precision), 8.1 FP32 TFLOPS, 320 GB/s GDDR6, 16 GB",),
        assumptions=("No bf16 on Turing: bf16 work runs at the fp16 rate",),
    ),
    "a100": HardwareSpec(
        name="NVIDIA A100 SXM 80GB",
        kind="gpu",
        peak_flops={"bf16": 312 * T, "fp16": 312 * T, "int8": 624 * T},
        mem_bw=2.039 * TBs,
        mem_capacity=80 * GB,
        link_bw=300 * GB,
        sources=("NVIDIA A100 datasheet: 312 TFLOPS BF16 dense, 2,039 GB/s, NVLink 600 GB/s total",),
    ),
    "h100": HardwareSpec(
        name="NVIDIA H100 SXM",
        kind="gpu",
        peak_flops={"bf16": 989 * T, "fp16": 989 * T, "fp8": 1979 * T, "int8": 1979 * T},
        mem_bw=3.35 * TBs,
        mem_capacity=80 * GB,
        link_bw=450 * GB,
        sources=("NVIDIA H100 SXM spec: ~990 TFLOPS BF16 / 1,979 FP8 dense, 3.35 TB/s HBM3, 80 GB, NVLink 900 GB/s total",),
    ),
    "h200": HardwareSpec(
        name="NVIDIA H200 SXM",
        kind="gpu",
        peak_flops={"bf16": 989 * T, "fp16": 989 * T, "fp8": 1979 * T, "int8": 1979 * T},
        mem_bw=4.8 * TBs,
        mem_capacity=141 * GB,
        link_bw=450 * GB,
        sources=("NVIDIA H200 spec: H100 compute, 141 GB HBM3e at 4.8 TB/s",),
    ),
    "b200": HardwareSpec(
        name="NVIDIA B200 (HGX, 180GB)",
        kind="gpu",
        peak_flops={"bf16": 2.25 * P, "fp16": 2.25 * P, "fp8": 4.5 * P, "fp4": 9 * P, "int8": 4.5 * P},
        mem_bw=7.7 * TBs,
        mem_capacity=180 * GB,
        link_bw=900 * GB,
        sources=("Lenovo Press LP2226, B200 180GB: 2.25 PF BF16 / 4.5 PF FP8 / 9 PF FP4 dense, 7.7 TB/s, NVLink5 1.8 TB/s total",),
    ),
    "groqchip1": HardwareSpec(
        name="Groq LPU v1 (GroqChip)",
        kind="lpu",
        peak_flops={"fp16": 188 * T, "int8": 750 * T},
        mem_bw=80 * TBs,  # on-die SRAM
        mem_capacity=230e6,
        link_bw=100 * GB,
        link_latency=0.5e-6,
        launch_overhead=0.0,  # statically scheduled, no kernel launches
        eff_compute=0.80,
        eff_memory=0.90,
        tile_m=1,
        tile_n=1,
        usable_memory=0.90,
        sources=("GroqChip product brief via flopper.io: 230 MB SRAM, 80 TB/s on-die, 188 FP16 TFLOPS, 750 INT8 TOPS, 16 RealScale links",),
        assumptions=(
            "Chip-to-chip bandwidth 100 GB/s per direction (not published per link)",
            "0.5 us per chip-to-chip hop",
        ),
    ),
    "groq3": HardwareSpec(
        name="NVIDIA Groq 3 LPU",
        kind="lpu",
        peak_flops={"fp8": 1.5 * P, "bf16": 0.75 * P, "fp16": 0.75 * P, "int8": 1.5 * P},
        mem_bw=150 * TBs,
        mem_capacity=500e6,
        link_bw=312.5 * GB,  # 640 Tb/s across 256 chips
        link_latency=0.5e-6,
        launch_overhead=0.0,
        eff_compute=0.80,
        eff_memory=0.90,
        tile_m=1,
        tile_n=1,
        sources=(
            "SDxCentral (GTC 2026): ~500 MB SRAM per LPU, 22-150 TB/s SRAM bandwidth, 256 LPUs per LPX rack at 640 Tb/s scale-up",
        ),
        assumptions=(
            "SRAM bandwidth taken at the top of the quoted 22-150 TB/s range",
            "Compute not published: assumed 1.5 PFLOPS FP8 / 0.75 PFLOPS BF16 (2x GroqChip v1 INT8)",
            "0.5 us per chip-to-chip hop",
        ),
    ),
}


def get(name: str) -> HardwareSpec:
    try:
        return CATALOG[name.lower()]
    except KeyError as e:
        raise KeyError(f"unknown hardware {name!r}; choose from {sorted(CATALOG)}") from e
