import torch

from sglang_omni.platforms import PlatformEnum, ResolvedPlatformSpec

CUDA_PLATFORM_SPEC = ResolvedPlatformSpec(PlatformEnum.CUDA, "cuda", "nccl")
CPU_PLATFORM_SPEC = ResolvedPlatformSpec(PlatformEnum.CPU, "cpu", "gloo")
XPU_PLATFORM_SPEC = ResolvedPlatformSpec(PlatformEnum.XPU, "xpu", "xccl")


def accelerator_device_type() -> str | None:
    """The available accelerator's device type, or None if there is none.

    Lets a test that needs *an* accelerator -- rather than CUDA specifically --
    skip on the type instead of hardcoding "cuda".
    """
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    return None


ACCELERATOR = accelerator_device_type()
