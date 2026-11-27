"""Build and runtime capability checks for Mojo's GPU factorization."""

from gpu.host import DeviceContext
from sys import is_defined
from sys.info import has_amd_gpu_accelerator, has_nvidia_gpu_accelerator


@always_inline
fn gpu_build_enabled() -> Bool:
    return is_defined["STRUT_GPU"]() and (has_amd_gpu_accelerator() or has_nvidia_gpu_accelerator())


fn require_gpu_backend() raises:
    @parameter
    if gpu_build_enabled():
        if DeviceContext.number_of_devices() == 0:
            raise Error("No compatible GPU devices are visible to this Strut build.")
    else:
        raise Error(
            "This Strut executable has no GPU target. "
            "Run uv run python scripts/build_mojo_solver.py --gpu on a supported GPU host."
        )
