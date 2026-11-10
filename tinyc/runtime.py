"""NVRTC compilation and kernel launch through cuda-python. Buffers are torch CUDA tensors.

`compile_cubin` needs no GPU (NVRTC is a pure compiler), so codegen can be validated on a machine
whose GPU is busy. Loading and launching need a CUDA context (torch's primary context is reused).
"""
from __future__ import annotations

import ctypes

import numpy as np

from cuda.bindings import driver as cu
from cuda.bindings import nvrtc


class CompileError(RuntimeError):
    pass


def _chk_nvrtc(res, what=""):
    err = res[0] if isinstance(res, tuple) else res
    if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        raise CompileError(f"NVRTC {what}: {err}")
    return res[1:] if isinstance(res, tuple) and len(res) > 1 else None


def _chk_cu(res, what=""):
    err = res[0] if isinstance(res, tuple) else res
    if err != cu.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"CUDA driver {what}: {err}")
    return res[1:] if isinstance(res, tuple) and len(res) > 1 else None


def compile_cubin(src: str, arch: str = "sm_120") -> bytes:
    """Compile CUDA C to a cubin for `arch`. Raises CompileError with the NVRTC log on failure."""
    (prog,) = _chk_nvrtc(nvrtc.nvrtcCreateProgram(src.encode(), b"tinyc.cu", 0, [], []), "create")
    opts = [f"--gpu-architecture={arch}".encode(), b"--std=c++17"]
    res = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
    if res[0] != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        (n,) = _chk_nvrtc(nvrtc.nvrtcGetProgramLogSize(prog), "log size")
        log = b" " * n
        nvrtc.nvrtcGetProgramLog(prog, log)
        raise CompileError(f"{res[0]}\n{log.decode(errors='replace')}\n--- source ---\n{src}")
    (n,) = _chk_nvrtc(nvrtc.nvrtcGetCUBINSize(prog), "cubin size")
    cubin = b" " * n
    _chk_nvrtc(nvrtc.nvrtcGetCUBIN(prog, cubin), "get cubin")
    nvrtc.nvrtcDestroyProgram(prog)
    return cubin


class LoadedKernel:
    """A compiled kernel loaded into the current CUDA context."""

    def __init__(self, cubin: bytes, name: str = "k"):
        import torch

        torch.cuda.init()
        torch.zeros(1, device="cuda")  # make sure torch's primary context is current
        (self.module,) = _chk_cu(cu.cuModuleLoadData(cubin), "module load")
        (self.func,) = _chk_cu(cu.cuModuleGetFunction(self.module, name.encode()), "get function")

    def launch(self, grid, block, ptrs: list[int], ints: list[int], stream: int = 0):
        args = [np.array([p], dtype=np.uint64) for p in ptrs] + [np.array([v], dtype=np.int32) for v in ints]
        holder = np.array([a.ctypes.data for a in args], dtype=np.uint64)
        _chk_cu(cu.cuLaunchKernel(self.func, *grid, *block, 0, stream, holder.ctypes.data, 0), "launch")


def device_arch() -> str:
    import torch

    major, minor = torch.cuda.get_device_capability()
    return f"sm_{major}{minor}"
