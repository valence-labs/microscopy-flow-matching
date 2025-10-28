# Forked from github.com/Charl-AI/tetrachromatic under MIT license.

import atexit
import functools
import logging
import os
from typing import Literal

import torch
import torch.distributed as dist
from submitit.helpers import TorchDistributedEnvironment

log = logging.getLogger(__name__)


def singleton(cls):
    """Decorate a class to turn it into a singleton."""
    instances = {}

    @functools.wraps(cls)
    def wrapper(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return wrapper


@singleton
class Distributed:
    def __init__(self):
        dist_env = TorchDistributedEnvironment()
        dist_env.export(set_cuda_visible_devices=True, overwrite=True)

        log.info(msg="Setting up Distributed environment")
        log.info(msg=f"Master: {dist_env.master_addr}:{dist_env.master_port}")
        log.info(msg=f"Rank: {dist_env.rank}")
        log.info(msg=f"Local rank: {dist_env.local_rank}")
        log.info(msg=f"Visible device ID: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
        log.info(msg=f"World size: {dist_env.world_size}")
        log.info(msg=f"Local world size: {dist_env.local_world_size}")

        dist.init_process_group(
            backend="nccl",
            world_size=dist_env.world_size,
            rank=dist_env.rank,
        )
        assert dist_env.rank == dist.get_rank()
        assert dist_env.world_size == dist.get_world_size()

        # assume one process per GPU, each process gets GPU index == local rank
        assert dist_env.local_rank == int(os.environ.get("CUDA_VISIBLE_DEVICES"))  # type: ignore

        self._rank = dist_env.rank
        self._local_rank = dist_env.local_rank
        self._world_size = dist_env.world_size
        self._local_world_size = dist_env.local_world_size
        self._device = torch.device(f"cuda:{self._local_rank}")
        torch.cuda.set_device(self._device)

        atexit.register(self._cleanup)  # cleanup hook on program exit
        self.barrier()

    def __enter__(self):
        log.info("Entering Distributed context. All processes synchronizing.")
        self.barrier()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        del exc_type, exc_val, exc_tb
        self._cleanup()

    def _cleanup(self):
        if dist.is_initialized():
            log.info(f"Rank {self.rank} destroying process group.")
            dist.destroy_process_group()

    @property
    def world_size(self) -> int:
        """The number of processes participating in the job.
        Usually equal to num_nodes * num_gpus per node."""
        return self._world_size

    @property
    def rank(self) -> int:
        """The rank of the current process. Takes a range of [0, world_size).
        It is conventional to treat rank 0 as the master process."""
        return self._rank

    @property
    def device(self) -> torch.device:
        """The GPU device associated with the current process."""
        return self._device

    def barrier(self):
        """Synchronise all processes."""
        if dist.is_initialized():
            dist.barrier()

    def gather_concat(self, x: torch.Tensor, dim: int = 0) -> torch.Tensor:
        """Gather a tensor from all processes and concat along a given dimension.

        Example:
        ```python
        D = Distributed()
        outputs = model(x)  # shape (B, C, H, W)
        outputs = D.gather_concat(outputs)  # shape (world_size * B, C, H, W)
        ```
        """
        x = x.contiguous()  # important: torch all_gather is buggy if non-contiguous
        if dist.is_initialized():
            x_list = [torch.empty_like(x) for _ in range(dist.get_world_size())]
            dist.all_gather(x_list, x)
            return torch.cat(x_list, dim=dim)
        else:
            return x

    def all_reduce(
        self,
        x: torch.Tensor,
        op: Literal["sum", "avg", "product", "min", "max"] = "sum",
    ) -> torch.Tensor:
        """Reduce a tensor by summing or averaging across all processes.

        Example:
        ```python
        D = Distributed()
        x = torch.randn(10, device=D.device) + 1  # shape (10,), mu=1, var=1 on each process
        y = D.all_reduce(x, op="sum")  # shape (10,), mu=world_size, var=world_size
        z = D.all_reduce(x, op="avg")  # shape (10,), mu=1, var=(1 / world_size)
        ```
        """
        if dist.is_initialized():
            x = x.clone()
            match op:
                case "sum":
                    reduce_op = dist.ReduceOp.SUM
                case "avg":
                    reduce_op = dist.ReduceOp.AVG
                case "product":
                    reduce_op = dist.ReduceOp.PRODUCT
                case "min":
                    reduce_op = dist.ReduceOp.MIN
                case "max":
                    reduce_op = dist.ReduceOp.MAX
                case _:
                    raise ValueError(f"Invalid reduction operation: {op}")
            dist.all_reduce(x, op=reduce_op)
            return x
        else:
            return x


def rank_zero(barrier: bool = False):
    """Decorate a function to only call it on rank zero processes.
    Can optionally force sync afterwards with `dist.barrier()`.

    The decorated function returns None on all non-zero processes.
    It is thus best to use this decorator with side-effect functions
    like saving and logging where a return value is not needed.

    Usage:
    ```python
    @rank_zero()
    def my_print(msg):
        print(msg)

    @rank_zero(barrier=True)
    def write_to_disk(filename, data):
        with open(filename, "w") as f:
            f.write(data)
        time.sleep(1) # Simulate a slow I/O operation
    ```
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = None
            is_dist = dist.is_available() and dist.is_initialized()
            rank0 = not is_dist or dist.get_rank() == 0
            if rank0:
                result = func(*args, **kwargs)
            if barrier and is_dist:
                dist.barrier()
            return result

        return wrapper

    return decorator
