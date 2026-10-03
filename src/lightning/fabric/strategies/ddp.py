# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
from contextlib import AbstractContextManager, nullcontext
from datetime import timedelta
from typing import Any, Literal, Optional, Union

import torch
import torch.distributed
from lightning_utilities.core.rank_zero import rank_zero_only as utils_rank_zero_only
from torch import Tensor
from torch.nn import Module
from torch.nn.parallel.distributed import DistributedDataParallel
from typing_extensions import override

from lightning.fabric.accelerators.accelerator import Accelerator
from lightning.fabric.plugins.collectives.torch_collective import default_pg_timeout
from lightning.fabric.plugins.environments.cluster_environment import ClusterEnvironment
from lightning.fabric.plugins.io.checkpoint_io import CheckpointIO
from lightning.fabric.plugins.precision import Precision
from lightning.fabric.strategies.launchers.multiprocessing import _MultiProcessingLauncher
from lightning.fabric.strategies.launchers.subprocess_script import _SubprocessScriptLauncher
from lightning.fabric.strategies.parallel import ParallelStrategy
from lightning.fabric.strategies.registry import _StrategyRegistry
from lightning.fabric.strategies.strategy import TBroadcast, _BackwardSyncControl
from lightning.fabric.utilities.distributed import (
    ReduceOp,
    _distributed_is_initialized,
    _get_default_process_group_backend_for_device,
    _init_dist_connection,
    _sync_ddp_if_available,
)
from lightning.fabric.utilities.distributed import group as _group
from lightning.fabric.utilities.rank_zero import rank_zero_info, rank_zero_only

_log = logging.getLogger(__name__)

# Convergence status reported by each rank when the DDP group tears down.
# A rank that finished its work normally is ``converged``; a rank that exited
# early due to an exception is ``failed``; ranks that never reach teardown
# (crash / hang / timeout) are reported as missing by their peers.
_STATUS_CONVERGED = "converged"
_STATUS_FAILED = "failed"

_DDP_FORK_ALIASES = (
    "ddp_fork",
    "ddp_notebook",
)


class DDPStrategy(ParallelStrategy):
    """Strategy for multi-process single-device training on one or multiple nodes."""

    def __init__(
        self,
        accelerator: Optional[Accelerator] = None,
        parallel_devices: Optional[list[torch.device]] = None,
        cluster_environment: Optional[ClusterEnvironment] = None,
        checkpoint_io: Optional[CheckpointIO] = None,
        precision: Optional[Precision] = None,
        process_group_backend: Optional[str] = None,
        timeout: Optional[timedelta] = default_pg_timeout,
        start_method: Literal["popen", "spawn", "fork", "forkserver"] = "popen",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            accelerator=accelerator,
            parallel_devices=parallel_devices,
            cluster_environment=cluster_environment,
            checkpoint_io=checkpoint_io,
            precision=precision,
        )
        self._num_nodes = 1
        self._process_group_backend: Optional[str] = process_group_backend
        self._timeout: Optional[timedelta] = timeout
        self._start_method = start_method
        self._backward_sync_control = _DDPBackwardSyncControl()
        self._ddp_kwargs = kwargs
        # Convergence / teardown bookkeeping for the group exit report.
        self._converged: bool = True
        self._teardown_done: bool = False
        self._exit_report: Optional[dict[str, Any]] = None

    @property
    @override
    def root_device(self) -> torch.device:
        assert self.parallel_devices is not None
        return self.parallel_devices[self.local_rank]

    @property
    def converged(self) -> bool:
        """Whether this rank has converged (finished its work without error)."""
        return self._converged

    def mark_failed(self) -> None:
        """Mark this rank as having exited abnormally (did not converge).

        Call this when the process exits early due to an exception so that the
        group exit report produced by :meth:`teardown` can reflect the
        non-converged status of this rank.

        """
        self._converged = False

    @property
    def num_nodes(self) -> int:
        return self._num_nodes

    @num_nodes.setter
    def num_nodes(self, num_nodes: int) -> None:
        # note that world ranks is related to num_nodes, when resetting it, need to reset world ranks
        self._num_nodes = num_nodes

    @property
    def num_processes(self) -> int:
        return len(self.parallel_devices) if self.parallel_devices is not None else 0

    @property
    @override
    def distributed_sampler_kwargs(self) -> dict[str, Any]:
        return {"num_replicas": (self.num_nodes * self.num_processes), "rank": self.global_rank}

    @property
    def process_group_backend(self) -> Optional[str]:
        return self._process_group_backend

    @override
    def _configure_launcher(self) -> None:
        assert self.cluster_environment is not None
        if self._start_method == "popen":
            self._launcher = _SubprocessScriptLauncher(self.cluster_environment, self.num_processes, self.num_nodes)
        else:
            self._launcher = _MultiProcessingLauncher(self, start_method=self._start_method)

    @override
    def setup_environment(self) -> None:
        super().setup_environment()
        self._setup_distributed()

    @override
    def setup_module(self, module: Module) -> DistributedDataParallel:
        """Wraps the model into a :class:`~torch.nn.parallel.distributed.DistributedDataParallel` module."""
        device_ids = self._determine_ddp_device_ids()
        ctx: Union[torch.cuda.StreamContext, nullcontext] = nullcontext()

        # https://pytorch.org/docs/stable/notes/cuda.html#id5
        if device_ids is not None:
            capturing = torch.cuda.is_current_stream_capturing()
            if capturing:
                # DDP must be initialized on a side-stream for CUDA graph whole-network capture.
                # The resulting AccumulateGrad stream mismatch is intentional in this case.
                # See: https://pytorch.org/docs/stable/notes/cuda.html#id5
                ctx = torch.cuda.stream(torch.cuda.Stream())
                torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(False)
            else:
                # Default stream avoids AccumulateGrad stream mismatch warnings during normal training.
                ctx = torch.cuda.stream(torch.cuda.default_stream())
        with ctx:
            return DistributedDataParallel(module=module, device_ids=device_ids, **self._ddp_kwargs)

    @override
    def module_to_device(self, module: Module) -> None:
        module.to(self.root_device)

    @override
    def all_reduce(
        self, tensor: Tensor, group: Optional[Any] = None, reduce_op: Optional[Union[ReduceOp, str]] = "mean"
    ) -> Tensor:
        """Reduces a tensor from several distributed processes to one aggregated tensor.

        Args:
            tensor: the tensor to sync and reduce
            group: the process group to gather results from. Defaults to all processes (world)
            reduce_op: the reduction operation. Defaults to 'mean'/'avg'.
                Can also be a string 'sum' to calculate the sum during reduction.

        Return:
            reduced value, except when the input was not a tensor the output remains is unchanged

        """
        if isinstance(tensor, Tensor):
            return _sync_ddp_if_available(tensor, group, reduce_op=reduce_op)
        return tensor

    @override
    def barrier(self, *args: Any, **kwargs: Any) -> None:
        if not _distributed_is_initialized():
            return
        if torch.distributed.get_backend() == "nccl":
            torch.distributed.barrier(device_ids=self._determine_ddp_device_ids())
        else:
            # Handle PyTorch bug where barrier() fails on CPU with "PrivateUse1HooksInterface" error
            try:
                torch.distributed.barrier()
            except RuntimeError as e:
                if "PrivateUse1HooksInterface" in str(e):
                    # Fallback: Use all_reduce as barrier - all processes must participate
                    # This achieves the same synchronization effect as barrier()
                    dummy_tensor = torch.tensor(0.0, device=self.root_device)
                    torch.distributed.all_reduce(dummy_tensor)
                else:
                    raise

    @override
    def broadcast(self, obj: TBroadcast, src: int = 0) -> TBroadcast:
        if not _distributed_is_initialized():
            return obj

        obj = [obj]
        torch.distributed.broadcast_object_list(obj, src, group=_group.WORLD)
        return obj[0]

    @override
    def teardown(self) -> None:
        """Tear down the distributed strategy and emit a group exit report.

        The report gathers the convergence status of every rank. The exchange
        is best-effort: if a rank is missing or times out, the remaining ranks
        log what they know and proceed with cleanup instead of blocking
        forever. Idempotent -- calling this more than once is a no-op.

        """
        if self._teardown_done:
            return
        self._teardown_done = True
        try:
            self._report_group_exit()
        except Exception:
            # The report must never block or abort cleanup. The report method
            # handles expected failures itself; this guards against the rest.
            _log.exception("Failed to produce the DDP group exit report. Proceeding with cleanup.")
        finally:
            super().teardown()

    def _report_group_exit(self) -> None:
        """Gather convergence statuses from all ranks and log the exit report."""
        own_status = _STATUS_CONVERGED if self._converged else _STATUS_FAILED

        if not _distributed_is_initialized():
            self._exit_report = self._build_exit_report(all_statuses=None, gather_succeeded=False)
            _log.info(
                "DDP group exit report: rank %d/%d status=%s (distributed group not initialized)",
                self.global_rank, self.world_size, own_status,
            )
            return

        world_size = self.world_size
        gathered: list[Optional[str]] = [None] * world_size
        gather_succeeded = True
        try:
            torch.distributed.all_gather_object(gathered, own_status, group=_group.WORLD)
        except Exception as exc:
            # A rank may have exited early or timed out. Do not block forever:
            # record what we know and let cleanup proceed.
            gather_succeeded = False
            _log.warning(
                "DDP group exit report: could not gather convergence statuses from all ranks"
                " (%s). Known status for rank %d/%d: %s. Proceeding with cleanup.",
                exc, self.global_rank, world_size, own_status,
            )

        self._exit_report = self._build_exit_report(
            all_statuses=gathered if gather_succeeded else None,
            gather_succeeded=gather_succeeded,
        )

        if gather_succeeded and self.global_rank == 0:
            self._log_exit_report(gathered)
        elif not gather_succeeded:
            _log.info(
                "DDP group exit report: rank %d/%d status=%s (aggregate report unavailable)",
                self.global_rank, world_size, own_status,
            )

    def _build_exit_report(
        self, all_statuses: Optional[list[Optional[str]]], gather_succeeded: bool
    ) -> dict[str, Any]:
        return {
            "world_size": self.world_size,
            "rank": self.global_rank,
            "status": _STATUS_CONVERGED if self._converged else _STATUS_FAILED,
            "all_statuses": all_statuses,
            "gather_succeeded": gather_succeeded,
        }

    @staticmethod
    def _log_exit_report(all_statuses: list[Optional[str]]) -> None:
        world_size = len(all_statuses)
        n_converged = sum(1 for status in all_statuses if status == _STATUS_CONVERGED)
        status_str = ", ".join(f"rank {i}={status}" for i, status in enumerate(all_statuses))
        if n_converged == world_size:
            rank_zero_info(
                f"DDP group exit report: all {world_size} ranks converged. [{status_str}]"
            )
        else:
            rank_zero_info(
                f"DDP group exit report: {n_converged}/{world_size} ranks converged. [{status_str}]"
            )

    @override
    def get_module_state_dict(self, module: Module) -> dict[str, Union[Any, Tensor]]:
        if isinstance(module, DistributedDataParallel):
            module = module.module
        return super().get_module_state_dict(module)

    @override
    def load_module_state_dict(
        self, module: Module, state_dict: dict[str, Union[Any, Tensor]], strict: bool = True
    ) -> None:
        if isinstance(module, DistributedDataParallel):
            module = module.module
        super().load_module_state_dict(module=module, state_dict=state_dict, strict=strict)

    @classmethod
    @override
    def register_strategies(cls, strategy_registry: _StrategyRegistry) -> None:
        entries = (
            ("ddp", "popen"),
            ("ddp_spawn", "spawn"),
            ("ddp_fork", "fork"),
            ("ddp_notebook", "fork"),
        )
        for name, start_method in entries:
            strategy_registry.register(
                name,
                cls,
                description=f"DDP strategy with `start_method={start_method!r}`",
                start_method=start_method,
            )
        strategy_registry.register(
            "ddp_find_unused_parameters_true",
            cls,
            description="Alias for `find_unused_parameters_true` and `start_method='popen'`",
            find_unused_parameters=True,
            start_method="popen",
        )

    def _setup_distributed(self) -> None:
        self._set_world_ranks()
        self._process_group_backend = self._get_process_group_backend()
        assert self.cluster_environment is not None
        _init_dist_connection(
            self.cluster_environment,
            self._process_group_backend,
            timeout=self._timeout,
            device_id=self.root_device if self.root_device.type != "cpu" else None,
        )

    def _get_process_group_backend(self) -> str:
        return self._process_group_backend or _get_default_process_group_backend_for_device(self.root_device)

    def _set_world_ranks(self) -> None:
        if self.cluster_environment is not None:
            self.cluster_environment.set_global_rank(self.node_rank * self.num_processes + self.local_rank)
            self.cluster_environment.set_world_size(self.num_nodes * self.num_processes)
        # `LightningEnvironment.set_global_rank` will do this too, but we cannot rely on that implementation detail
        # additionally, for some implementations, the setter is a no-op, so it's safer to access the getter
        rank_zero_only.rank = utils_rank_zero_only.rank = self.global_rank

    def _determine_ddp_device_ids(self) -> Optional[list[int]]:
        return None if self.root_device.type == "cpu" else [self.root_device.index]


class _DDPBackwardSyncControl(_BackwardSyncControl):
    @override
    def no_backward_sync(self, module: Module, enabled: bool) -> AbstractContextManager:
        """Blocks gradient synchronization inside the :class:`~torch.nn.parallel.distributed.DistributedDataParallel`
        wrapper."""
        if not enabled:
            return nullcontext()

        if not isinstance(module, DistributedDataParallel):
            raise TypeError(
                "Blocking backward sync is only possible if the module passed to"
                f" `{self.__class__.__name__}.no_backward_sync` is wrapped in `DistributedDataParallel`."
                f" Got: {module.__class__.__name__}."
            )
        return module.no_sync()
