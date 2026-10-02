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
"""Tests for the DDP group exit report emitted by ``DDPStrategy.teardown``."""

import json
import os
import socket
import time
from datetime import timedelta
from unittest import mock

import torch
import torch.multiprocessing as mp

from lightning.fabric.accelerators.cpu import CPUAccelerator
from lightning.fabric.plugins.environments import LightningEnvironment
from lightning.fabric.plugins.precision import Precision
from lightning.fabric.strategies import DDPStrategy
from lightning.fabric.utilities.distributed import _distributed_is_initialized


def _make_strategy(**kwargs):
    defaults = {
        "accelerator": CPUAccelerator(),
        "parallel_devices": [torch.device("cpu")],
        "cluster_environment": LightningEnvironment(),
        "precision": Precision(),
    }
    defaults.update(kwargs)
    return DDPStrategy(**defaults)


def test_exit_report_defaults_and_set_exit_status():
    """The exit status defaults to ok/converged and can be customized, including extra fields."""
    strategy = DDPStrategy()
    assert strategy.exit_report is None
    entry = strategy._local_exit_report_entry()
    assert entry["status"] == "ok"
    assert entry["converged"] is True
    assert entry["rank"] == 0
    assert entry["world_size"] == 1
    assert "timestamp" in entry

    strategy.set_exit_status(status="failed", converged=False, epochs=3)
    entry = strategy._local_exit_report_entry()
    assert entry["status"] == "failed"
    assert entry["converged"] is False
    assert entry["epochs"] == 3


def test_exit_report_teardown_single_process_and_idempotent():
    """Without a distributed group, teardown produces a local-only report, and repeated teardown calls are no-
    ops."""
    strategy = _make_strategy()
    strategy.set_exit_status(status="ok", converged=True, epochs=5)
    strategy.precision.teardown = mock.Mock(wraps=strategy.precision.teardown)
    with mock.patch("torch.distributed.new_group") as new_group_mock:
        strategy.teardown()
        strategy.teardown()
    # no collectives without an initialized process group
    new_group_mock.assert_not_called()
    # the old teardown path ran exactly once (idempotent)
    assert strategy.precision.teardown.call_count == 1
    report = strategy.exit_report
    assert report is not None
    assert len(report) == 1
    assert report[0]["status"] == "ok"
    assert report[0]["converged"] is True
    assert report[0]["epochs"] == 5


def _free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _exit_report_worker(rank, world_size, port, tmp_path, crash_rank, exit_report_timeout_s, extra_status):
    """Worker for the two-process CPU tests.

    Sets up a real gloo process group through ``DDPStrategy.setup_environment``, optionally simulates an abrupt
    crash on one rank, then tears down and persists the exit report of rank 0 to ``rank0.json``.

    """
    os.environ.update({
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": str(port),
        "RANK": str(rank),
        "LOCAL_RANK": str(rank),
        "WORLD_SIZE": str(world_size),
        "GROUP_RANK": "0",
        "NODE_RANK": "0",
    })
    strategy = _make_strategy(
        parallel_devices=[torch.device("cpu"), torch.device("cpu")],
        process_group_backend="gloo",
        timeout=timedelta(seconds=60),
        exit_report_timeout=timedelta(seconds=exit_report_timeout_s),
    )
    strategy.setup_environment()
    assert _distributed_is_initialized()

    if rank == crash_rank:
        # simulate a rank that dies abruptly without ever running teardown
        os._exit(1)

    strategy.set_exit_status(status="ok", converged=True, **extra_status)
    if crash_rank >= 0:
        time.sleep(3)  # give the crashing rank time to die before the survivor tears down

    start = time.monotonic()
    strategy.teardown()
    elapsed = time.monotonic() - start
    strategy.teardown()  # must be an idempotent no-op

    if rank == 0:
        payload = {
            "report": strategy.exit_report,
            "teardown_elapsed_seconds": elapsed,
            "teardown_complete": strategy._teardown_complete,
        }
        with open(os.path.join(tmp_path, "rank0.json"), "w") as f:
            json.dump(payload, f)
    torch.distributed.destroy_process_group()


def _launch_workers(world_size, port, tmp_path, crash_rank, exit_report_timeout_s, extra_status):
    # launch plain processes (not ``mp.spawn``) so that a crashing rank does not make the launcher
    # terminate the survivors before their teardown completes
    ctx = mp.get_context("spawn")
    processes = []
    for rank in range(world_size):
        process = ctx.Process(
            target=_exit_report_worker,
            args=(rank, world_size, port, str(tmp_path), crash_rank, exit_report_timeout_s, extra_status),
        )
        process.start()
        processes.append(process)
    return processes


def test_ddp_exit_report_normal_exit(tmp_path):
    """Two CPU processes exit normally: the report contains the convergence status of both ranks and teardown
    cleans up."""
    processes = _launch_workers(
        2, _free_port(), tmp_path, crash_rank=-1, exit_report_timeout_s=10, extra_status={"marker": "done"}
    )
    for process in processes:
        process.join(timeout=120)
        assert not process.is_alive(), "a rank is stuck: teardown blocked indefinitely"
        assert process.exitcode == 0

    payload = json.loads((tmp_path / "rank0.json").read_text())
    assert payload["teardown_complete"] is True
    report = payload["report"]
    assert len(report) == 2
    assert {entry["rank"] for entry in report} == {0, 1}
    assert all(entry["world_size"] == 2 for entry in report)
    assert all(entry["status"] == "ok" for entry in report)
    assert all(entry["converged"] is True for entry in report)
    assert all(entry["marker"] == "done" for entry in report)


def test_ddp_exit_report_crashed_rank_does_not_block_others(tmp_path):
    """One rank crashes before teardown: the surviving rank's teardown is bounded by ``exit_report_timeout``
    (no permanent blocking), emits a partial report, and still completes cleanup."""
    exit_report_timeout_s = 5
    processes = _launch_workers(
        2, _free_port(), tmp_path, crash_rank=1, exit_report_timeout_s=exit_report_timeout_s, extra_status={}
    )
    survivor, crashed = processes
    crashed.join(timeout=60)
    assert crashed.exitcode != 0  # rank 1 died abruptly as simulated

    start = time.monotonic()
    survivor.join(timeout=120)
    assert not survivor.is_alive(), "the surviving rank is permanently blocked in teardown"
    assert survivor.exitcode == 0, "the surviving rank did not clean up and exit normally"
    assert time.monotonic() - start < 60

    payload = json.loads((tmp_path / "rank0.json").read_text())
    assert payload["teardown_complete"] is True
    # teardown of the surviving rank was bounded by the exit report timeout, not the (much longer) pg timeout
    assert payload["teardown_elapsed_seconds"] < exit_report_timeout_s * 4
    report = payload["report"]
    # partial report: only the surviving rank reported
    assert len(report) == 1
    assert report[0]["rank"] == 0
    assert report[0]["status"] == "ok"
