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
"""Integration tests for the DDP group exit report.

Two CPU processes (gloo backend) simulate:

1. normal -- both ranks converge and tear down normally.
2. early_exception -- rank 0 exits early with an exception (marks itself
   failed) while rank 1 converges.
3. timeout -- rank 0 hangs and never reaches teardown; rank 1 must not
   block forever and must still clean up.

Each process writes its exit report to a JSON file so the main test process
can verify the report and the cleanup. Log messages are captured by pytest.
"""

import json
import os
import time
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.multiprocessing as mp

from lightning.fabric.accelerators.cpu import CPUAccelerator
from lightning.fabric.plugins.environments import LightningEnvironment
from lightning.fabric.plugins.environments.lightning import find_free_network_port
from lightning.fabric.strategies import DDPStrategy

_WORLD_SIZE = 2
_TIMEOUT_SEC = 3


def _build_strategy(timeout_sec):
    return DDPStrategy(
        parallel_devices=[torch.device("cpu"), torch.device("cpu")],
        cluster_environment=LightningEnvironment(),
        accelerator=CPUAccelerator(),
        timeout=timedelta(seconds=timeout_sec),
    )


def _ddp_exit_worker(rank, scenario, tmp_path, timeout_sec, master_port):
    """Entry point for each spawned process."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(_WORLD_SIZE)

    strategy = _build_strategy(timeout_sec)
    strategy.setup_environment()

    # Simulate an early exception on rank 0 for the early_exception scenario.
    if scenario == "early_exception" and rank == 0:
        try:
            raise RuntimeError("simulated failure on rank 0")
        except RuntimeError:
            strategy.mark_failed()

    # Simulate a hung rank that never reaches teardown for the timeout scenario.
    if scenario == "timeout" and rank == 0:
        time.sleep(timeout_sec + 2)
        return

    start = time.perf_counter()
    strategy.teardown()
    teardown_duration = time.perf_counter() - start

    # Idempotency: a second teardown must be a no-op.
    report_before = strategy._exit_report
    strategy.teardown()
    assert strategy._exit_report is report_before
    assert strategy._teardown_done

    report_file = tmp_path / f"report_rank_{rank}.json"
    with open(report_file, "w") as f:
        json.dump(
            {
                "rank": rank,
                "report": strategy._exit_report,
                "teardown_duration": teardown_duration,
                "world_size_env": os.environ.get("WORLD_SIZE"),
            },
            f,
        )


def _run_scenario(scenario, tmp_path, timeout_sec=_TIMEOUT_SEC):
    master_port = find_free_network_port()
    mp.spawn(
        _ddp_exit_worker,
        args=(scenario, tmp_path, timeout_sec, master_port),
        nprocs=_WORLD_SIZE,
        join=True,
    )


def _load_reports(tmp_path):
    reports = []
    for rank in range(_WORLD_SIZE):
        report_file = tmp_path / f"report_rank_{rank}.json"
        if report_file.exists():
            with open(report_file) as f:
                reports.append(json.load(f))
    return reports


def test_ddp_exit_report_normal(tmp_path):
    """Both ranks converge; the report shows all ranks converged and cleanup runs."""
    _run_scenario("normal", tmp_path)

    reports = _load_reports(tmp_path)
    assert len(reports) == _WORLD_SIZE

    for report in reports:
        assert report["report"]["gather_succeeded"] is True
        assert report["report"]["all_statuses"] == ["converged", "converged"]
        assert report["report"]["status"] == "converged"
        # Cluster environment teardown removed WORLD_SIZE.
        assert report["world_size_env"] is None

    # The report data is verified from the JSON file.
    # Log messages are captured by pytest (stderr).


def test_ddp_exit_report_early_exception(tmp_path):
    """Rank 0 exits early with an exception; the report shows it as failed."""
    _run_scenario("early_exception", tmp_path)

    reports = _load_reports(tmp_path)
    assert len(reports) == _WORLD_SIZE

    by_rank = {r["rank"]: r for r in reports}
    assert by_rank[0]["report"]["status"] == "failed"
    assert by_rank[1]["report"]["status"] == "converged"
    for report in reports:
        assert report["report"]["gather_succeeded"] is True
        assert report["report"]["all_statuses"] == ["failed", "converged"]
        assert report["world_size_env"] is None

    # The report data is verified from the JSON file.
    # Log messages are captured by pytest (stderr).


def test_ddp_exit_report_timeout_boundary(tmp_path):
    """A hung rank must not block the other rank forever; cleanup still runs."""
    _run_scenario("timeout", tmp_path)

    # Rank 0 hung and never wrote a report; rank 1 did.
    reports = _load_reports(tmp_path)
    assert len(reports) == 1
    report = reports[0]
    assert report["rank"] == 1

    # The gather failed because rank 0 never showed up.
    assert report["report"]["gather_succeeded"] is False
    assert report["report"]["all_statuses"] is None
    assert report["report"]["status"] == "converged"

    # Rank 1 must not have blocked forever: it should return shortly after the
    # process-group timeout (plus a small overhead).
    assert report["teardown_duration"] < _TIMEOUT_SEC + 5

    # Cleanup still ran: cluster environment teardown removed WORLD_SIZE.
    assert report["world_size_env"] is None

    # The warning about the missing rank was logged (captured by pytest stderr).


def test_ddp_exit_report_teardown_idempotent_unit():
    """Teardown is idempotent at the unit level (no distributed group needed)."""
    strategy = DDPStrategy(parallel_devices=[torch.device("cpu")])
    strategy.cluster_environment = LightningEnvironment()
    strategy.accelerator = CPUAccelerator()

    assert strategy._teardown_done is False
    strategy.teardown()
    assert strategy._teardown_done is True
    first_report = strategy._exit_report
    assert first_report is not None
    assert first_report["gather_succeeded"] is False  # no group initialized

    # Second teardown is a no-op and does not alter the report.
    strategy.teardown()
    assert strategy._teardown_done is True
    assert strategy._exit_report is first_report


if __name__ == "__main__":
    # Required for the spawn start method on Windows.
    pytest.main([__file__, "-v", "-s"])
