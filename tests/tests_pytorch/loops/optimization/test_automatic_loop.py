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
from collections.abc import Iterator, Mapping
from contextlib import nullcontext
from typing import Generic, TypeVar

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from lightning.pytorch import Trainer
from lightning.pytorch.demos.boring_classes import BoringModel
from lightning.pytorch.loops.optimization.automatic import ClosureResult
from lightning.pytorch.loops.progress import _OptimizerProgress
from lightning.pytorch.utilities.exceptions import MisconfigurationException


def test_closure_result_deepcopy():
    closure_loss = torch.tensor(123.45)
    result = ClosureResult(closure_loss)

    assert closure_loss.data_ptr() == result.closure_loss.data_ptr()
    # the `loss` is cloned so the storage is different
    assert closure_loss.data_ptr() != result.loss.data_ptr()

    copy = result.asdict()
    assert result.loss == copy["loss"]
    assert copy.keys() == {"loss"}

    # no copy
    assert id(result.loss) == id(copy["loss"])
    assert result.loss.data_ptr() == copy["loss"].data_ptr()


def test_closure_result_apply_accumulation():
    closure_loss = torch.tensor(25.0)
    result = ClosureResult.from_training_step_output(closure_loss, 5)
    assert result.loss == 5


T = TypeVar("T")


class OutputMapping(Generic[T], Mapping[str, T]):
    def __init__(self, d: dict[str, T]) -> None:
        self.d: dict[str, T] = d

    def __iter__(self) -> Iterator[str]:
        return iter(self.d)

    def __len__(self) -> int:
        return len(self.d)

    def __getitem__(self, key: str) -> T:
        return self.d[key]


@pytest.mark.parametrize(
    "case",
    [
        (5.0, "must return a Tensor, a dict, or None"),
        ({"a": 5}, "the 'loss' key needs to be present"),
        (OutputMapping({"a": 5}), "the 'loss' key needs to be present"),
    ],
)
def test_warning_invalid_trainstep_output(tmp_path, case):
    output, match = case

    class InvalidTrainStepModel(BoringModel):
        def training_step(self, batch, batch_idx):
            return output

    model = InvalidTrainStepModel()
    trainer = Trainer(default_root_dir=tmp_path, fast_dev_run=1)

    with pytest.raises(MisconfigurationException, match=match):
        trainer.fit(model)


@pytest.mark.parametrize("world_size", [1, 2])
def test_skip_training_step_not_allowed(world_size, tmp_path):
    """Test that skipping the training_step in distributed training is not allowed."""

    class TestModel(BoringModel):
        def training_step(self, batch, batch_idx):
            return None

    model = TestModel()
    trainer = Trainer(
        default_root_dir=tmp_path,
        max_steps=1,
        barebones=True,
    )
    trainer.strategy.world_size = world_size  # mock world size without launching processes
    error_context = (
        pytest.raises(RuntimeError, match="Skipping the `training_step` .* is not supported")
        if world_size > 1
        else nullcontext()
    )
    with error_context:
        trainer.fit(model)


def test_skipped_progress_state_dict_backward_compatible():
    """Old checkpoints (without skipped-step counters) must load without errors."""
    progress = _OptimizerProgress()
    state = progress.state_dict()
    assert set(state["skipped"]) == {"invalid_gradient", "accumulation"}

    # state as saved before the skipped tracking existed
    state.pop("skipped")
    _OptimizerProgress().load_state_dict(state)


class _FixedDataset(Dataset):
    """Deterministic small dataset returning ``(x, y)`` pairs."""

    def __init__(self, n: int = 6) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(42)
        self.x = torch.randn(n, 4, generator=g)
        self.y = torch.randn(n, 2, generator=g)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


class _NanGradAccumulationModel(BoringModel):
    """Tiny model with 2-batch gradient accumulation that skips the optimizer step on NaN gradients.

    The optimizer step at ``nan_window`` receives NaN gradients (loss multiplied by NaN) and must be skipped.
    """

    def __init__(self, nan_window: int = 1) -> None:
        super().__init__()
        self.layer = torch.nn.Linear(4, 2)
        self.nan_window = nan_window
        self.window_idx = 0
        self.weight_snapshots: list[torch.Tensor] = []
        self.initial_state = {k: v.detach().clone() for k, v in self.layer.state_dict().items()}

    def train_dataloader(self):
        return DataLoader(_FixedDataset(), batch_size=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self(x).sub(y).pow(2).mean()
        # poison the gradient of the first batch inside the selected accumulation window
        if self.window_idx == self.nan_window and batch_idx % 2 == 0:
            loss = loss * float("nan")
        return loss

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure=None):
        optimizer_closure()
        if all(torch.isfinite(p.grad).all() for p in self.parameters() if p.grad is not None):
            optimizer.step()
        # the hook only runs at accumulation boundaries (batch_idx 1, 3, 5 -> one snapshot per window)
        self.weight_snapshots.append(self.layer.weight.detach().clone())
        self.window_idx += 1

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)

    def val_dataloader(self):
        return None


def test_optimizer_step_skipped_counter_nan_and_accumulation(tmp_path):
    """NaN gradients at an accumulation boundary count as ``invalid_gradient`` skips, intermediate accumulation
    batches count as ``accumulation`` skips, only real ``optimizer.step`` calls advance ``global_step`` and
    weights stay untouched when the step is skipped."""
    model = _NanGradAccumulationModel(nan_window=1)
    initial_weight = model.initial_state["weight"].clone()

    trainer = Trainer(
        default_root_dir=tmp_path,
        max_epochs=1,
        accumulate_grad_batches=2,
        enable_model_summary=False,
        enable_checkpointing=False,
        logger=False,
        limit_val_batches=0,
        accelerator="cpu",
        devices=1,
    )
    trainer.fit(model)

    progress = trainer.fit_loop.epoch_loop.automatic_optimization.optim_progress
    # 6 batches, accumulation window 2 -> 3 step boundaries; the middle one was skipped due to NaN
    assert progress.skipped_optimizer_steps_invalid_gradient == 1
    assert progress.skipped_optimizer_steps_accumulation == 3
    assert progress.skipped_optimizer_steps == 4
    assert progress.optimizer_steps == 2
    assert trainer.global_step == 2

    # weights at the 3 boundaries: window 0 stepped, window 1 skipped (unchanged), window 2 stepped again
    assert len(model.weight_snapshots) == 3
    assert not torch.equal(model.weight_snapshots[0], initial_weight)  # window 0 performed a step
    assert torch.equal(model.weight_snapshots[1], model.weight_snapshots[0])  # NaN window: unchanged
    assert not torch.equal(model.weight_snapshots[2], model.weight_snapshots[1])  # training resumes
    assert torch.isfinite(model.weight_snapshots[2]).all()  # no NaN leaked into the parameters


def test_skipped_weights_match_manual_reference(tmp_path):
    """The accumulated steps on finite gradients must produce exactly the weights computed manually, while the NaN
    window leaves the weights identical to the previous step."""
    model = _NanGradAccumulationModel(nan_window=1)

    trainer = Trainer(
        default_root_dir=tmp_path,
        max_epochs=1,
        accumulate_grad_batches=2,
        enable_model_summary=False,
        enable_checkpointing=False,
        logger=False,
        limit_val_batches=0,
        accelerator="cpu",
        devices=1,
    )
    trainer.fit(model)

    # replay windows 0 and 2 (finite) manually with the same initialization, data and optimizer settings
    ref = torch.nn.Linear(4, 2)
    ref.load_state_dict({k: v.clone() for k, v in model.initial_state.items()})
    optimizer = torch.optim.SGD(ref.parameters(), lr=0.1)
    dataset = _FixedDataset()
    for i, j in ((0, 1), (4, 5)):
        optimizer.zero_grad()
        loss = torch.zeros(())
        for idx in (i, j):
            x, y = dataset[idx]
            loss = loss + (ref(x.unsqueeze(0)) - y).pow(2).mean() / 2
        loss.backward()
        assert torch.isfinite(ref.weight.grad).all()
        optimizer.step()

    assert torch.allclose(model.layer.weight, ref.weight, atol=1e-6)
    assert torch.allclose(model.layer.bias, ref.bias, atol=1e-6)
