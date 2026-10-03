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
"""Tests for optimizer skip-step counting in automatic optimization.

The automatic optimization loop counts every optimizer step that is skipped, distinguishing two reasons:

- ``invalid_gradients``: the computed gradients were non-finite (NaN/Inf), so the optimizer step was skipped to avoid
  corrupting the weights.
- ``accumulation``: the gradient accumulation window had not been reached, so the closure ran (forward + backward) but
  ``optimizer.step`` was not called.

The boundary is tightened so that ``global_step`` (i.e. ``optimizer.step.total.completed``) only advances when an actual
optimizer step occurred.
"""

import torch

from lightning.pytorch import Trainer
from lightning.pytorch.demos.boring_classes import BoringModel


class NaNGradientModel(BoringModel):
    """A model that produces a NaN loss (and thus NaN gradients) for specific batches."""

    def __init__(self, nan_batches):
        super().__init__()
        self.nan_batches = set(nan_batches)

    def training_step(self, batch, batch_idx):
        loss = self.step(batch)
        if batch_idx in self.nan_batches:
            loss = loss * float("nan")
        return {"loss": loss}


def _run_training(model, tmp_path, n_batches=4, accumulate_grad_batches=2):
    trainer = Trainer(
        default_root_dir=tmp_path,
        max_epochs=1,
        limit_train_batches=n_batches,
        limit_val_batches=0,
        accumulate_grad_batches=accumulate_grad_batches,
        enable_progress_bar=False,
        logger=False,
        enable_checkpointing=False,
    )
    trainer.fit(model)
    return trainer


def test_skip_counting_invalid_gradients_with_accumulation(tmp_path):
    """Two-step accumulation with a NaN gradient on the second batch of the first window.

    The first window (batches 0 and 1) accumulates NaN gradients, so the optimizer step is skipped. The second window
    (batches 2 and 3) is valid, so the step completes.
    """
    model = NaNGradientModel(nan_batches={1})
    initial_weight = model.layer.weight.detach().clone()

    trainer = _run_training(model, tmp_path)

    optim_progress = trainer.fit_loop.epoch_loop.automatic_optimization.optim_progress

    # 4 batches, accumulate_grad_batches=2:
    #   batch 0: accumulation (skip.accumulation)
    #   batch 1: step with NaN gradients (skip.invalid_gradients)
    #   batch 2: accumulation (skip.accumulation)
    #   batch 3: step with valid gradients (completed)
    assert optim_progress.optimizer.skip.accumulation.total.completed == 2
    assert optim_progress.optimizer.skip.invalid_gradients.total.completed == 1
    assert optim_progress.optimizer.step.total.completed == 1
    # global_step only counts actual optimizer steps
    assert optim_progress.optimizer_steps == 1
    assert trainer.global_step == 1

    # weights must not be corrupted by NaN
    assert torch.isfinite(model.layer.weight).all()
    # weights must have been updated (only from the completed step)
    assert not torch.allclose(model.layer.weight, initial_weight)


def test_skip_counting_all_valid_gradients(tmp_path):
    """Two-step accumulation with all valid gradients: no invalid gradient skips, accumulation skips are counted."""
    model = NaNGradientModel(nan_batches=set())
    initial_weight = model.layer.weight.detach().clone()

    trainer = _run_training(model, tmp_path)

    optim_progress = trainer.fit_loop.epoch_loop.automatic_optimization.optim_progress

    #   batch 0: accumulation (skip.accumulation)
    #   batch 1: step (completed)
    #   batch 2: accumulation (skip.accumulation)
    #   batch 3: step (completed)
    assert optim_progress.optimizer.skip.accumulation.total.completed == 2
    assert optim_progress.optimizer.skip.invalid_gradients.total.completed == 0
    assert optim_progress.optimizer.step.total.completed == 2
    assert optim_progress.optimizer_steps == 2
    assert trainer.global_step == 2

    assert torch.isfinite(model.layer.weight).all()
    assert not torch.allclose(model.layer.weight, initial_weight)


def test_skip_counting_all_nan_gradients(tmp_path):
    """Two-step accumulation with NaN gradients on every batch: no completed steps, weights not corrupted."""
    model = NaNGradientModel(nan_batches={0, 1, 2, 3})
    initial_weight = model.layer.weight.detach().clone()

    trainer = _run_training(model, tmp_path)

    optim_progress = trainer.fit_loop.epoch_loop.automatic_optimization.optim_progress

    #   batch 0: accumulation with NaN gradients (skip.accumulation)
    #   batch 1: step with NaN gradients (skip.invalid_gradients)
    #   batch 2: accumulation with NaN gradients (skip.accumulation)
    #   batch 3: step with NaN gradients (skip.invalid_gradients)
    assert optim_progress.optimizer.skip.accumulation.total.completed == 2
    assert optim_progress.optimizer.skip.invalid_gradients.total.completed == 2
    assert optim_progress.optimizer.step.total.completed == 0
    assert optim_progress.optimizer_steps == 0
    assert trainer.global_step == 0

    # weights must not be corrupted by NaN and must not have changed (no completed step)
    assert torch.isfinite(model.layer.weight).all()
    assert torch.allclose(model.layer.weight, initial_weight)


def test_skip_counting_no_accumulation(tmp_path):
    """Without accumulation (accumulate_grad_batches=1), there are no accumulation skips. A NaN gradient on batch 1
    causes an invalid gradient skip; the other batches complete."""
    model = NaNGradientModel(nan_batches={1})
    initial_weight = model.layer.weight.detach().clone()

    trainer = _run_training(model, tmp_path, accumulate_grad_batches=1)

    optim_progress = trainer.fit_loop.epoch_loop.automatic_optimization.optim_progress

    #   batch 0: step (completed)
    #   batch 1: step with NaN gradients (skip.invalid_gradients)
    #   batch 2: step (completed)
    #   batch 3: step (completed)
    assert optim_progress.optimizer.skip.accumulation.total.completed == 0
    assert optim_progress.optimizer.skip.invalid_gradients.total.completed == 1
    assert optim_progress.optimizer.step.total.completed == 3
    assert optim_progress.optimizer_steps == 3
    assert trainer.global_step == 3

    assert torch.isfinite(model.layer.weight).all()
    assert not torch.allclose(model.layer.weight, initial_weight)
