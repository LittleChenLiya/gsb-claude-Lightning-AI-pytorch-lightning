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
"""Minimal, reproducible example for the optimizer *skipped-step* counters in automatic optimization.

Scenario
--------
* A tiny CPU model is trained for 6 batches with ``accumulate_grad_batches=2`` -> 3 optimizer-step boundaries.
* The gradients of the middle accumulation window (batches 2-3) are poisoned with a NaN. The user-defined
  ``optimizer_step`` hook executes the closure, detects the non-finite gradients and refuses to call
  ``optimizer.step`` (the same pattern Lightning recommends for skipping steps).

Expected results
----------------
* 2 actual ``optimizer.step`` calls -> ``global_step == 2`` and 2 completed optimizer steps.
* 1 step-boundary batch skipped because of the invalid (NaN) gradient -> weights unchanged for that window.
* 3 in-between batches that only accumulated gradients because the 2-batch window was not yet reached.
* Training recovers after the NaN window and the final weights match a manual reference computation.

Run with::

    python examples/pytorch/optimization/skipped_optimizer_steps.py
"""

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from lightning.pytorch import LightningModule, Trainer


class FixedDataset(Dataset):
    def __init__(self, n: int = 6) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(42)
        self.x = torch.randn(n, 4, generator=g)
        self.y = torch.randn(n, 2, generator=g)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


class NaNAccumulationModel(LightningModule):
    def __init__(self, nan_window: int = 1) -> None:
        super().__init__()
        self.layer = nn.Linear(4, 2)
        self.nan_window = nan_window
        self.window_idx = 0
        self.boundary_weights: list[torch.Tensor] = []
        self.initial_weight = self.layer.weight.detach().clone()
        self.initial_bias = self.layer.bias.detach().clone()

    def train_dataloader(self):
        return DataLoader(FixedDataset(), batch_size=1)

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self(x).sub(y).pow(2).mean()
        # poison the gradient of the first batch in the selected accumulation window
        if self.window_idx == self.nan_window and batch_idx % 2 == 0:
            loss = loss * float("nan")
        return loss

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure=None):
        optimizer_closure()
        if all(torch.isfinite(p.grad).all() for p in self.parameters() if p.grad is not None):
            optimizer.step()
        # this hook only runs at accumulation boundaries (batch_idx 1, 3, 5)
        self.boundary_weights.append(self.layer.weight.detach().clone())
        self.window_idx += 1

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)


def main() -> None:
    model = NaNAccumulationModel(nan_window=1)
    trainer = Trainer(
        max_epochs=1,
        accumulate_grad_batches=2,
        enable_model_summary=False,
        enable_progress_bar=False,
        enable_checkpointing=False,
        logger=False,
        limit_val_batches=0,
        accelerator="cpu",
        devices=1,
    )
    trainer.fit(model)

    progress = trainer.fit_loop.epoch_loop.automatic_optimization.optim_progress
    print("\n==================== counters ====================")
    print(f"completed optimizer steps : {progress.optimizer_steps}")
    print(f"global_step               : {trainer.global_step}")
    print(f"skipped (invalid gradient): {progress.skipped_optimizer_steps_invalid_gradient}")
    print(f"skipped (accumulation)    : {progress.skipped_optimizer_steps_accumulation}")
    print(f"skipped (total)           : {progress.skipped_optimizer_steps}")

    assert progress.optimizer_steps == 2, progress.optimizer_steps
    assert trainer.global_step == 2, trainer.global_step
    assert progress.skipped_optimizer_steps_invalid_gradient == 1
    assert progress.skipped_optimizer_steps_accumulation == 3
    assert progress.skipped_optimizer_steps == 4

    print("\n==================== weights =====================")
    w0, w1, w2 = model.boundary_weights
    print(f"window 0 changed weights  : {not torch.equal(w0, model.initial_weight)}")
    print(f"NaN window left weights   : unchanged = {torch.equal(w1, w0)}")
    print(f"window 2 changed weights  : {not torch.equal(w2, w1)}")
    print(f"final weights are finite  : {bool(torch.isfinite(w2).all())}")
    assert not torch.equal(w0, model.initial_weight)
    assert torch.equal(w1, w0)
    assert not torch.equal(w2, w1)
    assert torch.isfinite(w2).all()

    # manual reference: replay windows 0 and 2 (finite) with the same init, data and SGD settings
    ref = nn.Linear(4, 2)
    with torch.no_grad():
        ref.weight.copy_(model.initial_weight)
        ref.bias.copy_(model.initial_bias)
    opt = torch.optim.SGD(ref.parameters(), lr=0.1)
    dataset = FixedDataset()
    for i, j in ((0, 1), (4, 5)):
        opt.zero_grad()
        loss = torch.zeros(())
        for idx in (i, j):
            x, y = dataset[idx]
            loss = loss + (ref(x.unsqueeze(0)) - y).pow(2).mean() / 2
        loss.backward()
        opt.step()

    print("\n================ reference =======================")
    match = bool(torch.allclose(model.layer.weight, ref.weight, atol=1e-6))
    print(f"final weights match manual reference: {match}")
    assert torch.allclose(model.layer.weight, ref.weight, atol=1e-6)
    assert torch.allclose(model.layer.bias, ref.bias, atol=1e-6)
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
