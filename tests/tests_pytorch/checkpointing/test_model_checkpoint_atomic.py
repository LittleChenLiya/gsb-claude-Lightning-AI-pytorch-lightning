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
"""Tests for atomic checkpoint saving (temporary file + atomic replace)."""

import hashlib
import os
from unittest import mock

import pytest
import torch

import lightning.fabric.utilities.cloud_io as cloud_io
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.demos.boring_classes import BoringModel


def _md5(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def _make_trainer(tmp_path, max_epochs: int = 1, **ckpt_kwargs):
    ckpt = ModelCheckpoint(dirpath=tmp_path, **ckpt_kwargs)
    trainer = Trainer(
        default_root_dir=tmp_path,
        max_epochs=max_epochs,
        limit_train_batches=4,
        limit_val_batches=0,
        accelerator="cpu",
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        callbacks=[ckpt],
    )
    return trainer, ckpt


def _inject_write_failure():
    """Patch ``torch.save`` (as used by ``cloud_io``) to write a partial chunk then fail."""
    real_save = torch.save

    def failing_save(obj, f, *args, **kwargs):
        f.write(b"PARTIAL-DATA")
        f.flush()
        raise OSError("injected filesystem fault: simulated disk full")

    cloud_io.torch.save = failing_save
    return real_save


def test_model_checkpoint_atomic_save_normal(tmp_path):
    """A normal save writes the checkpoint atomically with no temp leftovers."""
    trainer, _ = _make_trainer(tmp_path, save_top_k=-1, every_n_epochs=1, save_weights_only=True)
    trainer.fit(BoringModel())

    files = os.listdir(tmp_path)
    ckpt_files = [f for f in files if f.endswith(".ckpt")]
    assert len(ckpt_files) == 1
    ckpt_path = os.path.join(tmp_path, ckpt_files[0])

    # no temp leftovers
    assert not [f for f in files if ".tmp" in f]
    # checkpoint loads
    loaded = torch.load(ckpt_path, weights_only=True)
    assert "state_dict" in loaded


def test_model_checkpoint_atomic_save_failure_preserves_previous(tmp_path):
    """A failed write must not overwrite the previous valid checkpoint; temp files are recycled."""
    # first, a valid checkpoint
    trainer, _ = _make_trainer(tmp_path, save_top_k=-1, every_n_epochs=1, save_weights_only=True)
    trainer.fit(BoringModel())
    ckpt_path = os.path.join(tmp_path, [f for f in os.listdir(tmp_path) if f.endswith(".ckpt")][0])
    before_hash = _md5(ckpt_path)

    # inject a write failure mid-write
    real_save = _inject_write_failure()
    try:
        trainer2, _ = _make_trainer(tmp_path, save_top_k=-1, every_n_epochs=1, save_weights_only=True)
        with pytest.raises(OSError, match="injected filesystem fault"):
            trainer2.fit(BoringModel())
    finally:
        cloud_io.torch.save = real_save

    # previous checkpoint untouched
    assert os.path.exists(ckpt_path)
    assert _md5(ckpt_path) == before_hash
    # no temp leftovers
    assert not [f for f in os.listdir(tmp_path) if ".tmp" in f]
    # previous checkpoint still loads
    loaded = torch.load(ckpt_path, weights_only=True)
    assert "state_dict" in loaded


def test_model_checkpoint_atomic_save_recovery(tmp_path):
    """After a failed write, a subsequent save succeeds and produces a valid checkpoint."""
    trainer, _ = _make_trainer(tmp_path, save_top_k=-1, every_n_epochs=1, save_weights_only=True)
    trainer.fit(BoringModel())
    ckpt_path = os.path.join(tmp_path, [f for f in os.listdir(tmp_path) if f.endswith(".ckpt")][0])
    before_hash = _md5(ckpt_path)

    # inject a write failure mid-write
    real_save = _inject_write_failure()
    try:
        trainer2, _ = _make_trainer(tmp_path, save_top_k=-1, every_n_epochs=1, save_weights_only=True)
        with pytest.raises(OSError, match="injected filesystem fault"):
            trainer2.fit(BoringModel())
    finally:
        cloud_io.torch.save = real_save

    # recovery: save again without injection
    trainer3, _ = _make_trainer(tmp_path, max_epochs=2, save_top_k=-1, every_n_epochs=1, save_weights_only=True)
    trainer3.fit(BoringModel())

    files = os.listdir(tmp_path)
    ckpt_files = [f for f in files if f.endswith(".ckpt")]
    assert len(ckpt_files) >= 2
    new_path = os.path.join(tmp_path, ckpt_files[-1])
    assert _md5(new_path) != before_hash
    loaded = torch.load(new_path, weights_only=True)
    assert "state_dict" in loaded
    # no temp leftovers after recovery
    assert not [f for f in files if ".tmp" in f]


def test_model_checkpoint_remote_path_uses_direct_save(tmp_path):
    """Remote paths (non-local) fall back to the direct save path (no ``os.replace``)."""
    ckpt = ModelCheckpoint(dirpath="memory://ckpt", save_top_k=-1, every_n_epochs=1, save_weights_only=True)
    trainer = Trainer(
        default_root_dir=tmp_path,
        max_epochs=1,
        limit_train_batches=2,
        limit_val_batches=0,
        accelerator="cpu",
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        callbacks=[ckpt],
    )
    with mock.patch.object(trainer, "save_checkpoint", wraps=trainer.save_checkpoint) as save_mock:
        trainer.fit(BoringModel())

    # the remote path is passed directly (no temp file)
    args, _ = save_mock.call_args
    assert args[0].startswith("memory://")
    assert ".tmp" not in args[0]


def test_model_checkpoint_link_checkpoint_no_temp_leftovers(tmp_path):
    """The link path (copy fallback) is created atomically with no temp leftovers."""
    trainer = mock.Mock()
    file = tmp_path / "file.ckpt"
    file.touch()
    link = tmp_path / "last.ckpt"

    # force the copy fallback by making symlink fail
    with mock.patch("lightning.pytorch.callbacks.model_checkpoint.os.symlink", side_effect=OSError):
        ModelCheckpoint._link_checkpoint(trainer, filepath=str(file), linkpath=str(link))

    assert os.path.isfile(link)
    assert not [f for f in os.listdir(tmp_path) if ".tmp" in f]
