# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from copy import deepcopy
from pathlib import Path

from ..utils import BUTTON_MAPPING, NUM_CLASSES
from .model_config import ENCODER, TRANSFORMER


def experiment_config() -> dict:
    """Full Brain2Qwerty V1 configuration (SpanishBCBL, MEG)."""
    workspace = Path(__file__).resolve().parents[2] / "workspace"
    study_path = os.environ.get("BRAIN2QWERTY_STUDIES", str(workspace / "data"))
    cache = os.environ.get("BRAIN2QWERTY_CACHE", str(workspace / "cache"))
    results = os.environ.get("BRAIN2QWERTY_RESULTS", str(workspace / "runs"))
    return {
        "output_dir": results,
        "seed": 33,
        "n_epochs": 300,
        "patience": 30,
        "save_checkpoints": True,
        "data": {
            "study": {
                "name": "Pinet2024Meg",
                "path": study_path,
                "infra": {"folder": cache},
                "infra_timelines": {"folder": cache, "cluster": None},
            },
            "transforms": [
                {"name": "SpanishBCBLPreprocessing"},
                {"name": "Brain2QwertyV1Splitter", "seed": 1},
            ],
            "neuro": {
                "name": "MegExtractor",
                "frequency": 50,
                "filter": (0.1, 20.0),
                "baseline": (0.0, 0.2),
                "apply_proj": False,
                "clamp": 5,
                "scaler": "RobustScaler",
                "allow_maxshield": True,  # some SpanishBCBL recordings are MaxShield raw
                "infra": {"folder": cache, "cluster": None},
            },
            "feature": {
                "name": "LabelEncoder",
                "aggregation": "trigger",
                "predefined_mapping": BUTTON_MAPPING,
                "event_types": "Keystroke",
                "event_field": "button",
                "return_one_hot": False,
            },
            "num_classes": NUM_CLASSES,
            "start": -0.2,
            "duration": 0.5,
            "batch_size": 64,
            "val_batch_size": 2048,
            "test_batch_size": 2048,
            "num_workers": 16,
            "pin_memory": True,
            "persistent_workers": True,
        },
        "brain_model_config": deepcopy(ENCODER),
        "transformer_config": deepcopy(TRANSFORMER),
        "loss": {"name": "CrossEntropyLoss"},
        "optimizer": {
            "name": "LightningOptimizer",
            "optimizer": {"name": "AdamW", "lr": 5e-5, "kwargs": {"weight_decay": 1e-4}},
            "scheduler": {
                "name": "OneCycleLR",
                # we found this LR to be more stable than the one reported in paper
                "kwargs": {"max_lr": 5e-5, "pct_start": 0.1},
            },
            "interval": "step",
        },
    }


def debug_config() -> dict:
    """Smoke-test config: one timeline, 2 epochs, single GPU, no checkpoints."""
    cfg = experiment_config()
    cfg["data"]["study"]["query"] = "timeline_index == 0"
    cfg["n_epochs"] = 2
    cfg["patience"] = 2
    cfg["devices"] = 1
    cfg["save_checkpoints"] = False
    return cfg
