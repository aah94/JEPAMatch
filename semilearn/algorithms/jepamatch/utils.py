# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
import torch

from semilearn.algorithms.flexmatch.utils import FlexMatchThresholdingHook
from semilearn.core.hooks import Hook

__all__ = ["FlexMatchThresholdingHook", "get_alpha_schedule", "JEPADiagnosticsHook"]


def get_alpha_schedule(current_step, total_steps, warmup_ratio=0.25):
    """
    Curriculum-to-representation schedule for Adaptive Class-wise SIGReg (paper Sec. 4.2).

    Returns alpha in [0, 1]:
      - [0, warmup_ratio * T):            alpha = 0.0   (Warmup Phase: global SIGReg)
      - [warmup_ratio * T, (1-warmup_ratio) * T): cosine ramp 0.0 -> 1.0
      - [(1-warmup_ratio) * T, T]:         alpha = 1.0   (Main Phase: fully class-wise)

    `warmup_ratio` corresponds to the paper's T_warm (as a fraction of total
    training iterations T); the symmetric ramp reduces to the original
    fixed quarter/three-quarter split at the default warmup_ratio=0.25.
    """
    q1 = warmup_ratio * total_steps
    q3 = (1.0 - warmup_ratio) * total_steps

    if current_step < q1:
        return 0.0
    elif current_step >= q3:
        return 1.0
    else:
        progress = (current_step - q1) / (q3 - q1)
        return 0.5 * (1.0 - torch.cos(torch.tensor(torch.pi * progress))).item()


class JEPADiagnosticsHook(Hook):
    """
    Optional periodic dump of per-step pseudo-labeling diagnostics
    (utilization, accuracy, class-count imbalance) produced by JEPAMatch's
    train_step, used to reproduce the paper's convergence / pseudo-label
    quality / max-class-count figures. Fully opt-in via `args.save_train_info`
    so it never affects other algorithms or default JEPAMatch runs.
    """

    def __init__(self, save_every=1000):
        self.save_every = save_every
        self.records = []

    def after_train_step(self, algorithm):
        info = getattr(algorithm, "last_train_info", None)
        if info is None:
            return
        self.records.append(info)

        if algorithm.it % self.save_every == 0:
            save_path = os.path.join(algorithm.save_dir, algorithm.save_name)
            os.makedirs(save_path, exist_ok=True)
            torch.save(self.records, os.path.join(save_path, "train_diagnostics.pt"))

    def after_run(self, algorithm):
        save_path = os.path.join(algorithm.save_dir, algorithm.save_name)
        os.makedirs(save_path, exist_ok=True)
        torch.save(self.records, os.path.join(save_path, "train_diagnostics.pt"))
