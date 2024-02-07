"""Adaloss scheduler for sigma parameter of gaussian heatmaps"""

from collections import deque

import numpy as np


class AdalossScheduler:
    """
    Scheduler for sigma parameter of gaussian heatmaps. In essence the scheduler increases the
    problem difficulty by decreasing the sigma parameter. Implicitly by doing this, it creates a
    training curriculum.
        Proposed in "Adaloss: Adaptive Loss Function for Landmark Localization" - Teixeira et al.
            (2019)

    Args:
        nb_landmarks (int): number of landmarks
        rho (float, optional): [description]. Defaults to 0.9.
        window (int, optional): [description]. Defaults to 3.
        non_increasing (bool, optional): [description]. Defaults to False.
    """

    def __init__(
        self, nb_landmarks: int, rho: float = 0.9, window: int = 3, non_increasing: bool = False
    ):
        self.nb_landmarks = nb_landmarks
        self.rho = rho
        self.window = window
        self.non_increasing = non_increasing
        self.losses: list[deque[float]] = [deque([]) for _ in range(nb_landmarks)]
        self.prev_variances: list[float] = []
        self.counter = 0

    def __call__(self, losses, sigmas):
        self.counter += 1
        for i in range(self.nb_landmarks):
            self.losses[i].append(losses[i])
            if self.counter == self.window:
                self.prev_variances.append(
                    (1 / self.window) * (losses[i] - np.mean(self.losses[i])) ** 2
                )
            if self.counter > self.window:
                self.losses[i].popleft()
                variance = (1 / self.window) * (losses[i] - np.mean(self.losses[i])) ** 2
                adjustment = self.rho * (1 - self.prev_variances[i] / variance)
                if self.non_increasing:
                    adjustment = min(adjustment, 0.0)
                sigmas[i, 0] += adjustment
                sigmas[i, 1] += adjustment
                self.prev_variances[i] = variance
