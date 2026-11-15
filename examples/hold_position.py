"""A simulator smoke policy you can replace with an action-producing model."""

import numpy as np


class HoldPosition:
    def reset(self, seed):
        self.target = None

    def act(self, observation, instruction):
        if self.target is None:
            self.target = np.array(observation["joint_position"], dtype=float, copy=True)
            self.target[[6, 13]] = self.target[[6, 13]].clip(0.0, 1.0)
        return self.target.copy()


def create_policy(config):
    if config:
        raise ValueError("The hold-position example has no configuration")
    return HoldPosition()
