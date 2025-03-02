import numpy as np

import gymnasium
from gymnasium.spaces import Box
from gymnasium.envs.mujoco.humanoid_v5 import HumanoidEnv


class RaceingHumanoidsEnv(HumanoidEnv):
    def __init__(self, *args, n_agents=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_agents = n_agents
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(self.observation_space.shape[0]*n_agents,), dtype=np.float64
        )

gymnasium.envs.register(
    id="RaceingHumanoids-v5",
    entry_point="customHumEnv:RaceingHumanoidsEnv",
)