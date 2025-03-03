import numpy as np

import gymnasium
from gymnasium.spaces import Box
from gymnasium.envs.mujoco.humanoid_v5 import HumanoidEnv


class RaceingHumanoidsEnv(HumanoidEnv):
    def __init__(self, *args, n_agents=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_agents = n_agents
        obs_size = self.data.qpos.size + self.data.qvel.size
        obs_size -= 2 * self._exclude_current_positions_from_observation * n_agents
        obs_size += self.data.cinert[1:].size *  self._include_cinert_in_observation
        obs_size += self.data.cvel[1:].size *  self._include_cvel_in_observation
        obs_size += (self.data.qvel.size - 6 * n_agents) *  self._include_qfrc_actuator_in_observation
        obs_size += self.data.cfrc_ext[1:].size *  self._include_cfrc_ext_in_observation
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )
        # self.observation_space = Box(
        #     low=-np.inf, high=np.inf, shape=(self.observation_space.shape[0]*n_agents,), dtype=np.float64
        # )
        self.observation_structure = {
            "skipped_qpos": 2 * self._exclude_current_positions_from_observation,
            "qpos": self.data.qpos.size
            - 2 * self._exclude_current_positions_from_observation * n_agents,
            "qvel": self.data.qvel.size,
            "cinert": self.data.cinert[1:].size * self._include_cinert_in_observation,
            "cvel": self.data.cvel[1:].size * self._include_cvel_in_observation,
            "qfrc_actuator": (self.data.qvel.size - 6 * n_agents)
            * self._include_qfrc_actuator_in_observation,
            "cfrc_ext": self.data.cfrc_ext[1:].size * self._include_cfrc_ext_in_observation,
            "ten_length": 0,
            "ten_velocity": 0,
        }

gymnasium.envs.register(
    id="RaceingHumanoids-v5",
    entry_point="customHumEnv:RaceingHumanoidsEnv",
)