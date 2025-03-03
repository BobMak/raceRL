import numpy as np

import gymnasium
from gymnasium.spaces import Box
from gymnasium.envs.mujoco.humanoid_v5 import HumanoidEnv


class RacingHumanoidsEnv(HumanoidEnv):
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

    def _get_obs(self):
        all = []
        poslen = self.data.qpos.size // self.n_agents
        vellen = self.data.qvel.size // self.n_agents
        cominlen = (self.data.cinert.shape[0]-1) // self.n_agents
        comvlen = (self.data.cvel.shape[0] - 1) // self.n_agents
        actlen = self.data.qvel.size // self.n_agents
        extlen = self.data.cfrc_ext.shape[0] // self.n_agents
        cinert = self.data.cinert[1:]
        cvel = self.data.cvel[1:]
        cfrc = self.data.cfrc_ext[1:]
        for i in range(self.n_agents):
            position = self.data.qpos[i*poslen:(i+1)*poslen].flatten()
            velocity = self.data.qvel[i*vellen:(i+1)*vellen].flatten()

            if self._include_cinert_in_observation is True:
                com_inertia = cinert[i*cominlen:(i+1)*cominlen].flatten()
            else:
                com_inertia = np.array([])
            if self._include_cvel_in_observation is True:
                com_velocity = cvel[i*comvlen:(i+1)*comvlen].flatten()

            else:
                com_velocity = np.array([])

            if self._include_qfrc_actuator_in_observation is True:
                actuator_forces = self.data.qfrc_actuator[6+i*actlen:(i+1)*actlen].flatten()

            else:
                actuator_forces = np.array([])
            if self._include_cfrc_ext_in_observation is True:
                external_contact_forces = cfrc[i*extlen:(i+1)*extlen].flatten()
            else:
                external_contact_forces = np.array([])

            if self._exclude_current_positions_from_observation:
                # position = position[2*self.n_agents:]
                position = position[2:]

            a = np.concatenate(
                (
                    position,
                    velocity,
                    com_inertia,
                    com_velocity,
                    actuator_forces,
                    external_contact_forces,
                )
            )
            all.append(a)
        return np.concatenate(all)
    
gymnasium.envs.register(
    id="RacingHumanoids-v5",
    entry_point="customHumEnv:RacingHumanoidsEnv",
)