import numpy as np
import gymnasium
from gymnasium.spaces import Box
from gymnasium.envs.mujoco.ant_v5 import AntEnv


class RacingAntsEnv(AntEnv):
    def __init__(self, *args, n_agents=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_agents = n_agents

        obs_size = self.data.qpos.size + self.data.qvel.size
        obs_size -= 2 * self._exclude_current_positions_from_observation * n_agents
        obs_size += self.data.cfrc_ext[1:].size * self._include_cfrc_ext_in_observation

        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )

        self.observation_structure = {
            "skipped_qpos": 2 * self._exclude_current_positions_from_observation ,
            "qpos": self.data.qpos.size
            - 2 * self._exclude_current_positions_from_observation * n_agents,
            "qvel": self.data.qvel.size,
            "cfrc_ext": self.data.cfrc_ext[1:].size * self._include_cfrc_ext_in_observation,
        }

    def _get_obs(self):
        all = []
        poslen = self.data.qpos.size // self.n_agents
        vellen = self.data.qvel.size // self.n_agents
        cfrc = self.data.cfrc_ext[1:]
        extlen = self.data.cfrc_ext.shape[0] // self.n_agents

        for i in range(self.n_agents):
            position = self.data.qpos[i*poslen:(i+1)*poslen].flatten()
            velocity = self.data.qvel[i*vellen:(i+1)*vellen].flatten()

            if self._exclude_current_positions_from_observation:
                position = position[2:]

            if self._include_cfrc_ext_in_observation:
                contact_force = cfrc[i*extlen:(i+1)*extlen].flatten()
                a = np.concatenate((position, velocity, contact_force))
            else:
                a = np.concatenate((position, velocity))
            
            all.append(a)
        return np.concatenate(all)
    
gymnasium.envs.register(
    id="RacingAnts-v5",
    entry_point="customAntEnv:RacingAntsEnv",
)