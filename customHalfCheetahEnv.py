import numpy as np
import gymnasium
from gymnasium.spaces import Box
from gymnasium.envs.mujoco.half_cheetah_v5 import HalfCheetahEnv


class RacingHalfCheetahsEnv(HalfCheetahEnv):
    def __init__(self, *args, n_agents=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_agents = n_agents

        obs_size = (
            self.data.qpos.size
            + self.data.qvel.size
            - self._exclude_current_positions_from_observation * n_agents
        )
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )

        self.observation_structure = {
            "skipped_qpos": 1 * self._exclude_current_positions_from_observation * n_agents,
            "qpos": self.data.qpos.size
            - 1 * self._exclude_current_positions_from_observation * n_agents,
            "qvel": self.data.qvel.size,
        }

    def _get_obs(self):
        all = []
        poslen = self.data.qpos.size // self.n_agents
        vellen = self.data.qvel.size // self.n_agents
        
        for i in range(self.n_agents):
            position = self.data.qpos[i*poslen:(i+1)*poslen].flatten()
            velocity = self.data.qvel[i*vellen:(i+1)*vellen].flatten()

            if self._exclude_current_positions_from_observation:
                # position = position[2*self.n_agents:]
                position = position[1:]

            a = np.concatenate(
                (
                    position,
                    velocity,
                )
            ).ravel()
            all.append(a)
        return np.concatenate(all)
    
gymnasium.envs.register(
    id="RacingHalfCheetahs-v5",
    entry_point="customHalfCheetahEnv:RacingHalfCheetahsEnv",
)