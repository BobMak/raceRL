import numpy as np
import gymnasium
from gymnasium.spaces import Box
from gymnasium.envs.mujoco.hopper_v5 import HopperEnv


class RacingHoppersEnv(HopperEnv):
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
            velocity = np.clip(self.data.qvel[i*vellen:(i+1)*vellen].flatten(), -10, 10)

            if self._exclude_current_positions_from_observation:
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
    id="RacingHoppers-v5",
    entry_point="customHopperEnv:RacingHoppersEnv",
)