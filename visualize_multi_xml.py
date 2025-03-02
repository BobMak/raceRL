
import numpy as np
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

class TwoAgentMujocoEnv(MujocoEnv):
    # Custom metadata for the environment
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 20,
    }

    def __init__(self, model_path, frame_skip=5):
        super().__init__(model_path=model_path, frame_skip=frame_skip)

        # Assuming each agent has a state space that is a Box (modify to your needs)
        self.agent_observation_space = Box(low=-np.inf, high=np.inf, shape=(self.observation_space.shape[0],), dtype=np.float32)

        # Assuming each agent has the same action space (modify to your needs)
        self.agent_action_space = self.action_space

    def reset_model(self):
        # Reset the Mujoco simulation
        self.sim.reset()

        # Reset individual agent states (modify to your needs)
        self.agent_1_state = np.zeros(self.agent_observation_space.shape)
        self.agent_2_state = np.zeros(self.agent_observation_space.shape)

        return self._get_obs(), {}

    def step(self, action):
        # Handle actions for both agents
        agent_1_action, agent_2_action = action[0], action[1]

        # Take a step for each agent (this depends on how you want each agent to interact with the environment)
        self.agent_1_state = self._apply_action(self.agent_1_state, agent_1_action)
        self.agent_2_state = self._apply_action(self.agent_2_state, agent_2_action)

        # Collect observations for each agent (could be from the same state or different)
        observation = self._get_obs()

        # Calculate rewards, done status, and info (simplified here)
        reward = self._compute_reward()
        done = False  # Set when the agents should stop (e.g., based on episode length)
        info = {}

        return observation, reward, done, info

    def _apply_action(self, state, action):
        # Apply action to agent's state (you'll need to implement this based on the specifics of your model)
        new_state = state + action  # This is just a placeholder; adapt it for your environment
        return new_state

    def _get_obs(self):
        # Combine the observations for both agents (this could be from a shared or separate space)
        return np.concatenate([self.agent_1_state, self.agent_2_state])

    def render(self, mode="human"):
        # Render logic for two agents (depending on Mujoco setup)
        if mode == "human":
            super().render(mode=mode)
        elif mode == "rgb_array":
            # Render the environment as an image (you could combine both agent views here)
            return super().render(mode=mode)
        elif mode == "depth_array":
            # Render depth info
            return super().render(mode=mode)

# Now use the environment with two agents
env = TwoAgentMujocoEnv(model_path="half_cheetah_multi.xml", frame_skip=5)

# Reset the environment
observation, _ = env.reset()

# Simulate steps for both agents
done = False
while not done:
    action = [env.agent_action_space.sample(), env.agent_action_space.sample()]  # Sample actions for both agents
    observation, reward, done, info = env.step(action)

    # Render the environment (you can choose the mode here)
    env.render(mode="human")

# Close the environment when done
env.close()


