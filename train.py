from stable_baselines3 import SAC
import gymnasium as gym


env = gym.make('Humanoid-v5')
model = SAC('MlpPolicy', env, verbose=1, learning_starts=0, ent_coef=0.2)
model.learn(total_timesteps=1000)
model.save("humanoid")