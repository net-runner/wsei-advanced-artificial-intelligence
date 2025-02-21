import gymnasium as gym
import time
from stable_baselines3 import PPO, TD3, SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env

env = make_vec_env('BipedalWalker-v3', n_envs=4)

policy_kwargs = dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])])

model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1, learning_rate=1e-4, n_steps=1024, batch_size=64, gae_lambda=0.95, gamma=0.99, ent_coef=0.0)

model.learn(total_timesteps=1_000_000)

model.save("ppo_bipedalwalker")

model = PPO.load("ppo_bipedalwalker")

env = gym.make('BipedalWalker-v3', render_mode="human")

obs, _ = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    if done or truncated:
        obs, _ = env.reset()
    time.sleep(0.01)
env.close()

policy_kwargs = dict(net_arch=[dict(pi=[128, 128, 128], vf=[128, 128, 128])])

model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1, learning_rate=3e-4, n_steps=2048, batch_size=256, gae_lambda=0.9, gamma=0.98, ent_coef=0.01)

model.learn(total_timesteps=1_000_000)

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward} +/- {std_reward}")