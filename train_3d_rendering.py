import gym
import gfootball.env as football_env
import pandas as pd
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

ENV_NAME = '1_vs_1_easy'
# REPRESENTATION = 'simple115v2'
REPRESENTATION = 'extracted'

RENDER = False
NUM_ENVS = 4
TOTAL_TIMESTEPS = 100000
MODEL_SAVE_PATH = f"train5/ppo_sb3_{ENV_NAME}_fast"
TRAIN_INTERVAL = 10000
EVAL_EPISODES = 5

policy_kwargs = dict(
    net_arch=[dict(pi=[256, 256], vf=[256, 256])]
)

reward_records = []
loss_records = []

os.makedirs('train5', exist_ok=True)

def make_env():
    def _init():
        env = football_env.create_environment(
            env_name=ENV_NAME,
            representation=REPRESENTATION,
            rewards='scoring,checkpoints',
            render=RENDER
        )
        return env
    return _init

def evaluate(model, timesteps, episodes=1, render=True):
    eval_env = football_env.create_environment(
        env_name=ENV_NAME,
        representation='extracted',  # required for 3D rendering
        rewards='scoring,checkpoints',
        render=render  # <- this is sufficient for 3D window
    )
    for ep in range(episodes):
        obs = eval_env.reset()
        done = False
        episode_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)
            if render:
                import time
                time.sleep(0.03)  # slow down for human watching
            episode_reward += reward
        print(f"🎮 [Eval] Episode {ep + 1}: reward = {episode_reward}")
        reward_records.append({'timesteps': timesteps, 'episode': ep + 1, 'reward': episode_reward})


if __name__ == "__main__":
    env = SubprocVecEnv([make_env() for _ in range(NUM_ENVS)])

    model = PPO(
        policy='MlpPolicy',
        env=env,
        verbose=1,
        n_steps=256,
        batch_size=256,
        learning_rate=4e-5,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        max_grad_norm=0.76,
        n_epochs=2,
        policy_kwargs=policy_kwargs,
    )

    timesteps = 0
    while timesteps < TOTAL_TIMESTEPS:
        model.learn(total_timesteps=TRAIN_INTERVAL, reset_num_timesteps=False)
        timesteps += TRAIN_INTERVAL
        model.save(f"{MODEL_SAVE_PATH}_{timesteps}")
        print(f"Saved model at {timesteps} timesteps.")

        # Log and print detailed losses
        if hasattr(model, 'logger'):
            try:
                loss_info = model.logger.name_to_value
                record = {
                    'timesteps': timesteps,
                    'approx_kl': loss_info.get('train/approx_kl', None),
                    'clip_fraction': loss_info.get('train/clip_fraction', None),
                    'entropy_loss': loss_info.get('train/entropy_loss', None),
                    'policy_gradient_loss': loss_info.get('train/policy_gradient_loss', None),
                    'value_loss': loss_info.get('train/value_loss', None),
                    'total_loss': loss_info.get('train/loss', None)
                }
                loss_records.append(record)
                print(f"Loss at {timesteps}: {record}")
            except Exception as e:
                print(f"Error retrieving loss info: {e}")

        # evaluate(model, timesteps, episodes=EVAL_EPISODES)
        evaluate(model, timesteps, episodes=1, render=True)

    pd.DataFrame(reward_records).to_csv('train5/reward_log.csv', index=False)
    pd.DataFrame(loss_records).to_csv('train5/loss_log.csv', index=False)

    print("Training complete. Logs saved")
