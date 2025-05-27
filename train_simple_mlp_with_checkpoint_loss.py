from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
import gfootball.env as football_env
import os
import pandas as pd

ENV_NAME = '1_vs_1_easy'
REPRESENTATION = 'simple115v2'
RENDER = False
NUM_ENVS = 5
TOTAL_TIMESTEPS = 1000000

OUTPUT_DIR = './output_simple115_mlp_checkpoint'
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "ppo_model")
TRAIN_INTERVAL = 10000
EVAL_EPISODES = 5

policy_kwargs = dict(
    net_arch=[dict(pi=[256, 256], vf=[256, 256])]
)

reward_records = []
loss_records = []

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'traces'), exist_ok=True)

def make_env():
    def _init():
        env = football_env.create_environment(
            env_name=ENV_NAME,
            representation=REPRESENTATION,
            render=RENDER,
            rewards='scoring,checkpoints',
            other_config_options={
                'dump_full_episodes': False,
                'write_video': False,
                'tracesdir': os.path.join(OUTPUT_DIR, 'traces')
            }
        )
        return env
    return _init

def evaluate(model, timesteps, episodes=5):
    eval_env = football_env.create_environment(
        env_name=ENV_NAME,
        representation=REPRESENTATION,
        render=False,
        rewards='scoring,checkpoints',
        other_config_options={
            'dump_full_episodes': False,
            'write_video': False,
            'tracesdir': os.path.join(OUTPUT_DIR, 'traces')
        }
    )
    for ep in range(episodes):
        obs = eval_env.reset()
        done = False
        episode_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)
            episode_reward += reward
        print(f"Evaluation Episode {ep + 1}: total reward {episode_reward}")
        reward_records.append({'timesteps': timesteps, 'episode': ep + 1, 'reward': episode_reward})

if __name__ == "__main__":
    env = SubprocVecEnv([make_env() for _ in range(NUM_ENVS)])

    model = PPO(
        policy='MlpPolicy',
        env=env,
        verbose=1,
        n_steps=512,
        batch_size=512,
        learning_rate=0.0002,
        gamma=0.997,
        gae_lambda=0.95,
        clip_range=0.115,
        ent_coef=0.00155,
        max_grad_norm=0.76,
        n_epochs=2,
        policy_kwargs=policy_kwargs,
    )

    timesteps = 0
    while timesteps < TOTAL_TIMESTEPS:
        model.learn(total_timesteps=TRAIN_INTERVAL, reset_num_timesteps=False)
        timesteps += TRAIN_INTERVAL

        model_path = f"{MODEL_SAVE_PATH}_{timesteps}.zip"
        model.save(model_path)
        print(f"Saved model at {timesteps} timesteps.")

        if hasattr(model, 'logger'):
            try:
                loss_info = model.logger.name_to_value
                loss_records.append({
                    'timesteps': timesteps,
                    'approx_kl': loss_info.get('train/approx_kl', None),
                    'clip_fraction': loss_info.get('train/clip_fraction', None),
                    'entropy_loss': loss_info.get('train/entropy_loss', None),
                    'policy_gradient_loss': loss_info.get('train/policy_gradient_loss', None),
                    'value_loss': loss_info.get('train/value_loss', None),
                    'total_loss': loss_info.get('train/loss', None)
                })
                print(f"Loss at {timesteps}: {loss_records[-1]}")
            except Exception as e:
                print(f"Error retrieving loss info: {e}")

        evaluate(model, timesteps, episodes=EVAL_EPISODES)

    pd.DataFrame(reward_records).to_csv(os.path.join(OUTPUT_DIR, 'reward_log.csv'), index=False)
    pd.DataFrame(loss_records).to_csv(os.path.join(OUTPUT_DIR, 'loss_log.csv'), index=False)

    print("Training complete. Logs saved to:", OUTPUT_DIR)
