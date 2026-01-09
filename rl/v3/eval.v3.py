import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from matcha_env_v3 import MatchaBalanceEnvUprightV3

MODEL_PATH = r"D:\prj\Matcha\logs_ppo_v4\run_20251211-164113\ppo_matcha_v4_final.zip"
VECNORM_PATH = r"D:\prj\Matcha\logs_ppo\v3_upright_balance_optimized\run_20251211-154331\vecnormalize.pkl"
URDF_PATH = r"D:\prj\Matcha\hardware\balance_robot.urdf"

N_EVAL_EPISODES = 20

# ----------------------------------------------------------------
# Load env + normalization
# ----------------------------------------------------------------
def make_env():
    return MatchaBalanceEnvUprightV3(
        urdf_path=URDF_PATH,
        render=False,
        max_episode_steps=2000
    )

env = DummyVecEnv([make_env])
env = VecNormalize.load(VECNORM_PATH, env)
env.training = False
env.norm_reward = False

# ----------------------------------------------------------------
# Load PPO model
# ----------------------------------------------------------------
model = PPO.load(MODEL_PATH, env=env)

# ----------------------------------------------------------------
# Evaluation loop
# ----------------------------------------------------------------
episode_rewards = []
episode_lengths = []
success_count = 0

for ep in range(N_EVAL_EPISODES):
    obs = env.reset()
    done = False
    total_reward = 0
    steps = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward[0]
        steps += 1

    episode_rewards.append(total_reward)
    episode_lengths.append(steps)

    if steps >= 1999:
        success_count += 1  # balanced full episode

# ----------------------------------------------------------------
# Print evaluation results
# ----------------------------------------------------------------
print("===== Evaluation Results =====")
print(f"Average reward:      {np.mean(episode_rewards):.2f}")
print(f"Average episode len: {np.mean(episode_lengths):.1f}")
print(f"Success rate:        {success_count}/{N_EVAL_EPISODES}  ({success_count / N_EVAL_EPISODES * 100:.1f}%)")
print("Std reward:", np.std(episode_rewards))
print("Std length:", np.std(episode_lengths))
