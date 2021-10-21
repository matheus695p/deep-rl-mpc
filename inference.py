import gym
import torch
import numpy as np
import pybullet_envs
from gym import wrappers
from src.evaluate import evaluate_policy
from src.utils import (mkdir)
from src.inference_module import TD3


env_name = "HalfCheetahBulletEnv-v0"
seed = 0
work_dir = mkdir('exp', 'brs')
monitor_dir = mkdir(work_dir, 'monitor')

file_name = "%s_%s_%s" % ("TD3", env_name, str(seed))
print("---------------------------------------")
print("Configuración: %s" % (file_name))
print("---------------------------------------")

eval_episodes = 10
save_env_vid = True

env = gym.make(env_name)

max_episode_steps = env._max_episode_steps
if save_env_vid:
    env = wrappers.Monitor(env, monitor_dir, force=True)
    # env = gym.wrappers.Monitor(
    #     env, monitor_dir, video_callable=lambda episode_id: True, force=True)

    env.reset()
env.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])
policy = TD3(state_dim, action_dim, max_action)

policy.load(file_name, './pytorch_models/')
_ = evaluate_policy(env, policy, eval_episodes=eval_episodes)
