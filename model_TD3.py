import os
import gym
import json
import pickle
import warnings
import numpy as np
import pandas as pd
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.bench import Monitor
from stable_baselines.common.noise import (NormalActionNoise,
                                           OrnsteinUhlenbeckActionNoise,
                                           AdaptiveParamNoiseSpec)
from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines import TD3

from src.environment import DataFrameEnv
from src.bounds import get_bounds_limits
from src.callbacks import SaveOnBestTrainingRewardCallback
warnings.filterwarnings("ignore")

### NEED TO CHANGE MODEL NAME AND mode.load WITH CORRESPONDANT AGENT

# env settings
seed = 0

model_name = "TD3"
env_name = "TabularOptimization-{model_name}"
file_name = "%s_%s_%s" % ("{model_name}", env_name, str(seed))

# Create log dir
log_dir = "/tmp/gym/"
os.makedirs(log_dir, exist_ok=True)
# agents dir
os.makedirs("agents/", exist_ok=True)

# dataframe training environment
df = pd.read_csv("data/data_energy_train.csv")
test_df = pd.read_csv("data/data_energy_test.csv")

# dataframe atributes
target = "pe"
control_variables = ["at", "v"]
features = list(df.columns)
features.remove(target)
print("features /  Action Space: ", features)
print("control_variables: ", control_variables)
print("target: ", target)

# max number of episodes
max_episodes = 5 * int(1e5) + 1

# load model environment
path_model = "env_model/model_env.pkl"
with open(path_model, "rb") as file:
    model = pickle.load(file)

# bounds limits for variables
bounds = get_bounds_limits(df, alpha=1)
# dataframe enviroments
env = DataFrameEnv(df, model, features, control_variables, target, bounds)

# recommendations, context and control variables
rec_control_variables = ["rec_" + i for i in control_variables]
context_variables = env.context_variables

# monitor the complete environment
env = Monitor(env, log_dir)
# The algorithms require a vectorized environment to run
env = DummyVecEnv([lambda: env])
# Create the callback: check every 1000 steps
callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)

# actor critic agents
# the noise objects for DDPG
n_actions = env.action_space.shape[-1]
param_noise = None
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))

agent = TD3('MlpPolicy', env, action_noise=action_noise, verbose=1, tensorboard_log="./tensorboard/")
agent.learn(total_timesteps=max_episodes)
agent.save(f"agents/{model_name}")

# reload agent
agent = TD3.load(f"agents/{model_name}")

# configuración del entorno
test_env = DataFrameEnv(test_df, model, features, control_variables, target, bounds)
env = DummyVecEnv([lambda: test_env])

results = pd.DataFrame()
for i in range(len(test_df)):
    obs = np.array(test_df[features].iloc[i])
    obs = np.reshape(obs, (1, len(features)))
    action, _states = agent.predict(obs)
    _, rewards, done, info = test_env.step(action)
    print(rewards)
    # every episode render the new environment
    test_env.render()

    # states actions
    action_df = pd.DataFrame(action, columns=["rec_" + i for i in control_variables])
    for col in action_df.columns:
        action_df[col] = action_df[col].apply(float)

    # observation dataframe
    obs_df = pd.DataFrame(obs, columns=features)
    rewards_df = pd.DataFrame([rewards], columns=["rewards"])

    di = pd.concat([action_df, obs_df, rewards_df], axis=1)
    results = pd.concat([results, di], axis=0)
results.reset_index(drop=True, inplace=True)
df_opt = results[rec_control_variables + context_variables]

results["optimized"] = test_env.model.predict(df_opt)
results["prediction"] = model.predict(results[features])
results["real"] = test_df[target]
results["uplift"] = (results["optimized"] - results["real"]) / results["real"] * 100
results["mape"] = (
    (results["real"] - results["prediction"]) / results["real"] * 100
).apply(abs)

print("==================================================")
print("MAPE: ", results["mape"].mean(), "[%]")
print("Average rewards: ", results["rewards"].mean(), "[%]")
print("Uplift: ", results["uplift"].mean(), "[%]")
print("==================================================")

results["model_name"] = model_name
results.to_csv(f"results/test_results/{model_name}.csv", index=False)
