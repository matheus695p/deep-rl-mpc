import gym
import time
import torch
import pickle
import warnings
import numpy as np
import pandas as pd
import pybullet_envs
from gym import wrappers
from src.td3 import TD3
from src.environment import DataFrameEnv
from src.replayBuffer import ReplayBuffer
from src.evaluate import evaluate_policy
from src.utils import create_folders, mkdir
from src.visualizations import training_td3_results
from src.bounds import get_bounds_limits

warnings.filterwarnings("ignore")

# env settings
seed = 0
env_name = "TabularOptimization"
file_name = "%s_%s_%s" % ("TD3", env_name, str(seed))

# dataframe training environment
df = pd.read_csv("data/data_energy.csv")

# dataframe atributes
target = "pe"
control_variables = ["at", "v"]
features = list(df.columns)
features.remove(target)
print("features /  Action Space: ", features)
print("control_variables: ", control_variables)
print("target: ", target)

# load model environment
path_model = "env_model/model_env.pkl"
with open(path_model, "rb") as file:
    model = pickle.load(file)
# bounds limits for variables
bounds = get_bounds_limits(df)
# Número de of iteraciones/timesteps durante las cuales el modelo elige una
# acción al azar, y después de las cuales comienza a usar la red de políticas
start_timesteps = 1e4
# Con qué frecuencia se realiza el paso de evaluación
# (después de cuántos pasos timesteps)
eval_freq = 1e3
# Número total de iteraciones/timesteps
max_timesteps = 5e6
# Check Boolean para saber si guardar o no el modelo pre-entrenado
save_models = True
# Ruido de exploración: desviación estándar del ruido de exploración gaussiano
expl_noise = 0.1
batch_size = 100  # Tamaño del bloque
# Factor de descuento gamma, utilizado en el cáclulo de la recompensa de
# descuento total
discount = 0.99
# Ratio de actualización de la red de objetivos
tau = 0.005
# Desviación estándar del ruido gaussiano añadido a las acciones
# para fines de exploración
policy_noise = 0.2
# Valor máximo de ruido gaussiano añadido a las acciones (política)
noise_clip = 0.5
# Número de iteraciones a esperar antes de actualizar la red de políticas
# (actor modelo)
policy_freq = 2

# configuración del entorno
env = DataFrameEnv(df, model, features, control_variables, target, bounds)

print(env.observation_space.shape)
print(env.action_space.shape)
print(env.observation_space)
print(env.action_space)

# crear folders si no existe
create_folders(save_models)

# reproductibilidad
env.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
env.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

# traer desde el entorno los valores del espacio de acciones
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

# crear la politica, replay buffer y como se van a evaluar
policy = TD3(state_dim, action_dim, max_action)
replay_buffer = ReplayBuffer()
evaluations = [evaluate_policy(env, policy)]

# crear carpetas de monitoreo
work_dir = mkdir("exp", "brs")
monitor_dir = mkdir(work_dir, "monitor")

# máximo numéro de episodios
max_episode_steps = env._max_episode_steps

# inicializar las variables de entrenamiento
total_timesteps = 0
timesteps_since_eval = 0
episode_num = 0
done = True
t0 = time.time()

# es para que existan antes del episodio cero
episode_reward = 0
episode_timesteps = 0

# Iniciamos el bucle principal con un total de 500,000 timesteps
while total_timesteps < max_timesteps:
    # Si el episodio ha terminado
    if done:
        # Si no estamos en la primera de las iteraciones, arrancamos
        # el proceso de entrenar el modelo
        if total_timesteps != 0:
            print(
                f"Total Timesteps: {total_timesteps}",
                f"Episode Num: {episode_num}",
                f"Reward: {episode_reward}",
            )
            # ajustar pesos de los actores y criticos
            policy.train(
                replay_buffer,
                episode_timesteps,
                batch_size,
                discount,
                tau,
                policy_noise,
                noise_clip,
                policy_freq,
            )

        # Evaluamos el episodio y guardamos la política si han pasado
        # las iteraciones necesarias
        if timesteps_since_eval >= eval_freq:
            timesteps_since_eval %= eval_freq
            evaluations.append(evaluate_policy(env, policy))
            policy.save(file_name, directory="./pytorch_models")
            np.save("./results/%s" % (file_name), evaluations)

        # Cuando el entrenamiento de un episodio finaliza, reseteamos el
        # entorno
        obs = env.reset()
        # Configuramos el valor de done a False
        done = False
        # Configuramos la recompensa y el timestep del episodio a cero
        episode_reward = 0
        episode_timesteps = 0
        episode_num += 1

    # Antes de los 10000 timesteps, ejectuamos acciones aleatorias
    if total_timesteps < start_timesteps:
        action = env.action_space.sample()

    else:  # Después de los 10000 timesteps, cambiamos al modelo
        action = policy.select_action(np.array(obs))
        # Si el valor de explore_noise no es 0, añadimos ruido a la acción
        # y lo recortamos en el rango adecuado
        if expl_noise != 0:
            action = (
                action + np.random.normal(0, expl_noise, size=env.action_space.shape[0])
            ).clip(env.action_space.low, env.action_space.high)

    # El agente ejecuta una acción en el entorno y alcanza el siguiente
    # estado y una recompensa
    new_obs, reward, done, _ = env.step(action)
    # Comprobamos si el episodio ha terminado
    done_bool = 0 if episode_timesteps + 1 == env._max_episode_steps else float(done)
    # Incrementamos la recompensa total
    episode_reward += reward
    # Almacenamos la nueva transición en la memoria de repetición de
    # experiencias (ReplayBuffer)
    replay_buffer.add((obs, new_obs, action, reward, done_bool))
    # Actualizamos el estado, el timestep del número de episodio,
    # el total de timesteps y el número de pasos desde la última
    # evaluación de la política
    obs = new_obs
    episode_timesteps += 1
    total_timesteps += 1
    timesteps_since_eval += 1

# Añadimos la última actualización de la política a la lista de evaluaciones
# previa y guardamos nuestro modelo
evaluations.append(evaluate_policy(env, policy))
# ver resultados
training_td3_results(evaluations, max_timesteps, name=env_name)

if save_models:
    policy.save("%s" % (file_name), directory="./pytorch_models")
np.save("./results/%s" % (file_name), evaluations)
