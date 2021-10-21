import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import pybullet_envs
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym import wrappers
from torch.autograd import Variable
from collections import deque
# Selección del dispositivo (CPU o GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    """
    Construimos una red neuronal para los actores del modelo y target,
    se saca el número de neuronas del paper que Adressing functions
    aproximators que esta
    en la carpeta documents/

    ...

    Attributes
    ----------
    object : herencia
        módulo de redes neuronales de pytorch
    Methods
    -------
    forward(x):
        propagar adelante la red.
    """

    def __init__(self, state_dim, action_dim, max_action):
        """
        Construcor de la red
        Parameters
        ----------
        state_dim : int
            dimension de los espacios de estados.
        action_dim : int
            numero de acciones posibles.
        max_action : int
            DESCRIPTION.

        Returns
        -------
        None.
        """
        super(Actor, self).__init__()
        self.layer_1 = nn.Linear(state_dim, 400)
        self.layer_2 = nn.Linear(400, 300)
        self.layer_3 = nn.Linear(300, action_dim)
        self.max_action = max_action

    def forward(self, x):
        """
        Propagar hacia adelante info (forward de la red), aplicando
        las funciones de activación
        Parameters
        ----------
        x : object
            red neuronal del actor.

        Returns
        -------
        x : object
            red neuronal del actor.
        """
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = self.max_action * torch.tanh(self.layer_3(x))
        return x


class Critic(nn.Module):
    """
    Construimos una red neuronal para los criticos, se saca el número
    de neuronas del paper que Adressing functions aproximators que esta
    en la carpeta documents/

    ...

    Attributes
    ----------
    object : herencia
        módulo de redes neuronales de pytorch
    Methods
    -------
    forward(x):
        propagar adelante la red.
    Q1(x, u):
        propagar adelante la red.
    """

    def __init__(self, state_dim, action_dim):
        """
        Construcor de la redes
        Parameters
        ----------
        state_dim : int
            dimension de los espacios de estados.
        action_dim : int
            numero de acciones posibles.
        Returns
        -------
        None.

        """
        super(Critic, self).__init__()
        # Definimos el primero de los Críticos como red neuronal profunda
        self.layer_1 = nn.Linear(state_dim + action_dim, 400)
        self.layer_2 = nn.Linear(400, 300)
        self.layer_3 = nn.Linear(300, 1)
        # Definimos el segundo de los Críticos como red neuronal profunda
        self.layer_4 = nn.Linear(state_dim + action_dim, 400)
        self.layer_5 = nn.Linear(400, 300)
        self.layer_6 = nn.Linear(300, 1)

    def forward(self, x, u):
        """
        Propagar hacia adelante info (forward de la red), aplicando
        las funciones de activación, además se debe realizar la concatenación
        de los estados con las acciones

        Parameters
        ----------
        x : tensor
            estados.
        u : tensor
            acciones.

        Returns
        -------
        x1 : object
            red de primero de criticos.
        x2 : object
            red de segundo de criticos.
        """
        xu = torch.cat([x, u], 1)
        # Propagación hacia adelante del primero de los Críticos
        x1 = F.relu(self.layer_1(xu))
        x1 = F.relu(self.layer_2(x1))
        x1 = self.layer_3(x1)
        # Propagación hacia adelante del segundo de los Críticos
        x2 = F.relu(self.layer_4(xu))
        x2 = F.relu(self.layer_5(x2))
        x2 = self.layer_6(x2)
        return x1, x2

    def Q1(self, x, u):
        """
        Método para devolver solo la propagación del primero de los criticos,
        esto para solo tener este valor Q. Esto es solo para hacer más
        entendible el código, dado que se puede usar fordward, pero es que
        solo necesitaremos hacer gradiente ascendente en Q1

        Parameters
        ----------
        x : tensor
            estados.
        u : tensor
            acciones.
        Returns
        -------
        x1 : object
            valor Q solo del primero de los criticos.

        """
        xu = torch.cat([x, u], 1)
        x1 = F.relu(self.layer_1(xu))
        x1 = F.relu(self.layer_2(x1))
        x1 = self.layer_3(x1)
        return x1


class TD3(object):
    """
    Construir todo el proceso de entrenamiento del módelo TD3 en una sola
    clase

    paso4: Esta parte la puede cagar revisar al final
    # batch_states, batch_next_states, batch_actions,
    batch_rewards, batch_dones = replay_buffer.sample(batch_size)


    ...

    Attributes
    ----------
    object : herencia
        objecto de entrenamiento del módelo

    Methods
    -------
    select_action(state):
        propagar adelante la red.
    train(replay_buffer, iterations, batch_size=100, discount=0.99,
          tau=0.005, policy_noise=0.2, noise_clipping=0.5, policy_freq=2):
        proceso de entrenamiento.
    save(filename, directory):
        guardar el módelo.
    load(filename, directory):
        cargar el módelo.

    """

    def __init__(self, state_dim, action_dim, max_action):
        """
        Constructor del entrenamiento del algoritmo

        Parameters
        ----------
        state_dim : int
            dimension de los espacios de estados.
        action_dim : int
            numero de acciones posibles.
        max_action : int
            máximo valor de las acciones.

        Returns
        -------
        None.

        """
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())
        self.max_action = max_action

    def select_action(self, state):
        """
        Seleccionar un acción en función del estado en que me encuentro
        proceso markoviano

        Parameters
        ----------
        state : array
            estado en el que me encuentro.

        Returns
        -------
        tensor de pytorch como estado.

        """
        state = torch.Tensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, iterations, batch_size=100,
              discount=0.99, tau=0.005, policy_noise=0.2,
              noise_clipping=0.5, policy_freq=2):
        """
        Método de entrenamiento

        Parameters
        ----------
        replay_buffer : list
            memoria.
        iterations : int
            número de veces que se va ejecutar el algoritmo.
        batch_size : int, optional
            batch size. The default is 100.
        discount : float, optional
            gamma del Q learning. The default is 0.99.
        tau : float, optional
            tau de la formula. The default is 0.005.
        policy_noise : float, optional
            ruido a añadir a la politica, para añadir exploración.
            The default is 0.2.
        noise_clipping : float, optional
            para que ruido no se vaya a la puta, recortar ruido.
            The default is 0.5.
        policy_freq : float, optional
            delay de las iteracion twin delayed. The default is 2.

        Returns
        -------
        None.

        """
        for it in range(iterations):

            # Paso 4: Tomamos una muestra de transiciones
            # (s, s’, a, r) de la memoria.
            memory = replay_buffer.sample(batch_size)

            # descomprimir
            batch_states = memory[0]
            batch_next_states = memory[1]
            batch_actions = memory[2]
            batch_rewards = memory[3]
            batch_dones = memory[4]
            # convertir a tensores para ser procesado por la red
            state = torch.Tensor(batch_states).to(device)
            next_state = torch.Tensor(batch_next_states).to(device)
            action = torch.Tensor(batch_actions).to(device)
            reward = torch.Tensor(batch_rewards).to(device)
            done = torch.Tensor(batch_dones).to(device)

            # Paso 5: A partir del estado siguiente s', el Actor del Target
            # ejecuta la siguiente acción a'.
            next_action = self.actor_target(next_state)

            # Paso 6: Añadimos ruido gaussiano a la siguiente acción a' y lo
            # cortamos para tenerlo en el rango de valores aceptado
            # por el entorno.
            noise = torch.Tensor(batch_actions).data.normal_(
                0, policy_noise).to(device)
            noise = noise.clamp(-noise_clipping, noise_clipping)
            next_action = (
                next_action + noise).clamp(-self.max_action, self.max_action)

            # Paso 7: Los dos Críticos del Target toman un par (s’, a’)
            # como entrada y devuelven dos Q-values Qt1(s’,a’) y Qt2(s’,a’)
            # como salida.
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)

            # Paso 8: Nos quedamos con el mínimo de los dos Q-values:
            # min(Qt1, Qt2). Representa el valor aproximado del estado
            # siguiente.
            target_Q = torch.min(target_Q1, target_Q2)

            # Paso 9: Obtenemos el target final de los dos Crítico del Modelo,
            # que es: Qt = r + γ * min(Qt1, Qt2), donde γ es el factor de
            # descuento.
            target_Q = reward + ((1-done) * discount * target_Q).detach()

            # Paso 10: Los dos Críticos del Modelo toman un par (s, a) como
            # entrada y devuelven dos Q-values Q1(s,a) y Q2(s,a) como salida.
            current_Q1, current_Q2 = self.critic(state, action)

            # Paso 11: Calculamos la pérdida procedente de los Crítico
            # del Modelo:
            # Critic Loss = MSE_Loss(Q1(s,a), Qt) + MSE_Loss(Q2(s,a), Qt)
            critic_loss = F.mse_loss(
                current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            # Paso 12: Propagamos hacia atrás la pérdida del crítico y
            # actualizamos los parámetros de los dos Crítico del Modelo
            # con un SGD.
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Paso 13: Cada dos iteraciones, actualizamos nuestro modelo
            # de Actor ejecutando el gradiente ascendente en la salida
            # del primer modelo crítico.
            if it % policy_freq == 0:
                actor_loss = - self.critic.Q1(state, self.actor(state)).mean()

                self.actor_optimizer.zero_grad()
                self.actor_optimizer.backward()
                self.actor_optimizer.step()

                # Paso 14: Todavía cada dos iteraciones, actualizamos
                # los pesos del Actor del Target usando el promedio Polyak.
                for param, target_param in zip(self.actor.parameters(),
                                               self.actor_target.parameters()):
                    target_param.data.copy_(
                        tau * param.data + (1-tau) * target_param.data)

                # Paso 15: Todavía cada dos iteraciones,
                # actualizamos los pesos del target del Crítico usando
                # el promedio Polyak.
                for param, target_param in zip(
                        self.critic.parameters(),
                        self.critic_target.parameters()):

                    target_param.data.copy_(
                        tau * param.data + (1-tau) * target_param.data)

    # Método para guardar el modelo entrenado
    def save(self, filename, directory):
        """
        Guardar los modelos de actores y de criticos
        Parameters
        ----------
        filename : string
            nombre.
        directory : string
            donde?.

        Returns
        -------
        None.

        """
        torch.save(self.actor.state_dict(), "%s/%s_actor.pth" %
                   (directory, filename))
        torch.save(self.critic.state_dict(), "%s/%s_critic.pth" %
                   (directory, filename))

    # Método para cargar el modelo entrenado
    def load(self, filename, directory):
        """
        Cargar los modelos de actores y de criticos
        Parameters
        ----------
        filename : string
            nombre.
        directory : string
            donde?.

        Returns
        -------
        None.

        """
        self.actor.load_state_dict(torch.load(
            "%s/%s_actor.pth" % (directory, filename)))
        self.critic.load_state_dict(torch.load(
            "%s/%s_critic.pth" % (directory, filename)))


def evaluate_policy(env, policy, eval_episodes=10):
    """
    Recive el entorno la politica y el número de episodios cada cuanto se
    evalua

    Parameters
    ----------
    env : gym.env
        entorno.
    policy : clase td3
        politicas.
    eval_episodes : int, optional
        número de cada cuanto evaluar. The default is 10.

    Returns
    -------
    avg_reward : float
        recompenza promedio en los episodios.
    """
    avg_reward = 0.
    for _ in range(eval_episodes):
        obs = env.reset()
        done = False
        while not done:
            action = policy.select_action(np.array(obs))
            obs, reward, done, _ = env.step(action)
            avg_reward += reward
    avg_reward /= eval_episodes
    print("------------------------------------------------")
    print("Recompensa promedio en el paso de Evaluación: %f" % (avg_reward))
    print("------------------------------------------------")
    return avg_reward
