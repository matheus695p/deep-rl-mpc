import torch
import torch.nn.functional as F
from src.actor import Actor
from src.critic import Critic
# from src.replayBuffer import ReplayBuffer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device, "para hacer entrenamiento")


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
        # actor del módelo
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        # actor del objectivo
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        # en el caso que se quiera cargar información en el actor del target
        self.actor_target.load_state_dict(self.actor.state_dict())
        # optimizador de los actores adaptative momentum
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

        # critico del módelo
        self.critic = Critic(state_dim, action_dim).to(device)
        # critico del objectivo target
        self.critic_target = Critic(state_dim, action_dim).to(device)
        # en el caso que se quiera cargar información en el actor del target
        self.critic_target.load_state_dict(self.critic.state_dict())
        # optimizador de los actores adaptative momentum
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

        # rango de las acciones
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
        # estado como tensor de pytorch y darselo a la gpu
        state = torch.Tensor(state.reshape(1, -1)).to(device)
        # hacer la predicción y retornar el numpy
        prediction = self.actor(state).cpu().data.numpy().flatten()
        return prediction

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
        # iterar sobre todas las transiciones
        for it in range(iterations):
            # Paso 4: Tomamos una muestra de transiciones (s, s’, a, r)
            # de la memoria.
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

            # Paso 5: A partir del estado siguiente s',
            # el Actor del Target ejecuta la siguiente acción a'.
            next_action = self.actor_target(next_state)

            # Paso 6: Añadimos ruido gaussiano a la siguiente acción a' y lo
            # cortamos para tenerlo en el rango de valores aceptado
            # por el entorno.
            noise = torch.Tensor(batch_actions).data.normal_(
                0, policy_noise).to(device)
            noise = noise.clamp(-noise_clipping, noise_clipping)
            # obtenemos siguiente acción
            next_action = (
                next_action + noise).clamp(-self.max_action, self.max_action)

            # Paso 7: Los dos Críticos del Target toman un par (s’, a’)
            # como entrada y devuelven dos Q-values Qt1(s’,a’) y
            # Qt2(s’,a’) como salida.
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)

            # Paso 8: Nos quedamos con el mínimo de los dos
            # Q-values: min(Qt1, Qt2). Representa el valor aproximado
            # del estado siguiente.
            target_Q = torch.min(target_Q1, target_Q2)

            # Paso 9: Obtenemos el target final de los dos Crítico del Modelo,
            # que es: Qt = r + γ * min(Qt1, Qt2), donde γ es el factor de
            # descuento.
            target_Q = reward + ((1-done) * discount * target_Q).detach()

            # Paso 10: Los dos Críticos del Modelo toman un par (s, a)
            # como entrada y devuelven dos Q-values Q1(s,a) y Q2(s,a)
            # como salida.
            current_Q1, current_Q2 = self.critic(state, action)

            # Paso 11: Calculamos la pérdida procedente de los Crítico
            # del Modelo:
            # Critic Loss = MSE_Loss(Q1(s,a), Qt) + MSE_Loss(Q2(s,a), Qt)
            critic_loss = F.mse_loss(
                current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            # Paso 12: Propagamos hacia atrás la pérdida del crítico y
            # actualizamos los parámetros de los dos Crítico del Modelo con
            # un SGD.
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Paso 13: Cada dos iteraciones, actualizamos nuestro modelo de
            # Actor ejecutando el gradiente ascendente en la salida del
            # primer modelo crítico.
            if it % policy_freq == 0:
                actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
                self.actor_optimizer.zero_grad()
                actor_loss.backward()  # OJO ME DEJÉ EL LOSS
                self.actor_optimizer.step()

                # Paso 14: Todavía cada dos iteraciones, actualizamos
                # los pesos del Actor del Target usando el promedio Polyak.
                for param, target_param in zip(
                        self.actor.parameters(),
                        self.actor_target.parameters()):

                    target_param.data.copy_(
                        tau * param.data + (1-tau) * target_param.data)

                # Paso 15: Todavía cada dos iteraciones, actualizamos los
                # pesos del target del Crítico usando el promedio Polyak.
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
