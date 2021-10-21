import numpy as np


class ReplayBuffer(object):
    """
    Dado que hay dos criticos por lado, prediciendo valores Q, se necesita
    una memoria de repeteción, esta clase hace esto, garda información
    de las transiciones

    ...

    Attributes
    ----------
    object : object
        objeto de memoria
    Methods
    -------
    add(transition):
        Agregar una transición en la memoria.
    sample(batch_size):
        Método de sample, toma de muestras aleatorias de la memoria.
    """

    def __init__(self, max_size=1e8):
        """
        Contructor de de la clase
        Parameters
        ----------
        max_size : int, optional
            Tamaño de la memoria. The default is 1e6.
        Returns
        -------
        Setting de la clase.
        """
        # almacenamiento
        self.storage = []
        # tamaño máximo
        self.max_size = max_size
        # pointer, indice para acceder a la memoria
        self.ptr = 0

    def add(self, transition):
        """
        Añade nuevas trancisiones a la memoria

        1 ) Verifica que la memoria no este llena e incrementar el valor del
        pointer para que reescriba la lista en el caso que se haya
        completado
        2) En el otro caso añade las transiciones
        Parameters
        ----------
        transition : list
            transiciones realizadas.
        Returns
        -------
        Añade nuevas transiciones en self.storage.
        """
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = transition
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(transition)

    def sample(self, batch_size):
        """
        Método de sample, toma de muestras aleatorias de la memoria

        Parameters
        ----------
        batch_size : int
            tamaño de la muestra.
        Returns
        -------
        s1 : np.array
            batch_states.
        s2 : np.array
            batch_next_states.
        s3 : np.array
            batch_actions.
        s4 : np.array
            batch_rewards.
        s5 : np.array
            batch_dones.
        """
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        # estados actuales
        batch_states = []
        # estados siguientes
        batch_next_states = []
        # acciones realizadas
        batch_actions = []
        # recompenzas del estado
        batch_rewards = []
        # 0 / 1 dependiendo si el episodio sigue activo
        batch_dones = []

        # para cada indice de la lista
        for i in ind:
            # para la posicion i extraer las varibles de la posición i
            state, next_state, action, reward, done = self.storage[i]
            # darle los valores como array para los tensores de pytorch
            batch_states.append(np.array(state, copy=False))
            batch_next_states.append(np.array(next_state, copy=False))
            batch_actions.append(np.array(action, copy=False))
            batch_rewards.append(np.array(reward, copy=False))
            batch_dones.append(np.array(done, copy=False))

        # convertir a array y sacar del método
        s1 = np.array(batch_states)
        s2 = np.array(batch_next_states)
        s3 = np.array(batch_actions)
        # reshape de los valores de recomenzas y dones
        s4 = np.array(batch_rewards).reshape(-1, 1)
        s5 = np.array(batch_dones).reshape(-1, 1)
        return s1, s2, s3, s4, s5
