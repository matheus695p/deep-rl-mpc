import torch
import torch.nn as nn
import torch.nn.functional as F


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
        # activación de la herencia
        super(Actor, self).__init__()
        # construcción de la red
        self.layer_1 = nn.Linear(state_dim, 400)
        self.layer_2 = nn.Linear(400, 300)
        self.layer_3 = nn.Linear(300, action_dim)
        # cortar el valor de las acciones para que esten dentro de un limite
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
        # acotar las acciones en un rango [-1, 1] * self.max_action
        x = self.max_action * torch.tanh(self.layer_3(x))
        return x
