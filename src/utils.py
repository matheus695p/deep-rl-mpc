import os


def create_folders(save_models):
    """
    Crear carpetas donde se guardan los modelos

    Parameters
    ----------
    save_models : boolean
        si quieres crear las carpetas o no.

    Returns
    -------
    crea dos directorios /results y /pytorch_models.

    """
    if not os.path.exists("./results"):
        os.makedirs("./results")
    if save_models and not os.path.exists("./pytorch_models"):
        os.makedirs("./pytorch_models")


def mkdir(base, name):
    """
    Crear un carpeta con un base y name que seria subcarpeta

    Parameters
    ----------
    base : string
        carpeta.
    name : string
        subcarpeta.
    Returns
    -------
    path : string
        donde crearla.

    """
    path = os.path.join(base, name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path
