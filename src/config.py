import warnings
from argparse import ArgumentParser
warnings.filterwarnings("ignore", category=DeprecationWarning)


def arguments_parser():
    """
    El parser de argumentos de parámetros que hay que setiar para entrenar
    una red deep renewal
    Returns
    -------
    args : argparser
        argparser con todos los parámetros del modelo.
    """
    # argumentos
    parser = ArgumentParser()
    parser.add_argument(
        "-f", "--fff", help="a dummy argument to fool ipython", default="1")
    # agregar donde correr y guardar datos
    parser.add_argument('--model-save-dir', type=str, default="saved_models")
    parser.add_argument('--learning-rate', type=float, default=1e-2)
    parser.add_argument('--max-epochs', type=int, default=10)
    # solo para Deep Renewal Processes
    parser.add_argument('--forecast-type', type=str, default="hybrid")

    args = parser.parse_args()
    return args
