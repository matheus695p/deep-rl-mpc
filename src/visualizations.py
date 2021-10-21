import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('dark_background')


def training_td3_results(evaluations, max_timesteps, name="algo"):
    """
    Resultados de entrenamientos
    Parameters
    ----------
    evaluations : list
        evaluaciones en función de los episodios.
    max_timesteps : int
        máximo de iteraciones.
    name : int
        titulo.

    Returns
    -------
    plot de los resultados.

    """

    letter_size = 20
    data = pd.DataFrame(evaluations, columns=["recomenpenzas"])
    delta = max_timesteps / (len(data) - 1)
    episodes_data = [i * delta for i in range(0, (len(data)))]
    data["episodios"] = pd.Series(episodes_data)
    fig, ax = plt.subplots(1, figsize=(20, 12))
    sns.lineplot(data=data, x="episodios", y="recomenpenzas")
    ax.set_xlabel('episodios', fontname="Arial", fontsize=letter_size)
    ax.set_ylabel('recomenpenzas', fontname="Arial",
                  fontsize=letter_size+2)
    ax.set_title(f"Historial de entrenamiento: {name}",
                 fontname="Arial", fontsize=letter_size+10)
    ax.legend(['recompenzas en función' + '\n' + 'de episodios',
               'predicción'], loc='upper left',
              prop={'size': letter_size+5})
    # Tamaño de los ejes
    for tick in ax.get_xticklabels():
        tick.set_fontsize(letter_size)
    for tick in ax.get_yticklabels():
        tick.set_fontsize(letter_size)
