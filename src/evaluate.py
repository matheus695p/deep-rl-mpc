import numpy as np


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
    avg_reward = 0
    for _ in range(eval_episodes):
        obs = env.reset()
        done = False
        while not done:
            action = policy.select_action(np.array(obs))
            obs, reward, done, _ = env.step(action)
            avg_reward += reward
    avg_reward /= eval_episodes
    print("-------------------------------------------------")
    print("Recompensa promedio en el paso de Evaluación: %f" % (avg_reward))
    print("-------------------------------------------------")
    return avg_reward
