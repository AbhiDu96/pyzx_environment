import numpy as np
from zx_env.env import zx_env


if __name__ == '__main__':

    env = zx_env(n_qubits=5, depth=250, max_steps=100)
    obs = env.reset()
    num_rules = env.action_space.n
    rule = np.random.randint(num_rules)
    position = 0

    obs = env.step(action=rule, position=position)
    pass