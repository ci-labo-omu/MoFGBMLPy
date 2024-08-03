import numpy as np
random_gen = None


def init_random_gen(seed_val):
    global random_gen
    random_gen = np.random.Generator(np.random.MT19937(seed=seed_val))


def get_random_gen():
    if random_gen is None:
        raise ValueError('Random generator not initialized')
    return random_gen
