import numpy as np
import pandas as pd
import torch
from .data_gen import surface_for_params, POINTS, COLUMNS
from .heston import HESTON_PARAM_NAMES, PARAM_RANGES

def synthetic_target(random_seed: int):
    ''' draw one parameter set from the uniform ranges and return its implied-vol surface as a DataFrame '''
    np.random.seed(random_seed)
    true_params = {name: np.random.uniform(low, high) for name, (low, high) in PARAM_RANGES.items()}
    target_surface = surface_for_params(**true_params)
    return true_params, target_surface


if __name__ == "__main__":
    params, target_surface = synthetic_target(random_seed=100)
    print("true params:", params)
    print("target surface:", target_surface)
