### Task 6: Simulation environment
### Takes any decision-making policy as input and evaluates its average
### performance over E independent experiments (days).

from helper.SystemCharacteristics import SystemCharacteristics
import numpy as np


def run_experiment(policy, day_data):
    """
    Run a single experiment (day) with the given policy.

    Args:
        policy: module or object with a select_action(state) function.
        day_data: dict with keys 'prices', 'occupancy_r1', 'occupancy_r2'
                  each a list of 10 hourly values.

    Returns:
        total_cost (float)
    """
    raise NotImplementedError


def evaluate(policy, experiments):
    """
    Evaluate a policy over multiple experiments.

    Args:
        policy: module or object with a select_action(state) function.
        experiments: list of day_data dicts (length E).

    Returns:
        avg_cost (float), costs (list of float)
    """
    raise NotImplementedError
