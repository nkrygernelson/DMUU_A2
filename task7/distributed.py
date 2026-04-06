### Task 7: Distributed Decision-Making
### Lagrangian decomposition / distributed optimization for the mall
### (N=15 stores, shared power cap P_mall).

import numpy as np


def fetch_data():
    """
    Returns the fixed data for Task 7.
    THIS CODE SHOULD NOT BE CHANGED BY STUDENTS.
    """

    num_timeslots = 10

    return {
        'num_timeslots': num_timeslots,
        'P_mall': 45,
        'Temperature_reference': 21,
        'initial_temperature': 21.0,
        'heating_max_power': 3.0,
        'heat_exchange_coeff': 0.6,
        'heating_efficiency_coeff': 1.0,
        'thermal_loss_coeff': 0.1,
        'heat_vent_coeff': 0.7,
        'heat_occupancy_coeff': 0.02,
        'outdoor_temperature': [
            3 * np.sin(2 * np.pi * t / num_timeslots - np.pi / 2)
            for t in range(num_timeslots)
        ],
    }


def solve_distributed():
    """Task 7: implement distributed optimization algorithm here."""
    raise NotImplementedError
