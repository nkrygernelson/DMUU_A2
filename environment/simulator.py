### Task 6: Simulation environment
### Takes any decision-making policy as input and evaluates its average
### performance over E independent experiments (days).

import numpy as np
import pandas as pd

from SystemCharacteristics import get_fixed_data
from helper.v2_Checks import check_and_sanitize_action

price_df = pd.read_csv('data/v2_PriceData.csv')
occ1_df  = pd.read_csv('data/OccupancyRoom1.csv')
occ2_df  = pd.read_csv('data/OccupancyRoom2.csv')


def load_experiments():
    """
    Load 100 experiment days from the pre-drawn CSV files.

    Returns:
        list of dicts with keys 'price_previous', 'prices',
        'occupancy_r1', 'occupancy_r2'.
    """
    experiments = []
    for i in range(len(price_df)):
        row = price_df.iloc[i]
        experiments.append({
            'price_previous': float(row.iloc[0]),
            'prices':         [float(v) for v in row.iloc[1:].values],
            'occupancy_r1':   [float(v) for v in occ1_df.iloc[i].values],
            'occupancy_r2':   [float(v) for v in occ2_df.iloc[i].values],
        })
    return experiments


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
    fixed_data = get_fixed_data()
    T      = fixed_data['num_timeslots']

    prices  = day_data['prices']
    occ_r1  = day_data['occupancy_r1']
    occ_r2  = day_data['occupancy_r2']
    price_prev = day_data.get('price_previous', prices[0])

    # System parameters
    p_max    = fixed_data['heating_max_power']
    xi_exh   = fixed_data['heat_exchange_coeff']
    xi_loss  = fixed_data['thermal_loss_coeff']
    xi_conv  = fixed_data['heating_efficiency_coeff']
    xi_cool  = fixed_data['heat_vent_coeff']
    xi_occ   = fixed_data['heat_occupancy_coeff']
    eta_occ  = fixed_data['humidity_occupancy_coeff']
    eta_vent = fixed_data['humidity_vent_coeff']
    p_vent   = fixed_data['ventilation_power']
    T_out    = fixed_data['outdoor_temperature']

    T_low    = fixed_data['temp_min_comfort_threshold']
    T_ok     = fixed_data['temp_OK_threshold']
    T_high   = fixed_data['temp_max_comfort_threshold']
    H_thresh = fixed_data['humidity_threshold']
    vent_min = fixed_data['vent_min_up_time']

    # Initial state
    T1            = fixed_data['T1']
    T2            = fixed_data['T2']
    H             = fixed_data['H']
    vent_counter  = fixed_data['vent_counter']
    low_r1        = fixed_data['low_override_r1']
    low_r2        = fixed_data['low_override_r2']

    total_cost = 0.0

    for t in range(T):
        price_t = prices[t]
        occ1    = occ_r1[t]
        occ2    = occ_r2[t]

        state = {
            "T1":              T1,
            "T2":              T2,
            "H":               H,
            "Occ1":            occ1,
            "Occ2":            occ2,
            "price_t":         price_t,
            "price_previous":  price_prev,
            "vent_counter":    vent_counter,
            "low_override_r1": low_r1,
            "low_override_r2": low_r2,
            "current_time":    t,
        }

        action = check_and_sanitize_action(policy, state, {1: p_max, 2: p_max})
        p1 = action["HeatPowerRoom1"]
        p2 = action["HeatPowerRoom2"]
        V  = action["VentilationON"]

        # Overrule controllers 

        # High-temperature overrule: heater must be OFF if room is too hot
        if T1 >= T_high:
            p1 = 0.0
        if T2 >= T_high:
            p2 = 0.0

        # Low-temperature overrule: heater forced to max while override is active
        if low_r1:
            p1 = p_max
        if low_r2:
            p2 = p_max

        # Humidity overrule: ventilation forced ON when humidity exceeds threshold
        if H > H_thresh:
            V = 1

        # Ventilation inertia: must stay ON for at least vent_min consecutive hours
        if 0 < vent_counter < vent_min:
            V = 1

        # Cost 
        total_cost += price_t * (p1 + p2 + p_vent * V)

        # State transition 
        T_out_t = T_out[t]

        T1_new = (T1
                  - xi_exh  * (T1 - T2)
                  - xi_loss * (T1 - T_out_t)
                  + xi_conv * p1
                  - xi_cool * V
                  + xi_occ  * occ1)

        T2_new = (T2
                  - xi_exh  * (T2 - T1)
                  - xi_loss * (T2 - T_out_t)
                  + xi_conv * p2
                  - xi_cool * V
                  + xi_occ  * occ2)

        H_new = float(np.clip(H + eta_occ * (occ1 + occ2) - eta_vent * V, 0.0, 100.0))

        # Ventilation counter: accumulates while ON, resets when OFF
        vent_counter = vent_counter + 1 if V == 1 else 0

        # Low-override flags: activate below T_low, deactivate once T_ok is reached
        if T1_new < T_low:
            low_r1 = 1
        elif T1_new >= T_ok:
            low_r1 = 0

        if T2_new < T_low:
            low_r2 = 1
        elif T2_new >= T_ok:
            low_r2 = 0

        T1, T2, H = T1_new, T2_new, H_new
        price_prev = price_t

    return total_cost


def evaluate(policy, experiments):
    """
    Evaluate a policy over multiple experiments.

    Args:
        policy: module or object with a select_action(state) function.
        experiments: list of day_data dicts (length E).

    Returns:
        avg_cost (float), costs (list of float)
    """
    costs = [run_experiment(policy, day_data) for day_data in experiments]
    return float(np.mean(costs)), costs
