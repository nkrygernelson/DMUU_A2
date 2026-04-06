### Task 6: Dummy policy — never turns on heaters or ventilation;
### leaves everything to the overrule controllers.


def select_action(state):
    HereAndNowActions = {
        "HeatPowerRoom1": 0,
        "HeatPowerRoom2": 0,
        "VentilationON": 0,
    }

    return HereAndNowActions
