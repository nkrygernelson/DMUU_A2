# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 11:14:31 2025

@author: geots
"""
from policies import sp_policy
from policies import adp_policy
# The state will be provided by the environment as the following dictionary

# state = {
#     "T1": ..., #Temperature of room 1
#     "T2": ..., #Temperature of room 2
#     "H": ..., #Humidity
#     "Occ1": ..., #Occupancy of room 1
#     "Occ2": ..., #Occupancy of room 2
#     "price_t": ..., #Price
#     "price_previous": ..., #Previous Price
#     "vent_counter": ..., #For how many consecutive hours has the ventilation been on 
#     "low_override_r1": ..., #Is the low-temperature overrule controller of room 1 active 
#     "low_override_r2": ..., #Is the low-temperature overrule controller of room 2 active 
#     "current_time": ... #What is the hour of the day
# }


def select_action(state):
    
    
    ### Here goes your code
    
    HereAndNowActions = {
    "HeatPowerRoom1" : 0, #replace 0 with your choice
    "HeatPowerRoom2" : 0, #replace 0 with your choice
    "VentilationON" : 0 #binary. replace 0 with 1 if your choice is ON
    }
    HereAndNowActions = sp_policy.select_action(state)
    #HereAndNowActions = adp_policy.select_action(state)    
    return HereAndNowActions


state = {
    "T1": 10, #Temperature of room 1
     "T2": 23, #Temperature of room 2
     "H": 60, #Humidity
     "Occ1":2, #Occupancy of room 1
     "Occ2": 2, #Occupancy of room 2
     "price_t": 35, #Price
     "price_previous": 34, #Previous Price
     "vent_counter": 2, #For how many consecutive hours has the ventilation been on 
     "low_override_r1": True, #Is the low-temperature overrule controller of room 1 active 
     "low_override_r2": False, #Is the low-temperature overrule controller of room 2 active 
     "current_time":2 #What is the hour of the day
}
print(select_action(state))