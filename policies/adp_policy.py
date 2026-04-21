import numpy as np
from scipy.optimize import linprog
from scipy.stats import multivariate_normal as mvn
from dataclasses import dataclass, field
import numpy as np
from processes.PriceProcessRestaurant import price_model
from processes.OccupancyProcessRestaurant import next_occupancy_levels
from sklearn.cluster import KMeans
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
from SystemCharacteristics import get_fixed_data


#this is what the state looks like
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

def gen_samples(current_r1_occ, current_r2_occ, current_price, prev_price, num_samples=100):
    occ_samples = []
    price_samples = []
    for i in range(num_samples):
        occ_samples.append(next_occupancy_levels(r1_current=current_r1_occ, r2_current=current_r2_occ))
        price_samples.append(price_model(current_price=current_price, previous_price=prev_price))
    price_samples = np.array(price_samples)[:,None]
    occ_samples = np.array(occ_samples)
    joint_samples  = np.hstack([occ_samples, price_samples])
    return price_samples, occ_samples, joint_samples

def gen_scenarios(joint_samples, K):
    #K is the number of scenarios
    n_clusters = K
    km = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(joint_samples)
    counts = np.bincount(km.labels_, minlength=n_clusters)
    probs = counts / counts.sum()
    return km, probs




def select_action(state):
    HereAndNowActions = {
        "HeatPowerRoom1": 0,
        "HeatPowerRoom2": 0,
        "VentilationON": 0,
    }

    return HereAndNowActions
### Initial Values
def sample_init_state():
    return None

def init_eta(N_stages, eta_dim):
    scale = 1.0
    mean = np.zeros(eta_dim)
    cov = scale*np.eye(N=eta_dim)
    return mvn(mean=mean, cov = cov).rvs(size=N_stages)

def value_approx(eta, x_n):
    """
    Approximate the value function of a state vector using a linear combination.
   
    Parameters:
    eta (np.ndarray): weights (expects a column vector N x 1)
    x_n (np.ndarray): 
    Returns:
    float: The approximated value.
    """
    return x_n @ eta.T



def eta_regression(eta, x_n, target):
    #least squares
    return None

def generate_scenarios(K):
    w_k = None #uncertainty realizations
    p_k = None #probabilities
    return w_k, p_k

K = 5 #number of scenarios
N = 10 # number of samples per time step
N_stages = 13
init_states = [sample_init_state() for i in range(N)]
eta_ts = init_eta(N_stages)
### Forwards pass, construct the samples used later to perform regression
x_ns = [] #samples of states T x N x D


for t in range(N_stages):
    x_nt = []
    for n in range(N):
        ### Sample from exogenous
        
        w_n = 
        p_n= None
        ###

        ### Solve Bellman


        
        
        ####
        
        x_nt.append([w_n, p_n, y_n])
    x_ns.append(x_nt)

####
targets = []
etas = []
for t in range(stages):
    T = stages-1 - t
    for n in range(N):
        w_n, p_n, y_n = x_ns[T][n]

        ### compute the target
        V_target = None
        targets.append(V_target)
        ###

        #least squares regression to update eta
        eta_t = eta_regression(eta, x_n, target)
        eta.append(eta_t)

etas.reverse()


    






