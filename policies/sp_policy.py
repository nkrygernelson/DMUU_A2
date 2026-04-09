from dataclasses import dataclass, field
import numpy as np
from processes.PriceProcessRestaurant import price_model
from processes.OccupancyProcessRestaurant import next_occupancy_levels
from sklearn.cluster import KMeans

@dataclass
class ScenarioNode:
    node_id: int
    stage: int
    parent: 'ScenarioNode | None'
    children: list = field(default_factory=list)
    scenarios: list = field(default_factory=list)
    state: dict = field(default_factory=dict)
    decision: dict = field(default_factory=dict)
    prob: float = 1.0


def build_scenario_tree(bf, num_stages):
    node_counter = 0
    root = ScenarioNode(node_id=node_counter, stage=0, parent=None)
    node_counter += 1
    all_nodes = [root]
    leaves = []
    current_level = [root]
    for stage in range(1, num_stages + 1):
        next_level = []
        for parent in current_level:
            for b in range(bf):
                child = ScenarioNode(node_id=node_counter, stage=stage, parent=parent, prob=1/bf)
                parent.children.append(child)
                all_nodes.append(child)
                next_level.append(child)
                node_counter += 1
        current_level = next_level
    leaves = current_level
    for s, leaf in enumerate(leaves):
        node = leaf
        while node is not None:
            node.scenarios.append(s)
            node = node.parent
    return root, all_nodes, leaves

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

def propagate_uncertainty(root, all_nodes, num_samples=200):
    """Propagate exogenous uncertainty (price, occupancy) through the tree."""
    for node in all_nodes:
        if not node.children:
            continue
        s = node.state
        _, _, joint = gen_samples(
            current_r1_occ=s["current_r1_occ"],
            current_r2_occ=s["current_r2_occ"],
            current_price=s["current_price"],
            prev_price=s["prev_price"],
            num_samples=num_samples,
        )
        n_clusters = len(node.children)
        km = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(joint)
        counts = np.bincount(km.labels_, minlength=n_clusters)
        probs = counts / counts.sum()

        for i, child in enumerate(node.children):
            child.state = {
                "current_r1_occ": km.cluster_centers_[i][0],
                "current_r2_occ": km.cluster_centers_[i][1],
                "current_price": km.cluster_centers_[i][2],
                "prev_price": s["current_price"],
            }
            child.prob = probs[i]
    return root, all_nodes

'''
the porbabilities are on the tree and now i must optimize over the tree.
How is this going to work?
I think that what we do is to iterate throught he tree and define the correct constraints and introduce the right variables.

Decisions:
p_r1_t
p_r2_t
v_t

Transition dynamics:
Temperature (depends on):
    Prev temp
    Room heating
    Occupancy
    Heat exchange between rooms
    Heat exchange to outside
Humidity:
    Prev Humidity
    Occupancy
    Ventialtion

Overrule:
    Low temp overrule: if temp of a room dips below T_low, heater is on max until t_ok is reached.
    High temp: If room temp above t_high, heater off for that hour
    Humidity: If Humidity (in the whole building) exceeds the the threshld the ventilator is turned on
Ventilation Inertia:
    If the ventilator is switched on at t, it remains on for three hours t, t+1 and t+2
'''

def select_action(state):
    ### Task 3: Stochastic Programming policy — implement here
    HereAndNowActions = {
        "HeatPowerRoom1": 0,
        "HeatPowerRoom2": 0,
        "VentilationON": 0,
    }
    return HereAndNowActions

num_stages = 2
bf = 2
starting_state_dict = {"current_r1_occ": 1,
                       "current_r2_occ": 1,
                       "current_price": 1,
                       "prev_price": 1}

root, all_nodes, leaves = build_scenario_tree(bf, num_stages)
all_nodes[0].state = starting_state_dict
root, all_nodes = propagate_uncertainty(root, all_nodes)

for i in range(num_stages):
    stage_nodes = [node for node in all_nodes if node.stage == i]
    for node in stage_nodes:
        print()
        #Need to add the Pyomo implmentation