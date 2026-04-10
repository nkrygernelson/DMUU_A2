from dataclasses import dataclass, field
import numpy as np
from processes.PriceProcessRestaurant import price_model
from processes.OccupancyProcessRestaurant import next_occupancy_levels
from sklearn.cluster import KMeans
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
from SystemCharacteristics import get_fixed_data
NUM_SLOTS = int(get_fixed_data()["num_timeslots"])
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



num_stages = 2
bf = 2
starting_state_dict = {"current_r1_occ": 1,
                       "current_r2_occ": 1,
                       "current_price": 1,
                       "prev_price": 1}

root, all_nodes, leaves = build_scenario_tree(bf, num_stages)
all_nodes[0].state = starting_state_dict
root, all_nodes = propagate_uncertainty(root, all_nodes)
########################################
def path_prob(node):
    p = 1.0
    while node.parent is not None:
        p *= node.prob
        node = node.parent
    return p

#we solve the linear pogram 
def build_and_solve_linear_program(state, root, all_nodes, leaves ):
    current_time = state["current_time"]
    m = pyo.ConcreteModel()
    nids = [node.node_id for node in all_nodes]
    m.NODES = pyo.Set(initialize = nids)
    fixed_data = get_fixed_data()

    xi_exh = fixed_data["heat_exchange_coeff"]
    xi_loss = fixed_data["thermal_loss_coeff"]
    xi_conv = fixed_data["heating_efficiency_coeff"]
    xi_cool = fixed_data["heat_vent_coeff"]
    xi_occ = fixed_data["heat_occupancy_coeff"]

    #humidity coefficients
    eta_occ = fixed_data["humidity_occupancy_coeff"]
    eta_vent = fixed_data["humidity_vent_coeff"]

    #Parameters
    p_vent = fixed_data["ventilation_power"]
    T_low = fixed_data["temp_min_comfort_threshold"]
    T_high = fixed_data["temp_max_comfort_threshold"]
    T_ok = fixed_data["temp_OK_threshold"]
    T_out = fixed_data["outdoor_temperature"] #outdoor temperature at time t
    H_high = fixed_data["humidity_threshold"]
    P_overline = fixed_data["heating_max_power"] #this is written as P_r but we only have 1 max power heating ?
    T_circ = -3 #read from Systemcharacticeristics
    M_low = T_low - T_circ
    M_high = T_ok - T_circ
    M_hum = 100 - H_high #this might course problems

    #define variables
    #these are the decision variables.
    m.p1      = pyo.Var(m.NODES, bounds=(0, P_overline))
    m.p2      = pyo.Var(m.NODES, bounds=(0, P_overline))
    m.V       = pyo.Var(m.NODES, within=pyo.Binary)
    #state variables
    m.temp1   = pyo.Var(m.NODES, bounds=(T_circ, 2*T_high))
    m.temp2   = pyo.Var(m.NODES, bounds=(T_circ, 2*T_high))
    m.z1_cold = pyo.Var(m.NODES, within=pyo.Binary)
    m.z1_hot  = pyo.Var(m.NODES, within=pyo.Binary)
    m.z2_cold = pyo.Var(m.NODES, within=pyo.Binary)
    m.z2_hot  = pyo.Var(m.NODES, within=pyo.Binary)
    m.ON      = pyo.Var(m.NODES, within=pyo.Binary)
    m.OFF     = pyo.Var(m.NODES, within=pyo.Binary)
    m.hum     = pyo.Var(m.NODES, bounds=(0, 100))
    m.cons = pyo.ConstraintList()
    obj_expr = 0.0
    #fix the root node
    #the root node has node_id
    rid = root.node_id

    # State from environment (not from fixed_data)
    m.temp1[rid].fix(state["T1"])
    m.temp2[rid].fix(state["T2"])
    m.hum[rid].fix(state["H"])

    # Low-override status from environment
    m.z1_cold[rid].fix(state["low_override_r1"])
    m.z2_cold[rid].fix(state["low_override_r2"])

    # Overrule init (same as Task A)
    m.cons.add(T_ok - m.temp1[rid] <= M_high * (1 + m.z1_cold[rid]))
    m.cons.add(T_ok - m.temp2[rid] <= M_high * (1 + m.z2_cold[rid]))

    # Ventilation init (ported from Task A, but using vent_counter)
    vc = state["vent_counter"]
    if 0 < vc < 3:
        # Inertia not expired — must stay on
        m.V[rid].fix(1)
        m.ON[rid].fix(0)
        m.OFF[rid].fix(0)
    elif vc == 0:
        # Was off — same as Task A: V[0] == ON[0], OFF[0] == 0
        m.cons.add(m.V[rid] == m.ON[rid])
        m.OFF[rid].fix(0)
    else:
        # vc >= 3: was on, free to turn off
        m.ON[rid].fix(0)
        m.cons.add(m.V[rid] == 1 - m.OFF[rid])
    for node in all_nodes:
        nid = node.node_id
        #we need to define the node cost
        #no cost for the leaves, no decisions on leaves
        if node.children:
            # --- High temperature overrule ---
            # r1
            m.cons.add(m.temp1[nid] - T_high <= M_high * m.z1_hot[nid])
            m.cons.add(m.p1[nid] <= P_overline * (1 - m.z1_hot[nid]))
            # r2
            m.cons.add(m.temp2[nid] - T_high <= M_high * m.z2_hot[nid])
            m.cons.add(m.p2[nid] <= P_overline * (1 - m.z2_hot[nid]))

            # --- Low temperature overrule ---
            # r1
            m.cons.add(T_low - m.temp1[nid] <= M_low * m.z1_cold[nid])
            m.cons.add(m.p1[nid] >= P_overline * m.z1_cold[nid])
            # r2
            m.cons.add(T_low - m.temp2[nid] <= M_low * m.z2_cold[nid])
            m.cons.add(m.p2[nid] >= P_overline * m.z2_cold[nid])
            # --- Humidity overrule ---
            m.cons.add(m.hum[nid] - H_high <= M_hum * m.V[nid])
            price = node.state["current_price"]
            wp = path_prob(node)
            obj_expr += wp * price * (m.p1[nid] + m.p2[nid] + p_vent * m.V[nid])
            m.cons.add(m.ON[nid] + m.OFF[nid] <= 1)

        if node.parent is not None:

            parent = node.parent
            pid = parent.node_id
        
            #the pyo variables are indexed by node_id.
            #The equivalent of the t-1 value of a variable is the value of the node's parent

            #temp exchange
            #temp1
            #should check that the sign are correct and that we aren't porting over the error from last time
            parent_time = current_time + parent.stage
            T_out_val = T_out[min(parent_time, len(T_out) - 1)]

            m.cons.add(
                m.temp1[nid] ==
                    m.temp1[pid]
                    - xi_exh * (m.temp1[pid] - m.temp2[pid])
                    - xi_loss * (m.temp1[pid] - T_out_val)
                    + xi_conv * m.p1[pid]
                    - xi_cool * m.V[pid]
                    + xi_occ * parent.state["current_r1_occ"]
            )

            #temp2
            m.cons.add(
                m.temp2[nid] ==
                    m.temp2[pid]
                    - xi_exh * (m.temp2[pid] - m.temp1[pid])
                    - xi_loss * (m.temp2[pid] - T_out_val)
                    + xi_conv * m.p2[pid]
                    - xi_cool * m.V[pid]
                    + xi_occ * parent.state["current_r2_occ"]
            )
            #humidity
            m.cons.add(
                m.hum[nid] ==
                    m.hum[pid]
                    + eta_occ * (parent.state["current_r1_occ"] + parent.state["current_r2_occ"])
                    - eta_vent * m.V[pid]
            )

            #ventilator 
            
            if node.stage >= 2:
                grandparent = parent.parent
                m.cons.add(
                    m.V[nid]>= m.ON[nid]+m.ON[pid] 
                    + m.ON[grandparent.node_id]
                        )
            elif node.stage == 1:
                m.cons.add(
                    m.V[nid] >= m.ON[nid] + m.ON[pid]
                )

            m.cons.add(m.OFF[nid] <= m.V[pid])
            m.cons.add(m.ON[nid] <= 1 - m.V[pid])
            m.cons.add(m.V[nid] == m.V[pid] + m.ON[nid] - m.OFF[nid])

            
            # the "stay on until T_ok" constraint references parent's z_cold:
            m.cons.add(T_ok - m.temp1[nid] <= M_high * (1 - m.z1_cold[pid] + m.z1_cold[nid]))
            m.cons.add(T_ok - m.temp2[nid] <= M_high * (1 - m.z2_cold[pid] + m.z2_cold[nid]))

    
        
        #we need to define the node cost
        #no cost for the leaves, no decisions on leaves
        

    m.objective = pyo.Objective(expr=obj_expr, sense=pyo.minimize)
    solver = pyo.SolverFactory("gurobi")
    solver.options["TimeLimit"] = 10
    solver.options["MIPGap"] = 0.02
    solver.options["OutputFlag"] = 0
    result = solver.solve(m, tee=False)

    return m, result

def select_action(state):
    NUM_SLOTS  = 13
    current_time = state["current_time"]
    remaining = NUM_SLOTS - current_time  # hours left

    # Tree sizing
    num_stages = min(3, remaining)
    bf = 3

    root, all_nodes, leaves = build_scenario_tree(bf, num_stages)

    # Set root exogenous state
    root.state = {
        "current_r1_occ": state["Occ1"],
        "current_r2_occ": state["Occ2"],
        "current_price":  state["price_t"],
        "prev_price":     state["price_previous"],
    }

    # Propagate uncertainty
    propagate_uncertainty(root, all_nodes, num_samples=150)

    # Build and solve
    m, result = build_and_solve_linear_program(state, root, all_nodes, leaves)

    # Extract root decisions
    rid = root.node_id
    if (result.solver.status == pyo.SolverStatus.ok and
        result.solver.termination_condition in
            (pyo.TerminationCondition.optimal,
             pyo.TerminationCondition.feasible)):
        p1_val = pyo.value(m.p1[rid])
        p2_val = pyo.value(m.p2[rid])
        v_val  = round(pyo.value(m.V[rid]))
    else:
        p1_val = 0.0
        p2_val = 0.0
        v_val  = 0

    return {
        "HeatPowerRoom1": p1_val,
        "HeatPowerRoom2": p2_val,
        "VentilationON":  v_val,
    }