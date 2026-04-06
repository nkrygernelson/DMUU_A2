from dataclasses import dataclass, field
import numpy as np
from processes.PriceProcessRestaurant import price_model


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
    all_nodes = []
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


def gen_samples(n_clusters, dim, num_samples=100):
    return NotImplemented
    


def select_action(state):
    ### Task 3: Stochastic Programming policy — implement here

    HereAndNowActions = {
        "HeatPowerRoom1": 0,
        "HeatPowerRoom2": 0,
        "VentilationON": 0,
    }

    return HereAndNowActions
