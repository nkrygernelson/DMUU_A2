from environment.simulator import load_experiments, evaluate
from policies import dummy_policy
from policies import sp_policy

experiments = load_experiments()
avg_cost, costs = evaluate(dummy_policy, experiments)
print(f"Average cost over {len(costs)} days: {avg_cost:.2f}")