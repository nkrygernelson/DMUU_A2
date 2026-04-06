# Rename this file to ADP_policy_[your group number].py before submission.
# Thin wrapper — all logic lives in policies/adp_policy.py.

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from policies.adp_policy import select_action  
