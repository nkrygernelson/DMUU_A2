# Rename this file to SP_policy_[your group number].py before submission.
# Thin wrapper — all logic lives in policies/sp_policy.py.

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from policies.sp_policy import select_action  
