"""
Python module serving as a project/extension template.
"""

# Register Gym environments.
# from .tasks import *

# Register UI extensions.
from.dofbot.dofbot import MY_DOFBOT_CFG
from.ur.ur import MY_UR10E_CFG
from.kinova.gen3lite import MY_GEN3LITE_CFG
import gymnasium as gym