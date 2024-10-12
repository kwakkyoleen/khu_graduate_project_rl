import omni.isaac.lab.sim as sim_utils
import os
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets.articulation import ArticulationCfg

EXTENTION_PATH = os.path.dirname(os.path.realpath(__file__))

MY_DOFBOT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{EXTENTION_PATH}/dofbotrgbd.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "joint1": 0.0,
            "joint2": 0.0,
            "joint3": 0.0,
            "joint4": 0.0,
            "Wrist_Twist_RevoluteJoint": 0.0,
            "Finger_Left_01_RevoluteJoint": 0.523599,
            "Finger_Right_01_RevoluteJoint": -0.523599
        },
    ),
    actuators={
        "dofbot_arm": ImplicitActuatorCfg(
            joint_names_expr=["joint[1-4]"],
            effort_limit=5.2,
            velocity_limit=57.29578,
            stiffness=1048,
            damping=53.0,
        ),
        "dofbot_gripper": ImplicitActuatorCfg(
            joint_names_expr=["Finger_Left_01_RevoluteJoint", "Finger_Right_01_RevoluteJoint"],
            effort_limit=0.12,
            velocity_limit=1000000.0,
            stiffness=6000.0,
            damping=1000.0,
        ),
        "dofbot_twist": ImplicitActuatorCfg(
            joint_names_expr=["Wrist_Twist_RevoluteJoint"],
            effort_limit=0.1,
            velocity_limit=1000000.0,
            stiffness=1000.0,
            damping=10.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of customed dofbot."""