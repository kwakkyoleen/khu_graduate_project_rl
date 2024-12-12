import omni.isaac.lab.sim as sim_utils
import os
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets.articulation import ArticulationCfg

EXTENTION_PATH = os.path.dirname(os.path.realpath(__file__))

print(f"usd 파일 경로 : {EXTENTION_PATH}")

MY_GEN3LITE_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{EXTENTION_PATH}/gen3litenc.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "J0": 0.0,
            "J1": 0.6,
            "J2": -2.4,
            "J3": 1.57,
            "J4": 1.8,
            "J5": 0.0,
        },
    ),
    actuators={
        "J0": ImplicitActuatorCfg(
            joint_names_expr=["J0"],
            velocity_limit=1.6,
            effort_limit=10.0,
            stiffness=3000.0,
            damping=2.0,
        ),
        "J1": ImplicitActuatorCfg(
            joint_names_expr=["J1"],
            velocity_limit=1.6,
            effort_limit=14.0,
            stiffness=50000.0,
            damping=0.0,
        ),
        "J2": ImplicitActuatorCfg(
            joint_names_expr=["J2"],
            velocity_limit=1.6,
            effort_limit=10.0,
            stiffness=50000.0,
            damping=0.0,
        ),
        "J3": ImplicitActuatorCfg(
            joint_names_expr=["J3"],
            velocity_limit=1.6,
            effort_limit=7.0,
            stiffness=750.0,
            damping=0.2,
        ),
        "J4": ImplicitActuatorCfg(
            joint_names_expr=["J4"],
            velocity_limit=1.6,
            effort_limit=7.0,
            stiffness=5000.0,
            damping=1.0,
        ),
        "J5": ImplicitActuatorCfg(
            joint_names_expr=["J5"],
            velocity_limit=1.6,
            effort_limit=7.0,
            stiffness=100.0,
            damping=0.0,
        ),
    },
)
"""Configuration of UR-10e arm using implicit actuator models."""
