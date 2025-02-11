import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Obstacle avoiding task env.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=128, help="Number of environments to simulate.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(headless=args_cli.headless)
simulation_app = app_launcher.app

"""Rest everything else."""

import gymnasium as gym
import torch
from collections.abc import Sequence

import warp as wp

from omni.isaac.lab.assets.rigid_object.rigid_object_data import RigidObjectData

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils.parse_cfg import parse_env_cfg

from mydofbot import MY_UR10E_CFG
from mydofbot.envcfg.obstacle_env import ObstacleEnvCfg

wp.init()


class ArmSm:
    def __init__(self, dt: float, num_envs: int, device: torch.device | str = "cpu"):
        # save parameters
        self.dt = float(dt)
        self.num_envs = num_envs
        self.device = device
        # initialize state machine
        self.sm_dt = torch.full((self.num_envs,), self.dt, device=self.device)
        self.sm_wait_time = torch.zeros((self.num_envs,), device=self.device)

        # desired state
        self.des_ee_pose = torch.zeros((self.num_envs, 7), device=self.device)

        # approach above object offset
        self.offset = torch.zeros((self.num_envs, 7), device=self.device)
        self.offset[:, 2] = 0.1
        self.offset[:, -1] = 1.0  # warp expects quaternion as (x, y, z, w)

        # convert to warp
        self.sm_dt_wp = wp.from_torch(self.sm_dt, wp.float32)
        self.sm_wait_time_wp = wp.from_torch(self.sm_wait_time, wp.float32)
        self.des_ee_pose_wp = wp.from_torch(self.des_ee_pose, wp.transform)
        self.offset_wp = wp.from_torch(self.offset, wp.transform)


def main():
    # parse configuration
    env_cfg: ObstacleEnvCfg = parse_env_cfg(
        "Obstacle-direct-v0",
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    # create environment
    env = gym.make("Obstacle-direct-v0", cfg=env_cfg)
    # reset environment at start
    env.reset()

    # create action buffers (position + quaternion)
    actions = torch.zeros(env.unwrapped.action_space.shape, device=env.unwrapped.device)
    print(env.unwrapped.action_space.shape)
    # desired object orientation (we only do position control of object)
    desired_orientation = torch.zeros((env.unwrapped.num_envs, 4), device=env.unwrapped.device)
    desired_orientation[:, 1] = 1.0
    # create state machine
    arm_sm = ArmSm(env_cfg.sim.dt * env_cfg.decimation, env.unwrapped.num_envs, env.unwrapped.device)

    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # step environment
            dones = env.step(actions)[-2]

            # # observations
            # # -- end-effector frame
            # ee_frame_sensor = env.unwrapped.scene["ee_frame"]
            # tcp_rest_position = ee_frame_sensor.data.target_pos_w[..., 0, :].clone() - env.unwrapped.scene.env_origins
            # tcp_rest_orientation = ee_frame_sensor.data.target_quat_w[..., 0, :].clone()
            # # -- object frame
            # object_data: RigidObjectData = env.unwrapped.scene["object"].data
            # object_position = object_data.root_pos_w - env.unwrapped.scene.env_origins
            # # -- target object frame
            # desired_position = env.unwrapped.command_manager.get_command("object_pose")[..., :3]

            # # advance state machine
            # actions = pick_sm.compute(
            #     torch.cat([tcp_rest_position, tcp_rest_orientation], dim=-1),
            #     torch.cat([object_position, desired_orientation], dim=-1),
            #     torch.cat([desired_position, desired_orientation], dim=-1),
            # )

            # # reset state machine
            # if dones.any():
            #     pick_sm.reset_idx(dones.nonzero(as_tuple=False).squeeze(-1))

    # close the environment
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()