from __future__ import annotations

import math
import torch
from collections.abc import Sequence

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, ArticulationCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.math import sample_uniform

from ..ur.ur import MY_UR10E_CFG


@configclass
class ObstacleEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2  # 랜더링 간격
    episode_length_s = 15.0  # 에피소드 길이
    robot_dof_angle_scales = 0.5  # [라디안] 1도는 0.0174라디안임
    num_actions = 1  # 액션의 갯수
    num_observations = 3  # 관찰 갯수
    num_states = 0
    num_envs = 128
    env_spacing = 4.0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # robot
    robot_cfg: ArticulationCfg = MY_UR10E_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    )
    dof_1_name = "shoulder_pan_joint"
    dof_2_name = "shoulder_lift_joint"
    dof_3_name = "elbow_joint"
    dof_4_name = "wrist_1_joint"
    dof_5_name = "wrist_2_joint"
    dof_6_name = "wrist_3_joint"

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=num_envs, env_spacing=env_spacing, replicate_physics=True
    )

    # reward scales
    rew_scale_distance = -0.2
    rew_scale_collision = -100.0
    rew_scale_success = 100.0


class ObstacleEnv(DirectRLEnv):
    cfg: ObstacleEnvCfg

    def __init__(self, cfg: ObstacleEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.dt = self.cfg.sim.dt * self.cfg.decimation

        # create auxiliary variables for computing applied action, observations and rewards
        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)

        self.robot_dof_angle_scales = torch.ones_like(self.robot_dof_lower_limits)

        self.robot_dof_targets = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot_cfg)
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        # 장애물에대한 collisions 추가해야할듯??
        self.scene.filter_collisions(global_prim_paths=[])
        # add articultion to scene
        self.scene.articulations["robot"] = self._robot
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone().clamp(
            -1.0, 1.0
        )  # 입력값은 -1에서 1사이로 받고 나중에 스케일링 하는 방식으로 ㄱㄱ
        targets = (
            self.robot_dof_targets
            + self.robot_dof_angle_scales * self.actions
        )
        self.robot_dof_targets[:] = torch.clamp(
            targets, self.robot_dof_lower_limits, self.robot_dof_upper_limits
        )

    def _apply_action(self) -> None:
        self._robot.set_joint_position_target(self.robot_dof_targets)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # 장애물과 부딛히면 terminated
        terminated = torch.tensor(False)
        # 시간이 너무 많이 지나면 truncated
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated
    
    def _get_rewards(self) -> torch.Tensor:
        # 엔드 이펙터와 타겟의 거리 * rew_scale_distance +
        # (장애물과 부딛혔으면 rew_scale_collision) +
        # (목표물에 도착했으면 rew_scale_success)
        return torch.tensor([])
    
    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
        # robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids] + sample_uniform(
            -0.125,
            0.125,
            (len(env_ids), self._robot.num_joints),
            self.device,
        )
        joint_pos = torch.clamp(joint_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        joint_vel = torch.zeros_like(joint_pos)
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

    def _get_observations(self) -> dict:
        # 현재 로봇팔 각도 + 카메라로 입력된 rgbd 데이터(다듬어진) + 타겟의 위치
        return {"policy": torch.tensor([])}