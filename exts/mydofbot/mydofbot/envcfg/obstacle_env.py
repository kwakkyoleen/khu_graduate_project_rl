from __future__ import annotations

import math
import torch
from collections.abc import Sequence

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import (
    Articulation,
    ArticulationCfg,
    RigidObject,
    RigidObjectCfg,
)
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.sim.spawners.shapes.shapes_cfg import (
    CuboidCfg,
)
from omni.isaac.lab.sim.spawners import spawn_cuboid
from omni.isaac.lab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.math import sample_uniform
from omni.isaac.lab.sensors import (
    TiledCamera,
    TiledCameraCfg,
    ContactSensorCfg,
    ContactSensor,
)

from ..kinova.gen3lite import MY_GEN3LITE_CFG


@configclass
class ObstacleEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 12  # 랜더링 간격
    episode_length_s = 15.0  # 에피소드 길이
    robot_dof_angle_scales = 0.5  # [라디안] 1도는 0.0174라디안임
    num_actions = 6  # 액션의 갯수
    num_observations = 3  # 관찰 갯수
    num_states = 0
    num_envs = 128
    env_spacing = 4.0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # robot
    robot_cfg: ArticulationCfg = MY_GEN3LITE_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    )
    dof_1_name = "J1"
    dof_2_name = "J2"
    dof_3_name = "J3"
    dof_4_name = "J4"
    dof_5_name = "J5"

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=num_envs, env_spacing=env_spacing, replicate_physics=True
    )

    # reward scales
    rew_scale_distance = 2
    rew_scale_collision = -100.0
    rew_scale_success = 500.0


# @configclass
# class ObstacleCfg(CuboidCfg):
#     prim_path: str = "/World/envs/env_.*/Obstacle"
#     position: tuple = (0.0, 0.5, 1.0)  # 장애물의 초기 위치
#     size: tuple = (0.2, 0.2, 0.2)  # 장애물의 크기 (박스 형태로 가정)
#     color: tuple = (0.0, 1.0, 0.0)


@configclass
class ObstacleCfg(RigidObjectCfg):
    prim_path: str = "/World/envs/env_.*/Obstacle"
    spawn = sim_utils.CuboidCfg(
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.0, 1.0, 0.0), metallic=0.2
        ),
        size=(0.2, 0.2, 0.2),
    )
    init_state = RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.5, 1.0))


# @configclass
# class TargetCfg(CuboidCfg):
#     prim_path: str = "/World/envs/env_.*/Target"
#     position: tuple = (0.8, 0.5, 0.5)  # 타겟의 초기 위치
#     size: tuple = (0.05, 0.05, 0.05)  # 타겟의 크기 (박스 형태로 가정)
#     color: tuple = (1.0, 0.0, 0.0)


@configclass
class TargetCfg(RigidObjectCfg):
    prim_path: str = "/World/envs/env_.*/Target"
    spawn = sim_utils.CuboidCfg(
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(1.0, 0.0, 0.0), metallic=0.2
        ),
        size=(0.05, 0.05, 0.05),
    )
    init_state = RigidObjectCfg.InitialStateCfg(pos=(0.8, 0.5, 0.5))


class ObstacleEnv(DirectRLEnv):
    cfg: ObstacleEnvCfg

    def __init__(self, cfg: ObstacleEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.dt = self.cfg.sim.dt * self.cfg.decimation

        # create auxiliary variables for computing applied action, observations and rewards
        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[
            0, :, 0
        ].to(device=self.device)
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[
            0, :, 1
        ].to(device=self.device)

        self.robot_dof_angle_scales = torch.ones_like(self.robot_dof_lower_limits)

        self.robot_dof_targets = torch.zeros(
            (self.num_envs, self._robot.num_joints), device=self.device
        )

        self.end_effector_idx = self._robot.find_bodies("END_EFFECTOR")[0][0]

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot_cfg)

        # 장애물 생성
        self.obstacle_cfg = ObstacleCfg()
        self.obstacle = RigidObject(cfg=self.obstacle_cfg)
        self.scene.rigid_objects["Obstacle"] = self.obstacle
        # self.obstacle.reset()
        # spawn_cuboid(
        #     prim_path=self.obstacle_cfg.prim_path,
        #     cfg=self.obstacle_cfg,
        # )

        # 타겟 생성
        self.target_cfg = TargetCfg()
        self.target_obj = RigidObject(cfg=self.target_cfg)
        self.scene.rigid_objects["Target_obj"] = self.target_obj
        # self.target_obj.reset()
        # spawn_cuboid(
        #     prim_path=self.target_cfg.prim_path,
        #     cfg=self.target_cfg,
        # )

        # 충돌 감지 설정
        self.contact_sensor = ContactSensor(
            cfg=ContactSensorCfg(
                prim_path="/World/envs/env_.*/Robot/.*",
                update_period=0.0,
                history_length=6,
                debug_vis=True,
            )
        )
        self.scene.sensors["contact_sensor"] = self.contact_sensor

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
        # target을 쓰는게 맞나 아님 현재 각도를 쓰는게 맞나
        targets = self.robot_dof_targets + self.robot_dof_angle_scales * self.actions
        # targets = (
        #     self._robot.data.joint_pos + self.robot_dof_angle_scales * self.actions
        # )
        self.robot_dof_targets[:] = torch.clamp(
            targets, self.robot_dof_lower_limits, self.robot_dof_upper_limits
        )

        self.target_obj.write_root_velocity_to_sim(
            torch.zeros_like(self.target_obj.data.root_vel_w)
        )

    def _apply_action(self) -> None:
        self._robot.set_joint_position_target(self.robot_dof_targets)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # 장애물과 부딛히면 terminated
        contact_list = torch.sum(self.contact_sensor.data.net_forces_w**2, dim=-1)  # type: ignore
        contact_list = torch.sqrt(contact_list)
        contact_list, _ = torch.max(contact_list, dim=1)

        terminated = contact_list > 1

        # terminated = torch.tensor(False)
        # 시간이 너무 많이 지나면 truncated
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    def _get_rewards(self) -> torch.Tensor:
        # 엔드 이펙터와 타겟의 거리 * rew_scale_distance +
        # (장애물과 부딛혔으면 rew_scale_collision) +
        # (목표물에 도착했으면 rew_scale_success)
        # 엔드 이펙터와 타겟의 위치
        robot_ef_pos = self._robot.data.body_pos_w[:, self.end_effector_idx]
        target_pos = self.scene.rigid_objects["Target_obj"].data.root_pos_w

        # 접촉 여부 : 일단 norm으로 구했는데 그냥 평균내도 될려나?
        contact_list = torch.sum(self.contact_sensor.data.net_forces_w**2, dim=-1)  # type: ignore
        contact_list = torch.sqrt(contact_list)
        contact_list, _ = torch.max(contact_list, dim=1)

        return self._compute_rewards(robot_ef_pos, target_pos, contact_list)

    def _compute_rewards(
        self,
        robot_ef_pos: torch.Tensor,
        target_pos: torch.Tensor,
        contact_list: torch.Tensor,
    ) -> torch.Tensor:
        target_disparity = target_pos - robot_ef_pos
        target_distance = torch.sqrt(torch.sum(target_disparity**2, dim=-1))
        target_distance_riverse = 1 / (1 + 10 * target_distance)

        collision_bool = contact_list > 1
        # print("col list : ", collision_list)
        # collision_bool = torch.any(collision_list)

        goal_bool = target_distance < 0.01

        computed_reward = (
            target_distance_riverse * self.cfg.rew_scale_distance
            + collision_bool.float() * self.cfg.rew_scale_collision
            + goal_bool.float() * self.cfg.rew_scale_success
        )
        # print("col bool : ", collision_bool)
        # print("col rd : ", collision_bool.float() * self.cfg.rew_scale_collision)
        # print("rd : ", computed_reward)
        return computed_reward

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
        # robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids] + sample_uniform(
            -0.125,
            0.125,
            (len(env_ids), self._robot.num_joints),
            self.device,
        )
        joint_pos = torch.clamp(
            joint_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits
        )
        joint_vel = torch.zeros_like(joint_pos)
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # target state
        rand_target_pose = sample_uniform(-0.7, 0.7, (len(env_ids), 7), self.device)
        rand_target_pose[:, 2] = sample_uniform(0.1, 0.7, (len(env_ids),), self.device)
        rand_target_pose[:, 0:3] += self._robot.data.root_pos_w[env_ids]
        self.target_obj.write_root_pose_to_sim(rand_target_pose, env_ids=env_ids)
        self.target_obj.write_root_velocity_to_sim(
            torch.zeros_like(self.target_obj.data.root_vel_w[env_ids]), env_ids=env_ids
        )

    def _get_observations(self) -> dict:
        # 현재 로봇팔 각도 + 카메라로 입력된 rgbd 데이터(다듬어진) + 타겟의 위치

        # body랑 root랑 뭔차이지??
        target_pos = self.scene.rigid_objects["Target_obj"].data.root_pos_w
        robot_pos = self.scene.articulations["robot"].data.root_pos_w

        target_pos_rel = target_pos - robot_pos

        joint_pos = self._robot.data.joint_pos
        ef_pos = self._robot.data.body_pos_w - robot_pos.unsqueeze(1)
        ef_pos = ef_pos.flatten(start_dim=1)
        ef_trans = self._robot.data.body_quat_w[:, self.end_effector_idx]
        combined_pos = torch.cat((joint_pos, ef_pos, ef_trans), dim=1)

        return {
            # "policy": torch.tensor([]),
            "joint": combined_pos,
            "target": target_pos_rel,
        }
