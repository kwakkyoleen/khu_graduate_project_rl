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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@configclass
class ObstacleEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2  # 랜더링 간격
    episode_length_s = 8.0  # 에피소드 길이
    robot_dof_angle_scales = 0.05  # [라디안] 1도는 0.0174라디안임
    num_actions = 6  # 액션의 갯수
    num_observations = 3  # 관찰 갯수
    num_states = 0
    num_envs = 128
    env_bundles = 16
    env_spacing = 4.0
    kp = 20
    kd = 1

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
    rew_scale_distance = 20
    rew_scale_time = -0.2
    rew_scale_collision = -400.0
    rew_scale_success = 5000.0
    rew_scale_acc = 0.009

    # angle scale
    angle_scale_factor = 0.1
    vel_scale_factor = 60


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
        size=(0.01, 0.01, 0.01),
    )
    init_state = RigidObjectCfg.InitialStateCfg(pos=(-0.43, 0.0, 0.3))


def make_rand_val(grade : int, env_ids: torch.Tensor) -> torch.Tensor:
    n = env_ids.shape[0]
    v1 = 0.1
    v2 = 3.14 / 4
    v3 = 3.14 * 2
    if grade == 0:
        v1 = 0.05
        v2 = 3.14 / 4
    elif grade == 1:
        v1 = 0.1
        v2 = 3.14 / 2
    elif grade == 2:
        v1 = 0.1
        v2 = 3 * 3.14 / 4
    elif grade == 3:
        v1 = 0.1
        v2 = 4 * 3.14 / 5
    elif grade >= 4:
        v1 = 0.1  # 내원 반지름 최대 2까지 증가
        v2 = 3.14
    r = torch.rand(n) * v1 + v1  # r2
    t1 = torch.rand(n) * v2 - (v2 / 2)  # theta1
    t2 = torch.rand(n) * v3 - (v3 / 2)  # theta2
    br = 0.37
    bz = 0.3

    col_0 = torch.cos(t1) * (r * torch.cos(t2) - br)
    col_1 = torch.sin(t1) * (r * torch.cos(t2) + br)
    col_2 = r * torch.sin(t2) + bz

    result = torch.stack((col_0, col_1, col_2), dim=1)
    return result


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

        self.robot_dof_angle_scales = torch.ones_like(self.robot_dof_lower_limits) * self.cfg.angle_scale_factor

        self.robot_dof_targets = torch.zeros(
            (self.num_envs, self._robot.num_joints), device=self.device
        )

        self._donecount = torch.zeros((self.num_envs), device=self.device)

        self.end_effector_idx = self._robot.find_bodies("END_EFFECTOR")[0][0]
        self._target_distance_prev = None

        self.grade = 0
        self.precision = 100

        self.now_joint_vel = torch.zeros_like(self._robot.data.joint_vel, device=self.device)
        self.prev_joint_vel = torch.zeros_like(self._robot.data.joint_vel, device=self.device)
        self.target_object_pos = torch.zeros_like(self.scene.rigid_objects["Target_obj"].data.root_pos_w)

        self.min_target_distance = torch.full((self.num_envs,), fill_value=100, device=self.device, dtype=torch.float)
        self.min_target_distance_back = torch.full((self.num_envs,), fill_value=100, device=self.device, dtype=torch.float)
        self.timesteps = torch.zeros((self.num_envs), device=self.device, dtype=torch.int)
        self.min_reach_time = torch.full((self.num_envs,), fill_value=480, device=self.device, dtype=torch.int)
        self.min_reach_time_back = torch.full((self.num_envs,), fill_value=480, device=self.device, dtype=torch.int)

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
        # self.actions = actions.clone().clamp(
        #     -1.0, 1.0
        # )  # 입력값은 -1에서 1사이로 받고 나중에 스케일링 하는 방식으로 ㄱㄱ
        # target을 쓰는게 맞나 아님 현재 각도를 쓰는게 맞나
        # print(f"action : {actions}")
        # targets = self.robot_dof_targets + self.robot_dof_angle_scales * self.actions
        # print(f"scaled : {self.robot_dof_angle_scales * self.actions}")
        # joint_vel = self._robot.data.joint_vel.clone()
        # target_vel = actions.clone()
        # disparity_angle = target_vel.clone()  # * (self.cfg.decimation / 120)
        # self.target_torque = self.cfg.kp * disparity_angle + self.cfg.kd * (target_vel - joint_vel)
        self.target_vel = actions.clone()  # * self.cfg.vel_scale_factor
        self.now_joint_vel = actions.clone()
        self.temp_target_pos = self._robot.data.joint_pos_target  # 확인하려고

        self.target_obj.write_root_velocity_to_sim(
            torch.zeros_like(self.target_obj.data.root_vel_w)
        )

        # timestep 증가
        self.timesteps += 1

    def _apply_action(self) -> None:
        # self._robot.set_joint_effort_target(self.target_torque)
        self._robot.set_joint_velocity_target(self.target_vel)
        self._robot.set_joint_position_target(self._robot.data.joint_pos.clone() + self.target_vel.clone() * (5 * self.cfg.decimation / 120))

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # 장애물과 부딛히면 terminated
        # contact_list = torch.sum(self.contact_sensor.data.net_forces_w**2, dim=-1)  # type: ignore
        # contact_list = torch.sqrt(contact_list)
        # contact_list, _ = torch.max(contact_list, dim=1)

        # mask_plus = contact_list > 1
        # mask_reset = contact_list <= 1
        # self._donecount[mask_plus] += 1
        # self._donecount[mask_reset] = 0

        # cancel = self._donecount > 5

        # 목표 도달시 끝
        robot_ef_pos = self._robot.data.body_pos_w[:, self.end_effector_idx].clone()
        target_pos = self.target_object_pos.clone()
        target_disparity = target_pos - robot_ef_pos
        target_distance = torch.sqrt(torch.sum(target_disparity**2, dim=-1))
        self.target_distance = target_distance
        goal_bool = target_distance < 0.01

        # 목표 도달한 시간 기록
        reach_idx = torch.nonzero(target_distance <= 0.01).squeeze()
        self.min_reach_time[reach_idx] = torch.min(self.min_reach_time[reach_idx], self.timesteps[reach_idx])

        # 목표까지 도달하면 바로 학습이 종료되게 설정해 놓았는데 이건 추후 분석이 필요할듯
        # terminated = cancel | goal_bool
        # terminated = cancel
        terminated = torch.zeros_like(goal_bool, dtype=torch.bool)
        self.min_target_distance = torch.min(self.min_target_distance, target_distance)

        # terminated = torch.tensor(False)
        # 시간이 너무 많이 지나면 truncated
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        # if target_distance[0] < 0.01:
        #     self.precision = 0.9 * self.precision + 0.1

        # if truncated[0] is True:
        #     self.precision = 0.9 * self.precision

        return terminated, truncated

    def _get_rewards(self) -> torch.Tensor:
        # 엔드 이펙터와 타겟의 거리 * rew_scale_distance +
        # (장애물과 부딛혔으면 rew_scale_collision) +
        # (목표물에 도착했으면 rew_scale_success)
        # 엔드 이펙터와 타겟의 위치
        robot_ef_pos = self._robot.data.body_pos_w[:, self.end_effector_idx].clone()
        target_pos = self.target_object_pos.clone()

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
        target_distance = torch.sum(target_disparity**2, dim=-1)
        # if self._target_distance_prev is None :
        #     self._target_distance_prev = target_distance
        # target_distance_disparity = self._target_distance_prev - target_distance

        collision_bool = contact_list > 1
        # print("col list : ", collision_list)
        # collision_bool = torch.any(collision_list)

        goal_bool = target_distance < 0.01

        # 각 가속도 평가
        now_joint_vel = self.now_joint_vel.clone()
        joint_acc = torch.mean(torch.abs((now_joint_vel - self.prev_joint_vel) / (self.cfg.decimation / 120)))
        self.prev_joint_vel = self.now_joint_vel

        # computed_reward = (
        #     torch.exp(-self.cfg.rew_scale_distance * target_distance)
        #     + collision_bool.float() * self.cfg.rew_scale_collision
        #     + goal_bool.float() * self.cfg.rew_scale_success
        # )
        computed_reward = (
            torch.exp(-self.cfg.rew_scale_distance * target_distance)
            - joint_acc * self.cfg.rew_scale_acc
        )
        self._target_distance_prev = target_distance
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
        self._robot.set_joint_velocity_target(joint_vel, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self.robot_dof_targets[env_ids] = joint_pos

        # target state
        # rand_target_pose = sample_uniform(-0.7, 0.7, (len(env_ids), 7), self.device)
        # rand_target_pose[:, 2] = sample_uniform(0.1, 0.7, (len(env_ids),), self.device)
        # rand_target_pose[:, 0:3] += self._robot.data.root_pos_w[env_ids]
        # self.target_obj.write_root_pose_to_sim(rand_target_pose, env_ids=env_ids)
        if env_ids is not None:
            rand_vals = make_rand_val(self.grade, env_ids).to(device=self.device) + self._robot.data.root_pos_w[env_ids]
            qurt = torch.zeros((env_ids.shape[0], 4), device=self.device)
            qurt[:, 3] = 1
            initial_target_pose = torch.cat((rand_vals, qurt), dim=1)
            self.target_object_pos[env_ids] = initial_target_pose[:, 0:3].clone()
            # print(initial_target_pose[0])
        else :
            initial_target_pose = torch.zeros((len(env_ids), 7), device=self.device)
            initial_target_pose[:, 0] = -0.5
            initial_target_pose[:, 2] = 0.5
            initial_target_pose[:, 6] = 1
            initial_target_pose[:, 0:3] += self._robot.data.root_pos_w[env_ids]
        self.target_obj.write_root_pose_to_sim(initial_target_pose, env_ids=env_ids)
        self.target_obj.write_root_velocity_to_sim(
            torch.zeros_like(self.target_obj.data.root_vel_w[env_ids]), env_ids=env_ids
        )

        # prev vel 초기화
        self.prev_joint_vel[env_ids] = torch.zeros_like(self._robot.data.joint_vel[env_ids])

        # count reset
        self._donecount[env_ids] = 0

        # precision 업데이트
        # new_precision = torch.mean(self.min_target_distance[env_ids].float()).item()
        self.precision = torch.mean(self.min_target_distance[self.num_envs - self.cfg.env_bundles :].float()).item()
        self.min_target_distance_back[env_ids] = self.min_target_distance[env_ids]
        self.min_target_distance[env_ids] = 100

        if self.precision < 0.01 and self.grade < 4 :
            self.precision = 100
            self.grade += 1

        # timestep 초기화
        self.timesteps[env_ids] = 0
        self.min_reach_time_back[env_ids] = self.min_reach_time[env_ids]
        self.min_reach_time[env_ids] = 480

    def _get_observations(self) -> dict:
        # 현재 로봇팔 각도 + 카메라로 입력된 rgbd 데이터(다듬어진) + 타겟의 위치

        robot_ef_pos = self._robot.data.body_pos_w[:, self.end_effector_idx].clone()
        target_pos = self.target_object_pos.clone()

        pos_disparity = target_pos - robot_ef_pos
        joint_pos = self._robot.data.joint_pos.clone()
        joint_vel = self._robot.data.joint_vel.clone()

        combined_pos = torch.cat((pos_disparity, joint_pos, joint_vel), dim=1)

        return {
            # "policy": torch.tensor([]),
            "joint": combined_pos,
        }
