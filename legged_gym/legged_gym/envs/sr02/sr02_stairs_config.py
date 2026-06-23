# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin
import math
from os import path as osp

from legged_gym.envs.sr02.sr02_config import Sr02RoughCfg, Sr02RoughCfgPPO


class Sr02StairsCfg( Sr02RoughCfg ):

    class init_state( Sr02RoughCfg.init_state ):
        pos = [0.0, 0.0, 0.52]  # x,y,z [m]

    class terrain( Sr02RoughCfg.terrain ):
        mesh_type = 'trimesh'  # "heightfield" # none, plane, heightfield or trimesh
        horizontal_scale = 0.1  # [m]
        vertical_scale = 0.005  # [m]
        border_size = 15  # [m]
        curriculum = True
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        # rough terrain only:
        measure_heights = True
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]  # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = False  # select a unique terrain type and pass all arguments
        terrain_kwargs = None  # Dict of arguments for selected terrain
        max_init_terrain_level = 4  # resume平地模型时先从低中难度台阶开始
        terrain_curriculum_move_up_ratio = 0.25
        terrain_curriculum_move_down_command_ratio = 0.20
        terrain_curriculum_move_down_cap_ratio = 0.15
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 10  # number of terrain rows (levels)
        num_cols = 20  # number of terrain cols (types)
        # terrain types: [flat, rough, smooth_slope, rough_slope, stairs_up, stairs_down, discrete_obstacles, stepping_stones, pit, gap]
        terrain_proportions = [
            0.10,  # flat: 保留少量平地，防止忘掉基础步态
            0.10,  # rough
            0.10,  # smooth_slope
            0.10,  # rough_slope
            0.30,  # stairs_up: 强化高台阶上行样本
            0.20,  # stairs_down
            0.10,  # discrete_obstacles: 随机块/坎路面，保留
            0.00,  # stepping_stones
            0.00,  # pit
            0.00,  # gap
        ]
        
        random_difficulty_range = [0.0, 1.0]
        slope_min_deg = 8.0
        slope_max_deg = 22.0
        stairs_step_height_min = 0.10
        stairs_step_height_max = 0.25
        stairs_step_width = 0.35
        discrete_obstacles_height_min = 0.03
        discrete_obstacles_height_max = 0.12
        discrete_obstacles_num_rectangles = 35
        discrete_obstacles_min_size = 0.35
        discrete_obstacles_max_size = 0.90
        discrete_obstacles_platform_size = 3.0
        # trimesh only:
        slope_treshold = 0.75  # slopes above this threshold will be corrected to vertical surfaces

    class commands( Sr02RoughCfg.commands ):
        curriculum = True
        max_forward_curriculum = 1.5
        max_backward_curriculum = 0.6
        max_lat_curriculum = 0.6
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        heading_command = False # 高台阶尽量正对前进，避免heading修正带来斜撞台阶边

        class ranges( Sr02RoughCfg.commands.ranges ):
            lin_vel_x = [-0.8, 1.2]  # min max [m/s]
            lin_vel_y = [-0.8, 0.8]  # min max [m/s]
            ang_vel_yaw = [-2.14, 2.14]  # min max [rad/s]
            heading = [-math.pi, math.pi]

    class termination:
        base_vel_violate_commands = True

        out_of_border = True

        fall_down = True

    class asset( Sr02RoughCfg.asset ):
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        privileged_contacts_on = ["base", "thigh", "calf", "foot"]

    class domain_rand( Sr02RoughCfg.domain_rand ):
        randomize_payload_mass = True
        payload_mass_range = [0.0, 3.0]

        randomize_com_displacement = True
        com_displacement_range = [-0.03, 0.03]

        randomize_link_mass = False
        link_mass_range = [0.9, 1.1]

        randomize_friction = True
        friction_range = [0.4, 1.25]

        randomize_restitution = False
        restitution_range = [0., 1.0]

        randomize_motor_strength = True
        motor_strength_range = [0.9, 1.1]

        randomize_kp = True
        kp_range = [0.9, 1.1]

        randomize_kd = True
        kd_range = [0.9, 1.1]

        base_init_pos_range = dict(
            x=[-0.5, 0.5],
            y=[-0.5, 0.5],
            z=[0.0, 0.02],
        )

        base_init_rot_range = dict(
            roll=[-0.08, 0.08],
            pitch=[-0.08, 0.08],
            yaw=[-0.0, 0.0],
        )

        base_init_vel_range = dict(
            x=[-0.15, 0.15],
            y=[-0.15, 0.15],
            z=[-0.15, 0.15],
            roll=[-0.15, 0.15],
            pitch=[-0.15, 0.15],
            yaw=[-0.15, 0.15],
        )

        dof_init_pos_ratio_range = [0.95, 1.05]

        randomize_dof_vel = False
        dof_init_vel_range = [-0.1, 0.1]

        disturbance = True
        disturbance_range = [-20.0, 20.0]
        disturbance_interval = 8

        push_robots = True
        push_interval_s = 16
        max_push_vel_xy = 0.5

        delay = True

    class rewards( Sr02RoughCfg.rewards ):
        class scales:
            # general
            termination = -100
            # velocity-tracking
            tracking_lin_vel = 2.0
            tracking_ang_vel = 1.5
            # root
            lin_vel_z = -4.0
            ang_vel_xy = -0.10
            orientation = -3.0
            stand_roll = -3.0
            base_height = -2.5
            # joint
            torques = -0.0001
            torque_limits = -0.0
            dof_vel = -0.0
            dof_acc = -1.5e-7
            stand_still = -0.10
            hip_pos = -0.08
            thigh_pose = 0.0
            calf_pose = 0.0
            dof_pos_limits = -0.0
            dof_vel_limits = -0.0
            joint_power = -2e-5
            feet_mirror = -0.035
            # action
            action_rate = -0.010
            smoothness = -0.004
            hip_action_magnitude = -0.0
            # contact
            collision = -3.0 #“小腿或大腿经常扫到台阶”，再重点加 collision 惩罚，避免学会抬腿过高来躲避台阶，导致能过台阶但不稳
            feet_contact_forces = -0.00005
            # others
            feet_air_time = 0.60
            trot_phase = 0.0
            contact_count = -0.06
            has_contact = 0.05
            feet_stumble = -3.5 # 脚尖撞台阶边然后摔优先调 feet_stumble / feet_clearance_terrain
            obstacle_contact_clearance = -4  # 足端撞到台阶侧面时的额外惩罚；鼓励抬脚避开台阶边，而不是直接撞上去
            feet_slide = -0.02
            backward_feet_clearance = -0.0
            feet_clearance_base = -0.5
            feet_clearance_terrain = -0.04
            feet_yaw_clearance_terrain = 0.6
            stuck = -0.25   
            upward = 0.0

        only_positive_rewards = False  # 是否把总奖励裁剪为非负；False保留负奖励信号，便于惩罚摔倒、碰撞、拖脚等坏行为
        tracking_sigma = 0.30  # 线速度跟踪奖励宽度，公式约为exp(-error^2/sigma)；越小越严格，速度误差稍大就掉奖励
        tracking_ang_vel_sigma = 0.12  # yaw角速度跟踪奖励宽度；越小越强调精确转向，小yaw速度也不容易被学成不转
        soft_dof_pos_limit = 0.95  # 关节位置软限位比例；超过URDF限位的95%开始惩罚，调低可更早远离关节极限
        soft_dof_vel_limit = 0.95  # 关节速度软限位比例；超过速度上限的95%开始惩罚，防止动作过猛或数值不稳定
        soft_torque_limit = 0.95  # 关节力矩软限位比例；超过力矩上限的95%开始惩罚，减少长期顶满力矩的策略
        base_height_target = 0.52  # 机身目标离地高度；base_height奖励会惩罚机身过低或过高，台阶任务过低容易拖腿/撞台阶
        feet_height_target_base = -0.42  # 足端相对base坐标系的目标高度；用于约束摆腿时足端相对机身的高度形态
        feet_height_target_terrain = 0.30  # 足端相对地形的目标清障高度；高台阶需要更高摆腿，过大可能导致跳跃、能耗高或步态不稳
        backward_feet_height_target = 0.08  # 后退时足端相对地形的目标清障高度；后退通常不需要像上台阶那样抬得很高
        feet_stumble_min_terrain_level = 0  # feet_stumble惩罚生效的最低地形难度等级；0表示从最低难度台阶/障碍就开始惩罚绊脚
        obstacle_contact_height_target = 0.32  # 足端低高度撞到台阶侧面时的目标清障高度；低于该高度发生侧向碰撞会被额外惩罚
        obstacle_contact_force_ratio = 2.5  # 判定足端撞障碍侧面的力比例阈值；横向接触力大于该倍数的竖向力时认为更像撞边
        obstacle_contact_force_threshold = 10.0  # 判定足端撞障碍侧面的最小横向接触力阈值；过滤轻微接触和接触力噪声
        trot_period = 0.5  # 小跑相位周期，单位秒；启用trot_phase奖励时用于定义对角步态节奏
        max_contact_force = 300.  # 足端最大接触力阈值；超过该值的接触力会触发feet_contact_forces惩罚

logs_root = osp.join(osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))), "logs")
class Sr02StairsCfgPPO( Sr02RoughCfgPPO ):

    class runner( Sr02RoughCfgPPO.runner ):
        policy_class_name = 'HIMActorCritic'
        # algorithm_class_name = 'HybridPPO'
        num_steps_per_env = 100  # per iteration
        max_iterations = 6000  # number of policy updates

        # logging
        save_interval = 100  # check for potential saves every this many iterations
        experiment_name = 'stairs_sr02'
        run_name = ''
        # load and resume
        resume = False
        load_run = -1
        checkpoint = -1  # -1 = last saved model
