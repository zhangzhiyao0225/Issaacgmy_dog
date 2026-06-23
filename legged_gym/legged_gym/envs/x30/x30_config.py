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
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class X30RoughCfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env ):
        num_envs = 4096  # 并行仿真的环境数量（需根据GPU显存调整）
        num_one_step_observations = 45  # 单步 观测向量 维度（原始传感器数据）
        num_observations = num_one_step_observations * 6    # 总 观测向量 维度（含6步历史）
        num_one_step_privileged_obs = 45 + 3 + 3 + 187  # 单步 特权观测向量 维度，（+3维线速度 + 3维随机扰动力 + 地形扫描187）
        num_privileged_obs = num_one_step_privileged_obs * 1    # 总 特权观测向量 维度，if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise
        num_actions = 12  # 动作空间维度（12个关节）
        env_spacing = 3.  # 环境之间的间距（单位：米），not used with heightfields/trimeshes
        send_timeouts = True  # 是否发送超时信号给算法，send time out information to the algorithm
        episode_length_s = 20  # 单次训练Episode的时长（秒），episode length in seconds

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.50]   # 初始位置（x,y,z）单位：米
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        default_joint_angles = {
            'FL_HipX_joint': 0.10,
            'FR_HipX_joint': 0.10,
            'HL_HipX_joint': 0.10,
            'HR_HipX_joint': 0.10,

            'FL_HipY_joint': -0.715,
            'FR_HipY_joint': -0.715,
            'HL_HipY_joint': -0.715,
            'HR_HipY_joint': -0.715,

            'FL_Knee_joint': 1.43,
            'FR_Knee_joint': 1.43,
            'HL_Knee_joint': 1.43,
            'HR_Knee_joint': 1.43,
        }


    class terrain( LeggedRobotCfg.terrain ):
        mesh_type = 'trimesh' # "heightfield" # none, plane, heightfield or trimesh
        horizontal_scale = 0.1 # [m]
        vertical_scale = 0.005 # [m]
        border_size = 15 # [m]
        curriculum = False
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        # rough terrain only:
        measure_heights = True
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = False # select a unique terrain type and pass all arguments
        terrain_kwargs = None # Dict of arguments for selected terrain
        max_init_terrain_level = 0 # 平地基础训练先从最低难度开始
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 10 # number of terrain rows (levels)
        num_cols = 20 # number of terrain cols (types)
        # terrain types: [flat, rough, smooth_slope, rough_slope, stairs_up, stairs_down, discrete_obstacles, stepping_stones, pit, gap]
        terrain_proportions = [
            0.80,  # flat: 先训练稳定平地步态
            0.20,  # rough: 少量轻微粗糙，避免过拟合纯平面
            0.00,  # smooth_slope
            0.00,  # rough_slope
            0.00,  # stairs_up
            0.00,  # stairs_down
            0.00,  # discrete_obstacles
            0.00,  # stepping_stones
            0.00,  # pit
            0.00,  # gap
        ]
        random_difficulty_range = [0.0, 0.3]
        slope_min_deg = 0.0
        slope_max_deg = 8.0
        stairs_step_height_min = 0.06
        stairs_step_height_max = 0.10
        stairs_step_width = 0.35
        stepping_stones_size_min = 0.45  # 踏石/石子最小边长
        stepping_stones_size_max = 0.75
        stepping_stones_distance_min = 0.08
        stepping_stones_distance_max = 0.18
        # terrain types: [flat, rough, smooth_slope, rough_slope, stairs_up, stairs_down, discrete_obstacles, stepping_stones, pit, gap]
        # terrain_proportions = [0.6, 0.4, 0.0, 0.0]

        # trimesh only:
        slope_treshold = 0.75 # slopes above this threshold will be corrected to vertical surfaces

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'  # 控制类型（P=位置控制，T=力矩控制）
        stiffness = {'HipX': 100.0, 'HipY': 100.0, 'Knee': 100.0} # 关节刚度（单位：N·m/rad）
        damping = {'HipX': 3.0, 'HipY': 3.0, 'Knee': 3.0}    # 关节阻尼（单位：N·m·s/rad）
        action_scale = 0.25  # 动作缩放因子（目标角度 = 动作 * scale + 默认角度）
        decimation = 20      # policy_dt = sim.dt * decimation = 0.02s
        hip_reduction = 1.0 # 髋关节扭矩缩放因子（用于平衡前后腿负载）

    class commands( LeggedRobotCfg.commands ):
        curriculum = True
        max_forward_curriculum = 1.5  # x_vel 限制 [-1.0, 1.5]
        max_backward_curriculum = 0.6
        max_lat_curriculum = 0.6  # y_vel 限制 [-0.6, 0.6]
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        heading_command = False # 平地基础模型直接训练yaw速度，保证0.15/0.2rad/s也能转

        class ranges( LeggedRobotCfg.commands.ranges ):
            lin_vel_x = [-1.0, 2.0]  # min max [m/s]
            lin_vel_y = [-1.0, 1.0]  # min max [m/s]
            ang_vel_yaw = [-3.14, 3.14]    # min max [rad/s]
            heading = [-math.pi, math.pi]

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/X30_description/x30_mjcf/X30_urdf/urdf/X30.urdf'
        name = "x30"
        foot_name = "FOOT"
        penalize_contacts_on = ["THIGH", "SHANK"]
        terminate_after_contacts_on = ["TORSO"]
        privileged_contacts_on = ["TORSO", "THIGH", "SHANK", "FOOT"]
        self_collisions = 1
        collapse_fixed_joints = False
        flip_visual_attachments = False

    class termination:
        base_vel_violate_commands = False  # 是否在终止条件中考虑 当地形等级>3时，base速度 与 命令速度差异过大(超过2m/s)（摔倒恢复训练关闭）

        out_of_border = True  # 是否在终止条件中考虑 走出边界外

        fall_down = True  # 是否在终止条件中考虑 跌落(base的z方向线速度 < -5)

    class domain_rand:
        # startup
        randomize_payload_mass = True  # 是否随机改变 base的质量（默认质量 ±）
        payload_mass_range = [0.0, 5.0]

        randomize_com_displacement = True  # 是否随机改变 base的质心偏移（xyz）
        com_displacement_range = [-0.05, 0.05]

        randomize_link_mass = False  # 是否随机更改env各刚体部位（除了base）的质量（默认质量 *）
        link_mass_range = [0.9, 1.1]

        # startup and reset
        randomize_friction = True  # 是否随机化env各刚体部位的 摩擦系数
        friction_range = [0.2, 1.25]

        randomize_restitution = False  # 是否随机化env各刚体部位的 弹性系数
        restitution_range = [0., 1.0]

        # reset
        randomize_motor_strength = True  # 是否随机化env的电机强度（输出的actions *）
        motor_strength_range = [0.9, 1.1]

        randomize_kp = True  # 是否 随机改变PD控制器的p增益（stiffness）
        kp_range = [0.9, 1.1]

        randomize_kd = True  # 是否 随机改变PD控制器的D增益（damping）
        kd_range = [0.9, 1.1]

        # 重置时随机改变base的 位置（初始位置 +），默认x,y方向为 [-1, 1]，z方向为 0，若更改则为下面的
        base_init_pos_range = dict(
            x=[-1.0, 1.0],
            y=[-1.0, 1.0],
            z=[0.0, 0.02],
        )
        # 重置时随机设置base的 方向（摔倒恢复模式都设为 [-3.14, 3.14]）
        base_init_rot_range = dict(
            roll=[-0.05, 0.05],
            pitch=[-0.05, 0.05],
            yaw=[-0.0, 0.0],
        )
        # 重置时随机设置base的 线速度、角速度，默认x,y,x,rool,pitch,roll方向为 [-0.5, 0.5]，若更改则为下面的
        base_init_vel_range = dict(
            x=[-0.1, 0.1],
            y=[-0.1, 0.1],
            z=[-0.1, 0.1],
            roll=[-0.1, 0.1],
            pitch=[-0.1, 0.1],
            yaw=[-0.1, 0.1],
        )

        dof_init_pos_ratio_range = [0.95, 1.05]  # 重置时随机改变 关节初始位置（初始关节位置 *），默认为 [0.5, 1.5]

        randomize_dof_vel = False  # 重置时设置 关节初始速度
        dof_init_vel_range = [-0.1, 0.1]  # 默认为 0.0

        # interval
        disturbance = True  # 是否给base施加一个随机扰动力（xyz方向）
        disturbance_range = [-30.0, 30.0]  # N
        disturbance_interval = 8

        push_robots = True  # 是否给base在水平方向施加一个线速度
        push_interval_s = 16  # step间隔 [s]
        max_push_vel_xy = 1.  # 施加的最大线速度 [1m/s]

        delay = True  # actions是否随机延迟一个 policy_dt

        recover_mode = False  # 是否开启摔倒恢复模式

    class rewards( LeggedRobotCfg.rewards ):
        class scales:
            # general
            termination = -0.0  # 仿真终止时的惩罚：未启用。设为负值（如-10.0）可在跌倒时给予额外惩罚
            # velocity-tracking
            tracking_lin_vel = 2.0  # commands 中XY方向的 线速度跟踪 奖励 (>= 0.1m/s时)
            tracking_ang_vel = 1.5  # commands 中yaw方向的 角速度跟踪 奖励，小yaw也要能跟踪
            # root
            lin_vel_z = -6.0  # base 的 Z 轴线速度 惩罚：防止高速前进时跳跃
            ang_vel_xy = -0.15  # base 的 XY 轴角速度 惩罚：抑制机身翻滚（roll, pitch）
            orientation = -4.0  # base 非水平姿态 惩罚（地面不平时，可减小）
            stand_roll = -4.0  # 原地静止时的侧倾惩罚，增强站立稳定性
            base_height = -2.5  # base 目标高度惩罚，约束机身保持在 base_height_target 附近
            # joint
            torques = -0.0001  # 关节扭矩过大 惩罚
            torque_limits = -0.0  # 关节扭矩接近极限 惩罚
            dof_vel = -0.0  # 关节速度过大 惩罚
            dof_acc = -1.8e-7  # 关节加速度 惩罚（若步态抖动，可增大惩罚）
            stand_still = -0.12  # base原地静止时的关节位置偏差惩罚，增强站立稳定性
            hip_pos = -0.1  # hip关节位置与默认位置的 偏差 惩罚，抑制后腿外叉；原地静止时惩罚更强
            front_feet_y_distance = -0.8  # 前进时前脚横向间距过小惩罚，防止前腿向中线夹窄
            thigh_pose = 0.0
            calf_pose = 0.0
            dof_pos_limits = -0.0  # 关节位置接近极限 惩罚
            dof_vel_limits = -0.0  # 关节速度接近极限 惩罚
            joint_power = -2.2e-5  # 关节高功率 惩罚：降低能耗（需平衡运动效率，过高惩罚会导致动作迟缓）
            feet_mirror = -0.035  # 斜对称腿的关节位置偏差 惩罚
            # action
            action_rate = -0.009  # action变化 惩罚
            smoothness = -0.004  # action二阶平滑性 惩罚（复杂地形，可适当降低）
            hip_action_magnitude = -0.0  # action 中的 髋关节hip（0,3,6,9）动作幅度 惩罚（防止 > 1.0）
            # contact
            collision = -2.0  # 大腿等非足端接触惩罚，抑制跪地/趴地
            feet_contact_forces = -0.00005  # 四足的接触力 > 100N 惩罚
            # others
            feet_air_time = 0.7  # 鼓励迈开腿；配合lin_vel_z/contact_count抑制跳跃
            trot_phase = -0.05  # 固定时钟相位暂时关闭，避免早期未站稳时强拉节律
            contact_count = -0.08  # 运动时鼓励约2只脚接触地面，抑制四脚一起蹦
            has_contact = 0.15  # (base 原地不动) 时的 四足触地个数 奖励
            feet_stumble = 0.0  # 四足接触到垂直表面 惩罚
            feet_slide = -0.025  # 脚接触地面具有相对base的速度 惩罚
            backward_feet_clearance = -0.8  # 后退时低高度足端移动惩罚，抑制后退拖地
            feet_clearance_base = -0.0  # 大速度下 四足距base目标距离 惩罚
            feet_clearance_terrain = -0.18  # 大速度下 四足离地目标高度 惩罚
            feet_yaw_clearance_terrain = 0.45  # (base原地旋转) 时 鼓励抬脚换步，减少拧身/拖脚
            stuck = -0.02  # base 卡住 惩罚
            upward = 0.0  # 重力投影向下 奖励（恢复训练时开启）

        reward_curriculum = False
        reward_curriculum_term = ["feet_edge"]
        reward_curriculum_schedule = [[4000, 10000, 0.1, 1.0]]

        only_positive_rewards = False   # 负奖励保留：为True时总奖励不低于零，避免早期训练频繁终止。复杂任务建议保持False
        tracking_sigma = 0.30  # 跟踪奖励的高斯分布标准差 = exp(-error^2 / sigma)
        tracking_ang_vel_sigma = 0.12  # 小yaw速度也要明显区分，避免0.15/0.2rad/s被学成不转
        soft_dof_pos_limit = 0.95   # 关节位置软限位：关节角度超过URDF限位95%时触发惩罚。调低（如0.9）可提前约束
        soft_dof_vel_limit = 0.95   # 关节速度软限位：超过最大速度95%时惩罚。保护电机模型不过载
        soft_torque_limit = 0.95    # 关节力矩软限位：超过额定扭矩95%时惩罚。防止仿真数值发散
        base_height_target = 0.50  # 机身目标高度
        feet_height_target_base = -0.46  # 足部距base的 相对距离目标（抬脚高度为0.15 以适应台阶地形）
        feet_height_target_terrain = 0.10  # 足部离地高度目标，平地步态保持清脚但不过度跳
        backward_feet_height_target = 0.12  # 后退时足端离地目标下限，低于该高度且相对base运动会被惩罚
        front_feet_y_distance_target = 0.32  # 前两脚在机身坐标系下的最小横向间距 [m]
        trot_period = 0.5  # 对角小跑周期 [s]
        max_contact_force = 300.    # 四足接触力 > 100N 时触发惩罚的阈值

    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
        clip_observations = 100.
        clip_actions = 100.

    class noise:
        add_noise = True
        noise_level = 1.0 # scales other values
        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1

    class sim( LeggedRobotCfg.sim ):
        dt = 0.001  # 物理仿真步长；配合 control.decimation=20 得到 0.02s 控制周期


class X30RoughCfgPPO( LeggedRobotCfgPPO ):
    seed = 1
    runner_class_name = 'HIMOnPolicyRunner'

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        # rnn_type = 'lstm'
        # rnn_hidden_size = 512
        # rnn_num_layers = 1

    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01  # 熵系数（鼓励探索）

        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        num_learning_epochs = 5
        num_mini_batches = 4  # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.e-3  # 5.e-4
        schedule = 'adaptive'  # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.

    class runner( LeggedRobotCfgPPO.runner ):
        policy_class_name = 'HIMActorCritic'
        algorithm_class_name = 'HIMPPO'
        num_steps_per_env = 100  # per iteration
        max_iterations = 3000  # number of policy updates

        # logging
        save_interval = 100  # check for potential saves every this many iterations
        experiment_name = 'flat_x30'
        run_name = ''
        # load and resume
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
