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

class Sr02RoughRecoverCfg( Sr02RoughCfg ):

    class init_state( Sr02RoughCfg.init_state ):
        pos = [0.0, 0.0, 0.34]  # fallen recovery starts close to the ground, not dropped from standing height

    class terrain( Sr02RoughCfg.terrain ):
        mesh_type = 'trimesh'  # "heightfield" # none, plane, heightfield or trimesh
        horizontal_scale = 0.1  # [m]
        vertical_scale = 0.005  # [m]
        border_size = 15  # [m]
        curriculum = False
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        # rough terrain only:
        measure_heights = True
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]  # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = False  # select a unique terrain type and pass all arguments
        terrain_kwargs = None  # Dict of arguments for selected terrain
        max_init_terrain_level = 0  # starting curriculum state
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 10  # number of terrain rows (levels)
        num_cols = 20  # number of terrain cols (types)
        # terrain types: [flat, rough, smooth_slope, rough_slope, stairs_up, stairs_down, discrete_obstacles, stepping_stones, pit, gap]
        terrain_proportions = [0.8, 0.2]
        # trimesh only:
        slope_treshold = 0.75  # slopes above this threshold will be corrected to vertical surfaces

    class commands( Sr02RoughCfg.commands ):
        curriculum = False
        max_forward_curriculum = 0.0
        max_backward_curriculum = 0.0
        max_lat_curriculum = 0.0
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        heading_command = False # if true: compute ang vel command from heading error

        class ranges( Sr02RoughCfg.commands.ranges ):
            lin_vel_x = [0.0, 0.0]  # 纯摔倒恢复：不带速度命令
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]
            heading = [0.0, 0.0]

    class asset( Sr02RoughCfg.asset ):
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf", "base"]
        terminate_after_contacts_on = []  # 倒地恢复需 取消base触地终止
        privileged_contacts_on = ["base", "thigh", "calf", "foot"]
        self_collisions = 0  # 1：禁用自身各部分之间的碰撞检测（提升性能）；0：启用

    class termination( Sr02RoughCfg.termination ):
        base_vel_violate_commands = False
        fall_down = False  # 翻身早期会有冲击/下落速度，先不要因为这个重置

    class control( Sr02RoughCfg.control ):
        # 打滚起立要留出关节摆幅，但不能太猛；过大容易学成瞬间甩身。
        action_scale = 0.40

    class domain_rand ( Sr02RoughCfg.domain_rand ):
        base_init_pos_range = dict(
            x=[-0.05, 0.05],
            y=[-0.05, 0.05],
            z=[0.0, 0.03],
        )
        base_init_rot_range = dict(
            roll=[-0.2, 0.2],
            pitch=[-0.2, 0.2],
            yaw=[-3.14, 3.14],
        )

        randomize_payload_mass = False
        randomize_com_displacement = False
        randomize_motor_strength = False
        randomize_kp = False
        randomize_kd = False
        disturbance = False
        push_robots = False
        delay = False

        recover_mode = True
        recover_init_mode = True
        recover_init_mode_prob = [0.45, 0.45, 0.10]  # 四脚朝天、侧躺、趴地/低姿
        recover_init_roll_noise = 0.25
        recover_init_pitch_noise = 0.18
        recover_init_yaw_noise = 3.14
        base_init_vel_range = dict(
            x=[-0.05, 0.05],
            y=[-0.05, 0.05],
            z=[-0.05, 0.05],
            roll=[-0.10, 0.10],
            pitch=[-0.10, 0.10],
            yaw=[-0.10, 0.10],
        )

    class rewards( Sr02RoughCfg.rewards ):
        class scales:
            # general
            termination = -0.0
            # velocity-tracking
            tracking_lin_vel = 0.0
            tracking_ang_vel = 0.0
            # root
            lin_vel_z_up = -0.60
            ang_vel_xy_up = -0.08
            orientation_up = -1.20
            base_height_up = -1.00
            upside_down = -1.0  # 四脚朝天/背部朝地惩罚，避免仰翻后奖励静默
            down_feet_contact = 1.5  # 仰翻时鼓励脚先够到地面，为翻身提供支点
            recover_roll_over = 3.0  # 仰翻/侧躺阶段鼓励先滚到趴姿
            recover_twist_penalty = -0.05  # 抑制 pitch/yaw 乱拧，保留主要翻身动作
            recover_to_prone = 3.0  # 阶段成功：先翻到肚子朝下
            recover_success = 5.0  # 最终要站起来，不能只趴着完成任务
            recover_ref_pose = 0.35  # 轻量参考姿态牵引，避免猛烈乱甩
            # joint
            torques = -5e-5
            torque_limits = -0.0
            dof_vel = -0.002
            dof_acc = -1.0e-7
            stand_nice = -0.03
            hip_pos_up = -0.02
            thigh_pose_up = -0.01
            calf_pose_up = -0.01
            dof_pos_limits = -0.0
            dof_vel_limits = -0.0
            joint_power = -1e-5
            feet_mirror_up = -0.01
            # action
            action_rate = -0.008
            smoothness = -0.004
            hip_action_magnitude = -0.0
            # contact
            collision_up = -0.35
            feet_contact_forces = -0.00008
            # others
            feet_air_time = 0.0
            has_contact = 0.0  # 避免奖励趴着不动/全脚触地，翻身阶段用 down_feet_contact 即可
            feet_stumble_up = -0.0
            feet_slide_up = -0.02
            feet_clearance_base_up = -0.0
            feet_clearance_terrain_up = -0.0
            feet_yaw_clearance_terrain = 0.0
            stuck = 0.0
            upward = 0.0  # 用 recover_to_prone 作为更明确的翻身阶段奖励

        only_positive_rewards = False  # 恢复任务需要保留正负差异，避免总奖励长期被裁成0
        tracking_sigma = 0.25  # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 0.95  # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 0.95
        soft_torque_limit = 0.95
        base_height_target = 0.53
        feet_height_target_base = -0.32
        feet_height_target_terrain = 0.15
        max_contact_force = 220.  # forces above this value are penalized
        recover_ref_sigma = 0.55
        recover_roll_speed_target = 0.45
        recover_roll_speed_sigma = 0.25
        # 参考姿态[FL_hip, FL_thigh, FL_calf, FR_hip, FR_thigh, FR_calf,
        #             RL_hip, RL_thigh, RL_calf, RR_hip, RR_thigh, RR_calf]
        # 四脚朝天：给左右两个可选翻身姿态，让策略像睡觉翻身一样自己选方向滚到趴姿。
        recover_ref_supine = [-0.45, -1.20, 2.25,
                               0.45,  1.20, 2.25,
                              -0.45, -1.20, 2.25,
                               0.45,  1.20, 2.25]
        recover_ref_supine_left = [-0.65, -1.15, 2.20,
                                    0.15,  0.55, 1.55,
                                   -0.65, -1.15, 2.20,
                                    0.15,  0.55, 1.55]
        recover_ref_supine_right = [-0.15, -0.55, 1.55,
                                     0.65,  1.15, 2.20,
                                    -0.15, -0.55, 1.55,
                                     0.65,  1.15, 2.20]
        # 侧躺：一侧收腿、一侧撑腿。
        recover_ref_side = [-0.35, -1.05, 2.05,
                             0.35,  1.05, 2.05,
                            -0.35, -1.05, 2.05,
                             0.35,  1.05, 2.05]
        recover_ref_side_left = [-0.45, -1.05, 2.05,
                                  0.10,  0.75, 1.70,
                                 -0.45, -1.05, 2.05,
                                  0.10,  0.75, 1.70]
        recover_ref_side_right = [-0.10, -0.75, 1.70,
                                   0.45,  1.05, 2.05,
                                  -0.10, -0.75, 1.70,
                                   0.45,  1.05, 2.05]
        # 趴地/低姿：收腿到可支撑抬身的蹲姿。
        recover_ref_prone = [-0.10, -0.95, 1.90,
                              0.10,  0.95, 1.90,
                             -0.10, -0.95, 1.90,
                              0.10,  0.95, 1.90]
        recover_ref_stand = [-0.05, -0.795, 1.60,
                              0.05,  0.795, 1.60,
                             -0.05, -0.795, 1.60,
                              0.05,  0.795, 1.60]

    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
        clip_observations = 100.
        clip_actions = 100.


logs_root = osp.join(osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))), "logs")
class Sr02RoughRecoverCfgPPO( Sr02RoughCfgPPO ):

    class policy( Sr02RoughCfgPPO.policy ):
        init_noise_std = 0.25

    class algorithm( Sr02RoughCfgPPO.algorithm ):
        entropy_coef = 0.0005

    class runner( Sr02RoughCfgPPO.runner ):
        policy_class_name = 'HIMActorCritic'
        algorithm_class_name = 'HIMPPO'
        num_steps_per_env = 100  # per iteration
        max_iterations = 10000  # number of policy updates

        # logging
        save_interval = 100  # check for potential saves every this many iterations
        experiment_name = 'recover_sr02'
        run_name = 'slow_roll_stand'
        # load and resume
        resume = False
        load_run = -1
        checkpoint = -1  # -1 = last saved model
