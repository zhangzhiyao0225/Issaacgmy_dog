from .x30_mgdp_config_stage1 import X30MGDPCfgStage1, X30MGDPCfgPPOStage1
import numpy as np


class X30MGDPCfgStage2(X30MGDPCfgStage1):
    encoder_ppo_ref = None

    class env(X30MGDPCfgStage1.env):
        env_gait = 1

    class camera(X30MGDPCfgStage1.camera):
        update_wm = True
        load_world_model_policy = True
        load_world_model_policy_file = "{LEGGED_GYM_ROOT_DIR}" + '/models/MGDP/x30_stage1/001'

    class terrain(X30MGDPCfgStage1.terrain):
        mesh_type = 'trimesh'
        curriculum = True
        max_init_terrain_level = 0
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 10
        num_cols = 10
        # Stage2: slopes and stairs after the stage1 world model is available.
        terrain_proportions = [0.15, 0.15, 0.35, 0.35, 0.0, 0.0, 0.0]
        x_min, x_max, x_step = -0.40, 1.30, 0.1
        y_min, y_max, y_step = -0.50, 0.60, 0.1
        measured_points_x = np.round(np.arange(x_min, x_max, x_step), 2)
        measured_points_y = np.round(np.arange(y_min, y_max, y_step), 2)
        num_point_x = measured_points_x.shape[0]
        num_point_y = measured_points_y.shape[0]

    class commands(X30MGDPCfgStage1.commands):
        curriculum = True
        heading_command = False
        zero_command = True
        max_curriculum = 1.5
        min_curriculum = 0.0

        class ranges(X30MGDPCfgStage1.commands.ranges):
            lin_vel_x = [0.0, 1.5]
            lin_vel_y = [-0.3, 0.3]
            ang_vel_yaw = [-0.8, 0.8]
            heading = [-3.14, 3.14]

    class init_state(X30MGDPCfgStage1.init_state):
        pos = [0.0, 0.0, 0.50]

    class domain_rand(X30MGDPCfgStage1.domain_rand):
        randomize_action_latency = True
        latency_range = [0.00, 0.02]
        push_robots = True
        push_interval_s = 16
        max_push_vel_xy = 0.6

    class rewards(X30MGDPCfgStage1.rewards):
        terrain_adaptive_reward = False
        base_height_target = 0.50

        class scales(X30MGDPCfgStage1.rewards.scales):
            tracking_lin_vel = 1.5
            tracking_ang_vel = 0.6
            lin_vel_z = -2.0
            ang_vel_xy = -0.10
            torques = -1e-5
            dof_acc = -2.5e-7
            action_rate = -0.01
            orientation = -0.4
            collision = -1.5
            motion_trot = -0.05
            feet_air_time = 0.7
            feet_stumble = -1.0
            stand_still = -0.05
            base_height = -0.3
            feet_edge = -0.5


class X30MGDPCfgPPOStage2(X30MGDPCfgPPOStage1):
    class runner(X30MGDPCfgPPOStage1.runner):
        experiment_name = 'x30_mgdp_stage2'
        max_iterations = 30000
        save_interval = 1000


X30MGDPCfgStage2.encoder_ppo_ref = X30MGDPCfgPPOStage2
