from .legged_robot_config_baseline import LeggedRobotBaseCfg, LeggedRobotBaseCfgPPO
import numpy as np


class X30MGDPCfgStage1(LeggedRobotBaseCfg):
    encoder_ppo_ref = None

    class env(LeggedRobotBaseCfg.env):
        num_envs = 4096
        num_observations = 45
        num_privileged_obs = 3 + 4 + 17 * 11
        train_type = "MGDP"
        num_env_morph_priv_obs = 33
        num_histroy_obs = 4
        graphics_device_num = 0
        use_history = True
        env_gait = 3
        episode_length_s = 20

    class camera(LeggedRobotBaseCfg.camera):
        render_compare_pre_vis = False
        render_compare_pre_map = False
        render_compare_pre_map_real_data = False
        debug_height_stats = False
        camera_type = 'warp'
        use_camera = True
        use_lidar = False
        world_model = True
        use_memory = True

        # X30 camera extrinsics in the base frame. Keep these aligned with the
        # mechanical LiDAR mount unless the camera bracket is measured separately.
        offset_translation = [0.433, 0.0, -0.055]
        offset_rotation = [136.0, 0.0, 90.0]
        offset_trans_rand_min = [0.0, 0.0, 0.0]
        offset_trans_rand_max = [0.0, 0.0, 0.0]
        offset_rot_rand_min = [0.0, 0.0, 0.0]
        offset_rot_rand_max = [0.0, 0.0, 0.0]

        noise_gaussian = 0.03
        noise_dropout = 0.1

        normalize = True
        resized = [16, 16]
        near_clip = 0.1
        far_clip = 5.0
        horizontal_fov = 67.0

        update_wm = True
        use_map_decoder = True
        disable_cudnn_wm = True

        wm_visual_mse_weight = 1.2
        wm_visual_l1_weight = 0.0
        wm_height_mse_weight = 1.0
        wm_height_l1_weight = 0.0
        wm_height_visual_mse_weight = 0.0
        wm_contrastive_weight = 0.3
        wm_best_visual_weight = 0.6
        wm_best_height_weight = 0.4

        wm_lr = 1e-4
        wm_decoder_lr_scale = 1.0
        wm_weight_decay = 1e-5
        wm_grad_clip = 1.0

        update_interval = 5

        load_world_model_policy = False
        load_world_model_policy_file = "{LEGGED_GYM_ROOT_DIR}" + '/models/MGDP/x30_stage1/001'

    class terrain(LeggedRobotBaseCfg.terrain):
        mesh_type = 'trimesh'
        measure_heights = True
        curriculum = False
        max_init_terrain_level = 0
        horizontal_scale = 0.1
        vertical_scale = 0.005
        border_size = 15
        terrain_length = 8.
        terrain_width = 8.
        # Keep the training mesh larger than the IsaacGym env grid; otherwise
        # high num_envs runs spawn part of the batch outside the terrain.
        num_rows = 16
        num_cols = 40
        terrain_proportions = [0.80, 0.20, 0.0, 0.0, 0.0, 0.0, 0.0]
        slope_treshold = 0.75

        x_min, x_max, x_step = -0.40, 1.30, 0.1
        y_min, y_max, y_step = -0.50, 0.60, 0.1
        measured_points_x = np.round(np.arange(x_min, x_max, x_step), 2)
        measured_points_y = np.round(np.arange(y_min, y_max, y_step), 2)
        num_point_x = measured_points_x.shape[0]
        num_point_y = measured_points_y.shape[0]

    class commands(LeggedRobotBaseCfg.commands):
        curriculum = True
        heading_command = False
        zero_command = True
        max_curriculum = 0.5
        min_curriculum = 0.0
        resampling_time = 10.0

        class ranges(LeggedRobotBaseCfg.commands.ranges):
            lin_vel_x = [0.0, 0.5]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [-0.3, 0.3]
            heading = [-3.14, 3.14]

    class init_state(LeggedRobotBaseCfg.init_state):
        pos = [0.0, 0.0, 0.50]
        default_joint_angles = {
            'FL_HipX_joint': 0.0,
            'FR_HipX_joint': 0.0,
            'HL_HipX_joint': 0.0,
            'HR_HipX_joint': 0.0,

            'FL_HipY_joint': -0.68,
            'FR_HipY_joint': -0.68,
            'HL_HipY_joint': -0.68,
            'HR_HipY_joint': -0.68,

            'FL_Knee_joint': 1.45,
            'FR_Knee_joint': 1.45,
            'HL_Knee_joint': 1.45,
            'HR_Knee_joint': 1.45,
        }

    class control(LeggedRobotBaseCfg.control):
        control_type = "P"
        stiffness = {'HipX': 200.0, 'HipY': 200.0, 'Knee': 200.0}
        damping = {'HipX': 5.0, 'HipY': 5.0, 'Knee': 5.0}
        action_scale = 0.25
        decimation = 20

    class asset(LeggedRobotBaseCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/X30_description/x30_mjcf/X30_urdf/urdf/X30.urdf'
        name = "x30_mgdp"
        asset_name = ["x30_mgdp"]
        foot_name = "FOOT"
        penalize_contacts_on_narrow = ["TORSO", "THIGH", "SHANK"]
        terminate_after_contacts_on_narrow = []
        penalize_contacts_on = ["TORSO", "THIGH", "SHANK"]
        terminate_after_contacts_on = []
        self_collisions = 1
        collapse_fixed_joints = False
        flip_visual_attachments = False
        replace_cylinder_with_capsule = True

    class privInfo(LeggedRobotBaseCfg.privInfo):
        enableMeasuredVel = True
        enablePayload = False
        enableFriction = False
        enableStiffnessDamping = False
        enableMotorStrength = False
        enablemMotorOffsets = False
        enableCom = False
        enableLimb_mass = False
        enableForce = False
        enableFootContact = False
        enableFootHeight = True
        enableMaxFootHeight = False
        enableMeasuredHeight = True

    class termination:
        min_base_height = 0.18
        max_lin_vel_z = 5.0
        max_projected_gravity_xy = 0.95

    class domain_rand(LeggedRobotBaseCfg.domain_rand):
        randomize_base_mass = True
        added_mass_range = [0.0, 3.0]
        randomize_link_mass = False
        link_mass_range = [0.9, 1.1]
        randomize_friction = True
        added_friction_range = [0.4, 1.2]
        randomize_restitution = False
        restitution_range = [0.0, 0.3]
        randomize_com = True
        added_com_range = [-0.03, 0.03]
        randomize_motor_strength = True
        added_motor_strength = [0.9, 1.1]
        randomize_lag_timesteps = False
        added_lag_timesteps = 3
        added_lag_timesteps_sacle = [0, 2]
        randomize_motor_offset = False
        added_motor_offset = [-0.01, 0.01]
        dof_init_pos_ratio_range = [0.95, 1.05]
        randomize_dof_vel = False
        dof_init_vel_range = [-0.1, 0.1]
        randomize_action_latency = False
        latency_range = [0.00, 0.02]
        push_robots = False
        push_interval_s = 16
        max_push_vel_xy = 0.5

    class rewards(LeggedRobotBaseCfg.rewards):
        terrain_adaptive_reward = False
        base_height_target = 0.50
        max_contact_force = 300.0
        soft_dof_pos_limit = 0.95
        soft_dof_vel_limit = 0.95
        soft_torque_limit = 0.95
        gait_threshold = [0.0, 1.2]
        lin_vel_clip = 0.1
        only_positive_rewards = True
        foot_height_target = 0.10
        max_base_height_error = 0.5
        max_lin_vel_z_penalty = 3.0

        class scales(LeggedRobotBaseCfg.rewards.scales):
            alive = 0.5
            tracking_lin_vel = 1.2
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.10
            torques = -1e-5
            dof_acc = -2.5e-7
            action_rate = -0.015
            orientation = -0.6
            collision = -1.5
            motion_trot = -0.05
            feet_air_time = 0.7
            feet_stumble = -0.5
            stand_still = -0.1
            base_height = -0.5

    class normalization(LeggedRobotBaseCfg.normalization):
        class obs_scales(LeggedRobotBaseCfg.normalization.obs_scales):
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
        clip_observations = 100.0
        clip_actions = 100.0

    class noise(LeggedRobotBaseCfg.noise):
        add_noise = True
        add_privileged_noise = False
        add_height_noise = True
        noise_level = 1.0

        class noise_scales(LeggedRobotBaseCfg.noise.noise_scales):
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1

    class sim(LeggedRobotBaseCfg.sim):
        dt = 0.001
        substeps = 1


class X30MGDPCfgPPOStage1(LeggedRobotBaseCfgPPO):
    class policy(LeggedRobotBaseCfgPPO.policy):
        init_noise_std = 0.3
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu'

    class runner(LeggedRobotBaseCfgPPO.runner):
        run_name = ''
        max_iterations = 30000
        resume = False
        save_interval = 1000
        experiment_name = 'x30_mgdp_stage1'

    class Encoder(LeggedRobotBaseCfgPPO.Encoder):
        checkpoint_model = None
        camera_dim = X30MGDPCfgStage1.camera.resized
        HistoryLen = 4
        output_dims = 32
        encoder_output = 7
        pool = 1
        MLPModule_info = {
            'input_dims': 45 * HistoryLen,
            'hidden_dims': [256, 128, encoder_output],
        }
        CNNModule_info = {
            'input_channels': 2,
            'hidden_channels': [32, 64, 64],
            'output_channels': 64,
            'pool': 2,
        }
        MapModule_info = {
            'input_channels': 1,
            'hidden_channels': [16, 32, 64],
            'output_channels': 32,
            'pool': 2,
        }
        GRUModule_info = {
            'input_dims': output_dims * 2,
            'rnn_type': 'gru',
            'rnn_num_layers': 1,
            'rnn_hidden_dims': output_dims * 2,
        }
        height_encoder_type = "CNN"
        cnn_mlp_units = [16]
        DecoderModule_info = [64, 128, 256]


X30MGDPCfgStage1.encoder_ppo_ref = X30MGDPCfgPPOStage1
