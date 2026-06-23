# SPDX-License-Identifier: BSD-3-Clause
import glob

from legged_gym.envs.base.legged_robot_config import MOTION_FILES_DIR
from legged_gym.envs.sr02.sr02_stairs_config import Sr02StairsCfg, Sr02StairsCfgPPO


MOTION_FILES = []
MOTION_FILES.extend(glob.glob(str(MOTION_FILES_DIR / 'mocap_motions_aliengo/left*.txt')))
MOTION_FILES.extend(glob.glob(str(MOTION_FILES_DIR / 'mocap_motions_aliengo/right*.txt')))
MOTION_FILES.extend(glob.glob(str(MOTION_FILES_DIR / 'mocap_motions_aliengo/trot*.txt')))


class Sr02StairsAmpCfg(Sr02StairsCfg):
    class env(Sr02StairsCfg.env):
        using_amp = True

    class asset(Sr02StairsCfg.asset):
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        privileged_contacts_on = ["base", "thigh", "calf", "foot"]


class Sr02StairsAmpCfgPPO(Sr02StairsCfgPPO):
    seed = 1
    runner_class_name = 'HybridPolicyRunner'

    class algorithm(Sr02StairsCfgPPO.algorithm):
        amp_replay_buffer_size = 1000000

    class runner(Sr02StairsCfgPPO.runner):
        policy_class_name = 'HIMActorCritic'
        algorithm_class_name = 'HybridPPO'
        max_iterations = 5000
        experiment_name = 'stairs_amp_sr02'
        resume = False
        load_run = -1
        checkpoint = -1

        amp_reward_coef = 0.02
        amp_motion_files = MOTION_FILES
        amp_num_preload_transitions = 2000000
        amp_task_reward_lerp = 0.5
        amp_discr_hidden_dims = [1024, 512]

        min_normalized_std = [0.05, 0.02, 0.05] * 4
