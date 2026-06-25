#!/usr/bin/env python3
import copy
import json
import os
import sys
from collections import OrderedDict
from pathlib import Path

LEGGED_GYM_ROOT_DIR = str(Path(__file__).resolve().parent.parent.parent)
RSL_RL_ROOT_DIR = str(Path(__file__).resolve().parent.parent.parent.parent / "rsl_rl")
LIDAR_SENSOR_DIR = str(Path(__file__).resolve().parent.parent.parent.parent / "LidarSensor")
ISAAC_GYM_DIR = str(Path(__file__).resolve().parent.parent.parent.parent / "isaacgym" / "python")
sys.path.append(LEGGED_GYM_ROOT_DIR)
sys.path.append(RSL_RL_ROOT_DIR)
sys.path.append(LIDAR_SENSOR_DIR)
sys.path.append(ISAAC_GYM_DIR)

import isaacgym  # noqa: F401
import torch
import torch.nn.functional as F

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import get_args, task_registry
from legged_gym.utils.helpers import get_load_path, update_class_from_dict


class HIMPolicyExporter(torch.nn.Module):
    def __init__(self, actor_critic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor).cpu()
        self.estimator = copy.deepcopy(actor_critic.estimator.encoder).cpu()
        self.num_one_step_obs = int(actor_critic.num_one_step_obs)

    def forward(self, obs_history):
        parts = self.estimator(obs_history)[:, 0:19]
        vel, latent = parts[..., :3], parts[..., 3:]
        latent = F.normalize(latent, dim=-1, p=2.0)
        actor_obs = torch.cat((obs_history[:, : self.num_one_step_obs], vel, latent), dim=-1)
        return self.actor(actor_obs)


class ActorPolicyExporter(torch.nn.Module):
    def __init__(self, actor_critic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor).cpu()

    def forward(self, obs):
        return self.actor(obs)


def build_policy_exporter(actor_critic):
    if hasattr(actor_critic, "estimator") and hasattr(actor_critic, "num_one_step_obs"):
        return HIMPolicyExporter(actor_critic)
    return ActorPolicyExporter(actor_critic)


def load_logged_config_if_requested(args, env_cfg, train_cfg):
    if not args.load_cfg:
        return

    log_root = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", train_cfg.runner.experiment_name)
    if args.load_run is None or args.load_run == "-1":
        run_dir = os.path.dirname(get_load_path(log_root, load_run=-1, checkpoint=-1))
    elif os.path.isabs(args.load_run):
        run_dir = args.load_run
    else:
        run_dir = os.path.join(log_root, args.load_run)

    config_path = os.path.join(run_dir, "config.json")
    print(f"[INFO] Loading config from: {config_path}")
    with open(config_path, "r") as f:
        logged_cfg = json.load(f, object_pairs_hook=OrderedDict)
    update_class_from_dict(env_cfg, logged_cfg, strict=True)
    update_class_from_dict(train_cfg, logged_cfg, strict=True)


def prepare_export_env_cfg(env_cfg):
    env_cfg.env.num_envs = 1
    env_cfg.terrain.num_rows = min(env_cfg.terrain.num_rows, 1)
    env_cfg.terrain.num_cols = min(env_cfg.terrain.num_cols, 1)
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.max_init_terrain_level = 0
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.disturbance = False
    env_cfg.domain_rand.randomize_payload_mass = False
    env_cfg.domain_rand.randomize_com_displacement = False
    env_cfg.domain_rand.randomize_link_mass = False
    env_cfg.domain_rand.randomize_kp = False
    env_cfg.domain_rand.randomize_kd = False
    env_cfg.domain_rand.randomize_motor_strength = False


def export_policy_pt(args):
    args.headless = True
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    load_logged_config_if_requested(args, env_cfg, train_cfg)
    prepare_export_env_cfg(env_cfg)

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    train_cfg.runner.resume = True
    log_root = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", train_cfg.runner.experiment_name)
    runner, train_cfg = task_registry.make_alg_runner(
        env=env,
        name=args.task,
        args=args,
        train_cfg=train_cfg,
        save_cfg=False,
    )

    loaded_model_path = get_load_path(
        log_root,
        load_run=train_cfg.runner.load_run,
        checkpoint=train_cfg.runner.checkpoint,
    )
    export_dir = os.path.join(os.path.dirname(loaded_model_path), "exported")
    output_path = args.output if args.output else os.path.join(export_dir, "policy.pt")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    exporter = build_policy_exporter(runner.alg.actor_critic).eval().cpu()
    scripted = torch.jit.script(exporter)
    scripted.save(output_path)

    obs_dim = env_cfg.env.num_observations
    dummy_obs = torch.zeros(1, obs_dim, dtype=torch.float32)
    with torch.no_grad():
        action = scripted(dummy_obs)

    print(f"[INFO] Loaded checkpoint: {loaded_model_path}")
    print(f"[INFO] Exported TorchScript policy: {output_path}")
    print(f"[INFO] Test forward ok: obs {tuple(dummy_obs.shape)} -> action {tuple(action.shape)}")


def main():
    args = get_args(
        [
            dict(name="--output", type=str, default="", help="output .pt path; default: <run>/exported/policy.pt"),
            dict(name="--load_cfg", action="store_true", default=False, help="load env/train cfg from <run>/config.json"),
        ]
    )
    export_policy_pt(args)


if __name__ == "__main__":
    main()
