#!/usr/bin/env python3
import os
import sys
from pathlib import Path

LEGGED_GYM_ROOT_DIR = str(Path(__file__).resolve().parent.parent.parent)
RSL_RL_ROOT_DIR = str(Path(__file__).resolve().parent.parent.parent.parent / "rsl_rl")
LidarSensor_DIR = str(Path(__file__).resolve().parent.parent.parent.parent / "LidarSensor")
ISAAC_GYM_DIR = str(Path(__file__).resolve().parent.parent.parent.parent / "isaacgym" / "python")
sys.path.append(LEGGED_GYM_ROOT_DIR)
sys.path.append(RSL_RL_ROOT_DIR)
sys.path.append(LidarSensor_DIR)
sys.path.append(ISAAC_GYM_DIR)

import isaacgym  # noqa: F401
import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import export_policy_as_jit, get_args, task_registry
from legged_gym.utils.helpers import get_load_path


def export_jit_to_onnx(jit_path, onnx_path, obs_dim=270, opset=14):
    print(f"[INFO] Loading TorchScript: {jit_path}")
    model = torch.jit.load(jit_path, map_location="cpu")
    model.eval()

    dummy = torch.zeros(1, obs_dim, dtype=torch.float32)
    with torch.no_grad():
        action = model(dummy)
        print(f"[INFO] Test forward ok. output shape: {tuple(action.shape)}")

    os.makedirs(os.path.dirname(os.path.abspath(onnx_path)), exist_ok=True)
    print(f"[INFO] Exporting ONNX: {onnx_path}")
    torch.onnx.export(
        model,
        (dummy,),
        onnx_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["obs"],
        output_names=["action"],
        dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
    )
    print("[INFO] ONNX export done.")


def export_from_task(args):
    args.headless = True
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 25)
    env_cfg.terrain.num_rows = min(env_cfg.terrain.num_rows, 5)
    env_cfg.terrain.num_cols = min(env_cfg.terrain.num_cols, 5)
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.max_init_terrain_level = 0
    env_cfg.commands.curriculum = False
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

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    train_cfg.runner.resume = True

    log_root = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", train_cfg.runner.experiment_name)
    ppo_runner, train_cfg = task_registry.make_alg_runner(
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
    print(f"[INFO] Exporting checkpoint: {loaded_model_path}")

    export_dir = os.path.join(os.path.dirname(loaded_model_path), "exported")
    export_policy_as_jit(ppo_runner.alg.actor_critic, export_dir)
    jit_path = os.path.join(export_dir, "policy.pt")

    obs_dim = args.obs_dim if args.obs_dim is not None else env_cfg.env.num_observations
    onnx_path = args.onnx if args.onnx else os.path.join(export_dir, "policy.onnx")
    export_jit_to_onnx(jit_path, onnx_path, obs_dim=obs_dim, opset=args.opset)
    print(f"[INFO] TorchScript: {jit_path}")
    print(f"[INFO] ONNX: {onnx_path}")


def main():
    args = get_args(
        [
            dict(name="--jit", type=str, default="", help="input TorchScript .pt; if set, only convert jit to onnx"),
            dict(name="--onnx", type=str, default="", help="output ONNX path; default: <run>/exported/policy.onnx"),
            dict(name="--obs-dim", type=int, default=None, help="ONNX input observation dim; default: task cfg num_observations"),
            dict(name="--opset", type=int, default=14, help="ONNX opset version"),
        ]
    )

    if args.jit:
        onnx_path = args.onnx if args.onnx else "policy.onnx"
        obs_dim = args.obs_dim if args.obs_dim is not None else 270
        export_jit_to_onnx(args.jit, onnx_path, obs_dim=obs_dim, opset=args.opset)
    else:
        export_from_task(args)


if __name__ == "__main__":
    main()
