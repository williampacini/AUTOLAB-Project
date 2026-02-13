#!/usr/bin/env python3
"""Collect scripted expert demonstrations from robosuite environments.

Runs scripted policies that use ground-truth object positions to generate
successful demonstrations and saves them as HDF5 files matching the format
expected by ``convert_to_lerobot.py``.

Usage:
    # Collect all environments with default demo counts
    python data/collect_demos.py --all

    # Single environment with custom count
    python data/collect_demos.py --env Lift --n-demos 50

    # NutAssembly with more demos (recommended)
    python data/collect_demos.py --env NutAssembly --n-demos 75

    # Quick test (2 demos)
    python data/collect_demos.py --env Lift --n-demos 2
"""

import argparse
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np

# Headless rendering — must be set before robosuite import
os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

import robosuite as suite

try:
    # robosuite >= 1.5: composite controller architecture
    from robosuite.controllers import load_composite_controller_config

    _USE_COMPOSITE = True
except ImportError:
    # robosuite < 1.5: simple controller config
    from robosuite.controllers import load_controller_config

    _USE_COMPOSITE = False

from scripted_policies import get_scripted_policy, reset_policy

ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Environment configs
# ---------------------------------------------------------------------------

ENV_CONFIGS = {
    "Lift": {
        "max_steps": 200,
        "n_demos_default": 50,
        "task_description": "pick up the red cube and lift it",
    },
    "Stack": {
        "max_steps": 400,
        "n_demos_default": 50,
        "task_description": "pick up the red cube and stack it on the green cube",
    },
    "PickPlaceSingle": {
        "max_steps": 400,
        "n_demos_default": 50,
        "task_description": "pick up the object and place it in the bin",
    },
    "Door": {
        "max_steps": 300,
        "n_demos_default": 50,
        "task_description": "open the door by turning the handle",
    },
    "NutAssemblySingle": {
        "max_steps": 400,
        "n_demos_default": 50,
        "task_description": "pick up the nut and place it on the peg",
    },
    "NutAssembly": {
        "max_steps": 600,
        "n_demos_default": 75,
        "task_description": "pick up each nut and place it on the correct peg",
    },
}


def make_env(env_name, camera_res=128):
    """Create a robosuite environment configured for demo collection.

    Key settings:
      - ``use_object_obs=True``: exposes ground-truth object positions
      - Two cameras: agentview + robot0_eye_in_hand (for SmolVLA)
      - OSC_POSE controller → 7D action space
    """
    if _USE_COMPOSITE:
        controller_config = load_composite_controller_config(controller="BASIC")
    else:
        controller_config = load_controller_config(default_controller="OSC_POSE")

    env = suite.make(
        env_name=env_name,
        robots="Panda",
        controller_configs=controller_config,
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        use_object_obs=True,
        camera_names=["agentview", "robot0_eye_in_hand"],
        camera_heights=camera_res,
        camera_widths=camera_res,
        reward_shaping=True,
        control_freq=20,
    )
    return env


def collect_demos(env_name, n_demos, output_dir, camera_res=128, max_attempts=5):
    """Collect *n_demos* successful demonstrations for *env_name*.

    Only saves episodes where the task reward indicates success.
    Retries up to *max_attempts* per demo slot on failure.

    Returns:
        Path to saved HDF5 file.
    """
    config = ENV_CONFIGS[env_name]
    max_steps = config["max_steps"]
    task_desc = config["task_description"]
    policy_fn = get_scripted_policy(env_name)

    env = make_env(env_name, camera_res=camera_res)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    hdf5_path = output_dir / f"{env_name}.hdf5"

    print(f"\n{'='*60}")
    print(f"  Collecting {n_demos} demos for {env_name}")
    print(f"  Task: {task_desc}")
    print(f"  Max steps/ep: {max_steps}  |  Max attempts: {max_attempts}")
    print(f"  Output: {hdf5_path}")
    print(f"{'='*60}")

    t_start = time.time()

    with h5py.File(str(hdf5_path), "w") as f:
        f.attrs["env_name"] = env_name
        f.attrs["task_description"] = task_desc
        f.attrs["n_demos"] = n_demos
        f.attrs["camera_res"] = camera_res
        f.attrs["control_freq"] = 20
        f.attrs["action_dim"] = 7
        data_grp = f.create_group("data")

        demo_idx = 0
        total_attempts = 0

        while demo_idx < n_demos:
            attempt = 0
            saved = False

            while attempt < max_attempts and not saved:
                total_attempts += 1
                seed = 1000 * demo_idx + attempt
                np.random.seed(seed)

                # Reset stateful policies
                reset_policy(env_name)
                obs = env.reset()

                # Storage for this episode
                ep_actions = []
                ep_agentview = []
                ep_wrist = []
                ep_eef_pos = []
                ep_eef_quat = []
                ep_gripper_qpos = []
                ep_joint_pos = []

                success = False
                total_reward = 0.0
                for step in range(max_steps):
                    action = policy_fn(obs, env)
                    action = np.clip(action, -1, 1)

                    # Record observation-action pair (obs at time t, action at time t)
                    ep_actions.append(action.copy())
                    ep_agentview.append(obs["agentview_image"].copy())
                    ep_wrist.append(obs["robot0_eye_in_hand_image"].copy())
                    ep_eef_pos.append(obs["robot0_eef_pos"].copy())
                    ep_eef_quat.append(obs["robot0_eef_quat"].copy())
                    ep_gripper_qpos.append(obs["robot0_gripper_qpos"].copy())
                    ep_joint_pos.append(obs["robot0_joint_pos"].copy())

                    obs, reward, done, info = env.step(action)
                    total_reward += reward

                    # Check success via environment's internal check
                    try:
                        task_success = env._check_success()
                    except AttributeError:
                        # Shaped rewards accumulate; a single-step threshold
                        # gives false positives. Use cumulative reward instead.
                        task_success = bool(total_reward > 1.0)

                    if task_success:
                        success = True
                        break

                if success:
                    # Save successful episode
                    demo_grp = data_grp.create_group(f"demo_{demo_idx}")
                    demo_grp.create_dataset(
                        "actions", data=np.array(ep_actions, dtype=np.float32)
                    )
                    obs_grp = demo_grp.create_group("obs")
                    obs_grp.create_dataset(
                        "agentview_image",
                        data=np.array(ep_agentview, dtype=np.uint8),
                    )
                    obs_grp.create_dataset(
                        "robot0_eye_in_hand_image",
                        data=np.array(ep_wrist, dtype=np.uint8),
                    )
                    obs_grp.create_dataset(
                        "robot0_eef_pos", data=np.array(ep_eef_pos, dtype=np.float32)
                    )
                    obs_grp.create_dataset(
                        "robot0_eef_quat",
                        data=np.array(ep_eef_quat, dtype=np.float32),
                    )
                    obs_grp.create_dataset(
                        "robot0_gripper_qpos",
                        data=np.array(ep_gripper_qpos, dtype=np.float32),
                    )
                    obs_grp.create_dataset(
                        "robot0_joint_pos",
                        data=np.array(ep_joint_pos, dtype=np.float32),
                    )

                    elapsed = time.time() - t_start
                    print(
                        f"  [{env_name}] Demo {demo_idx + 1}/{n_demos} "
                        f"saved ({step + 1} steps, attempt {attempt + 1}) "
                        f"[{elapsed:.0f}s elapsed]"
                    )
                    demo_idx += 1
                    saved = True
                else:
                    attempt += 1
                    if attempt >= max_attempts:
                        print(
                            f"  [{env_name}] WARNING: Failed {max_attempts} attempts "
                            f"for demo slot {demo_idx}, advancing anyway"
                        )
                        demo_idx += 1  # skip this slot to avoid infinite loop
                        break

        # Update actual count in case some were skipped
        f.attrs["n_demos_actual"] = demo_idx

    elapsed = time.time() - t_start
    success_rate = demo_idx / max(total_attempts, 1) * 100
    print(f"\n  Done: {demo_idx} demos in {elapsed:.1f}s "
          f"({success_rate:.0f}% success rate)")
    print(f"  Saved: {hdf5_path}")

    env.close()
    return hdf5_path


def main():
    parser = argparse.ArgumentParser(
        description="Collect scripted expert demonstrations from robosuite"
    )
    parser.add_argument(
        "--env", type=str, default=None,
        choices=list(ENV_CONFIGS.keys()),
        help="Single environment to collect demos for",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Collect demos for all environments",
    )
    parser.add_argument(
        "--n-demos", type=int, default=None,
        help="Number of demos to collect (overrides env default)",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=str(ROOT / "data" / "robosuite_demos"),
        help="Output directory for HDF5 files",
    )
    parser.add_argument(
        "--camera-res", type=int, default=128,
        help="Camera resolution (default: 128)",
    )
    parser.add_argument(
        "--max-attempts", type=int, default=5,
        help="Max retry attempts per demo (default: 5)",
    )
    args = parser.parse_args()

    if not args.all and not args.env:
        parser.error("Specify --env ENV_NAME or --all")

    envs = list(ENV_CONFIGS.keys()) if args.all else [args.env]

    print(f"robosuite {suite.__version__}")
    print(f"Environments: {envs}")
    print(f"Output: {args.output_dir}")

    all_paths = []
    for env_name in envs:
        n = args.n_demos or ENV_CONFIGS[env_name]["n_demos_default"]
        path = collect_demos(
            env_name, n,
            output_dir=args.output_dir,
            camera_res=args.camera_res,
            max_attempts=args.max_attempts,
        )
        all_paths.append(path)

    print(f"\n{'='*60}")
    print(f"  All done! {len(all_paths)} HDF5 files saved:")
    for p in all_paths:
        print(f"    {p}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
