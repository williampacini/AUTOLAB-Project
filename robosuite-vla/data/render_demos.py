#!/usr/bin/env python3
"""Render image observations from raw robomimic state-only HDF5 files.

Replays MuJoCo states through robosuite to capture camera images and
proprioceptive observations. Produces HDF5 files compatible with
convert_to_lerobot.py.

Usage:
    python data/render_demos.py --input data/robomimic_nut_assembly/.../demo_v15.hdf5
    python data/render_demos.py --input demo_v15.hdf5 --camera-size 128 --max-demos 5
"""

import argparse
import json
import os
from pathlib import Path

# Headless rendering — must be before any robosuite import
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import h5py
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent


def parse_env_args(hdf5_path):
    """Parse environment args from HDF5 file-level attributes.

    Returns:
        dict with 'env_name' and optionally 'env_kwargs', or None
    """
    with h5py.File(str(hdf5_path), "r") as f:
        for group in [f.get("data"), f]:
            if group is not None and "env_args" in group.attrs:
                env_args_str = group.attrs["env_args"]
                if isinstance(env_args_str, bytes):
                    env_args_str = env_args_str.decode()
                return json.loads(env_args_str)
    return None


def infer_env_name(hdf5_path, env_args=None):
    """Determine the robosuite environment name.

    Tries env_args first, then falls back to path-based heuristics.
    """
    if env_args and "env_name" in env_args:
        return env_args["env_name"]

    path_str = str(hdf5_path).lower()
    if "square" in path_str:
        return "NutAssemblySquare"
    if "round" in path_str:
        return "NutAssemblyRound"
    if "nut" in path_str:
        return "NutAssembly"
    if "lift" in path_str:
        return "Lift"
    if "can" in path_str:
        return "PickPlaceCan"

    return None


def create_env_for_rendering(env_name, camera_names, camera_size, env_kwargs=None):
    """Create a robosuite env configured for state replay and rendering.

    Only adopts safe kwargs from env_args that affect physics/task,
    not rendering setup.
    """
    import robosuite as suite

    kwargs = dict(
        env_name=env_name,
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        use_object_obs=True,
        camera_names=camera_names,
        camera_heights=camera_size,
        camera_widths=camera_size,
        reward_shaping=True,
        horizon=2000,
    )

    if env_kwargs:
        safe_keys = [
            "controller_configs", "single_object_mode",
            "nut_type", "control_freq",
        ]
        for k in safe_keys:
            if k in env_kwargs:
                kwargs[k] = env_kwargs[k]

    env = suite.make(**kwargs)
    return env


def set_sim_state(env, state_vec):
    """Set simulator state, handling different MuJoCo bindings.

    Works with both mujoco-py (set_state_from_flattened) and native mujoco.
    """
    if hasattr(env.sim, "set_state_from_flattened"):
        env.sim.set_state_from_flattened(state_vec)
    else:
        nq = env.sim.model.nq
        nv = env.sim.model.nv
        env.sim.data.qpos[:] = state_vec[:nq]
        env.sim.data.qvel[:] = state_vec[nq:nq + nv]
        if len(state_vec) > nq + nv:
            na = env.sim.model.na
            if na > 0:
                env.sim.data.act[:] = state_vec[nq + nv:nq + nv + na]
    env.sim.forward()


def validate_state_dim(env, state_dim):
    """Check that state vector dimension matches the environment.

    Returns (expected_dim, message) tuple.
    """
    nq = env.sim.model.nq
    nv = env.sim.model.nv
    na = getattr(env.sim.model, "na", 0)
    expected = nq + nv + (na if na > 0 else 0)
    expected_no_act = nq + nv

    if state_dim == expected or state_dim == expected_no_act:
        return True, f"OK (nq={nq}, nv={nv}, na={na})"

    return False, (
        f"State dim mismatch: HDF5 has {state_dim}, "
        f"env expects {expected} (nq={nq}, nv={nv}, na={na})"
    )


def replay_demo_states(env, states, camera_names):
    """Replay MuJoCo states and capture observations at each timestep.

    Args:
        env: robosuite environment
        states: (T+1, D) flattened MuJoCo states (states[0] is initial)
        camera_names: list of camera names

    Returns:
        dict with observation arrays: {cam}_image, robot0_eef_pos, etc.
    """
    n_obs = len(states) - 1  # T observations to match T actions

    obs_dict = {f"{cam}_image": [] for cam in camera_names}
    obs_dict.update({
        "robot0_eef_pos": [],
        "robot0_eef_quat": [],
        "robot0_gripper_qpos": [],
        "robot0_joint_pos": [],
    })

    for t in range(n_obs):
        set_sim_state(env, states[t])

        # Get observations from environment
        obs = env._get_observations(force_update=True)

        for cam in camera_names:
            img_key = f"{cam}_image"
            if img_key in obs:
                obs_dict[img_key].append(obs[img_key].copy())
            else:
                available = [k for k in obs if "image" in k]
                raise KeyError(
                    f"Camera '{img_key}' not in observations. Available: {available}"
                )

        for key in ["robot0_eef_pos", "robot0_eef_quat",
                     "robot0_gripper_qpos", "robot0_joint_pos"]:
            if key in obs:
                obs_dict[key].append(obs[key].copy())

    result = {}
    for key, frames in obs_dict.items():
        if frames:
            result[key] = np.stack(frames)

    return result


def render_hdf5(input_path, output_path, camera_names, camera_size,
                max_demos=None, env_name_override=None, compress=True):
    """Render image observations for all demos in an HDF5 file.

    Reads states from input HDF5, replays through robosuite, captures
    images and proprioception, writes to output HDF5.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    env_args = parse_env_args(input_path)

    if env_name_override:
        env_name = env_name_override
    else:
        env_name = infer_env_name(input_path, env_args)
        if env_name is None:
            raise ValueError(
                "Could not determine environment name. "
                "Provide --env-name or ensure HDF5 has env_args."
            )

    env_kwargs = env_args.get("env_kwargs", {}) if env_args else {}

    print(f"  Environment: {env_name}")
    print(f"  Cameras: {camera_names}")
    print(f"  Resolution: {camera_size}x{camera_size}")
    print(f"  Compress: {compress}")

    env = create_env_for_rendering(env_name, camera_names, camera_size, env_kwargs)
    env.reset()

    state_validated = False

    with h5py.File(str(input_path), "r") as f_in:
        demo_keys = sorted(
            [k for k in f_in["data"].keys() if k.startswith("demo")],
            key=lambda x: int(x.split("_")[1]),
        )
        if max_demos is not None:
            demo_keys = demo_keys[:max_demos]

        print(f"  Rendering {len(demo_keys)} demos...")

        with h5py.File(str(output_path), "w") as f_out:
            data_grp = f_out.create_group("data")

            # Copy file-level attributes
            for attr_name in f_in["data"].attrs:
                data_grp.attrs[attr_name] = f_in["data"].attrs[attr_name]

            data_grp.attrs["rendered_camera_names"] = json.dumps(camera_names)
            data_grp.attrs["rendered_camera_size"] = camera_size

            failed_demos = []

            for demo_key in tqdm(demo_keys, desc="  Rendering"):
                demo_in = f_in["data"][demo_key]

                if "states" not in demo_in:
                    print(f"  WARNING: {demo_key} has no 'states', skipping")
                    failed_demos.append(demo_key)
                    continue

                states = demo_in["states"][:]
                actions = demo_in["actions"][:]

                # Validate state dim once
                if not state_validated:
                    ok, msg = validate_state_dim(env, states.shape[1])
                    print(f"  State validation: {msg}")
                    if not ok:
                        print("  ERROR: Cannot replay — state dimensions don't match.")
                        env.close()
                        return
                    state_validated = True

                # Reset env to establish scene geometry
                env.reset()

                try:
                    obs = replay_demo_states(env, states, camera_names)
                except Exception as e:
                    print(f"  WARNING: Failed to render {demo_key}: {e}")
                    failed_demos.append(demo_key)
                    continue

                # Validate obs count == action count
                n_actions = len(actions)
                for key, arr in obs.items():
                    if len(arr) != n_actions:
                        obs[key] = arr[:n_actions]

                # Write output
                demo_grp = data_grp.create_group(demo_key)
                demo_grp.create_dataset("actions", data=actions)

                if "rewards" in demo_in:
                    demo_grp.create_dataset("rewards", data=demo_in["rewards"][:])
                if "dones" in demo_in:
                    demo_grp.create_dataset("dones", data=demo_in["dones"][:])

                for attr_name in demo_in.attrs:
                    demo_grp.attrs[attr_name] = demo_in.attrs[attr_name]

                obs_grp = demo_grp.create_group("obs")
                for key, arr in obs.items():
                    if "image" in key:
                        obs_grp.create_dataset(
                            key, data=arr,
                            compression="gzip" if compress else None,
                            compression_opts=4 if compress else None,
                            chunks=(1,) + arr.shape[1:],
                        )
                    else:
                        obs_grp.create_dataset(key, data=arr)

            n_success = len(demo_keys) - len(failed_demos)
            data_grp.attrs["n_demos"] = n_success

    env.close()

    print(f"\n  Rendered HDF5 saved to: {output_path}")
    print(f"  Successful: {n_success}/{len(demo_keys)}")
    if failed_demos:
        print(f"  Failed demos: {failed_demos[:10]}{'...' if len(failed_demos) > 10 else ''}")


def verify_rendered_hdf5(hdf5_path, camera_names):
    """Verify that a rendered HDF5 file has the expected structure."""
    hdf5_path = Path(hdf5_path)
    if not hdf5_path.exists():
        print(f"  File not found: {hdf5_path}")
        return False

    with h5py.File(str(hdf5_path), "r") as f:
        if "data" not in f:
            print("  FAIL: No 'data' group")
            return False

        demos = sorted([k for k in f["data"].keys() if k.startswith("demo")])
        if not demos:
            print("  FAIL: No demos")
            return False

        demo = f["data"][demos[0]]
        if "obs" not in demo:
            print(f"  FAIL: No 'obs' group in {demos[0]}")
            return False

        obs_keys = sorted(demo["obs"].keys())
        n_actions = len(demo["actions"])
        print(f"  Demos: {len(demos)}")
        print(f"  Obs keys: {obs_keys}")
        print(f"  Actions in first demo: {n_actions}")

        for cam in camera_names:
            img_key = f"{cam}_image"
            if img_key not in demo["obs"]:
                print(f"  FAIL: Missing {img_key}")
                return False
            shape = demo["obs"][img_key].shape
            print(f"  {img_key}: {shape}")

        print("  OK")
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Render image observations from raw robomimic HDF5"
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Input HDF5 path (raw states, no images)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output HDF5 path (default: {input}_rendered.hdf5)")
    parser.add_argument("--camera-size", type=int, default=256,
                        help="Camera resolution (square, default: 256)")
    parser.add_argument("--max-demos", type=int, default=None,
                        help="Max demos to render (for testing)")
    parser.add_argument("--env-name", type=str, default=None,
                        help="Override environment name from HDF5")
    parser.add_argument("--camera-names", type=str, nargs="+",
                        default=["agentview", "robot0_eye_in_hand"],
                        help="Camera names to render")
    parser.add_argument("--no-compress", action="store_true",
                        help="Disable gzip compression for images")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip if output file already exists")
    parser.add_argument("--verify-only", action="store_true",
                        help="Only verify an existing rendered HDF5")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        return

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / f"{input_path.stem}_rendered.hdf5"

    if args.verify_only:
        print("Verifying rendered HDF5...")
        verify_rendered_hdf5(output_path, args.camera_names)
        return

    if args.skip_existing and output_path.exists():
        print(f"Output already exists, skipping: {output_path}")
        return

    print("=" * 60)
    print("  Render Image Observations from Raw Demo States")
    print("=" * 60)
    print(f"  Input:  {input_path}")
    print(f"  Output: {output_path}")
    print()

    render_hdf5(
        input_path=input_path,
        output_path=output_path,
        camera_names=args.camera_names,
        camera_size=args.camera_size,
        max_demos=args.max_demos,
        env_name_override=args.env_name,
        compress=not args.no_compress,
    )

    print("\nVerifying output...")
    verify_rendered_hdf5(output_path, args.camera_names)


if __name__ == "__main__":
    main()
