#!/usr/bin/env python3
"""Verify the success rate of the scripted Lift policy.

This script runs N episodes of robosuite's Lift task using the simple
proportional controller from robosuite_sim.ipynb and reports the actual
success rate.  It also saves successful episodes to HDF5 for downstream use.

Usage (on Colab or any machine with robosuite installed):
    python data/verify_lift_success_rate.py --episodes 200 --max-steps 300

    # Also save successful demos to HDF5:
    python data/verify_lift_success_rate.py --episodes 200 --output data/robosuite_demos/lift.hdf5

    # Compare original vs improved policy:
    python data/verify_lift_success_rate.py --episodes 200 --compare
"""

import argparse
import os
import sys

# Headless rendering — must be set BEFORE importing mujoco/robosuite
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")

import numpy as np

try:
    import robosuite as suite
except ImportError:
    print("ERROR: robosuite is not installed.")
    print("Install with: pip install robosuite")
    sys.exit(1)

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------

def scripted_lift_policy(obs):
    """Simple proportional controller from robosuite_sim.ipynb.

    Action space (7D for Panda with OSC_POSE controller):
      [dx, dy, dz, dax, day, daz, gripper]
      gripper: -1 = close, +1 = open
    """
    ee_pos = obs["robot0_eef_pos"]   # (3,) end-effector position
    cube_pos = obs["cube_pos"]       # (3,) cube position
    delta = cube_pos - ee_pos
    dist = np.linalg.norm(delta)

    action = np.zeros(7)
    if dist > 0.02:
        # Phase 1: Move toward the cube
        action[:3] = delta * 10.0  # proportional control
        action[6] = 1.0            # keep gripper open
    else:
        # Phase 2: Close gripper and lift
        action[2] = 1.0            # move up
        action[6] = -1.0           # close gripper

    return np.clip(action, -1, 1)


def improved_lift_policy(obs):
    """Improved scripted policy with pre-grasp hovering.

    Fixes key failure modes of the original:
    1. Approaches from above (hover first, then descend)
    2. Pauses to close gripper before lifting
    3. Uses a lower proportional gain to avoid overshooting
    """
    ee_pos = obs["robot0_eef_pos"]
    cube_pos = obs["cube_pos"]

    # Pre-grasp position: above the cube
    hover_height = 0.06  # hover 6cm above cube
    pre_grasp = cube_pos.copy()
    pre_grasp[2] += hover_height

    delta_hover = pre_grasp - ee_pos

    delta_cube = cube_pos - ee_pos
    dist_xy = np.linalg.norm(delta_cube[:2])  # horizontal distance
    dist_z = ee_pos[2] - cube_pos[2]          # height above cube

    action = np.zeros(7)

    if dist_xy > 0.02 or dist_z > hover_height + 0.02:
        # Phase 1: Move to hover position above cube
        action[:3] = delta_hover * 5.0
        action[6] = 1.0  # open gripper
    elif dist_z > 0.015:
        # Phase 2: Descend to cube
        action[2] = -0.5
        action[6] = 1.0  # keep open
    elif obs.get("robot0_gripper_qpos", np.array([0.04, -0.04]))[0] > 0.01:
        # Phase 3: Close gripper (pause to grip)
        action[6] = -1.0
    else:
        # Phase 4: Lift
        action[2] = 1.0
        action[6] = -1.0

    return np.clip(action, -1, 1)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def run_evaluation(policy_fn, policy_name, n_episodes, max_steps, seed_start=0,
                   hdf5_path=None):
    """Run N episodes and return success statistics.

    If hdf5_path is provided, successful episodes are saved to HDF5.
    """
    env = suite.make(
        env_name="Lift",
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        use_object_obs=True,
        camera_names="agentview",
        camera_heights=128,
        camera_widths=128,
        reward_shaping=True,
    )

    # Prepare HDF5 writer
    h5_file = None
    if hdf5_path and HAS_H5PY:
        os.makedirs(os.path.dirname(hdf5_path) or ".", exist_ok=True)
        h5_file = h5py.File(hdf5_path, "w")
        h5_file.create_group("data")
        h5_file["data"].attrs["env"] = "Lift"
        h5_file["data"].attrs["policy"] = policy_name
        demo_count = 0

    successes = 0
    total_rewards = []
    episode_details = []

    print(f"\n{'='*60}")
    print(f"  Evaluating: {policy_name}")
    print(f"  Episodes: {n_episodes}, Max steps: {max_steps}")
    if hdf5_path:
        print(f"  Saving successful demos to: {hdf5_path}")
    print(f"{'='*60}\n")

    for ep in range(n_episodes):
        env.seed(seed_start + ep)
        obs = env.reset()
        ep_reward = 0.0
        success = False

        # Collect trajectory data for this episode
        ep_actions = []
        ep_images = []
        ep_ee_pos = []
        ep_gripper_qpos = []

        for step in range(max_steps):
            action = policy_fn(obs)

            # Record before stepping
            ep_actions.append(action.copy())
            if "agentview_image" in obs:
                ep_images.append(obs["agentview_image"].copy())
            if "robot0_eef_pos" in obs:
                ep_ee_pos.append(obs["robot0_eef_pos"].copy())
            if "robot0_gripper_qpos" in obs:
                ep_gripper_qpos.append(obs["robot0_gripper_qpos"].copy())

            obs, reward, done, info = env.step(action)
            ep_reward += reward

            if done:
                success = True
                break

        successes += int(success)
        total_rewards.append(ep_reward)
        episode_details.append({
            "episode": ep,
            "seed": seed_start + ep,
            "success": success,
            "reward": ep_reward,
            "steps": step + 1,
        })

        # Save successful episodes to HDF5
        if success and h5_file is not None:
            demo_grp = h5_file["data"].create_group(f"demo_{demo_count}")
            demo_grp.attrs["seed"] = seed_start + ep
            demo_grp.attrs["num_samples"] = len(ep_actions)
            demo_grp.create_dataset("actions", data=np.array(ep_actions))
            obs_grp = demo_grp.create_group("obs")
            if ep_images:
                obs_grp.create_dataset("agentview_image", data=np.array(ep_images))
            if ep_ee_pos:
                obs_grp.create_dataset("robot0_eef_pos", data=np.array(ep_ee_pos))
            if ep_gripper_qpos:
                obs_grp.create_dataset("robot0_gripper_qpos", data=np.array(ep_gripper_qpos))
            h5_file.flush()
            demo_count += 1

        status = "SUCCESS" if success else "FAIL"
        running_rate = successes / (ep + 1) * 100
        print(f"  Episode {ep:3d}: {status:7s}  reward={ep_reward:7.2f}  "
              f"steps={step+1:4d}  running_sr={running_rate:.1f}%")

    env.close()

    # Finalize HDF5
    if h5_file is not None:
        h5_file["data"].attrs["total_demos"] = demo_count
        h5_file["data"].attrs["total_attempts"] = n_episodes
        h5_file["data"].attrs["success_rate"] = successes / max(n_episodes, 1)
        h5_file.close()
        print(f"\n  Saved {demo_count} successful demos to {hdf5_path}")
        if demo_count == 0:
            print(f"  WARNING: No successful demos! The HDF5 'data' group is empty.")
            print(f"  This means the scripted policy had a 0% success rate.")
            print(f"  Consider using --compare to try the improved policy.")

    success_rate = successes / n_episodes
    avg_reward = np.mean(total_rewards)

    print(f"\n{'='*60}")
    print(f"  RESULTS: {policy_name}")
    print(f"{'='*60}")
    print(f"  Success rate: {successes}/{n_episodes} = {success_rate:.1%}")
    print(f"  Avg reward:   {avg_reward:.2f}")
    print(f"  Std reward:   {np.std(total_rewards):.2f}")
    print(f"{'='*60}\n")

    return {
        "policy": policy_name,
        "success_rate": success_rate,
        "successes": successes,
        "total": n_episodes,
        "avg_reward": avg_reward,
        "episodes": episode_details,
    }


# ---------------------------------------------------------------------------
# HDF5 reader (safe — guards against empty data)
# ---------------------------------------------------------------------------

def inspect_hdf5(hdf5_path):
    """Read and inspect an HDF5 demo file, guarding against empty data."""
    if not HAS_H5PY:
        print("h5py not installed, cannot inspect HDF5")
        return

    if not os.path.exists(hdf5_path):
        print(f"File not found: {hdf5_path}")
        return

    with h5py.File(hdf5_path, "r") as f:
        if "data" not in f:
            print(f"No 'data' group in {hdf5_path}")
            return

        demo_keys = sorted(f["data"].keys())
        total_demos = len(demo_keys)
        print(f"\nHDF5: {hdf5_path}")
        print(f"  Total demos: {total_demos}")

        if total_demos == 0:
            print("  WARNING: No demos in file!")
            print("  The scripted policy collected 0 successful episodes.")
            sr = f["data"].attrs.get("success_rate", "N/A")
            attempts = f["data"].attrs.get("total_attempts", "N/A")
            print(f"  Success rate: {sr}")
            print(f"  Total attempts: {attempts}")
            return

        # Show first demo
        demo = f["data"][demo_keys[0]]
        print(f"  First demo ({demo_keys[0]}):")
        print(f"    Steps: {demo.attrs.get('num_samples', 'N/A')}")
        if "actions" in demo:
            print(f"    Actions shape: {demo['actions'].shape}")
        if "obs" in demo:
            for k in sorted(demo["obs"].keys()):
                print(f"    obs/{k}: {demo['obs'][k].shape}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Verify Lift scripted policy success rate")
    parser.add_argument("--episodes", type=int, default=200,
                        help="Number of episodes to run (default: 200)")
    parser.add_argument("--max-steps", type=int, default=300,
                        help="Max steps per episode (default: 300)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Starting seed (default: 0)")
    parser.add_argument("--compare", action="store_true",
                        help="Also run improved policy for comparison")
    parser.add_argument("--output", type=str, default=None,
                        help="Save successful demos to this HDF5 path")
    parser.add_argument("--inspect", type=str, default=None,
                        help="Inspect an existing HDF5 file (no simulation)")
    args = parser.parse_args()

    # Inspect-only mode
    if args.inspect:
        inspect_hdf5(args.inspect)
        return

    print(f"robosuite version: {suite.__version__}")

    # Test the original scripted policy (from robosuite_sim.ipynb)
    original = run_evaluation(
        scripted_lift_policy, "Original Scripted Policy",
        args.episodes, args.max_steps, args.seed,
        hdf5_path=args.output,
    )

    if args.compare:
        improved_output = None
        if args.output:
            base, ext = os.path.splitext(args.output)
            improved_output = f"{base}_improved{ext}"

        improved = run_evaluation(
            improved_lift_policy, "Improved Scripted Policy",
            args.episodes, args.max_steps, args.seed,
            hdf5_path=improved_output,
        )

        print(f"\n{'='*60}")
        print(f"  COMPARISON SUMMARY")
        print(f"{'='*60}")
        print(f"  Original:  {original['success_rate']:.1%} "
              f"({original['successes']}/{original['total']})")
        print(f"  Improved:  {improved['success_rate']:.1%} "
              f"({improved['successes']}/{improved['total']})")
        print(f"{'='*60}")

    # Final verdict on the 20% claim
    sr = original["success_rate"]
    print(f"\n--- VERDICT ---")
    print(f"Measured success rate: {sr:.1%}")
    if sr == 0:
        print("The ~20% claim is FALSE — actual rate is 0%.")
        print("The scripted proportional controller fails to complete any Lift episodes.")
        print("Failure modes:")
        print("  1. Direct approach to cube center pushes it instead of grasping")
        print("  2. Simultaneous close+lift causes the cube to slip")
        print("  3. No pre-grasp hover phase means unreliable finger placement")
    elif 0.15 <= sr <= 0.25:
        print("The ~20% claim appears CONSISTENT with measured results.")
    elif sr < 0.15:
        print(f"The ~20% claim appears OPTIMISTIC. Actual rate is {sr:.1%}.")
    else:
        print(f"The ~20% claim appears PESSIMISTIC. Actual rate is {sr:.1%}.")


if __name__ == "__main__":
    main()
