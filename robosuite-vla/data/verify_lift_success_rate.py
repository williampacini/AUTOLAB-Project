#!/usr/bin/env python3
"""Verify the ACTUAL success rate of scripted policies across robosuite envs.

The existing codebase has two bugs in success detection:
  BUG 1: `if done: success = True` — treats horizon expiry as success
  BUG 2: `success = bool(done and reward > 0)` — with reward_shaping=True,
         reward > 0 just means the arm got close, NOT that the task completed

The CORRECT check is: `env._check_success()`

This script runs episodes across all environments and compares all three
detection methods to show which claimed success rates are actually false.

Usage (paste in a Colab cell with ! prefix):
    !python data/verify_lift_success_rate.py --episodes 50
    !python data/verify_lift_success_rate.py --episodes 50 --envs Lift Stack PickPlaceSingle
    !python data/verify_lift_success_rate.py --episodes 50 --output data/robosuite_demos
"""

import argparse
import json
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
    print("WARNING: h5py not installed — cannot save demos to HDF5")


# ---------------------------------------------------------------------------
# Scripted policies per environment
# ---------------------------------------------------------------------------

def scripted_lift_policy(obs):
    """Original proportional controller from robosuite_sim.ipynb."""
    ee_pos = obs["robot0_eef_pos"]
    cube_pos = obs["cube_pos"]
    delta = cube_pos - ee_pos
    dist = np.linalg.norm(delta)

    action = np.zeros(7)
    if dist > 0.02:
        action[:3] = delta * 10.0
        action[6] = 1.0            # open
    else:
        action[2] = 1.0            # lift
        action[6] = -1.0           # close
    return np.clip(action, -1, 1)


def scripted_stack_policy(obs):
    """Proportional controller for Stack: pick cubeA, place on cubeB."""
    ee_pos = obs["robot0_eef_pos"]
    cubeA_pos = obs["cubeA_pos"]
    cubeB_pos = obs["cubeB_pos"]

    target_above_B = cubeB_pos.copy()
    target_above_B[2] += 0.08  # above cubeB

    # Check if we're holding cubeA (it's above table height)
    holding = cubeA_pos[2] > 0.82  # approximate table height check

    action = np.zeros(7)
    if not holding:
        delta = cubeA_pos - ee_pos
        dist = np.linalg.norm(delta)
        if dist > 0.02:
            action[:3] = delta * 10.0
            action[6] = 1.0
        else:
            action[2] = 1.0
            action[6] = -1.0
    else:
        delta = target_above_B - ee_pos
        dist = np.linalg.norm(delta)
        if dist > 0.03:
            action[:3] = delta * 5.0
            action[6] = -1.0  # keep gripping
        else:
            action[6] = 1.0   # release
    return np.clip(action, -1, 1)


def random_policy(obs, action_dim=7):
    """Random policy baseline — should have ~0% success on all tasks."""
    return np.random.uniform(-1, 1, size=action_dim)


# Map env names to their scripted policies
SCRIPTED_POLICIES = {
    "Lift": scripted_lift_policy,
    "Stack": scripted_stack_policy,
}


# ---------------------------------------------------------------------------
# Core evaluation — uses env._check_success() for ground truth
# ---------------------------------------------------------------------------

def run_episode(env, policy_fn, max_steps):
    """Run one episode, returning ground-truth success and diagnostic info.

    Returns dict with:
      - ground_truth: bool from env._check_success()
      - done_flag: bool — what the `done` return value said
      - last_reward: float — reward on the final step
      - total_reward: float — sum of all rewards
      - reward_positive: bool — whether total_reward > 0
      - buggy_done_only: bool — what `if done: success=True` would say
      - buggy_done_and_reward: bool — what `done and reward > 0` would say
      - steps: int — how many steps ran
    """
    obs = env.reset()
    total_reward = 0.0
    last_reward = 0.0
    done_flag = False
    ground_truth = False
    steps = 0

    # Trajectory buffers (for HDF5 saving)
    actions_buf = []
    images_buf = []

    for step in range(max_steps):
        action = policy_fn(obs)
        actions_buf.append(action.copy())
        if "agentview_image" in obs:
            images_buf.append(obs["agentview_image"].copy())

        obs, reward, done, info = env.step(action)
        total_reward += reward
        last_reward = reward
        steps = step + 1

        # Ground-truth success check — the ONLY correct method
        if env._check_success():
            ground_truth = True

        if done:
            done_flag = True
            break

    return {
        "ground_truth": ground_truth,
        "done_flag": done_flag,
        "last_reward": last_reward,
        "total_reward": total_reward,
        "reward_positive": total_reward > 0,
        "buggy_done_only": done_flag,               # BUG 1
        "buggy_done_and_reward": bool(done_flag and last_reward > 0),  # BUG 2
        "steps": steps,
        "actions": actions_buf,
        "images": images_buf,
    }


def evaluate_env(env_name, n_episodes, max_steps, seed_start=0, hdf5_dir=None):
    """Evaluate one environment with all detection methods."""
    policy_fn = SCRIPTED_POLICIES.get(env_name, None)
    policy_name = "scripted" if policy_fn else "random"
    if policy_fn is None:
        policy_fn = lambda obs: random_policy(obs)
        print(f"  No scripted policy for {env_name}, using random")

    env_kwargs = dict(
        env_name=env_name,
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

    # Stack needs specific obs keys
    if env_name in ("Stack",):
        pass  # default obs works

    env = suite.make(**env_kwargs)

    # HDF5 setup
    h5_file = None
    demo_count = 0
    hdf5_path = None
    if hdf5_dir and HAS_H5PY:
        os.makedirs(hdf5_dir, exist_ok=True)
        hdf5_path = os.path.join(hdf5_dir, f"{env_name.lower()}_demos.hdf5")
        h5_file = h5py.File(hdf5_path, "w")
        h5_file.create_group("data")
        h5_file["data"].attrs["env"] = env_name
        h5_file["data"].attrs["policy"] = policy_name

    counts = {
        "ground_truth": 0,
        "buggy_done_only": 0,
        "buggy_done_and_reward": 0,
    }
    episodes = []

    print(f"\n{'='*70}")
    print(f"  {env_name} — {policy_name} policy — {n_episodes} episodes")
    print(f"{'='*70}")

    for ep in range(n_episodes):
        np.random.seed(seed_start + ep)
        result = run_episode(env, policy_fn, max_steps)

        for key in counts:
            counts[key] += int(result[key])

        episodes.append({
            "episode": ep,
            "ground_truth": result["ground_truth"],
            "buggy_done_only": result["buggy_done_only"],
            "buggy_done_and_reward": result["buggy_done_and_reward"],
            "total_reward": result["total_reward"],
            "steps": result["steps"],
        })

        # Save ONLY ground-truth successes to HDF5
        if result["ground_truth"] and h5_file is not None:
            grp = h5_file["data"].create_group(f"demo_{demo_count}")
            grp.attrs["seed"] = seed_start + ep
            grp.attrs["num_samples"] = len(result["actions"])
            grp.create_dataset("actions", data=np.array(result["actions"]))
            if result["images"]:
                obs_grp = grp.create_group("obs")
                obs_grp.create_dataset("agentview_image",
                                       data=np.array(result["images"]))
            h5_file.flush()
            demo_count += 1

        # Progress every 10 episodes
        if (ep + 1) % 10 == 0 or ep == n_episodes - 1:
            gt_rate = counts["ground_truth"] / (ep + 1)
            bug1_rate = counts["buggy_done_only"] / (ep + 1)
            bug2_rate = counts["buggy_done_and_reward"] / (ep + 1)
            print(f"  [{ep+1:3d}/{n_episodes}]  "
                  f"ACTUAL={gt_rate:5.1%}  "
                  f"done_only(BUG1)={bug1_rate:5.1%}  "
                  f"done+reward(BUG2)={bug2_rate:5.1%}")

    env.close()

    # Finalize HDF5
    if h5_file is not None:
        h5_file["data"].attrs["total_demos"] = demo_count
        h5_file["data"].attrs["total_attempts"] = n_episodes
        h5_file["data"].attrs["success_rate"] = counts["ground_truth"] / max(n_episodes, 1)
        h5_file.close()
        print(f"\n  HDF5: {hdf5_path}")
        print(f"    Saved {demo_count} truly successful demos out of {n_episodes} attempts")
        if demo_count == 0:
            print(f"    WARNING: 0 successful demos — file is empty")

    rates = {k: v / n_episodes for k, v in counts.items()}

    print(f"\n  RESULTS for {env_name}:")
    print(f"  {'Method':<30s} {'Rate':>8s} {'Count':>8s}  Status")
    print(f"  {'-'*65}")
    print(f"  {'env._check_success() [TRUTH]':<30s} {rates['ground_truth']:>7.1%} "
          f"  {counts['ground_truth']:>3d}/{n_episodes:<3d}  {'correct':>8s}")

    for bug_name, label in [
        ("buggy_done_only", "if done: success=True"),
        ("buggy_done_and_reward", "done and reward > 0"),
    ]:
        status = "WRONG" if counts[bug_name] != counts["ground_truth"] else "ok"
        inflated = counts[bug_name] - counts["ground_truth"]
        note = f"  (+{inflated} false positives)" if inflated > 0 else ""
        print(f"  {label:<30s} {rates[bug_name]:>7.1%} "
              f"  {counts[bug_name]:>3d}/{n_episodes:<3d}  {status:>8s}{note}")

    return {
        "env": env_name,
        "policy": policy_name,
        "n_episodes": n_episodes,
        "rates": rates,
        "counts": counts,
        "hdf5_path": hdf5_path,
        "hdf5_demos": demo_count,
        "episodes": episodes,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Verify ACTUAL success rates across robosuite environments")
    parser.add_argument("--episodes", type=int, default=50,
                        help="Episodes per environment (default: 50)")
    parser.add_argument("--max-steps", type=int, default=300,
                        help="Max steps per episode (default: 300)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Starting seed (default: 0)")
    parser.add_argument("--envs", nargs="+",
                        default=["Lift", "Stack", "PickPlaceSingle"],
                        help="Environments to test")
    parser.add_argument("--output", type=str, default=None,
                        help="Directory to save HDF5 demo files")
    args = parser.parse_args()

    print(f"robosuite {suite.__version__}")
    print(f"Testing: {args.envs}")
    print(f"Episodes per env: {args.episodes}")
    print(f"Max steps: {args.max_steps}")

    all_results = []
    for env_name in args.envs:
        try:
            result = evaluate_env(
                env_name, args.episodes, args.max_steps,
                seed_start=args.seed, hdf5_dir=args.output,
            )
            all_results.append(result)
        except Exception as e:
            print(f"\n  ERROR on {env_name}: {e}")
            import traceback
            traceback.print_exc()

    # Summary table
    print(f"\n\n{'='*70}")
    print(f"  SUMMARY — Ground Truth vs Buggy Detection")
    print(f"{'='*70}")
    print(f"  {'Environment':<20s} {'ACTUAL':>8s} {'BUG1(done)':>12s} {'BUG2(done+r)':>14s} {'HDF5 demos':>12s}")
    print(f"  {'-'*70}")
    for r in all_results:
        hdf5_info = f"{r['hdf5_demos']}/{r['n_episodes']}" if r.get("hdf5_path") else "N/A"
        print(f"  {r['env']:<20s} "
              f"{r['rates']['ground_truth']:>7.1%} "
              f"{r['rates']['buggy_done_only']:>11.1%} "
              f"{r['rates']['buggy_done_and_reward']:>13.1%} "
              f"{hdf5_info:>12s}")

    print(f"\n  ACTUAL  = env._check_success()  [ground truth]")
    print(f"  BUG1    = if done: success=True  [treats horizon expiry as success]")
    print(f"  BUG2    = done and reward > 0    [reward shaping gives false positives]")

    # Save JSON summary
    if args.output:
        summary_path = os.path.join(args.output, "verification_results.json")
        json_results = []
        for r in all_results:
            json_results.append({
                "env": r["env"],
                "policy": r["policy"],
                "n_episodes": r["n_episodes"],
                "actual_success_rate": r["rates"]["ground_truth"],
                "buggy_done_only_rate": r["rates"]["buggy_done_only"],
                "buggy_done_and_reward_rate": r["rates"]["buggy_done_and_reward"],
                "actual_successes": r["counts"]["ground_truth"],
                "hdf5_path": r.get("hdf5_path"),
                "hdf5_demos_saved": r.get("hdf5_demos", 0),
            })
        with open(summary_path, "w") as f:
            json.dump(json_results, f, indent=2)
        print(f"\n  Results saved to {summary_path}")


if __name__ == "__main__":
    main()
