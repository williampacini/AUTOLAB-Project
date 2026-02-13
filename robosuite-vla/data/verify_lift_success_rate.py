#!/usr/bin/env python3
"""Verify the success rate of the scripted Lift policy.

This script runs N episodes of robosuite's Lift task using the simple
proportional controller from robosuite_sim.ipynb and reports the actual
success rate. Use it to verify the claimed ~20% success rate.

Usage (on Colab or any machine with robosuite installed):
    python data/verify_lift_success_rate.py --episodes 200 --max-steps 300
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
    dist_hover = np.linalg.norm(delta_hover)

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


def run_evaluation(policy_fn, policy_name, n_episodes, max_steps, seed_start=0):
    """Run N episodes and return success statistics."""
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

    successes = 0
    total_rewards = []
    episode_details = []

    print(f"\n{'='*60}")
    print(f"  Evaluating: {policy_name}")
    print(f"  Episodes: {n_episodes}, Max steps: {max_steps}")
    print(f"{'='*60}\n")

    for ep in range(n_episodes):
        env.seed(seed_start + ep)
        obs = env.reset()
        ep_reward = 0.0
        success = False

        for step in range(max_steps):
            action = policy_fn(obs)
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

        status = "SUCCESS" if success else "FAIL"
        running_rate = successes / (ep + 1) * 100
        print(f"  Episode {ep:3d}: {status:7s}  reward={ep_reward:7.2f}  "
              f"steps={step+1:4d}  running_sr={running_rate:.1f}%")

    env.close()

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
    args = parser.parse_args()

    print(f"robosuite version: {suite.__version__}")
    print(f"Environments: {suite.ALL_ENVIRONMENTS if hasattr(suite, 'ALL_ENVIRONMENTS') else 'N/A'}")

    # Test the original scripted policy (from robosuite_sim.ipynb)
    original = run_evaluation(
        scripted_lift_policy, "Original Scripted Policy",
        args.episodes, args.max_steps, args.seed,
    )

    if args.compare:
        improved = run_evaluation(
            improved_lift_policy, "Improved Scripted Policy",
            args.episodes, args.max_steps, args.seed,
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
    if 0.15 <= sr <= 0.25:
        print(f"The ~20% claim appears CONSISTENT with measured results.")
    elif sr < 0.15:
        print(f"The ~20% claim appears OPTIMISTIC. Actual rate is lower.")
    else:
        print(f"The ~20% claim appears PESSIMISTIC. Actual rate is higher.")


if __name__ == "__main__":
    main()
