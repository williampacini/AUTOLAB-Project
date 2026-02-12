#!/usr/bin/env python3
"""Collect NutAssembly demonstrations using a scripted pick-and-place policy.

Uses privileged state access (nut positions, peg positions) to generate
near-optimal two-nut assembly trajectories. Saves as HDF5 compatible with
the convert_to_lerobot.py pipeline.

Usage:
    python data/collect_nut_assembly.py --num-demos 50 --output data/nut_assembly_scripted/demos.hdf5
    python data/collect_nut_assembly.py --num-demos 5 --visualize  # Preview GIF of first demo
"""

import argparse
import os
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

# Headless rendering — must be before any robosuite import
os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

ROOT = Path(__file__).resolve().parent.parent


def create_env(camera_size=256):
    """Create a NutAssembly environment with two cameras."""
    import robosuite as suite

    env = suite.make(
        env_name="NutAssembly",
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        use_object_obs=True,
        camera_names=["agentview", "robot0_eye_in_hand"],
        camera_heights=camera_size,
        camera_widths=camera_size,
        reward_shaping=True,
        single_object_mode=0,  # Both round and square nuts
        horizon=1000,
    )
    return env


def get_nut_handle_pos(env, nut_name):
    """Get the handle position of a nut from the simulator.

    Args:
        env: robosuite environment
        nut_name: "RoundNut" or "SquareNut"
    """
    nut_obj = None
    for nut in env.nuts:
        if nut_name.lower() in nut.name.lower():
            nut_obj = nut
            break

    if nut_obj is None:
        raise ValueError(f"Nut '{nut_name}' not found. Available: {[n.name for n in env.nuts]}")

    # Get handle site position from the simulator
    handle_site = nut_obj.important_sites["handle"]
    site_id = env.sim.model.site_name2id(handle_site)
    return env.sim.data.site_xpos[site_id].copy()


def get_peg_pos(env, peg_id):
    """Get the top position of a peg.

    Args:
        env: robosuite environment
        peg_id: 1 for peg1 (square nut target) or 2 for peg2 (round nut target)
    """
    if peg_id == 1:
        body_id = env.peg1_body_id
    else:
        body_id = env.peg2_body_id
    return env.sim.data.body_xpos[body_id].copy()


class ScriptedNutAssemblyPolicy:
    """Scripted policy for NutAssembly using privileged state information.

    Executes a two-nut pick-and-place sequence:
      1. Pick RoundNut -> place on peg2
      2. Pick SquareNut -> place on peg1
    """

    def __init__(self, env, noise_std=0.01, k_p=3.0):
        self.env = env
        self.noise_std = noise_std
        self.k_p = k_p
        self.phase = "reach_round"
        self.grasp_counter = 0
        self.release_counter = 0
        self.last_target = None
        # Height offsets for approach/lift
        self.approach_height = 0.15
        self.grasp_height = 0.01
        self.lift_height = 0.20
        self.place_height = 0.02

    def reset(self):
        self.phase = "reach_round"
        self.grasp_counter = 0
        self.release_counter = 0
        self.last_target = None

    def get_action(self, obs):
        """Compute action based on current phase."""
        eef_pos = obs["robot0_eef_pos"]

        action = np.zeros(7)

        if self.phase == "reach_round":
            target = get_nut_handle_pos(self.env, "RoundNut")
            target[2] += self.approach_height  # Approach from above
            action = self._move_to(eef_pos, target, gripper_open=True)
            if np.linalg.norm(eef_pos - target) < 0.02:
                self.phase = "lower_round"

        elif self.phase == "lower_round":
            target = get_nut_handle_pos(self.env, "RoundNut")
            target[2] += self.grasp_height
            action = self._move_to(eef_pos, target, gripper_open=True)
            if np.linalg.norm(eef_pos - target) < 0.02:
                self.phase = "grasp_round"
                self.grasp_counter = 0

        elif self.phase == "grasp_round":
            action = self._move_to(eef_pos, self.last_target, gripper_close=True)
            self.grasp_counter += 1
            if self.grasp_counter >= 15:
                self.phase = "lift_round"

        elif self.phase == "lift_round":
            target = eef_pos.copy()
            target[2] = self.lift_height + 0.7  # Lift above table
            action = self._move_to(eef_pos, target, gripper_close=True)
            if eef_pos[2] > self.lift_height + 0.65:
                self.phase = "move_to_peg2"

        elif self.phase == "move_to_peg2":
            target = get_peg_pos(self.env, 2)
            target[2] += self.approach_height
            action = self._move_to(eef_pos, target, gripper_close=True)
            if np.linalg.norm(eef_pos[:2] - target[:2]) < 0.02:
                self.phase = "lower_to_peg2"

        elif self.phase == "lower_to_peg2":
            target = get_peg_pos(self.env, 2)
            target[2] += self.place_height
            action = self._move_to(eef_pos, target, gripper_close=True)
            if np.linalg.norm(eef_pos[:2] - target[:2]) < 0.02:
                self.phase = "release_round"
                self.release_counter = 0

        elif self.phase == "release_round":
            action = self._move_to(eef_pos, self.last_target, gripper_open=True)
            self.release_counter += 1
            if self.release_counter >= 15:
                self.phase = "retract_from_peg2"

        elif self.phase == "retract_from_peg2":
            target = eef_pos.copy()
            target[2] = self.lift_height + 0.7
            action = self._move_to(eef_pos, target, gripper_open=True)
            if eef_pos[2] > self.lift_height + 0.65:
                self.phase = "reach_square"

        elif self.phase == "reach_square":
            target = get_nut_handle_pos(self.env, "SquareNut")
            target[2] += self.approach_height
            action = self._move_to(eef_pos, target, gripper_open=True)
            if np.linalg.norm(eef_pos - target) < 0.02:
                self.phase = "lower_square"

        elif self.phase == "lower_square":
            target = get_nut_handle_pos(self.env, "SquareNut")
            target[2] += self.grasp_height
            action = self._move_to(eef_pos, target, gripper_open=True)
            if np.linalg.norm(eef_pos - target) < 0.02:
                self.phase = "grasp_square"
                self.grasp_counter = 0

        elif self.phase == "grasp_square":
            action = self._move_to(eef_pos, self.last_target, gripper_close=True)
            self.grasp_counter += 1
            if self.grasp_counter >= 15:
                self.phase = "lift_square"

        elif self.phase == "lift_square":
            target = eef_pos.copy()
            target[2] = self.lift_height + 0.7
            action = self._move_to(eef_pos, target, gripper_close=True)
            if eef_pos[2] > self.lift_height + 0.65:
                self.phase = "move_to_peg1"

        elif self.phase == "move_to_peg1":
            target = get_peg_pos(self.env, 1)
            target[2] += self.approach_height
            action = self._move_to(eef_pos, target, gripper_close=True)
            if np.linalg.norm(eef_pos[:2] - target[:2]) < 0.02:
                self.phase = "lower_to_peg1"

        elif self.phase == "lower_to_peg1":
            target = get_peg_pos(self.env, 1)
            target[2] += self.place_height
            action = self._move_to(eef_pos, target, gripper_close=True)
            if np.linalg.norm(eef_pos[:2] - target[:2]) < 0.02:
                self.phase = "release_square"
                self.release_counter = 0

        elif self.phase == "release_square":
            action = self._move_to(eef_pos, self.last_target, gripper_open=True)
            self.release_counter += 1
            if self.release_counter >= 15:
                self.phase = "done"

        elif self.phase == "done":
            pass  # No-op — episode should end via success check

        # Add noise for trajectory diversity
        if self.noise_std > 0:
            action[:6] += np.random.normal(0, self.noise_std, size=6)

        return np.clip(action, -1, 1)

    def _move_to(self, current, target, gripper_open=False, gripper_close=False):
        """Proportional controller to move end-effector toward target."""
        self.last_target = target.copy()
        action = np.zeros(7)
        delta = target - current
        action[:3] = self.k_p * delta

        if gripper_open:
            action[6] = -1.0
        elif gripper_close:
            action[6] = 1.0

        return np.clip(action, -1, 1)


def collect_episode(env, policy, max_steps=1000):
    """Run one episode and collect demonstration data.

    Returns:
        data: dict with obs, actions, success flag, or None if failed
    """
    obs = env.reset()
    policy.reset()

    agentview_frames = []
    wrist_frames = []
    eef_positions = []
    eef_quats = []
    gripper_qpos = []
    actions_list = []

    success = False

    for step in range(max_steps):
        action = policy.get_action(obs)

        # Record observations BEFORE taking action
        agentview = np.flip(obs["agentview_image"], axis=0).copy()
        wrist = np.flip(obs["robot0_eye_in_hand_image"], axis=0).copy()
        agentview_frames.append(agentview)
        wrist_frames.append(wrist)
        eef_positions.append(obs["robot0_eef_pos"].copy())
        eef_quats.append(obs["robot0_eef_quat"].copy())
        gripper_qpos.append(obs["robot0_gripper_qpos"].copy())
        actions_list.append(action.copy())

        obs, reward, done, info = env.step(action)

        if env._check_success():
            success = True
            break

    if not success:
        return None

    return {
        "agentview_image": np.stack(agentview_frames),
        "robot0_eye_in_hand_image": np.stack(wrist_frames),
        "robot0_eef_pos": np.stack(eef_positions),
        "robot0_eef_quat": np.stack(eef_quats),
        "robot0_gripper_qpos": np.stack(gripper_qpos),
        "actions": np.stack(actions_list),
        "success": success,
        "n_steps": len(actions_list),
    }


def save_hdf5(demos, output_path, task_language):
    """Save collected demonstrations as HDF5 file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(str(output_path), "w") as f:
        f.attrs["task_description"] = task_language
        f.attrs["env_name"] = "NutAssembly"
        f.attrs["robot"] = "Panda"
        f.attrs["n_demos"] = len(demos)

        grp = f.create_group("data")

        for i, demo in enumerate(demos):
            demo_grp = grp.create_group(f"demo_{i}")
            obs_grp = demo_grp.create_group("obs")

            obs_grp.create_dataset("agentview_image", data=demo["agentview_image"],
                                   compression="gzip", compression_opts=4)
            obs_grp.create_dataset("robot0_eye_in_hand_image", data=demo["robot0_eye_in_hand_image"],
                                   compression="gzip", compression_opts=4)
            obs_grp.create_dataset("robot0_eef_pos", data=demo["robot0_eef_pos"])
            obs_grp.create_dataset("robot0_eef_quat", data=demo["robot0_eef_quat"])
            obs_grp.create_dataset("robot0_gripper_qpos", data=demo["robot0_gripper_qpos"])
            demo_grp.create_dataset("actions", data=demo["actions"])

    print(f"Saved {len(demos)} demos to {output_path}")


def save_preview_gif(frames, path, fps=20):
    """Save a preview GIF from a list of frames."""
    from PIL import Image as PILImage

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    pil_frames = [PILImage.fromarray(f).resize((256, 256)) for f in frames]
    if pil_frames:
        pil_frames[0].save(
            str(path), save_all=True, append_images=pil_frames[1:],
            duration=max(1, 1000 // fps), loop=0, optimize=True,
        )
        print(f"Preview GIF saved: {path} ({len(frames)} frames)")


def main():
    parser = argparse.ArgumentParser(description="Collect NutAssembly demos via scripted policy")
    parser.add_argument("--num-demos", type=int, default=50,
                        help="Number of successful demos to collect")
    parser.add_argument("--max-attempts", type=int, default=200,
                        help="Maximum episode attempts (to handle failures)")
    parser.add_argument("--output", type=str,
                        default="data/nut_assembly_scripted/demos.hdf5",
                        help="Output HDF5 path")
    parser.add_argument("--camera-size", type=int, default=256,
                        help="Camera resolution (square)")
    parser.add_argument("--noise-std", type=float, default=0.01,
                        help="Gaussian noise added to actions for diversity")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=1000,
                        help="Max steps per episode")
    parser.add_argument("--visualize", action="store_true",
                        help="Save a preview GIF of the first demo")
    args = parser.parse_args()

    np.random.seed(args.seed)

    task_language = "Assemble the round nut and square nut onto their respective pegs"

    print("=" * 60)
    print("  NutAssembly Scripted Demo Collection")
    print("=" * 60)
    print(f"  Target demos: {args.num_demos}")
    print(f"  Camera size:  {args.camera_size}x{args.camera_size}")
    print(f"  Noise std:    {args.noise_std}")
    print(f"  Max steps:    {args.max_steps}")
    print(f"  Output:       {args.output}")
    print()

    env = create_env(camera_size=args.camera_size)
    policy = ScriptedNutAssemblyPolicy(env, noise_std=args.noise_std)

    demos = []
    attempts = 0

    pbar = tqdm(total=args.num_demos, desc="Collecting demos")
    while len(demos) < args.num_demos and attempts < args.max_attempts:
        attempts += 1
        demo = collect_episode(env, policy, max_steps=args.max_steps)
        if demo is not None:
            demos.append(demo)
            pbar.update(1)
            pbar.set_postfix(attempts=attempts, success_rate=f"{len(demos)/attempts:.1%}")
        else:
            pbar.set_postfix(attempts=attempts, failed=attempts - len(demos))

    pbar.close()
    env.close()

    print(f"\nCollected {len(demos)}/{args.num_demos} demos in {attempts} attempts")
    print(f"Success rate: {len(demos)/attempts:.1%}")

    if not demos:
        print("ERROR: No successful demos collected!")
        return

    # Save HDF5
    output_path = ROOT / args.output
    save_hdf5(demos, output_path, task_language)

    # Save preview GIF
    if args.visualize and demos:
        gif_path = output_path.parent / "preview.gif"
        # Use every 3rd frame to keep GIF small
        preview_frames = demos[0]["agentview_image"][::3]
        save_preview_gif(preview_frames, gif_path)

    # Print stats
    lengths = [d["n_steps"] for d in demos]
    print(f"\nEpisode lengths: min={min(lengths)}, max={max(lengths)}, mean={np.mean(lengths):.0f}")


if __name__ == "__main__":
    main()
