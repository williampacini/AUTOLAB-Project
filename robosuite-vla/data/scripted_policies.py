#!/usr/bin/env python3
"""Scripted expert policies for robosuite environments.

Each policy uses ground-truth object positions from the observation dictionary
(enabled by ``use_object_obs=True``) and a simple proportional controller to
generate expert demonstrations.  No learning or human teleoperation needed.

Supported environments:
  Lift, Stack, PickPlaceSingle, Door, NutAssemblySingle, NutAssembly

All grasp-from-above policies use the same robust pattern:
  approach → descend (open) → grasp (close, check qpos) → post-grasp
      ↑                              │
      └──── retry (open, retract) ←──┘  (if fingers closed on nothing)

Gripper convention (robosuite Panda + OSC_POSE):
  -1 = open, +1 = close
"""

import numpy as np


# ---------------------------------------------------------------------------
# Common helpers
# ---------------------------------------------------------------------------

_GRASP_STEPS = 15          # steps to hold close before checking
_EMPTY_GRIP_THRESH = 0.02  # finger-gap sum below this → missed


def reach_pos(ee_pos, target_pos, gain=10.0):
    """Proportional controller: move end-effector toward *target_pos*.

    Returns clipped (dx, dy, dz) action components in [-1, 1].
    """
    delta = target_pos - ee_pos
    return np.clip(delta * gain, -1, 1)


def is_near(a, b, threshold=0.02):
    """True if Euclidean distance between *a* and *b* < *threshold*."""
    return np.linalg.norm(np.asarray(a) - np.asarray(b)) < threshold


def _gripper_closed(obs, threshold=0.04):
    """Heuristic: gripper fingers are close together → object grasped."""
    qpos = obs.get("robot0_gripper_qpos", np.zeros(2))
    return (qpos[0] + qpos[1]) < threshold


def _quat_to_rot(q):
    """Convert quaternion [x, y, z, w] to 3x3 rotation matrix."""
    x, y, z, w = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
    ])


def _gripper_yaw_correction(obs, gain=3.0):
    """Compute ``action[5]`` to align gripper fingers with the nearest axis.

    The Panda gripper opens along the EEF y-axis.  For a top-down grasp of an
    axis-aligned object we want that finger axis projected onto the world XY
    plane to point along world X or world Y (nearest 90-degree snap).

    Returns a single float to add to ``action[5]`` (z-rotation in OSC_POSE).
    """
    quat = obs.get("robot0_eef_quat")
    if quat is None:
        return 0.0
    R = _quat_to_rot(quat)
    # Finger opening direction = EEF y-axis, projected onto world XY
    finger_xy = R[:2, 1]
    angle = np.arctan2(finger_xy[1], finger_xy[0])
    # Snap to nearest 90-degree increment
    target = round(angle / (np.pi / 2)) * (np.pi / 2)
    error = target - angle
    # Wrap to [-pi, pi]
    if error > np.pi:
        error -= 2 * np.pi
    elif error < -np.pi:
        error += 2 * np.pi
    return float(np.clip(error * gain, -1, 1))


def _missed_grasp(obs):
    """True if gripper fingers are fully closed (nothing between them)."""
    qpos = obs.get("robot0_gripper_qpos", np.zeros(2))
    return (qpos[0] + qpos[1]) < _EMPTY_GRIP_THRESH


# ---------------------------------------------------------------------------
# 1. Lift (stateful — retry on missed grasp)
# ---------------------------------------------------------------------------

class LiftPolicy:
    """Stateful scripted Lift policy.

    Descends with gripper open so fingers go around the cube sides, then
    closes.  If the grasp fails (fingers closed on nothing), retracts and
    retries up to *max_attempts* times.

    Phases: approach → descend → grasp → lift  (retry loops back to descend)
    """

    def __init__(self, max_attempts=2):
        self.max_attempts = max_attempts
        self.reset()

    def reset(self):
        self.phase = "approach"
        self._grasp_counter = 0
        self._attempts = 0

    def __call__(self, obs, env):
        ee = obs["robot0_eef_pos"]
        cube = obs["cube_pos"]

        action = np.zeros(7)
        xy_dist = np.linalg.norm(ee[:2] - cube[:2])
        z_diff = ee[2] - cube[2]

        # Always align gripper yaw
        action[5] = _gripper_yaw_correction(obs)

        if self.phase == "approach":
            target = cube.copy()
            target[2] += 0.05
            action[:3] = reach_pos(ee, target)
            action[6] = -1.0
            if xy_dist < 0.02:
                self.phase = "descend"

        elif self.phase == "descend":
            target = cube.copy()
            target[2] -= 0.01  # 1 cm below cube center
            action[:3] = reach_pos(ee, target)
            action[6] = -1.0  # open
            if z_diff < -0.005:
                self.phase = "grasp"
                self._grasp_counter = 0

        elif self.phase == "grasp":
            action[:3] = reach_pos(ee, cube, gain=2.0)
            action[6] = 1.0
            self._grasp_counter += 1
            if self._grasp_counter >= _GRASP_STEPS:
                if _missed_grasp(obs):
                    self._attempts += 1
                    if self._attempts < self.max_attempts:
                        self.phase = "retry"
                    else:
                        self.phase = "lift"
                else:
                    self.phase = "lift"

        elif self.phase == "lift":
            action[2] = 1.0
            action[6] = 1.0

        elif self.phase == "retry":
            target = cube.copy()
            target[2] += 0.05
            action[:3] = reach_pos(ee, target)
            action[6] = -1.0
            if z_diff > 0.04:
                self.phase = "descend"

        return np.clip(action, -1, 1)


_lift_policy = LiftPolicy()


def scripted_lift_policy(obs, env):
    """Move toward cube, grasp, lift — delegates to :class:`LiftPolicy`."""
    return _lift_policy(obs, env)


# ---------------------------------------------------------------------------
# 2. Stack (stateful — retry on missed grasp)
# ---------------------------------------------------------------------------

class StackPolicy:
    """Pick cubeA (red), stack on cubeB (green), release.

    Phases: approach → descend → grasp → retry? → lift_to_B → lower_to_B
            → release → retract
    """

    def __init__(self, max_attempts=2):
        self.max_attempts = max_attempts
        self.reset()

    def reset(self):
        self.phase = "approach"
        self._grasp_counter = 0
        self._attempts = 0

    def __call__(self, obs, env):
        ee = obs["robot0_eef_pos"]
        cubeA = obs["cubeA_pos"]
        cubeB = obs["cubeB_pos"]

        action = np.zeros(7)
        action[5] = _gripper_yaw_correction(obs)

        xy_dist = np.linalg.norm(ee[:2] - cubeA[:2])
        z_diff = ee[2] - cubeA[2]

        if self.phase == "approach":
            target = cubeA.copy()
            target[2] += 0.05
            action[:3] = reach_pos(ee, target)
            action[6] = -1.0
            if xy_dist < 0.02:
                self.phase = "descend"

        elif self.phase == "descend":
            target = cubeA.copy()
            target[2] -= 0.01
            action[:3] = reach_pos(ee, target)
            action[6] = -1.0
            if z_diff < -0.005:
                self.phase = "grasp"
                self._grasp_counter = 0

        elif self.phase == "grasp":
            action[:3] = reach_pos(ee, cubeA, gain=2.0)
            action[6] = 1.0
            self._grasp_counter += 1
            if self._grasp_counter >= _GRASP_STEPS:
                if _missed_grasp(obs):
                    self._attempts += 1
                    if self._attempts < self.max_attempts:
                        self.phase = "retry"
                    else:
                        self.phase = "lift_to_B"
                else:
                    self.phase = "lift_to_B"

        elif self.phase == "retry":
            target = cubeA.copy()
            target[2] += 0.05
            action[:3] = reach_pos(ee, target)
            action[6] = -1.0
            if z_diff > 0.04:
                self.phase = "descend"

        elif self.phase == "lift_to_B":
            target = cubeB.copy()
            target[2] += 0.10
            action[:3] = reach_pos(ee, target)
            action[6] = 1.0
            above_B_xy = np.linalg.norm(ee[:2] - cubeB[:2]) < 0.02
            if above_B_xy and ee[2] > cubeB[2] + 0.06:
                self.phase = "lower_to_B"

        elif self.phase == "lower_to_B":
            target = cubeB.copy()
            target[2] += 0.025
            action[:3] = reach_pos(ee, target, gain=5.0)
            action[6] = 1.0
            if ee[2] < target[2] + 0.01:
                self.phase = "release"

        elif self.phase == "release":
            action[6] = -1.0

        return np.clip(action, -1, 1)


_stack_policy = StackPolicy()


def scripted_stack_policy(obs, env):
    """Stack cubeA on cubeB — delegates to :class:`StackPolicy`."""
    return _stack_policy(obs, env)


# ---------------------------------------------------------------------------
# 3. PickPlaceSingle (stateful — retry on missed grasp)
# ---------------------------------------------------------------------------

_PP_OBJECTS = ["Can", "Milk", "Bread", "Cereal",
               "can", "milk", "bread", "cereal",
               "Can0", "Milk0", "Bread0", "Cereal0"]


def _find_object_pos(obs):
    """Find the active object position in PickPlaceSingle."""
    for name in _PP_OBJECTS:
        key = f"{name}_pos"
        if key in obs:
            return obs[key]
    # Broad search: any obs key ending in _pos that matches known objects
    for key in obs:
        if key.endswith("_pos") and any(
            obj.lower() in key.lower() for obj in ["can", "milk", "bread", "cereal"]
        ):
            return obs[key]
    if "object-state" in obs:
        return obs["object-state"][:3]
    raise KeyError("Cannot find object position in obs")


def _get_target_bin_pos(env):
    """Get target bin position from the simulation model.

    Searches for the target bin body using multiple naming conventions
    across robosuite versions (1.4 and 1.5+).
    """
    # Try direct env attributes first (robosuite stores bin positions)
    if hasattr(env, "target_bin_placements"):
        placements = env.target_bin_placements
        if len(placements) > 0:
            return np.array(placements[0][:3])

    # Try common body names across robosuite versions
    for name in ["bin2_body", "bin2", "bin1_body", "bin1",
                  "visual_bin_body", "table_visual_bin"]:
        try:
            bid = env.sim.model.body_name2id(name)
            return env.sim.data.body_xpos[bid].copy()
        except (ValueError, KeyError):
            continue

    # Broad search: find any body with "bin" in the name
    for i in range(env.sim.model.nbody):
        name = env.sim.model.body_id2name(i)
        if "bin" in name.lower():
            pos = env.sim.data.body_xpos[i].copy()
            if np.linalg.norm(pos) > 0.1:  # skip origin bodies
                return pos

    return np.array([0.18, 0.0, 0.82])


class PickPlacePolicy:
    """Pick up object, move to bin, release.

    Phases: approach → descend → grasp → retry? → lift → move_to_bin
            → lower_to_bin → release
    """

    def __init__(self, max_attempts=2):
        self.max_attempts = max_attempts
        self.reset()

    def reset(self):
        self.phase = "approach"
        self._grasp_counter = 0
        self._attempts = 0

    def __call__(self, obs, env):
        ee = obs["robot0_eef_pos"]
        obj_pos = _find_object_pos(obs)
        bin_pos = _get_target_bin_pos(env)

        action = np.zeros(7)
        action[5] = _gripper_yaw_correction(obs)

        xy_dist = np.linalg.norm(ee[:2] - obj_pos[:2])
        z_diff = ee[2] - obj_pos[2]

        if self.phase == "approach":
            target = obj_pos.copy()
            target[2] += 0.05
            action[:3] = reach_pos(ee, target)
            action[6] = -1.0
            if xy_dist < 0.02:
                self.phase = "descend"

        elif self.phase == "descend":
            target = obj_pos.copy()
            target[2] -= 0.01
            action[:3] = reach_pos(ee, target)
            action[6] = -1.0
            if z_diff < -0.005:
                self.phase = "grasp"
                self._grasp_counter = 0

        elif self.phase == "grasp":
            action[:3] = reach_pos(ee, obj_pos, gain=2.0)
            action[6] = 1.0
            self._grasp_counter += 1
            if self._grasp_counter >= _GRASP_STEPS:
                if _missed_grasp(obs):
                    self._attempts += 1
                    if self._attempts < self.max_attempts:
                        self.phase = "retry"
                    else:
                        self.phase = "lift"
                else:
                    self.phase = "lift"

        elif self.phase == "retry":
            target = obj_pos.copy()
            target[2] += 0.05
            action[:3] = reach_pos(ee, target)
            action[6] = -1.0
            if z_diff > 0.04:
                self.phase = "descend"

        elif self.phase == "lift":
            action[2] = 1.0
            action[6] = 1.0
            if ee[2] > obj_pos[2] + 0.10:
                self.phase = "move_to_bin"

        elif self.phase == "move_to_bin":
            target = bin_pos.copy()
            target[2] += 0.15
            action[:3] = reach_pos(ee, target)
            action[6] = 1.0
            if np.linalg.norm(ee[:2] - bin_pos[:2]) < 0.03:
                self.phase = "lower_to_bin"

        elif self.phase == "lower_to_bin":
            target = bin_pos.copy()
            target[2] += 0.05
            action[:3] = reach_pos(ee, target, gain=5.0)
            action[6] = 1.0
            if ee[2] < target[2] + 0.02:
                self.phase = "release"

        elif self.phase == "release":
            action[6] = -1.0

        return np.clip(action, -1, 1)


_pickplace_policy = PickPlacePolicy()


def scripted_pickplace_policy(obs, env):
    """Pick and place — delegates to :class:`PickPlacePolicy`."""
    return _pickplace_policy(obs, env)


# ---------------------------------------------------------------------------
# 4. Door
# ---------------------------------------------------------------------------

def scripted_door_policy(obs, env):
    """Reach door handle, grasp, pull to open.

    Obs keys: ``handle_pos``, ``hinge_qpos``, ``robot0_eef_pos``.
    Door uses a horizontal reach (not top-down), so it keeps the simpler
    stateless approach with yaw correction.
    """
    ee = obs["robot0_eef_pos"]
    handle = obs.get("handle_pos")
    if handle is None:
        handle = obs.get("door_handle_pos")
    if handle is None:
        handle = obs.get("object-state", np.zeros(10))[:3]
    hinge = obs.get("hinge_qpos", np.zeros(1))
    if isinstance(hinge, np.ndarray):
        hinge = float(hinge.flat[0])

    action = np.zeros(7)
    action[5] = _gripper_yaw_correction(obs)

    dist = np.linalg.norm(ee - handle)

    if dist > 0.02:
        # Phase 1: reach handle
        action[:3] = reach_pos(ee, handle)
        action[6] = -1.0
    elif hinge < 0.25:
        # Phase 2: grasp and pull
        action[6] = 1.0
        action[1] = -1.0
        action[2] = 0.15  # slight upward to keep contact
    else:
        # Door open enough
        action[6] = 1.0

    return np.clip(action, -1, 1)


# ---------------------------------------------------------------------------
# 5. NutAssemblySingle (stateful — retry on missed grasp)
# ---------------------------------------------------------------------------

_NUT_NAMES = ["RoundNut", "SquareNut",
              "RoundNut0", "SquareNut0",
              "roundnut", "squarenut",
              "round-nut", "square-nut"]


def _find_nut_pos(obs, names=None):
    """Return (nut_name, nut_pos) for the first nut found in obs."""
    names = names or _NUT_NAMES
    for name in names:
        key = f"{name}_pos"
        if key in obs:
            return name, obs[key]
    # Broad search: any key ending in _pos that contains "nut"
    for key in obs:
        if "nut" in key.lower() and key.endswith("_pos"):
            name = key[:-4]  # strip "_pos"
            return name, obs[key]
    if "object-state" in obs:
        return "unknown", obs["object-state"][:3]
    raise KeyError("Cannot find nut position in obs")


def _get_peg_pos(env, nut_name):
    """Get the target peg position for *nut_name* from the sim model.

    Searches for peg bodies using multiple naming conventions across
    robosuite versions (1.4 and 1.5+).
    """
    if "Square" in nut_name:
        candidates = ["peg1", "peg1_table", "peg1_body"]
    else:
        candidates = ["peg2", "peg2_table", "peg2_body"]
    for name in candidates:
        try:
            bid = env.sim.model.body_name2id(name)
            return env.sim.data.body_xpos[bid].copy()
        except (ValueError, KeyError):
            continue

    # Try numbered pegs generically
    for i in range(1, 5):
        for prefix in ["peg", "peg_", "peg_body"]:
            try:
                bid = env.sim.model.body_name2id(f"{prefix}{i}")
                return env.sim.data.body_xpos[bid].copy()
            except (ValueError, KeyError):
                continue

    # Broad search: find any body with "peg" in the name
    peg_bodies = []
    for i in range(env.sim.model.nbody):
        name = env.sim.model.body_id2name(i)
        if "peg" in name.lower():
            pos = env.sim.data.body_xpos[i].copy()
            peg_bodies.append((name, pos))

    if peg_bodies:
        # If multiple pegs found, pick based on nut type:
        # Square nuts typically go to the first peg, round to the second
        idx = 0 if "Square" in nut_name else min(1, len(peg_bodies) - 1)
        return peg_bodies[idx][1]

    raise RuntimeError(f"Cannot find peg body for {nut_name} in sim model")


class NutAssemblySinglePolicy:
    """Pick up the nut, align above the correct peg, insert, release.

    Phases: approach → descend → grasp → retry? → lift_to_peg
            → lower_to_peg → release
    """

    def __init__(self, max_attempts=2):
        self.max_attempts = max_attempts
        self.reset()

    def reset(self):
        self.phase = "approach"
        self._grasp_counter = 0
        self._attempts = 0

    def __call__(self, obs, env):
        ee = obs["robot0_eef_pos"]
        nut_name, nut_pos = _find_nut_pos(obs)
        peg_pos = _get_peg_pos(env, nut_name)

        action = np.zeros(7)
        action[5] = _gripper_yaw_correction(obs)

        xy_dist = np.linalg.norm(ee[:2] - nut_pos[:2])
        z_diff = ee[2] - nut_pos[2]

        if self.phase == "approach":
            target = nut_pos.copy()
            target[2] = max(target[2] + 0.05, ee[2])
            action[:3] = reach_pos(ee, target)
            action[6] = -1.0
            if xy_dist < 0.02:
                self.phase = "descend"

        elif self.phase == "descend":
            target = nut_pos.copy()
            target[2] -= 0.01
            action[:3] = reach_pos(ee, target)
            action[6] = -1.0
            if z_diff < -0.005:
                self.phase = "grasp"
                self._grasp_counter = 0

        elif self.phase == "grasp":
            action[:3] = reach_pos(ee, nut_pos, gain=2.0)
            action[6] = 1.0
            self._grasp_counter += 1
            if self._grasp_counter >= _GRASP_STEPS:
                if _missed_grasp(obs):
                    self._attempts += 1
                    if self._attempts < self.max_attempts:
                        self.phase = "retry"
                    else:
                        self.phase = "lift_to_peg"
                else:
                    self.phase = "lift_to_peg"

        elif self.phase == "retry":
            target = nut_pos.copy()
            target[2] += 0.05
            action[:3] = reach_pos(ee, target)
            action[6] = -1.0
            if z_diff > 0.04:
                self.phase = "descend"

        elif self.phase == "lift_to_peg":
            target = peg_pos.copy()
            target[2] += 0.12
            action[:3] = reach_pos(ee, target)
            action[6] = 1.0
            above_peg_xy = np.linalg.norm(ee[:2] - peg_pos[:2]) < 0.015
            if above_peg_xy and ee[2] > peg_pos[2] + 0.08:
                self.phase = "lower_to_peg"

        elif self.phase == "lower_to_peg":
            target = peg_pos.copy()
            target[2] += 0.02
            action[:3] = reach_pos(ee, target, gain=5.0)
            action[6] = 1.0
            if ee[2] < target[2] + 0.01:
                self.phase = "release"

        elif self.phase == "release":
            action[6] = -1.0

        return np.clip(action, -1, 1)


_nut_single_policy = NutAssemblySinglePolicy()


def scripted_nut_single_policy(obs, env):
    """Nut assembly single — delegates to :class:`NutAssemblySinglePolicy`."""
    return _nut_single_policy(obs, env)


# ---------------------------------------------------------------------------
# 6. NutAssembly (two nuts — stateful, retry on missed grasp)
# ---------------------------------------------------------------------------

class NutAssemblyPolicy:
    """Stateful scripted policy for two-nut assembly.

    Sequentially grasps each unplaced nut and inserts it onto its peg.
    After placing one nut, resets the grasp phases for the next.
    """

    def __init__(self, max_attempts=2):
        self.max_attempts = max_attempts
        self.reset()

    def reset(self):
        self.phase = "approach"
        self._grasp_counter = 0
        self._attempts = 0

    def _nut_on_peg(self, nut_pos, peg_pos):
        xy_dist = np.linalg.norm(nut_pos[:2] - peg_pos[:2])
        z_close = abs(nut_pos[2] - peg_pos[2]) < 0.04
        return xy_dist < 0.03 and z_close

    def _reset_grasp(self):
        """Reset grasp-specific state for the next nut."""
        self.phase = "approach"
        self._grasp_counter = 0
        self._attempts = 0

    def __call__(self, obs, env):
        ee = obs["robot0_eef_pos"]

        # Discover nuts and find next unplaced one
        nuts = []
        for name in _NUT_NAMES:
            key = f"{name}_pos"
            if key in obs:
                peg = _get_peg_pos(env, name)
                nuts.append((name, obs[key], peg))

        if not nuts:
            return np.zeros(7)

        placed = [self._nut_on_peg(n[1], n[2]) for n in nuts]

        target_nut = None
        for i, (name, pos, peg) in enumerate(nuts):
            if not placed[i]:
                target_nut = (name, pos, peg)
                break

        if target_nut is None:
            return np.zeros(7)

        nut_name, nut_pos, peg_pos = target_nut

        action = np.zeros(7)
        action[5] = _gripper_yaw_correction(obs)

        xy_dist = np.linalg.norm(ee[:2] - nut_pos[:2])
        z_diff = ee[2] - nut_pos[2]

        if self.phase == "approach":
            target = nut_pos.copy()
            target[2] = max(target[2] + 0.05, ee[2])
            action[:3] = reach_pos(ee, target)
            action[6] = -1.0
            if xy_dist < 0.02:
                self.phase = "descend"

        elif self.phase == "descend":
            target = nut_pos.copy()
            target[2] -= 0.01
            action[:3] = reach_pos(ee, target)
            action[6] = -1.0
            if z_diff < -0.005:
                self.phase = "grasp"
                self._grasp_counter = 0

        elif self.phase == "grasp":
            action[:3] = reach_pos(ee, nut_pos, gain=2.0)
            action[6] = 1.0
            self._grasp_counter += 1
            if self._grasp_counter >= _GRASP_STEPS:
                if _missed_grasp(obs):
                    self._attempts += 1
                    if self._attempts < self.max_attempts:
                        self.phase = "retry"
                    else:
                        self.phase = "lift_to_peg"
                else:
                    self.phase = "lift_to_peg"

        elif self.phase == "retry":
            target = nut_pos.copy()
            target[2] += 0.05
            action[:3] = reach_pos(ee, target)
            action[6] = -1.0
            if z_diff > 0.04:
                self.phase = "descend"

        elif self.phase == "lift_to_peg":
            target = peg_pos.copy()
            target[2] += 0.12
            action[:3] = reach_pos(ee, target)
            action[6] = 1.0
            above_peg_xy = np.linalg.norm(ee[:2] - peg_pos[:2]) < 0.015
            if above_peg_xy and ee[2] > peg_pos[2] + 0.08:
                self.phase = "lower_to_peg"

        elif self.phase == "lower_to_peg":
            target = peg_pos.copy()
            target[2] += 0.02
            action[:3] = reach_pos(ee, target, gain=5.0)
            action[6] = 1.0
            if ee[2] < target[2] + 0.01:
                self.phase = "release"

        elif self.phase == "release":
            action[6] = -1.0
            # After releasing, retract and reset for next nut
            if self._nut_on_peg(nut_pos, peg_pos):
                self.phase = "retract"

        elif self.phase == "retract":
            # Move up before approaching the next nut
            target = ee.copy()
            target[2] += 0.10
            action[:3] = reach_pos(ee, target)
            action[6] = -1.0
            if ee[2] > nut_pos[2] + 0.08:
                self._reset_grasp()

        return np.clip(action, -1, 1)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_nut_assembly_policy = NutAssemblyPolicy()

# All stateful policy instances for easy reset
_STATEFUL_POLICIES = {
    "Lift": _lift_policy,
    "Stack": _stack_policy,
    "PickPlaceSingle": _pickplace_policy,
    "NutAssemblySingle": _nut_single_policy,
    "NutAssembly": _nut_assembly_policy,
}


def get_scripted_policy(env_name):
    """Return the scripted policy callable for *env_name*.

    Returns:
        policy_fn(obs, env) -> np.ndarray of shape (7,)
    """
    policies = {
        "Lift": scripted_lift_policy,
        "Stack": scripted_stack_policy,
        "PickPlaceSingle": scripted_pickplace_policy,
        "Door": scripted_door_policy,
        "NutAssemblySingle": scripted_nut_single_policy,
        "NutAssembly": _nut_assembly_policy,
    }
    if env_name not in policies:
        raise ValueError(
            f"No scripted policy for '{env_name}'. "
            f"Available: {list(policies.keys())}"
        )
    return policies[env_name]


def reset_policy(env_name):
    """Reset stateful policy state (call before each new episode)."""
    policy = _STATEFUL_POLICIES.get(env_name)
    if policy is not None:
        policy.reset()
