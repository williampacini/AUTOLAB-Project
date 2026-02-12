#!/usr/bin/env python3
"""Scripted expert policies for robosuite environments.

Each policy uses ground-truth object positions from the observation dictionary
(enabled by ``use_object_obs=True``) and a simple proportional controller to
generate expert demonstrations.  No learning or human teleoperation needed.

Supported environments:
  Lift, Stack, PickPlaceSingle, Door, NutAssemblySingle, NutAssembly

Gripper convention (robosuite Panda + OSC_POSE):
  -1 = open, +1 = close
"""

import numpy as np


# ---------------------------------------------------------------------------
# Common helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 1. Lift
# ---------------------------------------------------------------------------

def scripted_lift_policy(obs, env):
    """Move toward cube, grasp, lift.

    Obs keys: ``cube_pos``, ``robot0_eef_pos``.
    Based on ``colab_notebooks/robosuite_sim.ipynb`` cell-15.

    Uses separate XY/Z checks so the EEF fully descends to cube height
    before closing (robot0_eef_pos is at the gripper center, not fingertips).
    """
    ee = obs["robot0_eef_pos"]
    cube = obs["cube_pos"]

    action = np.zeros(7)

    xy_dist = np.linalg.norm(ee[:2] - cube[:2])
    z_diff = ee[2] - cube[2]  # positive = EEF above cube

    if xy_dist > 0.02:
        # Phase 1: Align XY from above
        approach = cube.copy()
        approach[2] += 0.05
        action[:3] = reach_pos(ee, approach)
        action[6] = -1.0  # gripper open
    elif z_diff > 0.01:
        # Phase 2: Descend to cube height (XY aligned, still above)
        action[:3] = reach_pos(ee, cube)
        action[6] = -1.0  # gripper open
    else:
        # Phase 3: Close gripper and lift
        action[2] = 1.0
        action[6] = 1.0

    return np.clip(action, -1, 1)


# ---------------------------------------------------------------------------
# 2. Stack
# ---------------------------------------------------------------------------

def scripted_stack_policy(obs, env):
    """Pick cubeA (red), stack on cubeB (green), release.

    Obs keys: ``cubeA_pos``, ``cubeB_pos``, ``robot0_eef_pos``,
    ``robot0_gripper_qpos``.
    """
    ee = obs["robot0_eef_pos"]
    cubeA = obs["cubeA_pos"]
    cubeB = obs["cubeB_pos"]
    grasped = _gripper_closed(obs)

    action = np.zeros(7)

    cubeA_above_B = cubeA[2] > cubeB[2] + 0.03
    above_B_xy = np.linalg.norm(ee[:2] - cubeB[:2]) < 0.02

    if not grasped and not cubeA_above_B:
        # Phase 1: reach and grasp cubeA
        xy_dist = np.linalg.norm(ee[:2] - cubeA[:2])
        z_diff = ee[2] - cubeA[2]
        if xy_dist > 0.02:
            approach = cubeA.copy()
            approach[2] += 0.05
            action[:3] = reach_pos(ee, approach)
            action[6] = -1.0  # open
        elif z_diff > 0.01:
            action[:3] = reach_pos(ee, cubeA)
            action[6] = -1.0  # open
        else:
            action[6] = 1.0  # close
    elif grasped and not (above_B_xy and ee[2] > cubeB[2] + 0.06):
        # Phase 2: lift and move above cubeB
        target = cubeB.copy()
        target[2] += 0.10
        action[:3] = reach_pos(ee, target)
        action[6] = 1.0  # keep closed
    elif grasped:
        # Phase 3: lower onto cubeB
        target = cubeB.copy()
        target[2] += 0.025
        if ee[2] > target[2] + 0.01:
            action[:3] = reach_pos(ee, target, gain=5.0)
            action[6] = 1.0
        else:
            action[6] = -1.0  # release
    else:
        # cubeA already above cubeB and not grasped → done, hold
        action[6] = -1.0

    return np.clip(action, -1, 1)


# ---------------------------------------------------------------------------
# 3. PickPlaceSingle
# ---------------------------------------------------------------------------

_PP_OBJECTS = ["Can", "Milk", "Bread", "Cereal", "can", "milk", "bread", "cereal"]


def _find_object_pos(obs):
    """Find the active object position in PickPlaceSingle."""
    for name in _PP_OBJECTS:
        key = f"{name}_pos"
        if key in obs:
            return obs[key]
    # Fallback: first 3 values of object-state
    if "object-state" in obs:
        return obs["object-state"][:3]
    raise KeyError("Cannot find object position in obs")


def _get_target_bin_pos(env):
    """Get target bin position from the simulation model."""
    # PickPlaceSingle target is visual_bin_body or bin2
    for name in ["bin2_body", "bin2", "visual_bin_body"]:
        try:
            bid = env.sim.model.body_name2id(name)
            return env.sim.data.body_xpos[bid].copy()
        except ValueError:
            continue
    # Fallback: hardcoded approximate position
    return np.array([0.18, 0.0, 0.82])


def scripted_pickplace_policy(obs, env):
    """Pick up object, place in target bin, release.

    Obs keys: ``{Obj}_pos`` or ``object-state``, ``robot0_eef_pos``,
    ``robot0_gripper_qpos``. Bin position from sim model.
    """
    ee = obs["robot0_eef_pos"]
    obj_pos = _find_object_pos(obs)
    bin_pos = _get_target_bin_pos(env)
    grasped = _gripper_closed(obs)

    action = np.zeros(7)

    target_above_bin = bin_pos.copy()
    target_above_bin[2] += 0.15

    obj_lifted = obj_pos[2] > 0.88  # above table

    if not grasped and not obj_lifted:
        # Phase 1: reach and grasp
        xy_dist = np.linalg.norm(ee[:2] - obj_pos[:2])
        z_diff = ee[2] - obj_pos[2]
        if xy_dist > 0.02:
            approach = obj_pos.copy()
            approach[2] += 0.05
            action[:3] = reach_pos(ee, approach)
            action[6] = -1.0
        elif z_diff > 0.01:
            action[:3] = reach_pos(ee, obj_pos)
            action[6] = -1.0
        else:
            action[6] = 1.0
    elif grasped and np.linalg.norm(ee[:2] - bin_pos[:2]) > 0.03:
        # Phase 2: lift and move above bin
        if ee[2] < obj_pos[2] + 0.10:
            action[2] = 1.0  # lift first
        else:
            action[:3] = reach_pos(ee, target_above_bin)
        action[6] = 1.0
    else:
        # Phase 3: lower into bin and release
        drop = bin_pos.copy()
        drop[2] += 0.05
        if ee[2] > drop[2] + 0.02:
            action[:3] = reach_pos(ee, drop, gain=5.0)
            action[6] = 1.0
        else:
            action[6] = -1.0

    return np.clip(action, -1, 1)


# ---------------------------------------------------------------------------
# 4. Door
# ---------------------------------------------------------------------------

def scripted_door_policy(obs, env):
    """Reach door handle, grasp, pull to open.

    Obs keys: ``handle_pos``, ``hinge_qpos``, ``robot0_eef_pos``.
    Adaptive pull direction: tracks hinge_qpos change over time.
    """
    ee = obs["robot0_eef_pos"]
    # Key name varies across robosuite versions
    handle = (obs.get("handle_pos")
              or obs.get("door_handle_pos")
              or obs.get("object-state", np.zeros(10))[:3])
    hinge = obs.get("hinge_qpos", np.zeros(1))
    if isinstance(hinge, np.ndarray):
        hinge = float(hinge.flat[0])

    action = np.zeros(7)

    dist = np.linalg.norm(ee - handle)

    if dist > 0.02:
        # Phase 1: reach handle
        action[:3] = reach_pos(ee, handle)
        action[6] = -1.0  # open gripper
    elif hinge < 0.25:
        # Phase 2: grasp and pull
        action[6] = 1.0
        # Pull toward robot base (negative Y direction generally opens door)
        # Also try rotating the handle
        action[1] = -1.0
        action[2] = 0.15  # slight upward to keep contact
    else:
        # Door open enough, keep holding
        action[6] = 1.0

    return np.clip(action, -1, 1)


# ---------------------------------------------------------------------------
# 5. NutAssemblySingle
# ---------------------------------------------------------------------------

_NUT_NAMES = ["RoundNut", "SquareNut"]


def _find_nut_pos(obs, names=None):
    """Return (nut_name, nut_pos) for the first nut found in obs."""
    names = names or _NUT_NAMES
    for name in names:
        key = f"{name}_pos"
        if key in obs:
            return name, obs[key]
    # Fallback
    if "object-state" in obs:
        return "unknown", obs["object-state"][:3]
    raise KeyError("Cannot find nut position in obs")


def _get_peg_pos(env, nut_name):
    """Get the target peg position for *nut_name* from the sim model.

    Convention in robosuite NutAssembly:
      - peg1 / peg1_table → SquareNut target
      - peg2 / peg2_table → RoundNut target
    """
    if "Square" in nut_name:
        candidates = ["peg1", "peg1_table"]
    else:
        candidates = ["peg2", "peg2_table"]

    for name in candidates:
        try:
            bid = env.sim.model.body_name2id(name)
            return env.sim.data.body_xpos[bid].copy()
        except ValueError:
            continue

    # Fallback: try generic names
    for i in range(1, 5):
        for prefix in ["peg", "peg_"]:
            try:
                bid = env.sim.model.body_name2id(f"{prefix}{i}")
                return env.sim.data.body_xpos[bid].copy()
            except ValueError:
                continue

    raise RuntimeError(f"Cannot find peg body for {nut_name} in sim model")


def scripted_nut_single_policy(obs, env):
    """Pick up the nut, align above the correct peg, insert, release.

    Obs keys: ``{RoundNut,SquareNut}_pos``, ``robot0_eef_pos``,
    ``robot0_gripper_qpos``.  Peg position from ``env.sim``.
    """
    ee = obs["robot0_eef_pos"]
    nut_name, nut_pos = _find_nut_pos(obs)
    peg_pos = _get_peg_pos(env, nut_name)
    grasped = _gripper_closed(obs)

    action = np.zeros(7)

    above_peg_xy = np.linalg.norm(ee[:2] - peg_pos[:2]) < 0.015
    nut_lifted = nut_pos[2] > peg_pos[2] + 0.05

    if not grasped and not nut_lifted:
        # Phase 1: reach and grasp nut
        xy_dist = np.linalg.norm(ee[:2] - nut_pos[:2])
        z_diff = ee[2] - nut_pos[2]
        if xy_dist > 0.02:
            approach = nut_pos.copy()
            approach[2] = max(approach[2] + 0.05, ee[2])
            action[:3] = reach_pos(ee, approach)
            action[6] = -1.0
        elif z_diff > 0.01:
            action[:3] = reach_pos(ee, nut_pos)
            action[6] = -1.0
        else:
            action[6] = 1.0
    elif grasped and not above_peg_xy:
        # Phase 2: lift and align above peg
        target = peg_pos.copy()
        target[2] += 0.12
        action[:3] = reach_pos(ee, target)
        action[6] = 1.0
    elif grasped:
        # Phase 3: lower onto peg
        insert = peg_pos.copy()
        insert[2] += 0.02
        if ee[2] > insert[2] + 0.01:
            action[:3] = reach_pos(ee, insert, gain=5.0)
            action[6] = 1.0
        else:
            action[6] = -1.0  # release
    else:
        # Nut is on peg and released
        action[6] = -1.0

    return np.clip(action, -1, 1)


# ---------------------------------------------------------------------------
# 6. NutAssembly (two nuts — stateful)
# ---------------------------------------------------------------------------

class NutAssemblyPolicy:
    """Stateful scripted policy for two-nut assembly.

    Sequentially inserts each nut onto its matching peg.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.current_idx = 0  # 0 = first nut, 1 = second nut
        self.phase = "reach"
        self._grasp_steps = 0

    def _nut_on_peg(self, nut_pos, peg_pos):
        """Heuristic: nut is near peg XY and close to peg height."""
        xy_dist = np.linalg.norm(nut_pos[:2] - peg_pos[:2])
        z_close = abs(nut_pos[2] - peg_pos[2]) < 0.04
        return xy_dist < 0.03 and z_close

    def __call__(self, obs, env):
        ee = obs["robot0_eef_pos"]

        # Discover both nuts
        nuts = []
        for name in _NUT_NAMES:
            key = f"{name}_pos"
            if key in obs:
                peg = _get_peg_pos(env, name)
                nuts.append((name, obs[key], peg))

        if not nuts:
            return np.zeros(7)

        # Check which nuts are already placed
        placed = [self._nut_on_peg(n[1], n[2]) for n in nuts]

        # Find next unplaced nut
        target_nut = None
        for i, (name, pos, peg) in enumerate(nuts):
            if not placed[i]:
                target_nut = (name, pos, peg)
                break

        if target_nut is None:
            # Both placed — done
            return np.zeros(7)

        nut_name, nut_pos, peg_pos = target_nut
        grasped = _gripper_closed(obs)

        action = np.zeros(7)
        above_peg_xy = np.linalg.norm(ee[:2] - peg_pos[:2]) < 0.015
        nut_lifted = nut_pos[2] > peg_pos[2] + 0.05

        if not grasped and not nut_lifted:
            # Phase 1: reach and grasp
            xy_dist = np.linalg.norm(ee[:2] - nut_pos[:2])
            z_diff = ee[2] - nut_pos[2]
            if xy_dist > 0.02:
                approach = nut_pos.copy()
                approach[2] = max(approach[2] + 0.05, ee[2])
                action[:3] = reach_pos(ee, approach)
                action[6] = -1.0
            elif z_diff > 0.01:
                action[:3] = reach_pos(ee, nut_pos)
                action[6] = -1.0
            else:
                action[6] = 1.0
                self._grasp_steps += 1
        elif grasped and not above_peg_xy:
            # Phase 2: lift and align
            target = peg_pos.copy()
            target[2] += 0.12
            action[:3] = reach_pos(ee, target)
            action[6] = 1.0
            self._grasp_steps = 0
        elif grasped:
            # Phase 3: lower and insert
            insert = peg_pos.copy()
            insert[2] += 0.02
            if ee[2] > insert[2] + 0.01:
                action[:3] = reach_pos(ee, insert, gain=5.0)
                action[6] = 1.0
            else:
                action[6] = -1.0  # release
                self._grasp_steps = 0
        else:
            action[6] = -1.0

        return np.clip(action, -1, 1)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

# Pre-instantiate the stateful policy so reset() can be called externally
_nut_assembly_policy = NutAssemblyPolicy()


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
    if env_name == "NutAssembly":
        _nut_assembly_policy.reset()
