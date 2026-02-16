"""Register robosuite-native MuJoCo objects in LIBERO's OBJECTS_DICT.

This module creates LIBERO-compatible wrappers around robosuite's built-in
objects (Cube, RoundNutObject, SquareNutObject, etc.) so they can be
referenced in BDDL task files.

LIBERO discovers objects via its OBJECTS_DICT registry. Each entry maps a
category string (e.g. "akita_black_bowl") to an object class. We add new
entries for robosuite objects using the "robosuite_*" prefix to avoid
collisions with existing LIBERO objects.
"""

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_REGISTERED = False


def register_all():
    """Register all robosuite objects in LIBERO's registry.

    Safe to call multiple times — only registers once.
    """
    global _REGISTERED
    if _REGISTERED:
        return
    _REGISTERED = True

    try:
        from libero.libero.envs.base_object import OBJECTS_DICT
    except ImportError:
        logger.warning(
            "LIBERO not installed — skipping robosuite object registration. "
            "Install with: pip install -e LIBERO/"
        )
        return

    try:
        import robosuite
    except ImportError:
        logger.warning(
            "robosuite not installed — skipping object registration. "
            "Install with: pip install robosuite==1.4.1"
        )
        return

    # Get robosuite's asset directory
    robosuite_root = Path(robosuite.__file__).parent
    assets_dir = robosuite_root / "models" / "assets"

    n_registered = 0

    # --- Cube (Lift task) ---
    n_registered += _register_cube(OBJECTS_DICT, assets_dir)

    # --- Round Nut (NutAssembly) ---
    n_registered += _register_round_nut(OBJECTS_DICT, assets_dir)

    # --- Square Nut (NutAssembly) ---
    n_registered += _register_square_nut(OBJECTS_DICT, assets_dir)

    # --- Can ---
    n_registered += _register_can(OBJECTS_DICT, assets_dir)

    # --- Bottle ---
    n_registered += _register_bottle(OBJECTS_DICT, assets_dir)

    logger.info(f"Registered {n_registered} robosuite objects in LIBERO")


def _register_cube(objects_dict, assets_dir):
    """Register robosuite's Lift cube."""
    try:
        from robosuite.models.objects import BoxObject

        class RobosuiteCube:
            """LIBERO-compatible wrapper for robosuite's Lift cube.

            LIBERO expects objects with specific interfaces. This wrapper
            creates a BoxObject with the same dimensions as the Lift cube
            and exposes the interface LIBERO's placement system needs.
            """

            def __init__(self, name="robosuite_cube", joints=None, **kwargs):
                self._obj = BoxObject(
                    name=name,
                    size_min=[0.020, 0.020, 0.020],
                    size_max=[0.022, 0.022, 0.022],
                    rgba=[1, 0, 0, 1],  # Red cube like Lift
                    density=100,
                )
                self.name = name
                self.joints = joints or [{"type": "free", "damping": "0.0005"}]
                self.rotation = None
                self.rotation_axis = "z"

            @property
            def bottom_offset(self):
                return self._obj.bottom_offset

            @property
            def top_offset(self):
                return self._obj.top_offset

            @property
            def horizontal_radius(self):
                return self._obj.horizontal_radius

            def get_obj(self):
                return self._obj

        objects_dict["robosuite_cube"] = RobosuiteCube
        logger.debug("Registered: robosuite_cube")
        return 1
    except Exception as e:
        logger.warning(f"Failed to register robosuite_cube: {e}")
        return 0


def _register_round_nut(objects_dict, assets_dir):
    """Register robosuite's round nut for NutAssembly."""
    try:
        from robosuite.models.objects import MujocoXMLObject
        from robosuite.utils.mjcf_utils import xml_path_completion

        xml_path = xml_path_completion("objects/round-nut.xml")

        class RobosuiteRoundNut:
            """LIBERO wrapper for robosuite's round nut."""

            def __init__(self, name="robosuite_round_nut", joints=None, **kwargs):
                self._obj = MujocoXMLObject(
                    fname=xml_path,
                    name=name,
                    joints=[{"type": "free", "damping": "0.0005"}],
                    duplicate_collision_geoms=True,
                )
                self.name = name
                self.joints = joints or [{"type": "free", "damping": "0.0005"}]
                self.rotation = None
                self.rotation_axis = "z"

            @property
            def bottom_offset(self):
                return self._obj.bottom_offset

            @property
            def top_offset(self):
                return self._obj.top_offset

            @property
            def horizontal_radius(self):
                return self._obj.horizontal_radius

            def get_obj(self):
                return self._obj

        objects_dict["robosuite_round_nut"] = RobosuiteRoundNut
        logger.debug("Registered: robosuite_round_nut")
        return 1
    except Exception as e:
        logger.warning(f"Failed to register robosuite_round_nut: {e}")
        return 0


def _register_square_nut(objects_dict, assets_dir):
    """Register robosuite's square nut for NutAssembly."""
    try:
        from robosuite.models.objects import MujocoXMLObject
        from robosuite.utils.mjcf_utils import xml_path_completion

        xml_path = xml_path_completion("objects/square-nut.xml")

        class RobosuiteSquareNut:
            """LIBERO wrapper for robosuite's square nut."""

            def __init__(self, name="robosuite_square_nut", joints=None, **kwargs):
                self._obj = MujocoXMLObject(
                    fname=xml_path,
                    name=name,
                    joints=[{"type": "free", "damping": "0.0005"}],
                    duplicate_collision_geoms=True,
                )
                self.name = name
                self.joints = joints or [{"type": "free", "damping": "0.0005"}]
                self.rotation = None
                self.rotation_axis = "z"

            @property
            def bottom_offset(self):
                return self._obj.bottom_offset

            @property
            def top_offset(self):
                return self._obj.top_offset

            @property
            def horizontal_radius(self):
                return self._obj.horizontal_radius

            def get_obj(self):
                return self._obj

        objects_dict["robosuite_square_nut"] = RobosuiteSquareNut
        logger.debug("Registered: robosuite_square_nut")
        return 1
    except Exception as e:
        logger.warning(f"Failed to register robosuite_square_nut: {e}")
        return 0


def _register_can(objects_dict, assets_dir):
    """Register robosuite's can object."""
    try:
        from robosuite.models.objects import MujocoXMLObject
        from robosuite.utils.mjcf_utils import xml_path_completion

        xml_path = xml_path_completion("objects/can.xml")

        class RobosuiteCan:
            """LIBERO wrapper for robosuite's can object."""

            def __init__(self, name="robosuite_can", joints=None, **kwargs):
                self._obj = MujocoXMLObject(
                    fname=xml_path,
                    name=name,
                    joints=[{"type": "free", "damping": "0.0005"}],
                    duplicate_collision_geoms=True,
                )
                self.name = name
                self.joints = joints or [{"type": "free", "damping": "0.0005"}]
                self.rotation = None
                self.rotation_axis = "z"

            @property
            def bottom_offset(self):
                return self._obj.bottom_offset

            @property
            def top_offset(self):
                return self._obj.top_offset

            @property
            def horizontal_radius(self):
                return self._obj.horizontal_radius

            def get_obj(self):
                return self._obj

        objects_dict["robosuite_can"] = RobosuiteCan
        logger.debug("Registered: robosuite_can")
        return 1
    except Exception as e:
        logger.warning(f"Failed to register robosuite_can: {e}")
        return 0


def _register_bottle(objects_dict, assets_dir):
    """Register robosuite's bottle object."""
    try:
        from robosuite.models.objects import MujocoXMLObject
        from robosuite.utils.mjcf_utils import xml_path_completion

        xml_path = xml_path_completion("objects/bottle.xml")

        class RobosuiteBottle:
            """LIBERO wrapper for robosuite's bottle object."""

            def __init__(self, name="robosuite_bottle", joints=None, **kwargs):
                self._obj = MujocoXMLObject(
                    fname=xml_path,
                    name=name,
                    joints=[{"type": "free", "damping": "0.0005"}],
                    duplicate_collision_geoms=True,
                )
                self.name = name
                self.joints = joints or [{"type": "free", "damping": "0.0005"}]
                self.rotation = None
                self.rotation_axis = "z"

            @property
            def bottom_offset(self):
                return self._obj.bottom_offset

            @property
            def top_offset(self):
                return self._obj.top_offset

            @property
            def horizontal_radius(self):
                return self._obj.horizontal_radius

            def get_obj(self):
                return self._obj

        objects_dict["robosuite_bottle"] = RobosuiteBottle
        logger.debug("Registered: robosuite_bottle")
        return 1
    except Exception as e:
        logger.warning(f"Failed to register robosuite_bottle: {e}")
        return 0
