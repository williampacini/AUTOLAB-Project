"""Bridge module: register robosuite-native objects in LIBERO's object registry.

Import this module BEFORE creating any LIBERO OffScreenRenderEnv to make
robosuite objects (Cube, RoundNut, SquareNut, etc.) available in BDDL files.

Usage:
    import bddl_tasks.robosuite_bridge  # registers objects
    from libero.libero.envs import OffScreenRenderEnv
    env = OffScreenRenderEnv(bddl_file_name="my_task.bddl", ...)

The following objects become available for BDDL (:objects ...) sections:
    robosuite_cube        - The standard Lift cube
    robosuite_round_nut   - NutAssembly round nut
    robosuite_square_nut  - NutAssembly square nut
    robosuite_can         - Standard can object
    robosuite_bottle      - Standard bottle object
"""

from .register_objects import register_all

register_all()
