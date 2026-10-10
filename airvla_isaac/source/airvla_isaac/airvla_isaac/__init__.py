# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Python module serving as a project/extension template.
"""

def register_simulation_extensions():
    """Register simulator tasks explicitly after AppLauncher starts Isaac Sim.

    Dataset and policy utilities can then be imported in the separate LeRobot
    environment without loading simulator-only dependencies.
    """
    from . import tasks, ui_extension_example
    return tasks, ui_extension_example
