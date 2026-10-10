# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##


gym.register(
    id="Template-Airvla-Isaac-Direct-v0",
    entry_point=f"{__name__}.airvla_isaac_env:AirvlaIsaacEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.airvla_isaac_env_cfg:AirvlaIsaacEnvCfg",
    },
)