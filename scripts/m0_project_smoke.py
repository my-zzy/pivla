#!/usr/bin/env python3
"""Launch and step the generated direct-workflow example without installing it."""
import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
app = launcher.app

import gymnasium as gym
import torch
import airvla_isaac.tasks
from airvla_isaac.tasks.direct.airvla_isaac.airvla_isaac_env_cfg import AirvlaIsaacEnvCfg


def main():
    cfg = AirvlaIsaacEnvCfg()
    cfg.scene.num_envs = 1
    cfg.sim.device = args.device
    cfg.seed = 0
    cfg.robot_cfg.spawn.usd_path = str(Path(__file__).resolve().parents[1] / 'airvla_isaac/assets/m0/cartpole.usd')
    env = gym.make('Template-Airvla-Isaac-Direct-v0', cfg=cfg)
    try:
        obs, _ = env.reset()
        assert obs['policy'].shape == (1, 4)
        with torch.inference_mode():
            for _ in range(120):
                obs, reward, terminated, truncated, info = env.step(torch.zeros((1, 1), device=env.unwrapped.device))
                assert torch.isfinite(obs['policy']).all()
                assert torch.isfinite(reward).all()
        report = {'task': 'Template-Airvla-Isaac-Direct-v0', 'num_envs': 1, 'steps': 120,
                  'observation_shape': list(obs['policy'].shape), 'finite': True, 'status': 'passed'}
        output = Path(__file__).resolve().parents[1] / 'artifacts/M0/project_result.json'
        output.write_text(json.dumps(report, indent=2) + '\n')
        print('M0 project:', report, flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    try:
        main()
    finally:
        app.close(skip_cleanup=True)
