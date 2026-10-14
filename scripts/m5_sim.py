#!/usr/bin/env python3
"""M5 simulator entry point; inference stays in a separate LeRobot process."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import traceback
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--mode', choices=['replay', 'policy'], required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--ipc', type=Path)
AppLauncher.add_app_launcher_args(parser)
parser.add_argument('--config', type=Path, default=Path('configs/eval.yaml'))
args = parser.parse_args()
if not args.enable_cameras:
    parser.error('M5 requires --enable_cameras')
if args.mode == 'policy' and args.ipc is None:
    parser.error('Policy mode requires an inference worker IPC directory')
app = AppLauncher(args).app

import torch
import yaml
from airvla_isaac.sim.scenario_env import ScenarioEnv
from airvla_isaac.eval.evaluate import run_trial, save


def merge(base, updates):
    base = copy.deepcopy(base)
    for key, value in updates.items():
        base[key] = merge(base[key], value) if isinstance(value, dict) and key in base else value
    return base


def main():
    args.output.mkdir(parents=True, exist_ok=False)
    settings = yaml.safe_load(args.config.read_text())
    manifest = json.loads(Path(settings['manifest']).read_text())
    save(args.output/'settings.json', settings)
    selected = settings['replay_episode_ids'] if args.mode == 'replay' else settings['familiar_episode_ids']+settings['heldout_episode_ids']
    base = merge(yaml.safe_load(Path('configs/sim.yaml').read_text()), yaml.safe_load(Path('configs/manipulation.yaml').read_text()))
    env = ScenarioEnv(base, args.device, True)
    results = []
    for ep in selected:
        result = run_trial(env, manifest['episodes'][ep], settings, args.output/f'episode_{ep:03d}', args.mode, args.ipc)
        results.append(result)
        save(args.output/'result.json', {'status': 'measured', 'settings': settings, 'trials': results})
        if result['termination'] == 'error':
            # An interface/inference error needs investigation before more trials.
            raise RuntimeError(result['error'])
        if args.mode == 'replay' and not (result['success'] and result['max_replay_command_error'] < 1e-6
                                        and result['replay_action_count_matches']):
            raise RuntimeError('Expert replay failed: inspect conversion and physics before running learned control')


if __name__ == '__main__':
    try:
        with torch.inference_mode():
            main()
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush(); sys.stderr.flush()
        os._exit(1)
    app.close(skip_cleanup=True)
