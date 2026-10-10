#!/usr/bin/env python3
"""Supervise separately sandboxed LeRobot inference and Isaac Sim processes."""
import argparse
import json
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('artifacts/M5/closed_loop'))
    args = parser.parse_args()
    replay = json.loads(Path('artifacts/M5/replay/result.json').read_text())
    if not all(t['success'] and t['max_replay_command_error'] < 1e-6 and t['replay_action_count_matches']
               for t in replay['trials']):
        raise RuntimeError('Passing expert replay is required first')
    args.output.mkdir(parents=True, exist_ok=False)
    ipc = args.output/'ipc'
    ipc.mkdir()
    worker = None
    with (args.output/'worker.log').open('w') as log:
        try:
            worker = subprocess.Popen(['bash', 'scripts/run_lerobot.sh', 'scripts/m5_worker.py', '--ipc', str(ipc)],
                                      stdout=log, stderr=subprocess.STDOUT)
            started = time.monotonic()
            while not (ipc/'ready.json').exists():
                if worker.poll() is not None or time.monotonic()-started > 180:
                    raise RuntimeError('Worker failed to become ready; see worker.log')
                time.sleep(.2)
            with (args.output/'sim.log').open('w') as simlog:
                subprocess.run(['bash', 'scripts/run_isaac.sh', 'scripts/m5_sim.py', '--mode', 'policy',
                    '--enable_cameras', '--ipc', str(ipc), '--output', str(args.output/'trials')],
                    stdout=simlog, stderr=subprocess.STDOUT, check=True)
        finally:
            (ipc/'stop').touch()
            if worker is not None:
                try:
                    worker.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    worker.terminate()
                    worker.wait(timeout=10)


if __name__ == '__main__':
    main()
