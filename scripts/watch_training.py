#!/usr/bin/env python3
"""Read-only live view of a running or completed training directory. No GPU imports."""
import argparse
from pathlib import Path
import time

from rich.live import Live
from training_ui import TrainingDisplay, read_json, tail_records


def snapshot(output, display):
    config = read_json(output / 'run_config.json')
    settings = config.get('settings', {})
    result = read_json(output / 'result.json')
    display.total = result.get('probe_steps', config.get('probe_steps') or settings.get('steps', 0))
    display.batch = settings.get('batch_size', 0)
    display.records.clear()
    display.records.extend(tail_records(output / 'loss.jsonl'))
    evaluations = tail_records(output / 'evaluation.jsonl', limit=1)
    display.evaluation = evaluations[-1] if evaluations else {}
    status = read_json(output / 'status.json')
    state = status.get('status', 'waiting')
    phase = read_json(output / 'progress.json').get('phase')
    display.phase_name = {'passed': 'Finished · passed', 'failed': 'Failed'}.get(
        state, phase or ('Running' if state == 'running' else 'Waiting for startup'))
    try:
        age = max(0, time.time() - (output / 'loss.jsonl').stat().st_mtime)
        display.note = f'Last training metric {age:.0f}s ago'
        if state == 'running' and age > 30:
            display.note += ' · no recent step; may be evaluating, saving, or waiting'
    except OSError:
        display.note = 'No optimizer metrics yet; startup/evaluation may still be running.'
    if status.get('error'):
        display.note = status['error']
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--once', action='store_true', help='Print one snapshot and exit')
    parser.add_argument('--interval', type=float, default=1.)
    args = parser.parse_args()
    if not args.output.is_dir():
        parser.error('Output directory does not exist')
    if args.interval <= 0:
        parser.error('--interval must be positive')
    display = TrainingDisplay(0, 0, args.output)
    state = snapshot(args.output, display)
    if args.once or not display.console.is_terminal or state in ('passed', 'failed'):
        display.console.print(display.render())
        return
    # Ctrl-C affects only this log reader, never the training process.
    try:
        with Live(display.render(), console=display.console, auto_refresh=False) as live:
            while True:
                state = snapshot(args.output, display)
                live.update(display.render(), refresh=True)
                if state in ('passed', 'failed'):
                    break
                time.sleep(args.interval)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
