#!/usr/bin/env python3
"""Plot recorded flight and encode the three-camera video without launching Sim."""
import argparse
import json
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/M1')
args = parser.parse_args()
out = args.output
config = json.loads((out / 'config.json').read_text())
rows = [json.loads(line) for line in (out / 'flight.jsonl').read_text().splitlines()]
t = np.array([row['time_s'] for row in rows])
p = np.array([row['position_w'] for row in rows])
target = np.array(config['target']['position'])
force = np.array([row['force_b_n'] for row in rows])
torque = np.array([row['torque_b_nm'] for row in rows])
fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
axes[0].plot(t, p[:, 2], label='Measured altitude')
axes[0].axhline(target[2], color='black', linestyle='--', label='Target')
axes[0].set_ylabel('Altitude (m)')
axes[1].plot(t, np.linalg.norm(p - target, axis=1), label='Position error')
axes[1].axhline(config['position_tolerance_m'], color='red', linestyle='--', label='M1 limit')
axes[1].set_ylabel('Error (m)')
axes[2].plot(t, force[:, 2], label='Body +Z thrust')
axes[2].axhline(config['drone']['mass_kg'] * config['gravity'], color='black', linestyle='--', label='Weight')
axes[2].set_ylabel('Thrust (N)')
for i, axis in enumerate('xyz'):
    axes[3].plot(t, torque[:, i], label=axis)
axes[3].set_ylabel('Body torque (N m)')
axes[3].set_xlabel('Simulation time (s)')
for axis in axes:
    axis.axvspan(0, config['settle_s'], color='grey', alpha=.12)
    axis.axvline(config['settle_s'], color='grey', linestyle=':')
    axis.grid(alpha=.2)
    axis.legend(loc='upper right')
fig.suptitle('M1 force-driven hover | shaded region: settling')
fig.tight_layout()
fig.savefig(out / 'hover_plots.png', dpi=140)
plt.close(fig)
ffmpeg = '/home/user/miniconda3/envs/isaaclab/bin/ffmpeg'
result = json.loads((out / 'result.json').read_text())
subprocess.run([ffmpeg, '-hide_banner', '-loglevel', 'error', '-y',
                '-framerate', str(config['camera_hz']), '-i', str(out / 'frames/%04d.png'),
                '-frames:v', str(result['camera_frames_per_stream']),
                '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '20',
                '-movflags', '+faststart', str(out / 'hover.mp4')], check=True)
print(out / 'hover_plots.png')
print(out / 'hover.mp4')
