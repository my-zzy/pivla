#!/usr/bin/env python3
"""Plot contact/load evidence and encode an M2 episode without starting Sim."""
import argparse
import json
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, default=ROOT/'artifacts/M2')
parser.add_argument('--episode', type=int, default=0)
args = parser.parse_args()
output = args.output
episode = output / f'episode_{args.episode:03d}'
config = json.loads((output/'config.json').read_text())
rows = [json.loads(line) for line in (episode/'flight.jsonl').read_text().splitlines()]
result = json.loads((episode/'result.json').read_text())
t = np.array([r['time_s'] for r in rows])
drone = np.array([r['drone_state'] for r in rows])
block = np.array([r['block_state'] for r in rows])
target = np.array([r['target_position_w'] for r in rows])
force = np.array([r['force_b_n'][2] for r in rows])
empty_mass = config['drone']['mass_kg'] + 2*config['gripper']['finger_mass_kg']
fig, axes = plt.subplots(5, 1, figsize=(13, 13), sharex=True)
axes[0].plot(t, drone[:, 2], label='Drone altitude')
axes[0].plot(t, target[:, 2], '--', label='Commanded altitude')
axes[0].plot(t, block[:, 2], label='Block altitude')
axes[0].axhline(config['task']['table_top_m'], color='grey', linestyle=':', label='Table top')
axes[0].set_ylabel('Altitude (m)')
axes[1].plot(t, drone[:, 0], label='Drone X')
axes[1].plot(t, block[:, 0], label='Block X')
axes[1].plot(t, target[:, 0], '--', label='Commanded X')
axes[1].set_ylabel('Position X (m)')
axes[2].plot(t, force, label='Body Z thrust')
axes[2].axhline(empty_mass*config['gravity'], color='grey', linestyle='--', label='Empty weight')
axes[2].axhline((empty_mass+config['task']['block_mass_kg'])*config['gravity'],
                color='orange', linestyle='--', label='Loaded weight')
axes[2].set_ylabel('Thrust (N)')
axes[3].plot(t, [r['aperture_m'] for r in rows], label='Measured finger aperture')
axes[3].axhline(config['task']['block_size_m'][1], color='grey', linestyle='--', label='Block width')
axes[3].set_ylabel('Aperture (m)')
contacts = np.array([r['contact_force_w_n'] for r in rows])
for index, name in enumerate(('Left', 'Right')):
    axes[4].plot(t, np.linalg.norm(contacts[:, index], axis=1), label=f'{name} block contact')
axes[4].set_ylabel('Normal contact (N)')
axes[4].set_xlabel('Simulation time (s)')
start = 0.
for i, event in enumerate(result['events']):
    stop = event['end_time_s']
    for axis in axes:
        if i % 2 == 0:
            axis.axvspan(start, stop, color='grey', alpha=.09)
        axis.axvline(stop, color='grey', alpha=.3, linewidth=.7)
    axes[0].text((start+stop)/2, 1.02, event['stage'], transform=axes[0].get_xaxis_transform(),
                 ha='center', fontsize=8, rotation=25)
    start = stop
for axis in axes:
    axis.grid(alpha=.2)
    axis.legend(loc='upper right', fontsize=8)
fig.suptitle('M2 physical grasp and payload transport | unchanged flight gains', y=.995)
fig.tight_layout()
fig.savefig(output/'load_contact_plots.png', dpi=140)
plt.close(fig)
if result['camera_frames_per_stream']:
    subprocess.run(['/home/user/miniconda3/envs/isaaclab/bin/ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
        '-framerate', str(config['camera_hz']), '-i', str(episode/'frames/%04d.png'),
        '-frames:v', str(result['camera_frames_per_stream']), '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
        '-crf', '20', '-movflags', '+faststart', str(output/'pick_place.mp4')], check=True)
    print(output/'pick_place.mp4')
print(output/'load_contact_plots.png')
