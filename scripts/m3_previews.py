#!/usr/bin/env python3
"""Encode five full raw episodes and plot independently decoded action chunks."""
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys

# Reuse the installed plotting stack; do not install matplotlib into LeRobot.
# The caller's read-only sandbox and project-local cache settings stay in force.
if importlib.util.find_spec('matplotlib') is None:
    plotting_python = '/home/user/miniconda3/envs/isaaclab/bin/python'
    if sys.executable == plotting_python:
        raise RuntimeError('The existing plotting environment has no matplotlib')
    print('Using existing isaaclab Python for offline plots/video; Isaac Sim is not launched.', flush=True)
    os.execv(plotting_python, [plotting_python, '-B', str(Path(__file__).resolve()), *sys.argv[1:]])

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from airvla_isaac.policy.action_codec import decode_chunk, encode_chunk, episode_chunk

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--raw', type=Path, default=Path('artifacts/M3'))
parser.add_argument('--episode-ids', nargs='+', type=int, help='Preview a subset during collection; default is the five schema episodes.')
args = parser.parse_args()
raw = args.raw
schema = json.loads((raw/'collection_manifest.json').read_text())['schema']
dest = raw/'previews'
dest.mkdir(exist_ok=True)
selected = args.episode_ids if args.episode_ids is not None else schema['preview_episodes']
fig, axes = plt.subplots(len(selected), 2, figsize=(13, 3*len(selected)), squeeze=False)
records = []
for row_index, ep_id in enumerate(selected):
    episode = raw/f'episode_{ep_id:03d}'
    result = json.loads((episode/'result.json').read_text())
    rows = [json.loads(x) for x in (episode/'samples.jsonl').read_text().splitlines()]
    actions = np.array([r['action'] for r in rows], dtype=np.float32)
    states = np.array([r['state'] for r in rows], dtype=np.float32)
    times = np.array([r['time_s'] for r in rows])
    restored = np.zeros_like(actions)
    for start in range(0,len(rows),schema['chunk_size']):
        chunk, valid = episode_chunk(actions,start,schema['chunk_size'])
        decoded = decode_chunk(encode_chunk(chunk,states[start],valid),states[start],valid)
        restored[start:start+valid.sum()] = decoded[valid]
    for j,label in enumerate(('X','Y','Z')):
        axes[row_index,0].plot(times, actions[:,j], label=f'{label} raw')
        axes[row_index,0].plot(times[::5],restored[::5,j],'.',ms=2,label=f'{label} decoded')
    axes[row_index,0].set_ylabel(f'Episode {ep_id}\nSetpoint (m)')
    axes[row_index,1].plot(times,actions[:,6],label='Gripper raw')
    axes[row_index,1].plot(times[::5],restored[::5,6],'.',ms=2,label='Gripper decoded')
    axes[row_index,1].plot(times,states[:,7],label='Measured aperture (m)')
    axes[row_index,1].set_ylabel('Gripper command / aperture')
    video = dest/f'episode_{ep_id:03d}.mp4'
    subprocess.run(['/home/user/miniconda3/envs/isaaclab/bin/ffmpeg','-hide_banner','-loglevel','error','-y',
                    '-framerate',str(schema['camera_hz']),'-i',str(episode/'frames/%04d.png'),
                    '-frames:v',str(result['camera_frames_per_stream']),'-c:v','libx264','-crf','20',
                    '-pix_fmt','yuv420p','-movflags','+faststart',str(video)],check=True)
    # Every recorded frame, in order: down and external views side by side.
    # Split into two sheets so individual frames remain inspectable at original resolution.
    sheets = []
    for part,start in enumerate(range(0,result['camera_frames_per_stream'],144)):
        count = min(144,result['camera_frames_per_stream']-start)
        sheet = Image.new('RGB',(12*256,math.ceil(count/12)*146),'#172132')
        draw = ImageDraw.Draw(sheet)
        for k in range(count):
            index = start+k
            x,y = (k%12)*256,(k//12)*146
            for col,name in enumerate(('down','external')):
                with Image.open(episode/f'frames/{name}/{index:04d}.png') as im:
                    sheet.paste(im.resize((128,128)),(x+col*128,y+18))
            draw.text((x+4,y+2),f'E{ep_id}  {index/schema["camera_hz"]:.1f}s',fill='white')
        path = dest/f'episode_{ep_id:03d}_all_frames_{part}.jpg'
        sheet.save(path,quality=90)
        sheets.append(str(path))
    records.append({'episode_id':ep_id,'video':str(video),'frames':result['camera_frames_per_stream'],
                    'contact_sheets_all_frames':sheets,'max_action_error':float(np.max(np.abs(restored-actions)))})
for ax in axes.flat:
    ax.grid(alpha=.2)
    ax.legend(fontsize=6,loc='best')
for ax in axes[-1]:
    ax.set_xlabel('Simulation time (s)')
fig.suptitle('M3: absolute expert actions vs decoded chunks, with one observed anchor per chunk')
fig.tight_layout()
fig.savefig(raw/'action_reconstruction.png',dpi=130)
(raw/'preview_manifest.json').write_text(json.dumps({'status':'complete' if selected==schema['preview_episodes'] else 'partial',
    'episodes':records,'review_method':'All frames in chronological down/external contact sheets, plus full trajectory reconstruction; videos retain all three streams.'},indent=2)+'\n')
print(f'Saved {len(selected)} videos, all-frame contact sheets, and reconstruction plot')
