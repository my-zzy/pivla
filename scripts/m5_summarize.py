#!/usr/bin/env python3
"""Encode recorded M5 cameras and summarize physical outcomes and timing."""
import argparse
import json
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('artifacts/M5'))
    args = parser.parse_args()
    root = args.root
    offline = json.loads((root/'offline/result.json').read_text())
    replay = json.loads((root/'replay/result.json').read_text())
    policy = json.loads((root/'closed_loop/trials/result.json').read_text())
    selected = [('replay', root/'replay', t) for t in replay['trials']]+[
        ('policy', root/'closed_loop/trials', t) for t in policy['trials']]
    fig, axes = plt.subplots(len(selected), 3, figsize=(15, 3*len(selected)), squeeze=False)
    summaries = []
    for row_index, (mode, directory, trial) in enumerate(selected):
        episode = directory/f'episode_{trial["episode_id"]:03d}'
        video = episode/'rollout.mp4'
        subprocess.run(['/home/user/miniconda3/envs/isaaclab/bin/ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
            '-framerate', '5', '-i', str(episode/'frames/%04d.png'), '-frames:v', str(trial['camera_frames']),
            '-c:v', 'libx264', '-crf', '20', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(video)], check=True)
        probe = subprocess.run(['/home/user/miniconda3/envs/isaaclab/bin/ffprobe', '-v', 'error',
            '-show_entries', 'stream=width,height,nb_frames,duration', '-of', 'json', str(video)],
            check=True, text=True, capture_output=True)
        integrity = json.loads(probe.stdout)['streams'][0]
        assert int(integrity['nb_frames']) == trial['camera_frames']
        rows = [json.loads(s) for s in (episode/'flight.jsonl').read_text().splitlines()]
        commands = [json.loads(s) for s in (episode/'actions.jsonl').read_text().splitlines()]
        time_s = np.array([r['time_s'] for r in rows])
        drone = np.array([r['drone_state'] for r in rows])
        block = np.array([r['block_state'] for r in rows])
        targets = np.array([r['target_position_w'] for r in rows])
        cfg = json.loads((episode/'config.json').read_text())
        ax = axes[row_index, 0]
        ax.plot(drone[:, 0], drone[:, 1], label='Drone')
        ax.plot(block[:, 0], block[:, 1], label='Block')
        ax.plot(targets[:, 0], targets[:, 1], ':', label='Command')
        ax.scatter(*cfg['task']['bin_position'][:2], marker='s', label='Bin')
        ax.set(xlabel='World X (m)', ylabel='World Y (m)',
               title=f'{mode} E{trial["episode_id"]} — {"success" if trial["success"] else trial["termination"]}')
        ax.axis('equal')
        ax = axes[row_index, 1]
        ax.plot(time_s, drone[:, 2], label='Drone Z')
        ax.plot(time_s, block[:, 2], label='Block Z')
        ax.plot(time_s, targets[:, 2], ':', label='Command Z')
        ax.set(xlabel='Simulation time (s)', ylabel='Height (m)')
        ax = axes[row_index, 2]
        ax.step([r['time_s'] for r in commands], [r['command'][6] for r in commands], where='post', label='Close command')
        ax.plot(time_s, [all(r['contacts']) for r in rows], alpha=.6, label='Both contacts')
        ax.set(xlabel='Simulation time (s)', ylabel='Boolean', ylim=(-.1, 1.1))
        delays = trial['delays']
        metrics = {k: v for k, v in trial.items() if k not in ('delays', 'final_observation')}
        metrics['video'] = str(video)
        metrics['video_integrity'] = integrity
        metrics['min_drone_z_m'] = float(drone[:, 2].min())
        metrics['max_tracking_error_m'] = float(np.linalg.norm(drone[:, :3]-targets, axis=1).max())
        pick_pose = np.array([*cfg['task']['block_position'][:2], cfg['task']['grasp_height_m']])
        pick_errors = np.linalg.norm(drone[:, :3]-pick_pose, axis=1)
        closest = int(pick_errors.argmin())
        metrics['closest_pick_pose'] = {'error_m': float(pick_errors[closest]),
            'time_s': float(time_s[closest]), 'position_w': drone[closest, :3].tolist(),
            'gripper_open': rows[closest]['command_open'], 'contacts': rows[closest]['contacts']}
        metrics['sim_to_wall_ratio'] = trial['duration_s']/trial['wall_elapsed_s']
        if delays:
            metrics['timing'] = {name: {'median': float(np.median([d[name] for d in delays])),
                                      'p95': float(np.percentile([d[name] for d in delays], 95)),
                                      'max': float(max(d[name] for d in delays))}
                                for name in ('inference_wall_s', 'roundtrip_wall_s', 'simulation_delay_s')}
            metrics['all_responses_had_physics_progress'] = all(d['physics_steps_during_wait'] > 0 for d in delays)
        summaries.append(metrics)
    for ax in axes.flat:
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(root/'rollout_metrics.png', dpi=120)
    totals = {}
    for name, ids in [('familiar', policy['settings']['familiar_episode_ids']), ('heldout', policy['settings']['heldout_episode_ids'])]:
        trials = [t for t in policy['trials'] if t['episode_id'] in ids]
        totals[name] = {'trials': len(trials), 'successes': sum(t['success'] for t in trials),
                        'lifts': sum(t['lifted'] for t in trials), 'grasps': sum(t['grasp_contact'] for t in trials)}
    summary = {'status': 'measured', 'checkpoint': policy['settings']['checkpoint'],
               'scope': policy['settings']['scope'], 'settings': policy['settings'],
               'offline': offline['groups'], 'closed_loop_totals': totals, 'trials': summaries}
    (root/'result.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    print(json.dumps(totals, indent=2))


if __name__ == '__main__':
    main()
