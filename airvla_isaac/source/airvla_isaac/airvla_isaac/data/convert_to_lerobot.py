"""Export split-local LeRobot datasets with absolute actions and lossless RGB."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from ..policy.action_codec import encode_chunk, episode_chunk


def read_samples(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def compute_normalization(raw, manifest):
    schema = manifest['schema']
    states, actions = [], []
    ids = []
    for item in manifest['episodes']:
        if item['split'] != 'train':
            continue
        ids.append(item['episode_id'])
        rows = read_samples(raw/f'episode_{item["episode_id"]:03d}/samples.jsonl')
        state = np.array([r['state'] for r in rows], dtype=np.float32)
        absolute = np.array([r['action'] for r in rows], dtype=np.float32)
        states.append(state)
        for i in range(len(rows)):
            chunk, valid = episode_chunk(absolute, i, schema['chunk_size'])
            actions.append(encode_chunk(chunk, state[i], valid)[valid])
    stats = {'source_split': 'train', 'source_episode_ids': ids,
             'representation': 'One observed pose anchor per 50-step relative action chunk; invalid future steps excluded.'}
    for key, arrays in [('observation.state', states), ('action', actions)]:
        values = np.concatenate(arrays).astype(np.float64)
        mean, std = values.mean(axis=0), values.std(axis=0)
        std = np.where(std < schema['normalization_std_floor'], 1., std)
        stats[key] = {'mean': mean.tolist(), 'std': std.tolist(), 'min': values.min(axis=0).tolist(),
                      'max': values.max(axis=0).tolist(), 'count': len(values)}
    return stats


def export(raw, destination):
    manifest = json.loads((raw/'collection_manifest.json').read_text())
    schema = manifest['schema']
    destination.mkdir(parents=True, exist_ok=True)
    for split in ('train', 'validation'):
        if (destination/split).exists():
            raise FileExistsError(f'Export destination already exists: {destination/split}')
    # Freeze split membership before the library computes any per-dataset metadata.
    split_manifest = {split: [{'source_episode_id': item['episode_id'], 'scenario_id': item['scenario_id'],
                               'seed': item['seed'], 'dataset_episode_id': i,
                               'block_position': item['config']['task']['block_position'],
                               'bin_position': item['config']['task']['bin_position'],
                               'initial_drone_position': item['config']['drone']['initial_position'],
                               'initial_drone_yaw': item['config']['drone']['initial_yaw']}
                    for i,item in enumerate(x for x in manifest['episodes'] if x['split']==split)]
                    for split in ('train','validation')}
    save(raw/'split_manifest.json', split_manifest)
    save(raw/'schema.json', schema)
    features = {'observation.state': {'dtype': 'float32', 'shape': (8,), 'names': schema['state_names']},
                'action': {'dtype': 'float32', 'shape': (7,), 'names': schema['action_names']},
                'recording.time_s': {'dtype': 'float32', 'shape': (1,), 'names': ['simulation_time']},
                'recording.camera_capture_time_s': {'dtype': 'float32', 'shape': (1,), 'names': ['camera_capture_time']},
                'recording.camera_available_time_s': {'dtype': 'float32', 'shape': (1,), 'names': ['camera_available_time']}}
    for name in schema['image_names']:
        features['observation.images.'+name] = {'dtype': 'image', 'shape': tuple(schema['image_shape_hwc']),
                                              'names': ['height', 'width', 'channels']}
    export_result = {'status': 'running', 'splits': {}, 'source_hashes': {}}
    for split, entries in split_manifest.items():
        root = destination/split
        if root.exists():
            raise FileExistsError(f'Export destination already exists: {root}')
        ds = LeRobotDataset.create(schema['repo_id']+'_'+split, schema['fps'], features,
                                   root=root, robot_type='airvla_drone_gripper', use_videos=False,
                                   image_writer_threads=4, video_backend='pyav')
        frame_count = 0
        try:
            for entry in entries:
                episode = raw/f'episode_{entry["source_episode_id"]:03d}'
                result = json.loads((episode/'result.json').read_text())
                if result['status'] != 'passed':
                    raise ValueError(f'Refusing failed episode {episode}')
                sample_path = episode/'samples.jsonl'
                export_result['source_hashes'][str(sample_path.relative_to(raw))] = hashlib.sha256(sample_path.read_bytes()).hexdigest()
                rows = read_samples(sample_path)
                for i,row in enumerate(rows):
                    camera = row['camera']
                    if abs(row['time_s']-i/schema['fps']) > 1e-8 or camera['available_time_s'] > row['time_s']+1e-9:
                        raise ValueError('Invalid observation or camera availability timestamp')
                    frame = {'observation.state': np.array(row['state'], dtype=np.float32),
                             'action': np.array(row['action'], dtype=np.float32), 'task': schema['task'],
                             'recording.time_s': np.array([row['time_s']], dtype=np.float32),
                             'recording.camera_capture_time_s': np.array([camera['capture_time_s']], dtype=np.float32),
                             'recording.camera_available_time_s': np.array([camera['available_time_s']], dtype=np.float32)}
                    for name in schema['image_names']:
                        with Image.open(episode/camera['cameras'][name]['path']) as image:
                            frame['observation.images.'+name] = np.array(image.convert('RGB'))
                    ds.add_frame(frame)
                ds.save_episode()
                frame_count += len(rows)
                print(f'Export {split} episode {entry["source_episode_id"]}: {len(rows)} frames', flush=True)
        finally:
            ds.finalize()
        export_result['splits'][split] = {'root': str(root), 'repo_id': schema['repo_id']+'_'+split,
                                          'episodes': len(entries), 'frames': frame_count}
    stats = compute_normalization(raw, manifest)
    save(raw/'normalization.json', stats)
    save(destination/'normalization.json', stats)
    save(destination/'schema.json', schema)
    export_result['status'] = 'passed'
    export_result['normalization'] = 'Use normalization.json through ChunkDataset. Native meta/stats.json describes absolute stored actions and is not the chunk normalizer.'
    save(raw/'export_result.json', export_result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, default=Path('artifacts/M3'))
    parser.add_argument('--destination', type=Path, default=Path('datasets/airvla_m3'))
    args = parser.parse_args()
    export(args.raw.resolve(), args.destination.resolve())


if __name__ == '__main__':
    main()
