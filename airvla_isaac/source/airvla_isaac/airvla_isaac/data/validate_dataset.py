"""Validate raw-to-LeRobot alignment, split isolation, chunk masks and normalization."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

from .chunk_dataset import ChunkDataset
from .convert_to_lerobot import compute_normalization, read_samples, save
from ..policy.action_codec import decode_chunk, encode_chunk, episode_chunk, unnormalize, wrap_yaw, state_from_raw


def validate(raw, root):
    save(raw/'validation_report.json', {'status': 'running', 'dataset': str(root)})
    manifest = json.loads((raw/'collection_manifest.json').read_text())
    schema = manifest['schema']
    assert json.loads((root/'schema.json').read_text()) == schema
    splits = json.loads((raw/'split_manifest.json').read_text())
    saved_stats = json.loads((root/'normalization.json').read_text())
    recomputed = compute_normalization(raw, manifest)
    if saved_stats != recomputed:
        raise AssertionError('Training-only statistics do not reproduce')
    scenario_sets = {s:{e['scenario_id'] for e in entries} for s,entries in splits.items()}
    assert not scenario_sets['train'] & scenario_sets['validation']
    train_ids = [e['source_episode_id'] for e in splits['train']]
    assert saved_stats['source_episode_ids'] == train_ids
    report = {'status': 'running', 'checks': {}, 'splits': {}, 'max_roundtrip_error': 0.,
              'max_image_age_s': 0., 'image_pixel_comparisons': 0, 'native_loader_chunks_checked': 0,
              'raw_chunks_checked': 0, 'normalization_sha256': hashlib.sha256((root/'normalization.json').read_bytes()).hexdigest()}
    for split, entries in splits.items():
        loader = ChunkDataset(root, split)
        numeric = loader.dataset.hf_dataset.select_columns([
            'observation.state', 'action', 'episode_index', 'frame_index', 'timestamp',
            'recording.time_s', 'recording.camera_capture_time_s', 'recording.camera_available_time_s']).with_format('numpy')[:]
        total = 0
        for entry in entries:
            ep_id = entry['source_episode_id']
            ep = raw/f'episode_{ep_id:03d}'
            rows = read_samples(ep/'samples.jsonl')
            flight = read_samples(ep/'flight.jsonl')
            assert json.loads((ep/'result.json').read_text())['status']=='passed'
            indices = np.flatnonzero(numeric['episode_index']==entry['dataset_episode_id'])
            assert len(indices)==len(rows)
            assert np.all(np.diff(indices)==1)
            state = np.array([r['state'] for r in rows], dtype=np.float32)
            absolute = np.array([r['action'] for r in rows], dtype=np.float32)
            assert np.isfinite(state).all() and np.isfinite(absolute).all()
            np.testing.assert_array_equal(numeric['observation.state'][indices], state)
            np.testing.assert_array_equal(numeric['action'][indices], absolute)
            np.testing.assert_array_equal(absolute[:, 3:5], 0.)
            assert set(absolute[:, 6]) <= {0.,1.}
            assert np.all((state[:, 3]>=-np.pi)&(state[:, 3]<np.pi))
            np.testing.assert_allclose(numeric['timestamp'][indices], np.arange(len(rows))/schema['fps'], atol=1e-5)
            np.testing.assert_array_equal(numeric['frame_index'][indices], np.arange(len(rows)))
            for i,row in enumerate(rows):
                camera = row['camera']
                age = row['time_s']-camera['available_time_s']
                assert -1e-9 <= age <= 1/schema['camera_hz']+1e-9
                assert camera['capture_time_s'] <= camera['available_time_s'] <= row['time_s']+1e-9
                report['max_image_age_s'] = max(report['max_image_age_s'], age)
                np.testing.assert_allclose(state[i], state_from_raw(row['privileged']['drone_state'], row['privileged']['aperture_m']), atol=1e-7)
                # Independent physics trace: current observation precedes the first
                # step carrying the new command. The previous step supplies state(t).
                tick = round(row['time_s']*200)
                applied = flight[tick]
                np.testing.assert_allclose(applied['target_position_w'], absolute[i,:3], atol=1e-6)
                assert float(not applied['command_open']) == absolute[i,6]
                if tick:
                    np.testing.assert_allclose(flight[tick-1]['drone_state'], row['privileged']['drone_state'], atol=1e-6)
                for key, expected in [('recording.time_s',row['time_s']),
                                      ('recording.camera_capture_time_s',camera['capture_time_s']),
                                      ('recording.camera_available_time_s',camera['available_time_s'])]:
                    np.testing.assert_allclose(numeric[key][indices[i]], [expected], atol=1e-5)
                chunk, valid = episode_chunk(absolute, i, schema['chunk_size'])
                decoded = decode_chunk(encode_chunk(chunk, state[i], valid), state[i], valid)
                delta = decoded[valid]-chunk[valid]
                delta[:,5] = wrap_yaw(delta[:,5])
                report['max_roundtrip_error'] = max(report['max_roundtrip_error'], float(np.abs(delta).max()))
                assert np.max(np.abs(delta)) < 1e-6
                assert np.all(decoded[~valid]==0.)
                report['raw_chunks_checked'] += 1
            # Exercise LeRobot's actual chunk reader at boundaries and stage changes.
            inspect = {0, len(rows)//2, len(rows)-1, max(0,len(rows)-schema['chunk_size']+1)}
            inspect.update(i for i in range(1,len(rows)) if rows[i]['stage']!=rows[i-1]['stage'])
            for i in sorted(inspect):
                sample = loader[int(indices[i])]
                valid = ~sample['action_is_pad'].numpy()
                chunk, expected_valid = episode_chunk(absolute, i, schema['chunk_size'])
                np.testing.assert_array_equal(valid, expected_valid)
                restored = decode_chunk(unnormalize(sample['action'].numpy(), saved_stats['action'], valid), state[i], valid)
                np.testing.assert_allclose(restored[valid], chunk[valid], atol=1e-6)
                assert torch.all(sample['action'][~torch.from_numpy(valid)]==0)
                assert sample['task']==schema['task']
                assert sample['observation.state'].shape==(8,) and sample['action'].shape==(schema['chunk_size'],7)
                assert not any('privileged' in k or 'block_state' in k or 'contact' in k for k in sample)
                report['native_loader_chunks_checked'] += 1
                if ep_id in schema['preview_episodes']:
                    for name in schema['image_names']:
                        actual = sample['observation.images.'+name]
                        assert actual.shape==(3,256,256) and actual.dtype==torch.float32
                        pixels = (actual.permute(1,2,0).numpy()*255).round().astype(np.uint8)
                        with Image.open(ep/rows[i]['camera']['cameras'][name]['path']) as im:
                            np.testing.assert_array_equal(pixels, np.array(im.convert('RGB')))
                        report['image_pixel_comparisons'] += 1
            total += len(rows)
        batch = next(iter(DataLoader(loader, batch_size=4, shuffle=False, num_workers=0)))
        assert batch['action'].shape==(4,schema['chunk_size'],7)
        assert batch['observation.state'].shape==(4,8)
        assert batch['action_is_pad'].shape==(4,schema['chunk_size'])
        assert all(torch.isfinite(batch[k]).all() for k in ['action','observation.state'])
        report['splits'][split] = {'episodes': len(entries), 'frames': total,
                                  'batch_action_shape': list(batch['action'].shape),
                                  'batch_image_shape': list(batch['observation.images.front'].shape)}
    report['checks'] = {key: True for key in (
        'scenario_disjoint_splits', 'training_only_relative_statistics', 'absolute_disk_actions_match_raw',
        'observations_precede_commands', 'no_future_images', 'state_order_and_yaw', 'gripper_absolute',
        'single_anchor_roundtrip', 'no_episode_crossing', 'zero_masked_padding',
        'loader_batch_shapes', 'lossless_image_roundtrip', 'privileged_state_excluded')}
    report['status'] = 'passed'
    save(raw/'validation_report.json', report)
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, default=Path('artifacts/M3'))
    parser.add_argument('--dataset', type=Path, default=Path('datasets/airvla_m3'))
    args = parser.parse_args()
    try:
        validate(args.raw.resolve(), args.dataset.resolve())
    except BaseException as exc:
        save(args.raw.resolve()/'validation_report.json',
             {'status': 'failed', 'error': f'{type(exc).__name__}: {exc}', 'dataset': str(args.dataset.resolve())})
        raise
