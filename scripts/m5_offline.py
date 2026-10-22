#!/usr/bin/env python3
"""Evaluate a frozen checkpoint on familiar and held-out M3 episodes."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from m4_train import evaluate, save
from airvla_isaac.data.chunk_dataset import ChunkDataset
from airvla_isaac.policy.pi0_adapter import Pi0BatchAdapter, load_policy_strict, policy_config


def configure_normalization(dataset, adapter, source='dataset'):
    """Evaluate fixed recordings with the model's training-time normalization."""
    if source == 'dataset':
        assert dataset.schema == adapter.schema and dataset.stats == adapter.stats
    elif source == 'checkpoint':
        # Dataset provenance may differ, but observation/action semantics must not.
        provenance = {'repo_id', 'collection', 'split', 'preview_episodes'}
        contract = lambda schema: {k: v for k, v in schema.items() if k not in provenance}
        if contract(dataset.schema) != contract(adapter.schema):
            raise ValueError('Evaluation dataset and checkpoint have incompatible feature contracts')
        dataset.stats = adapter.stats
    else:
        raise ValueError(f'Unknown offline_normalization: {source}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('artifacts/M5/offline'))
    parser.add_argument('--config', type=Path, default=Path('configs/eval.yaml'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    settings = yaml.safe_load(args.config.read_text())
    checkpoint = Path(settings['checkpoint'])
    adapter = Pi0BatchAdapter(checkpoint, checkpoint/'tokenizer')
    cfg = policy_config(checkpoint)
    cfg.num_inference_steps = settings['inference_steps']
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = True
    policy = load_policy_strict(checkpoint, cfg).to('cuda').eval()
    policy.config.device = 'cuda'
    report = {'checkpoint': str(checkpoint), 'settings': settings, 'groups': {},
              'scope': 'Offline command prediction only. ' + settings['scope']}
    split_manifest = json.loads(Path('artifacts/M3/split_manifest.json').read_text())
    for split, name, selected in [('train', 'familiar', [0, 4]), ('validation', 'heldout', None)]:
        dataset = ChunkDataset(settings['dataset'], split)
        configure_normalization(dataset, adapter, settings.get('offline_normalization', 'dataset'))
        numeric = dataset.dataset.hf_dataset.select_columns(['episode_index', 'action']).with_format('numpy')[:]
        membership, actions = np.asarray(numeric['episode_index']), np.asarray(numeric['action'])
        group = {}
        for ep in sorted(set(membership.tolist())):
            if selected is not None and ep not in selected:
                continue
            rows = np.flatnonzero(membership == ep).tolist()
            indices = sorted(set(rows[::settings['offline_stride']] + [rows[-1]] +
                [i for i in rows if (i > rows[0] and actions[i, 6] != actions[i-1, 6]) or
                 (i < rows[-1] and actions[i, 6] != actions[i+1, 6])]))
            metrics, arrays, records = evaluate(policy, adapter, dataset, indices, settings['seed'])
            metrics['source_episode_id'] = next(r['source_episode_id'] for r in split_manifest[split]
                                                if r['dataset_episode_id'] == ep)
            # Class-balanced scores expose the weakness of always-open/closed guesses.
            valid = arrays['valid']
            truth = arrays['targets'][:, :, 6][valid] >= .5
            predicted = arrays['predictions'][:, :, 6][valid] >= .5
            metrics['gripper_open_accuracy'] = float((~predicted[~truth]).mean())
            metrics['gripper_closed_accuracy'] = float(predicted[truth].mean())
            metrics['gripper_balanced_accuracy'] = (metrics['gripper_open_accuracy'] + metrics['gripper_closed_accuracy'])/2
            stem = f'{split}_{ep:03d}'
            np.savez_compressed(args.output/f'{stem}.npz', **arrays)
            save(args.output/f'{stem}_rows.json', records)
            group[str(ep)] = metrics
            print(name, ep, metrics, flush=True)
        weights = np.array([m['valid_future_actions'] for m in group.values()])
        aggregate = {key: float(np.sqrt(np.average([m[key]**2 for m in group.values()], weights=weights)))
                     for key in ('position_rmse_m', 'yaw_rmse_rad')}
        aggregate['gripper_accuracy'] = float(np.average([m['gripper_accuracy'] for m in group.values()], weights=weights))
        aggregate['evaluated_chunks'] = sum(m['evaluated_chunks'] for m in group.values())
        aggregate['valid_future_actions'] = int(weights.sum())
        report['groups'][name] = {'episodes': group, 'aggregate': aggregate}
        save(args.output/'result.json', report)
    report['status'] = 'measured'
    save(args.output/'result.json', report)


if __name__ == '__main__':
    main()
