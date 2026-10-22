#!/usr/bin/env python3
"""Gate revised data on training-reset coverage, not just valid file formats."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from airvla_isaac.data.convert_to_lerobot import compute_normalization, read_samples, save


def yaw_coverage(stats, reference, design_range):
    current = stats['observation.state']
    baseline = reference['observation.state']
    normalized = [(y-current['mean'][3])/current['std'][3] for y in design_range]
    reference_max = max(abs((y-baseline['mean'][3])/baseline['std'][3]) for y in design_range)
    # Predeclared relative to the original training statistics; no held-out states.
    limit = 1.25 * reference_max
    checks = {
        'observed_yaw_covers_declared_training_range': current['min'][3] <= design_range[0]+1e-6
            and current['max'][3] >= design_range[1]-1e-6,
        'yaw_normalization_not_collapsed': max(map(abs, normalized)) <= limit,
    }
    return {'checks':checks, 'declared_range_rad':design_range,
            'normalized_range':normalized, 'max_abs_normalized_allowed':limit,
            'yaw_std_rad':current['std'][3], 'reference_yaw_std_rad':baseline['std'][3]}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw',type=Path,required=True)
    parser.add_argument('--dataset',type=Path)
    args=parser.parse_args()
    manifest=json.loads((args.raw/'collection_manifest.json').read_text())
    original=json.loads(Path('artifacts/M3/collection_manifest.json').read_text())
    reference=json.loads(Path('datasets/airvla_m3/normalization.json').read_text())
    stats=(json.loads((args.dataset/'normalization.json').read_text()) if args.dataset
           else compute_normalization(args.raw,manifest))
    report=yaw_coverage(stats,reference,original['schema']['collection']['drone_initial_yaw_range_rad'])
    originals={i['source_episode_id']:i for i in manifest['episodes'] if i.get('case')=='original'}
    expected={i['episode_id']:i for i in original['episodes'] if i['split']=='train'}
    checks=report['checks']
    checks['all_original_training_resets_preserved']=set(originals)==set(expected) and all(
        originals[k]['config']==v['config'] for k,v in expected.items())
    max_initial_error=0.
    for source,item in originals.items():
        first=read_samples(args.raw/f"episode_{item['episode_id']:03d}/samples.jsonl")[0]['state']
        old=read_samples(Path(f'artifacts/M3/episode_{source:03d}/samples.jsonl'))[0]['state']
        max_initial_error=max(max_initial_error,float(np.max(np.abs(np.array(first)-old))))
    checks['actual_original_initial_states_reproduced']=max_initial_error<1e-6
    stages=set();suffixes=[];fault_free=True
    for item in manifest['episodes']:
        path=args.raw/f"episode_{item['episode_id']:03d}/samples.jsonl"
        if item['split']=='train':
            rows=read_samples(path);stages.update(r['stage'] for r in rows)
            fault_free &= all(not r.get('fault_active',False) for r in rows)
            if item.get('case','').endswith('_suffix'):suffixes.append(item['episode_id'])
    checks['forced_fault_prefixes_excluded']=fault_free
    checks['four_successful_recovery_suffixes_present']=len(suffixes)>=4
    checks['all_manipulation_and_recovery_stages_present']={
        'approach','descend','grasp','lift','transport','release','retreat','recover'}<=stages
    validation=[i for i in manifest['episodes'] if i['split']=='validation']
    checks['original_validation_unchanged']=len(validation)==4 and all(
        hashlib.sha256((args.raw/f"episode_{i['episode_id']:03d}/samples.jsonl").read_bytes()).digest()==
        hashlib.sha256(Path(f"artifacts/M3/episode_{i['original_episode_id']:03d}/samples.jsonl").read_bytes()).digest()
        for i in validation)
    checks['training_validation_scenarios_disjoint']=not (
        {i['scenario_id'] for i in manifest['episodes'] if i['split']=='train'} & {i['scenario_id'] for i in validation})
    report.update(status='passed' if all(checks.values()) else 'failed',
                  max_original_initial_state_error=max_initial_error,stages=sorted(stages),
                  recovery_suffix_episode_ids=suffixes,normalization_source='train only',
                  scope='Coverage gate against the original training design; not proof of learned task success.')
    if args.dataset:
        report['dataset_root']=str(args.dataset.resolve())
        report['dataset_repo_id']=manifest['schema']['repo_id']
        report['normalization_sha256']=hashlib.sha256((args.dataset/'normalization.json').read_bytes()).hexdigest()
        report['training_rows']=stats['observation.state']['count']
    save(args.raw/'coverage_report.json',report)
    print(json.dumps(report,indent=2),flush=True)
    if not all(checks.values()):raise RuntimeError('Recovery coverage gate failed')


if __name__=='__main__':main()
