#!/usr/bin/env python3
"""Build a new export view from passing corrective training episodes and old validation."""
import argparse,copy,hashlib,json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collection',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--pilot',type=Path,required=True)
    args=parser.parse_args()
    source=args.collection.resolve();original=Path('artifacts/M3').resolve()
    result=json.loads((source/'result.json').read_text())
    manifest=json.loads((source/'collection_manifest.json').read_text())
    assert result['status']=='passed' and len(result['trials'])==len(manifest['episodes'])
    assert all(i['split']=='train' and i['fault'] is None and i['source_episode_id']<16 for i in manifest['episodes'])
    protocol=manifest['protocol']
    assert protocol in ('state_driven_recovery_v1','state_driven_recovery_v2')
    args.output.mkdir(parents=True,exist_ok=False)
    combined=copy.deepcopy(manifest);combined['schema']['repo_id']='local/airvla_m5_recovery_'+protocol.rsplit('_',1)[-1]
    combined['schema']['collection']={'protocol':protocol,'training_source':str(source),
        'validation_source':'Unchanged M3 validation episodes 16–19; evaluation only'}
    origins=[]
    for item in combined['episodes']:
        folder=source/f"episode_{item['episode_id']:03d}"
        (args.output/f"episode_{item['episode_id']:03d}").symlink_to(folder,target_is_directory=True)
        origins.append({'export_episode_id':item['episode_id'],'source':str(folder),'split':'train'})
    # Include only successful expert-controlled suffixes AFTER each pilot fault ends.
    pilot=args.pilot.resolve()
    pilot_manifest=json.loads((pilot/'collection_manifest.json').read_text())
    assert json.loads((pilot/'result.json').read_text())['status']=='passed'
    assert pilot_manifest['protocol']==protocol
    for item in pilot_manifest['episodes']:
        if item['fault'] is None:continue
        assert item['split']=='train' and item['source_episode_id']<16
        source_ep=pilot/f"episode_{item['episode_id']:03d}"
        rows=[json.loads(x) for x in (source_ep/'samples.jsonl').read_text().splitlines()]
        last_fault=max(i for i,r in enumerate(rows) if r['fault_active'])
        start=next(i for i in range(last_fault+1,len(rows)) if i%2==0)
        offset=rows[start]['time_s']; episode_id=len(combined['episodes'])
        dest=args.output/f'episode_{episode_id:03d}';dest.mkdir()
        (dest/'frames').symlink_to(source_ep/'frames',target_is_directory=True)
        suffix=copy.deepcopy(rows[start:])
        for r in suffix:
            assert not r['fault_active']
            r['time_s']-=offset
            for key in ['capture_time_s','available_time_s']:r['camera'][key]-=offset
        (dest/'samples.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in suffix))
        flight=[json.loads(x) for x in (source_ep/'flight.jsonl').read_text().splitlines()][round(offset*200):]
        for r in flight:
            for key in ['time_s','action_time_s','control_time_s']:r[key]-=offset
        (dest/'flight.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in flight))
        config=copy.deepcopy(item['config'])
        config['drone']['initial_position']=suffix[0]['state'][:3]
        config['drone']['initial_yaw']=suffix[0]['state'][3]
        # This is a recorded suffix, not an independently resettable simulation seed.
        (dest/'config.json').write_text(json.dumps(config,indent=2)+'\n')
        metadata={'source':str(source_ep),'source_start_s':offset,'fault_prefix_excluded':True,
                  'reset_scope':'Recorded recovery suffix; replay requires full physical-state restoration.'}
        (dest/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
        result=json.loads((source_ep/'result.json').read_text());assert result['status']=='passed'
        result.update(samples=len(suffix),duration_s=len(suffix)/10,physics_steps=len(flight),source_start_s=offset,
                      camera_frames=len({r['camera']['frame'] for r in suffix}),
                      events=[{**e,'end_time_s':e['end_time_s']-offset} for e in result['events'] if e['end_time_s']>=offset])
        (dest/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        copied=copy.deepcopy(item);copied.update(episode_id=episode_id,fault=None,case=item['case']+'_suffix',config=config)
        combined['episodes'].append(copied)
        origins.append({'export_episode_id':episode_id,'source':str(dest.resolve()),'split':'train',**metadata})
    old=json.loads((original/'collection_manifest.json').read_text())
    for item in old['episodes']:
        if item['split']!='validation':continue
        copied=copy.deepcopy(item);copied['original_episode_id']=copied['episode_id'];copied['episode_id']=len(combined['episodes'])
        combined['episodes'].append(copied)
        folder=original/f"episode_{item['episode_id']:03d}"
        (args.output/f"episode_{copied['episode_id']:03d}").symlink_to(folder,target_is_directory=True)
        origins.append({'export_episode_id':copied['episode_id'],'source':str(folder),'split':'validation'})
    for row in origins:
        folder=args.output/f"episode_{row['export_episode_id']:03d}"
        row['samples_sha256']=hashlib.sha256((folder/'samples.jsonl').read_bytes()).hexdigest()
    train_count=sum(i['split']=='train' for i in combined['episodes'])
    suffix_count=sum('source_start_s' in i for i in origins)
    combined['scope']=f'{train_count-suffix_count} training approach episodes plus {suffix_count} fault-free recovery suffixes; original validation preserved.'
    combined['schema']['preview_episodes']=[0,4,8,12,len(manifest['episodes']),train_count]
    combined['origins']=origins
    (args.output/'collection_manifest.json').write_text(json.dumps(combined,indent=2)+'\n')
    print('Prepared new export view:',args.output,flush=True)

if __name__=='__main__':main()
