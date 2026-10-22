#!/usr/bin/env python3
"""Prepare a deliberately narrow, high-contrast first-attempt learning experiment."""
import argparse,copy,json
from pathlib import Path
import yaml


def prepare():
    original=json.loads(Path('artifacts/M3/collection_manifest.json').read_text())
    base=copy.deepcopy(original['episodes'][0]['config'])
    base['task']['block_color']=[.025,.12,.95]
    base['drone']['initial_yaw']=0.
    base['drone']['reset_position_jitter_m']=0.
    block=base['task']['block_position']
    # One physical layout, distinct reset configurations. No scene-generalization claim.
    starts=[(0.,0.,1.4),(.025,0.,1.4),(-.025,0.,1.4),(0.,.025,1.4),
            (0.,-.025,1.4),(0.,0.,.95),(.012,-.012,1.4),(-.012,.012,1.4)]
    plan=[]
    for i,(dx,dy,z) in enumerate(starts):
        cfg=copy.deepcopy(base);cfg['seed']=8500+i
        cfg['drone']['initial_position']=[block[0]+dx,block[1]+dy,z]
        cfg['target']['position']=cfg['drone']['initial_position'].copy()
        plan.append({'episode_id':i,'source_episode_id':0,'scenario_id':i,'layout_id':0,
                     'split':'train' if i<6 else 'validation','case':'single_scene_first_attempt',
                     'fault':None,'seed':cfg['seed'],'config':cfg})
    schema=copy.deepcopy(original['schema']);schema['repo_id']='local/airvla_single_scene_v1'
    schema['collection']={'protocol':'single_scene_first_attempt_v1','scope':'One blue-block layout, zero yaw; validation differs only in initial position. No recovery faults or scene/OOD generalization.'}
    schema['split']={'train_scenarios':list(range(6)), 'validation_scenarios':[6,7],
                     'scope':'Distinct reset configurations of the same physical layout'}
    schema['preview_episodes']=[0,5,6]
    return {'protocol':'single_scene_first_attempt_v1','schema':schema,'episodes':plan,
            'scope':schema['collection']['scope']}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('artifacts/M5/single_scene_work/plan.json'))
    args=parser.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(prepare(),indent=2)+'\n')
    print(args.output)
