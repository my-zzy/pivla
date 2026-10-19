#!/usr/bin/env python3
"""Read-only analysis of recorded failures and fixed-noise input sensitivity."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from airvla_isaac.policy.pi0_adapter import CAMERA_MAP, Pi0BatchAdapter, load_policy_strict, policy_config
from airvla_isaac.policy.rollout import decode_prediction


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def trajectory_diagnostics(root):
    result = json.loads((root/'result.json').read_text())
    records = []
    for trial in result['trials']:
        ep = root/'closed_loop/trials'/f"episode_{trial['episode_id']:03d}"
        rows = read_rows(ep/'flight.jsonl')
        closures = [r for i,r in enumerate(rows) if not r['command_open'] and (i==0 or rows[i-1]['command_open'])]
        events = []
        for r in closures:
            drone, block = np.array(r['drone_state']), np.array(r['block_state'])
            events.append({'time_s': r['time_s'], 'horizontal_drone_block_error_m': float(np.linalg.norm(drone[:2]-block[:2])),
                           'drone_z_m': float(drone[2]), 'block_z_m': float(block[2]), 'aperture_m': r['aperture_m'], 'contacts': r['contacts']})
        records.append({'episode':trial['episode_id'],'termination':trial['termination'],
            'first_descent_below_1_2_s': next((r['time_s'] for r in rows if r['drone_state'][2]<1.2),None),
            'closed_two_finger_contact_s': sum(all(r['contacts']) and not r['command_open'] for r in rows)/200,
            'min_block_z_m':min(r['block_state'][2] for r in rows),'closures':events})
    return records


def demo_ambiguity():
    result=[]
    for ep in range(20):
        root=Path(f'artifacts/M3/episode_{ep:03d}');rows=read_rows(root/'samples.jsonl')
        # Predeclared pair: settled hover at 4 s, beginning of descent at 8.4 s.
        a,b=rows[40],rows[84]
        image_mae={}
        for name in CAMERA_MAP:
            ia=np.asarray(Image.open(root/a['camera']['cameras'][name]['path']),dtype=float)
            ib=np.asarray(Image.open(root/b['camera']['cameras'][name]['path']),dtype=float)
            image_mae[name]=float(np.abs(ia-ib).mean()/255)
        sa,sb=np.array(a['state']),np.array(b['state'])
        result.append({'episode':ep,'split':'train' if ep<16 else 'validation','times_s':[a['time_s'],b['time_s']],
            'stages':[a['stage'],b['stage']], 'position_difference_m':float(np.linalg.norm(sa[:3]-sb[:3])),
            'velocity_difference_m_s':float(np.linalg.norm(sa[4:7]-sb[4:7])),
            'yaw_difference_rad':float(abs(sa[3]-sb[3])), 'aperture_difference_m':float(abs(sa[7]-sb[7])),
            'image_mae_0_1':image_mae,
            'first_10_target_z_difference_rms_m':float(np.sqrt(np.mean((np.array([r['action'][2] for r in rows[40:50]])-np.array([r['action'][2] for r in rows[84:94]]))**2)))})
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--step',choices=['007500','010000'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    root=Path('artifacts/M5')/('baseline_step_'+args.step)
    summary=json.loads((root/'result.json').read_text());checkpoint=Path(summary['checkpoint'])
    save(args.output/'trajectory.json',trajectory_diagnostics(root))
    save(args.output/'demo_ambiguity.json',demo_ambiguity())
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
    adapter=Pi0BatchAdapter(checkpoint,checkpoint/'tokenizer');cfg=policy_config(checkpoint)
    cfg.num_inference_steps=summary['settings']['inference_steps']
    policy=load_policy_strict(checkpoint,cfg).to('cuda').eval();policy.config.device='cuda'
    records=[];arrays={}
    modes=['baseline','repeat','gray_all','gray_down','other_time_rgb','mean_state','zero_velocity']
    for trial in summary['trials']:
        ep=trial['episode_id'];delays=read_rows(root/'closed_loop/trials'/f'episode_{ep:03d}'/'inference.jsonl')
        selected={min(delays,key=lambda d:abs(d['observation_time_s']-t))['request_id'] for t in [0,8,15,30,40,60]}
        for delay in delays:
            request_id=delay['request_id']
            if request_id not in selected: continue
            with np.load(root/'closed_loop/ipc'/f'request_{request_id}.npz') as f: data={k:f[k].copy() for k in f.files}
            # Opposite endpoint from the same episode; intentionally mismatched RGB/state.
            donor=delays[-1] if delay['observation_time_s']<trial['duration_s']/2 else delays[0]
            with np.load(root/'closed_loop/ipc'/f"request_{donor['request_id']}.npz") as f: donor_images={k:f[k].copy() for k in CAMERA_MAP}
            predictions={};normalized={}
            for mode in modes:
                images={k:data[k] for k in CAMERA_MAP}
                if mode=='gray_all': images={k:np.full_like(v,128) for k,v in images.items()}
                if mode=='gray_down': images['down']=np.full_like(images['down'],128)
                if mode=='other_time_rgb': images=donor_images
                state=data['state'].copy()
                if mode=='zero_velocity': state[4:7]=0
                batch=adapter.from_observation(state,images)
                if mode=='mean_state': batch['observation.state']=torch.zeros_like(batch['observation.state'])
                torch.manual_seed(int(data['seed']))
                with torch.inference_mode(): pred=policy.predict_action_chunk(batch)[0].float().cpu().numpy()
                normalized[mode]=pred
                # Hold original decoding anchor fixed: measure model-input sensitivity only.
                predictions[mode]=decode_prediction(pred,data['state'],adapter.stats)
                arrays[request_id+'_'+mode]=predictions[mode]
            saved=json.loads((root/'closed_loop/ipc'/f'response_{request_id}.json').read_text())['prediction']
            reload_error=float(np.max(np.abs(normalized['baseline']-np.asarray(saved))))
            repeat_error=float(np.max(np.abs(normalized['baseline']-normalized['repeat'])))
            assert repeat_error==0,repeat_error
            assert reload_error<1e-4,(request_id,reload_error)
            changes={}
            for mode in modes[2:]:
                delta=predictions[mode][:10]-predictions['baseline'][:10]
                changes[mode]={'position_rms_change_m':float(np.sqrt(np.mean(np.sum(delta[:,:3]**2,axis=1)))),
                    'gripper_flip_fraction':float(np.mean((predictions[mode][:10,6]>=.5)!=(predictions['baseline'][:10,6]>=.5)))}
            records.append({'episode':ep,'request_id':request_id,'time_s':delay['observation_time_s'],
                'donor_request_id':donor['request_id'],'reload_max_abs_error':reload_error,'repeat_max_abs_error':repeat_error,'changes':changes})
            print(args.step,request_id,'verified; input interventions measured',flush=True)
    aggregate={m:{'median_position_rms_change_m':float(np.median([r['changes'][m]['position_rms_change_m'] for r in records])),
                  'p90_position_rms_change_m':float(np.percentile([r['changes'][m]['position_rms_change_m'] for r in records],90)),
                  'mean_gripper_flip_fraction':float(np.mean([r['changes'][m]['gripper_flip_fraction'] for r in records]))} for m in modes[2:]}
    np.savez_compressed(args.output/'predictions.npz',**arrays)
    save(args.output/'result.json',{'checkpoint':str(checkpoint),'samples':len(records),'aggregate':aggregate,'records':records,
        'scope':'Input sensitivity on selected recorded on-policy observations, with identical noise and fixed original decoding anchors. Interventions may be out of distribution; differences do not measure task success or prove a modality is correctly grounded.'})
    print(json.dumps(aggregate,indent=2),flush=True)

if __name__=='__main__':main()
