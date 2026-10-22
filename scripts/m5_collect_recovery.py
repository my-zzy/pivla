#!/usr/bin/env python3
"""Separate state-driven expert pilot and training-only corrective recordings."""
import argparse,copy,json,os,sys,traceback
from pathlib import Path
from isaaclab.app import AppLauncher
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--mode',choices=['pilot','collect'],required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--pilot-result',type=Path)
parser.add_argument('--protocol',choices=['v1','v2'],default='v2')
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args()
if not args.enable_cameras:parser.error('--enable_cameras required')
app=AppLauncher(args).app
import numpy as np
import torch
import yaml
from airvla_isaac.sim.scenario_env import ScenarioEnv
from airvla_isaac.data.recovery_expert import RecoveryExpert
from airvla_isaac.data.recovery_plan import recovery_plan
from airvla_isaac.data.collect_demos import EpisodeRecorder
from airvla_isaac.policy.action_codec import state_from_raw
from airvla_isaac.eval.metrics import Outcome


def save(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')


def main():
    args.output.mkdir(parents=True,exist_ok=False)
    manifest=json.loads(Path('artifacts/M3/collection_manifest.json').read_text())
    settings=yaml.safe_load(Path('configs/eval.yaml').read_text())
    if args.mode=='collect':
        if args.pilot_result is None:raise ValueError('Passing pilot required')
        pilot=json.loads(args.pilot_result.read_text())
        expected=recovery_plan(manifest,'pilot',args.protocol)
        pilot_manifest=json.loads((args.pilot_result.parent/'collection_manifest.json').read_text())
        assert pilot['status']=='passed' and len(pilot['trials'])==len(expected)
        assert pilot_manifest['episodes']==expected, 'Pilot must match this collection protocol'
    plan=recovery_plan(manifest,args.mode,args.protocol)
    save(args.output/'collection_manifest.json',{'protocol':'state_driven_recovery_'+args.protocol,'schema':manifest['schema'],'episodes':plan,
        'scope':'Original training resets plus approach corrections. Forced-fault prefixes are excluded; successful post-fault suffixes may be exported.'})
    env=ScenarioEnv(plan[0]['config'],args.device,True);results=[]
    for item in plan:
        env.configure(item['config']);env.target_yaw[:]=0.;env.warmup_rendering()
        expert=RecoveryExpert(item['config']);out=args.output/f"episode_{item['episode_id']:03d}"
        recorder=EpisodeRecorder(out,item['config']);save(out/'config.json',item['config']);save(out/'metadata.json',item)
        outcome=Outcome(item['config']['task'],settings);fault_started=None;fault_finished=False;latest=None;count=0
        try:
            with (out/'samples.jsonl').open('w') as samples:
                while env.time<100.:
                    if env.step_count%env.camera_interval==0:latest=recorder.capture(env,expert.stage)
                    if env.step_count%20==0:
                        obs=env.observation();target,opened=expert.update(env.time,obs)
                        if expert.done:break
                        # Pilot-only actuator fault. Record actual command, never a hidden override.
                        trigger=(item['fault']=='missed_closure' and expert.stage=='grasp') or (item['fault']=='lost_contact' and expert.stage=='lift')
                        if trigger and fault_started is None:fault_started=env.time
                        if fault_started is not None and not fault_finished:
                            if env.time-fault_started<2.5:opened=True
                            else:fault_finished=True
                        row={'time_s':env.time,'state':state_from_raw(obs['drone_state'],obs['aperture_m']).tolist(),
                            'action':[*target,0.,0.,0.,float(not opened)],'stage':expert.stage,'camera':latest,'privileged':obs,
                            'fault_active':fault_started is not None and not fault_finished}
                        assert latest['available_time_s']<=env.time+1e-9
                        samples.write(json.dumps(row)+'\n');count+=1
                        env.target_position[:]=env.target_position.new_tensor(target);env.gripper.command(opened)
                    env.step();obs=env.observation();recorder.record(env,obs,expert.stage);outcome.update(obs,env.dt)
                    state=np.array(obs['drone_state'])
                    if not np.isfinite(state).all() or not .45<=state[2]<=2.5 or np.linalg.norm(state[:2])>2.5:raise RuntimeError('Unsafe drone state')
                checks={'expert_finished':expert.done,'physical_pick_place':outcome.success,'physics_ticks':env.physics_steps==env.step_count,
                        'action_rate':count==round(env.time*10),'recovery_exercised':not item['fault'] or (fault_started is not None and expert.retries>0)}
                result={'episode_id':item['episode_id'],'case':item['case'],'source_episode_id':item['source_episode_id'],
                    'status':'passed' if all(checks.values()) else 'failed','checks':checks,'duration_s':env.time,
                    'samples':count,'camera_frames':recorder.frame_count,'retries':expert.retries,'events':expert.events,**outcome.result()}
                save(out/'result.json',result);results.append(result);print(result,flush=True)
                save(args.output/'result.json',{'protocol':'state_driven_recovery_'+args.protocol,'status':'passed' if len(results)==len(plan) and all(r['status']=='passed' for r in results) else 'partial','trials':results,'planned_episodes':len(plan)})
                if not all(checks.values()):raise RuntimeError('Pilot/collection checks failed')
        except BaseException as e:
            save(out/'failure.json',{'error':str(e),'time_s':env.time,'stage':expert.stage,'events':expert.events});raise
        finally:recorder.close()

if __name__=='__main__':
    try:
        with torch.inference_mode():main()
    except BaseException:
        traceback.print_exc();sys.stdout.flush();sys.stderr.flush();os._exit(1)
    app.close(skip_cleanup=True)
