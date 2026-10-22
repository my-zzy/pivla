#!/usr/bin/env python3
"""Collect fixed-layout high-contrast demonstrations; require first-attempt success."""
import argparse,copy,json,os,sys,traceback
from pathlib import Path
from isaaclab.app import AppLauncher
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--plan',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args()
if not args.enable_cameras:parser.error('--enable_cameras required')
app=AppLauncher(args).app
import numpy as np
import torch
import yaml
from airvla_isaac.sim.scenario_env import ScenarioEnv
from airvla_isaac.data.recovery_expert import RecoveryExpert
from airvla_isaac.data.collect_demos import EpisodeRecorder
from airvla_isaac.policy.action_codec import state_from_raw
from airvla_isaac.eval.metrics import Outcome


def save(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')


def main():
    args.output.mkdir(parents=True,exist_ok=False)
    manifest=json.loads(args.plan.read_text())
    assert manifest['protocol']=='single_scene_first_attempt_v1'
    plan=manifest['episodes']
    assert all(item['fault'] is None for item in plan)
    settings=yaml.safe_load(Path('configs/eval.yaml').read_text())
    save(args.output/'collection_manifest.json',manifest)
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
                        row={'time_s':env.time,'state':state_from_raw(obs['drone_state'],obs['aperture_m']).tolist(),
                            'action':[*target,0.,0.,0.,float(not opened)],'stage':expert.stage,'camera':latest,'privileged':obs,
                            'fault_active':False}
                        assert latest['available_time_s']<=env.time+1e-9
                        samples.write(json.dumps(row)+'\n');count+=1
                        env.target_position[:]=env.target_position.new_tensor(target);env.gripper.command(opened)
                    env.step();obs=env.observation();recorder.record(env,obs,expert.stage);outcome.update(obs,env.dt)
                    state=np.array(obs['drone_state'])
                    if not np.isfinite(state).all() or not .45<=state[2]<=2.5 or np.linalg.norm(state[:2])>2.5:raise RuntimeError('Unsafe drone state')
                checks={'expert_finished':expert.done,'physical_pick_place':outcome.success,'physics_ticks':env.physics_steps==env.step_count,
                        'action_rate':count==round(env.time*10),'first_attempt':expert.retries==0}
                result={'episode_id':item['episode_id'],'case':item['case'],'source_episode_id':item['source_episode_id'],
                    'status':'passed' if all(checks.values()) else 'failed','checks':checks,'duration_s':env.time,
                    'samples':count,'camera_frames':recorder.frame_count,'retries':expert.retries,'events':expert.events,**outcome.result()}
                save(out/'result.json',result);results.append(result);print(result,flush=True)
                save(args.output/'result.json',{'protocol':manifest['protocol'],'status':'passed' if len(results)==len(plan) and all(r['status']=='passed' for r in results) else 'partial','trials':results,'planned_episodes':len(plan)})
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
