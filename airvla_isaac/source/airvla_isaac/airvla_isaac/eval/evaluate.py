"""Simulator rollouts with a separate inference process and explicit queue holds."""
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
from ..data.collect_demos import EpisodeRecorder
from ..policy.action_codec import state_from_raw
from ..policy.rollout import ActionQueue, checked_action, replay_chunk
from .metrics import Outcome


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def run_trial(env, item, settings, output, mode, ipc=None):
    config = item['config']
    env.configure(config)
    env.target_yaw[:] = config['target']['yaw']
    if mode == 'policy':
        initial = state_from_raw(env.observation()['drone_state'], env.observation()['aperture_m'])
        env.target_position[:] = env.target_position.new_tensor(initial[:3])
        env.target_yaw[:] = float(initial[3])
    env.warmup_rendering()
    recorder = EpisodeRecorder(output, config)
    save(output/'config.json', config)
    save(output/'metadata.json', {k: v for k, v in item.items() if k != 'config'})
    stats = json.loads((Path(settings['checkpoint'])/'normalization.json').read_text())
    outcome = Outcome(config['task'], settings)
    queue = ActionQueue(settings)
    interval = config['physics_hz']//settings['action_hz']
    if config['physics_hz'] % settings['action_hz']:
        raise ValueError('Action rate must divide physics rate')
    latest = None
    request_count, fallback_ticks, executed, max_replay_error = 0, 0, 0, 0.
    delays, pending = [], None
    source_rows = []
    if mode == 'replay':
        source = Path('artifacts/M3')/f'episode_{item["episode_id"]:03d}'/'samples.jsonl'
        source_rows = [json.loads(line) for line in source.read_text().splitlines()]
        absolute = np.asarray([r['action'] for r in source_rows], dtype=np.float32)
    start_wall = time.monotonic()
    termination = 'timeout'
    failure = None
    try:
        with (output/'actions.jsonl').open('w') as action_log, (output/'inference.jsonl').open('w') as inference_log:
            while env.time < settings['timeout_s']:
                if env.step_count % env.camera_interval == 0:
                    latest = recorder.capture(env, mode)
                if env.step_count % interval == 0:
                    obs = env.observation()
                    anchor = state_from_raw(obs['drone_state'], obs['aperture_m'])
                    action = None
                    if mode == 'replay':
                        index = env.step_count//interval
                        if index >= len(source_rows):
                            termination = 'replay_finished'
                            break
                        if not queue.actions:
                            chunk = absolute[index:index+settings['action_horizon']]
                            decoded = replay_chunk(chunk, np.asarray(source_rows[index]['state'], dtype=np.float32), stats)
                            max_replay_error = max(max_replay_error, float(np.abs(decoded-chunk).max()))
                            queue.actions.extend(decoded[:settings['execute_actions']])
                        action = queue.pop()
                        max_replay_error = max(max_replay_error, float(np.abs(action-absolute[index]).max()))
                    else:
                        if pending is not None:
                            path = ipc/f'response_{pending["request_id"]}.json'
                            if path.exists():
                                response = json.loads(path.read_text())
                                if 'error' in response:
                                    raise RuntimeError('Inference worker: '+response['error'])
                                queue.receive(response['request_id'], response['prediction'], pending['anchor'], stats)
                                delay = {'request_id': pending['request_id'], 'observation_time_s': pending['sim_time'],
                                    'camera_time_s': pending['camera_time'], 'arrival_time_s': env.time,
                                    'physics_steps_during_wait': env.step_count-pending['physics_step'],
                                    'simulation_delay_s': env.time-pending['sim_time'],
                                    'roundtrip_wall_s': time.monotonic()-pending['wall_time'],
                                    'inference_wall_s': response['inference_wall_s']}
                                delays.append(delay)
                                inference_log.write(json.dumps(delay)+'\n'); inference_log.flush()
                                pending = None
                            elif time.monotonic()-pending['wall_time'] > settings['worker_timeout_wall_s']:
                                raise RuntimeError('Inference worker timed out')
                        if not queue.actions and pending is None:
                            request_id = f'{item["episode_id"]:03d}_{request_count:05d}'
                            queue.request(request_id)
                            assert latest is not None and latest['available_time_s'] <= env.time+1e-9
                            data = {'state': anchor, 'seed': settings['seed']+item['episode_id']*10000+request_count}
                            for name in ('front', 'down', 'external'):
                                with Image.open(output/latest['cameras'][name]['path']) as im:
                                    data[name] = np.asarray(im.convert('RGB'))
                            pending = {'request_id': request_id, 'anchor': anchor.copy(), 'sim_time': env.time,
                                'wall_time': time.monotonic(), 'physics_step': env.step_count,
                                'camera_time': latest['capture_time_s']}
                            temporary = ipc/f'request_{request_id}.tmp'
                            with temporary.open('wb') as handle:
                                np.savez(handle, **data)
                            temporary.replace(ipc/f'request_{request_id}.npz')
                            request_count += 1
                        action = queue.pop()
                    if action is None:
                        fallback_ticks += 1
                    else:
                        action = checked_action(action, settings)
                        env.target_position[:] = env.target_position.new_tensor(action[:3])
                        env.target_yaw[:] = float(action[5])
                        env.gripper.command(bool(action[6] < .5))
                        executed += 1
                    action_log.write(json.dumps({'time_s': env.time, 'state': anchor.tolist(),
                        'command': [*env.target_position[0].cpu().tolist(), 0., 0., float(env.target_yaw[0]),
                                    float(not env.gripper.command_open)],
                        'hold': action is None, 'queue_remaining': len(queue.actions),
                        'camera_time_s': latest['capture_time_s']})+'\n')
                env.step()
                obs = env.observation()
                recorder.record(env, obs, mode)
                state = np.asarray(obs['drone_state'])
                lo, hi = settings['safe_z_m']
                if not np.isfinite(state).all() or not lo <= state[2] <= hi or np.linalg.norm(state[:2]) > settings['safe_xy_radius_m']:
                    termination = 'unsafe_state'
                    break
                outcome.update(obs, env.dt)
                if outcome.success and mode == 'policy':
                    termination = 'success'
                    break
                if env.step_count % (config['physics_hz']*10) == 0:
                    print(f'M5 {mode} E{item["episode_id"]} {env.time:.1f}s z={state[2]:.3f} '
                          f'block_z={obs["block_state"][2]:.3f} lifted={outcome.lifted} requests={request_count}', flush=True)
                if mode == 'policy':
                    # Best effort: never skip physics/controller ticks to catch up after rendering.
                    delay = env.time-(time.monotonic()-start_wall)
                    if delay > 0:
                        time.sleep(min(delay, env.dt))
    except (ValueError, RuntimeError) as exc:
        termination = 'unsafe_command' if str(exc).startswith('Predicted target outside test volume:') else 'error'
        failure = f'{type(exc).__name__}: {exc}'
    finally:
        recorder.close()
    result = {'episode_id': item['episode_id'], 'scenario_id': item['scenario_id'], 'split': item['split'],
        'mode': mode, **outcome.result(), 'termination': termination, 'error': failure,
        'duration_s': env.time, 'wall_elapsed_s': time.monotonic()-start_wall,
        'physics_steps': env.physics_steps, 'exact_physics_ticks': env.physics_steps == env.step_count,
        'camera_frames': recorder.frame_count, 'executed_actions': executed,
        'hold_action_ticks': fallback_ticks, 'hold_simulation_s': fallback_ticks/settings['action_hz'],
        'requests': request_count, 'completed_requests': len(delays), 'delays': delays,
        'max_replay_command_error': max_replay_error if mode == 'replay' else None,
        'final_observation': env.observation(),
        'time_mode': settings['simulation_time_mode'] if mode == 'policy' else 'unpaced_fixed_dt_replay'}
    if source_rows:
        result['replay_action_count_matches'] = executed == len(source_rows)
    save(output/'result.json', result)
    print('M5 trial result:', {k: v for k, v in result.items() if k not in ('delays', 'final_observation')}, flush=True)
    return result
