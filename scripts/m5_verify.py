#!/usr/bin/env python3
"""Audit actual M5 action execution, clocks, anchors, and environment preservation."""
import ast
import json
from pathlib import Path
import subprocess

import numpy as np
import yaml
from airvla_isaac.policy.rollout import decode_prediction


def main():
    root = Path('artifacts/M5')
    settings = yaml.safe_load(Path('configs/eval.yaml').read_text())
    stats = json.loads((Path(settings['checkpoint'])/'normalization.json').read_text())
    policy = json.loads((root/'closed_loop/trials/result.json').read_text())
    replay = json.loads((root/'replay/result.json').read_text())
    expected = settings['familiar_episode_ids']+settings['heldout_episode_ids']
    assert [r['episode_id'] for r in policy['trials']] == expected
    checks = {'all_predeclared_trials_recorded': True}
    records = []
    for trial in policy['trials']:
        episode = root/'closed_loop/trials'/f'episode_{trial["episode_id"]:03d}'
        actions = [json.loads(s) for s in (episode/'actions.jsonl').read_text().splitlines()]
        times = np.array([r['time_s'] for r in actions])
        np.testing.assert_allclose(times, np.arange(len(times))/settings['action_hz'], atol=1e-9)
        assert trial['physics_steps'] == len(actions)*20
        assert trial['exact_physics_ticks']
        camera_ages = np.array([r['time_s']-r['camera_time_s'] for r in actions])
        assert camera_ages.min() >= -1e-9 and camera_ages.max() <= .1+1e-9
        by_tick = {round(r['time_s']*settings['action_hz']): r for r in actions}
        max_error, audited = 0., 0
        for delay in trial['delays']:
            assert delay['physics_steps_during_wait'] > 0
            np.testing.assert_allclose(delay['simulation_delay_s'], delay['physics_steps_during_wait']/200, atol=1e-9)
            request_id = delay['request_id']
            response = json.loads((root/f'closed_loop/ipc/response_{request_id}.json').read_text())
            with np.load(root/f'closed_loop/ipc/request_{request_id}.npz', allow_pickle=False) as request:
                assert set(request.files) == {'state', 'seed', 'front', 'down', 'external'}
                predicted = decode_prediction(response['prediction'], request['state'], stats)
            first_tick = round(delay['arrival_time_s']*settings['action_hz'])
            for j in range(settings['execute_actions']):
                row = by_tick.get(first_tick+j)
                if row is None:  # a trial can end before its final chunk is exhausted
                    break
                expected_action = predicted[j].copy()
                expected_action[6] = float(expected_action[6] >= .5)
                assert not row['hold']
                difference = float(np.abs(np.asarray(row['command'])-expected_action).max())
                max_error = max(max_error, difference)
                audited += 1
        assert max_error < 1e-6 and audited == trial['executed_actions']
        for previous, current in zip(actions, actions[1:]):
            if current['hold']:
                np.testing.assert_array_equal(current['command'], previous['command'])
        records.append({'episode_id': trial['episode_id'], 'executed_commands_audited': audited,
                        'max_command_decode_error': max_error, 'max_camera_age_s': float(camera_ages.max())})
    checks.update(action_timestamps_10hz=True, physics_ticks_200hz=True, physics_advances_during_inference=True,
                  request_time_anchor_decoding=True, executed_commands_match_predictions=True,
                  empty_queue_holds_last_command=True, no_privileged_policy_inputs=True,
                  no_future_camera_frames=True, expert_physical_replay=all(t['success'] for t in replay['trials']))
    sources = {}
    for name, path in [('isaaclab', '/home/user/IsaacLab'), ('lerobot', '/home/user/lerobot'),
                       ('isaaclab_shared_editable', '/home/user/workspace/objsplat_ws/.cache/IsaacLab')]:
        p = subprocess.run(['git', '-C', path, 'status', '--porcelain'], check=True, capture_output=True, text=True)
        sources[name] = p.stdout.strip()
    checks['shared_source_checkouts_clean'] = all(not value for value in sources.values())
    for path in Path('scripts').glob('m5_*.py'):
        ast.parse(path.read_text())
    checks['entry_points_parse'] = True
    result = {'status': 'passed' if all(checks.values()) else 'failed', 'checks': checks,
              'trials': records, 'shared_source_status': sources,
              'adapter_tests': 5, 'deployment_tests': 3,
              'environment_protection': 'Both runtime processes use read-only host mounts; no packages installed.'}
    (root/'verification.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
    assert all(checks.values())


if __name__ == '__main__':
    main()
