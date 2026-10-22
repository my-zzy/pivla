"""Training-only recovery plans, including the original reset distribution."""
import copy


def recovery_plan(manifest, mode, protocol='v2'):
    training = {i['episode_id']: i for i in manifest['episodes'] if i['split'] == 'train'}
    plan = []

    def add(source, case, fault=None, yaw=None):
        item = training[source]  # Never silently admit a validation episode.
        cfg = copy.deepcopy(item['config'])
        if case != 'original':
            cfg['seed'] = 8100 + len(plan)
            block = cfg['task']['block_position']
            position = [block[0], block[1], 1.4]
            if case == 'lateral':
                position[0] += .16
                position[1] -= .12
            elif case == 'low_approach':
                position = [block[0] - .13, block[1] + .10, 1.04]
            elif case == 'near_grasp':
                position = [block[0] + .06, block[1] - .04, .90]
            elif case == 'shifted_block':
                cfg['task']['block_position'][0] += .07
                position[0] -= .08
                position[1] += .06
            cfg['drone']['initial_position'] = position
            cfg['target']['position'] = position.copy()
        if yaw is not None:
            cfg['drone']['initial_yaw'] = yaw
        plan.append({'episode_id': len(plan), 'source_episode_id': source,
                     'scenario_id': item['scenario_id'], 'split': 'train',
                     'case': case, 'fault': fault, 'seed': cfg['seed'], 'config': cfg})

    if protocol == 'v1':
        cases = [('aligned', None), ('lateral', None), ('low_approach', None),
                 ('missed_closure', 'missed_closure'), ('lost_contact', 'lost_contact')]
        if mode == 'collect':
            cases = [('aligned', None), ('lateral', None), ('low_approach', None), ('shifted_block', None)]
        for source in ([0] if mode == 'pilot' else [0, 4, 8, 12]):
            for case, fault in cases:
                add(source, case, fault, yaw=0.)
                plan[-1]['seed'] = plan[-1]['config']['seed'] = 7900 + len(plan) - 1
        return plan
    if protocol != 'v2' or mode not in ('pilot', 'collect'):
        raise ValueError('Expected protocol v1/v2 and mode pilot/collect')

    # Use the pre-existing training design, never held-out observed yaw values.
    lo, hi = manifest['schema']['collection']['drone_initial_yaw_range_rad']
    if mode == 'pilot':
        add(0, 'original')
        add(4, 'original')
        add(0, 'lateral', yaw=lo)
        add(4, 'lateral', yaw=hi)
        add(0, 'near_grasp', yaw=lo)
        add(4, 'near_grasp', yaw=hi)
        for source, yaw in [(0, lo), (4, hi)]:
            add(source, 'missed_closure', 'missed_closure', yaw)
            add(source, 'lost_contact', 'lost_contact', yaw)
    else:
        for source in sorted(training):
            add(source, 'original')
        # Additional low/offset starts supplement rather than replace the originals.
        for index, source in enumerate((0, 4, 8, 12)):
            add(source, 'lateral', yaw=lo if index % 2 == 0 else hi)
            add(source, 'near_grasp', yaw=hi if index % 2 == 0 else lo)
    return plan
