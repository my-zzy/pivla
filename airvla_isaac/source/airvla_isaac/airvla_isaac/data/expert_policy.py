"""Privileged-state pick/place expert with measured transitions and timeouts."""
import numpy as np


class ScriptedExpert:
    stages = ('approach', 'descend', 'grasp', 'lift', 'loaded_hold', 'transport', 'release', 'retreat')

    def __init__(self, config, lift_only=False):
        self.cfg = config
        self.task = config['task']
        self.stage = 'approach'
        self.entered = 0.
        self.dwell = 0.
        self.events = []
        self.done = False
        self.lift_only = lift_only
        self.last_time = 0.
        self.target = np.array(config['target']['position'], dtype=float)

    def in_bin(self, observation):
        pos = np.array(observation['block_state'][:3])
        vel = np.array(observation['block_state'][7:10])
        base = np.array(self.task['bin_position'])
        half = np.array(self.task['bin_inner_size_m'])/2
        # Conservative enclosing sphere for the cube, including any rotation.
        radius = np.linalg.norm(self.task['block_size_m'])/2
        return bool(np.all(np.abs(pos[:2]-base[:2]) + radius < half)
                    and abs(pos[2] - (base[2]+self.task['block_size_m'][2]/2)) < .025
                    and np.linalg.norm(vel) < .04)

    def update(self, t, obs):
        dt = t - self.last_time
        self.last_time = t
        age = t - self.entered
        if age > self.task['stage_timeout_s']:
            raise RuntimeError(f'Stage {self.stage} timed out at {t:.3f}s: {obs}')
        drone = np.array(obs['drone_state'])
        block = np.array(obs['block_state'])
        pick = np.array(self.task['block_position'])
        goal = np.array([*pick[:2], self.task['flight_height_m']])
        if self.stage in ('descend', 'grasp'):
            goal[2] = self.task['grasp_height_m']
        if self.stage in ('transport', 'release', 'retreat'):
            goal[:2] = self.task['bin_position'][:2]
        if self.stage == 'retreat':
            goal[1] -= .5
        delta = goal-self.target
        distance = np.linalg.norm(delta)
        self.target += delta * min(1., self.task['setpoint_speed_m_s'] * dt / max(distance, 1e-9))
        error = np.linalg.norm(drone[:3] - goal)
        slow = np.linalg.norm(drone[7:10]) < .035
        contacts = all(obs['contacts'])
        if self.stage == 'approach':
            condition = error < .015 and slow and age >= 8.
        elif self.stage == 'descend':
            condition = error < .012 and slow and age >= 3.
        elif self.stage == 'grasp':
            condition = contacts and .07 < obs['aperture_m'] < .095 and age >= 1.
        elif self.stage in ('lift', 'loaded_hold'):
            condition = (contacts and block[2] > self.task['table_top_m'] + .30
                         and error < self.task['loaded_position_tolerance_m'] and slow)
            condition &= age >= (8. if self.stage == 'lift' else self.task['hover_measure_s'] + 2.)
        elif self.stage == 'transport':
            if self.task.get('release_footprint_margin_m') is not None:
                # A payload can shift the PD equilibrium laterally. Gate release
                # on the actual block footprint plus clearance inside the bin.
                radius = np.linalg.norm(self.task['block_size_m'])/2
                clearance = (np.array(self.task['bin_inner_size_m'])/2
                             - np.abs(block[:2]-goal[:2]) - radius)
                aligned = bool(np.all(clearance > self.task['release_footprint_margin_m']))
            else:
                aligned = np.linalg.norm(drone[:2] - goal[:2]) < .025
            condition = contacts and aligned and slow and block[2] > .80
        elif self.stage == 'release':
            condition = self.in_bin(obs) and not any(obs['contacts']) and obs['aperture_m'] > .12 and age >= 2.
        else:
            condition = self.in_bin(obs) and error < .025 and slow and age >= 3.
        self.dwell = self.dwell + dt if condition else 0.
        if self.dwell >= .5:
            previous = self.stage
            self.events.append({'stage': previous, 'end_time_s': t, 'duration_s': age,
                                'drone_position': drone[:3].tolist(), 'block_position': block[:3].tolist()})
            if previous == 'retreat' or (previous == 'loaded_hold' and self.lift_only):
                self.done = True
            else:
                self.stage = self.stages[self.stages.index(previous) + 1]
                self.entered, self.dwell = t, 0.
            print(f'M2 {t:6.2f}s: {previous} -> {"done" if self.done else self.stage}', flush=True)
        opened = self.stage in ('approach', 'descend', 'release', 'retreat')
        return self.target.tolist(), opened
