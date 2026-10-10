"""Physical outcome metrics; privileged fields are never policy inputs."""
import numpy as np


def in_bin(observation, task, settings):
    block = np.asarray(observation['block_state'])
    base = np.asarray(task['bin_position'])
    radius = np.linalg.norm(task['block_size_m'])/2
    return bool(np.all(np.abs(block[:2]-base[:2])+radius < np.asarray(task['bin_inner_size_m'])/2)
                and abs(block[2]-(base[2]+task['block_size_m'][2]/2)) < settings['bin_z_tolerance_m']
                and np.linalg.norm(block[7:10]) < settings['settled_speed_m_s'])


class Outcome:
    def __init__(self, task, settings):
        self.task, self.settings = task, settings
        self.lifted = False
        self.grasp_contact = False
        self.success_dwell = 0.
        self.max_block_z = 0.
        self.success = False

    def update(self, obs, dt):
        self.max_block_z = max(self.max_block_z, obs['block_state'][2])
        contact = all(obs['contacts']) and not obs['command_open']
        self.grasp_contact |= contact
        self.lifted |= contact and obs['block_state'][2] > self.task['table_top_m']+self.settings['lift_above_table_m']
        placed = self.lifted and in_bin(obs, self.task, self.settings) and obs['command_open'] and not any(obs['contacts'])
        self.success_dwell = self.success_dwell+dt if placed else 0.
        self.success = self.success_dwell >= self.settings['success_dwell_s']

    def result(self):
        return {'success': bool(self.success), 'grasp_contact': bool(self.grasp_contact),
                'lifted': bool(self.lifted), 'max_block_z_m': float(self.max_block_z),
                'success_dwell_s': self.success_dwell}
