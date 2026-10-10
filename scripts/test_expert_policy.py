#!/usr/bin/env python3
"""CPU checks for safety-relevant expert transitions, independently of Sim."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
path = ROOT/'airvla_isaac/source/airvla_isaac/airvla_isaac/data/expert_policy.py'
spec = importlib.util.spec_from_file_location('expert_policy', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ExpertChecks(unittest.TestCase):
    def setUp(self):
        self.cfg = yaml.safe_load((ROOT/'configs/manipulation.yaml').read_text())
        self.expert = module.ScriptedExpert(self.cfg)
        self.obs = {'drone_state': [-.65, 0., .83, 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                    'block_state': [-.65, 0., .54, 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                    'contacts': [False, False], 'aperture_m': .08}

    def test_grasp_needs_both_contacts(self):
        self.expert.stage = 'grasp'
        for step in range(1, 31):
            self.obs['contacts'] = [True, False]
            self.expert.update(step*.1, self.obs)
        self.assertEqual(self.expert.stage, 'grasp')
        self.obs['contacts'] = [True, True]
        for step in range(31, 37):
            self.expert.update(step*.1, self.obs)
        self.assertEqual(self.expert.stage, 'lift')

    def test_release_needs_settled_object_inside_bin(self):
        self.expert.stage = 'release'
        self.obs['aperture_m'] = .14
        for step in range(1, 31):
            self.expert.update(step*.1, self.obs)
        self.assertEqual(self.expert.stage, 'release')
        self.obs['block_state'][:3] = [.65, 0., .54]
        self.obs['block_state'][7] = .5
        self.assertFalse(self.expert.in_bin(self.obs))
        self.obs['block_state'][7] = 0.
        for step in range(31, 37):
            self.expert.update(step*.1, self.obs)
        self.assertEqual(self.expert.stage, 'retreat')

    def test_timeout_is_a_failure(self):
        with self.assertRaisesRegex(RuntimeError, 'approach timed out'):
            self.expert.update(self.cfg['task']['stage_timeout_s']+1, self.obs)

    def test_m3_release_gate_checks_payload_clearance(self):
        self.expert.task['release_footprint_margin_m'] = .05
        self.expert.stage = 'transport'
        self.obs['contacts'] = [True, True]
        self.obs['drone_state'][:3] = [.65, .04, 1.24]
        self.obs['block_state'][:3] = [.65, .17, .95]
        for step in range(1,11):
            self.expert.update(step*.1,self.obs)
        self.assertEqual(self.expert.stage,'transport')
        self.obs['block_state'][1] = .04
        for step in range(11,18):
            self.expert.update(step*.1,self.obs)
        self.assertEqual(self.expert.stage,'release')

    def test_setpoint_speed_is_bounded(self):
        self.expert.stage = 'transport'
        previous = self.expert.target.copy()
        target, _ = self.expert.update(.1, self.obs)
        self.assertLessEqual(np.linalg.norm(np.array(target)-previous),
                             self.cfg['task']['setpoint_speed_m_s']*.1+1e-9)


if __name__ == '__main__':
    unittest.main()
