"""Regression checks for the narrow-yaw dataset that passed format validation."""
import copy
import json
from pathlib import Path
import unittest

from m5_check_recovery_coverage import yaw_coverage


class CoverageTests(unittest.TestCase):
    def setUp(self):
        self.reference=json.loads(Path('datasets/airvla_m3/normalization.json').read_text())

    def test_previous_recovery_data_is_rejected(self):
        old=json.loads(Path('datasets/airvla_m5_recovery/normalization.json').read_text())
        checks=yaw_coverage(old,self.reference,[-.15,.15])['checks']
        self.assertFalse(checks['observed_yaw_covers_declared_training_range'])
        self.assertFalse(checks['yaw_normalization_not_collapsed'])

    def test_extreme_samples_alone_do_not_hide_collapsed_scaling(self):
        old=json.loads(Path('datasets/airvla_m5_recovery/normalization.json').read_text())
        old['observation.state']['min'][3]=-.15
        old['observation.state']['max'][3]=.15
        checks=yaw_coverage(old,self.reference,[-.15,.15])['checks']
        self.assertTrue(checks['observed_yaw_covers_declared_training_range'])
        self.assertFalse(checks['yaw_normalization_not_collapsed'])

    def test_reference_scaling_with_full_design_support_passes(self):
        stats=copy.deepcopy(self.reference)
        stats['observation.state']['min'][3]=-.15
        stats['observation.state']['max'][3]=.15
        self.assertTrue(all(yaw_coverage(stats,self.reference,[-.15,.15])['checks'].values()))


if __name__=='__main__':unittest.main()
