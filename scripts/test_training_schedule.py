#!/usr/bin/env python3
"""Check training schedule boundaries and evaluation's isolation from training RNG."""
import unittest
import hashlib
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import torch
from m4_train import check_dataset_coverage, evaluate, learning_rate_at


class ScheduleTests(unittest.TestCase):
    def test_training_refuses_stale_or_failed_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            normalization=root/'normalization.json';normalization.write_text('{}')
            (root/'schema.json').write_text('{"repo_id":"local/coverage_test"}')
            path=root/'coverage.json'
            report={'status':'passed','checks':{'coverage':True},'dataset_repo_id':'local/coverage_test',
                    'normalization_sha256':hashlib.sha256(normalization.read_bytes()).hexdigest()}
            path.write_text(json.dumps(report))
            cfg={'dataset':str(root),'dataset_coverage_report':str(path)}
            check_dataset_coverage(cfg)
            normalization.write_text('{"changed":true}')
            with self.assertRaises(ValueError):check_dataset_coverage(cfg)
            normalization.write_text('{}');report['checks']['coverage']=False
            path.write_text(json.dumps(report))
            with self.assertRaises(ValueError):check_dataset_coverage(cfg)

    def test_warmup_and_decay_boundaries(self):
        cfg = {'learning_rate': 5e-5, 'steps': 10000, 'warmup_steps': 500,
               'lr_schedule': 'cosine', 'min_lr_ratio': .1}
        self.assertAlmostEqual(learning_rate_at(1, cfg), 1e-7)
        self.assertAlmostEqual(learning_rate_at(500, cfg), 5e-5)
        self.assertAlmostEqual(learning_rate_at(5250, cfg), 2.75e-5)
        self.assertAlmostEqual(learning_rate_at(10000, cfg), 5e-6)
        self.assertAlmostEqual(learning_rate_at(12000, cfg), 5e-6)
        self.assertEqual(learning_rate_at(1, {'learning_rate': 1e-4}), 1e-4)

    def test_evaluation_preserves_training_rng_and_mode(self):
        class Model(torch.nn.Module):
            def predict_action_chunk(self, batch):
                return torch.randn(1, 50, 7)

        class Adapter:
            def __call__(self, rows):
                return rows

            def decode(self, prediction, anchors, mask):
                return prediction.numpy()

        row = {'action': torch.zeros(50, 7), 'anchor_state': torch.zeros(8),
               'action_is_pad': torch.zeros(50, dtype=torch.bool), 'episode_index': 0, 'frame_index': 0}
        policy = Model().train()
        torch.manual_seed(120)
        before = torch.get_rng_state().clone()
        with patch('torch.cuda.is_available', return_value=False):
            first = evaluate(policy, Adapter(), [row], [0], 77)
            torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)
            self.assertTrue(policy.training)
            second = evaluate(policy, Adapter(), [row], [0], 77)
            self.assertEqual(first[0], second[0])
            policy.eval()
            evaluate(policy, Adapter(), [row], [0], 77)
            self.assertFalse(policy.training)


if __name__ == '__main__':
    unittest.main()
