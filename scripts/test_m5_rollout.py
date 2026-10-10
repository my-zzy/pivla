#!/usr/bin/env python3
"""Check the live deployment boundary against recorded M3 data and queue timing."""
import json
from pathlib import Path
import unittest
import numpy as np
import yaml
from airvla_isaac.policy.action_codec import encode_chunk, normalize
from airvla_isaac.policy.rollout import ActionQueue, replay_chunk
from airvla_isaac.eval.metrics import Outcome


class DeploymentTests(unittest.TestCase):
    def setUp(self):
        self.settings = yaml.safe_load(Path('configs/eval.yaml').read_text())
        self.stats = json.loads((Path(self.settings['checkpoint'])/'normalization.json').read_text())

    def test_every_recorded_replay_command(self):
        for ep in (0, 4, 16, 17, 18, 19):
            rows = [json.loads(s) for s in Path(f'artifacts/M3/episode_{ep:03d}/samples.jsonl').read_text().splitlines()]
            actions = np.array([r['action'] for r in rows], dtype=np.float32)
            for i in range(0, len(rows), 10):
                chunk = actions[i:i+50]
                actual = replay_chunk(chunk, np.asarray(rows[i]['state'], dtype=np.float32), self.stats)
                np.testing.assert_allclose(actual, chunk, atol=1e-6, rtol=0)

    def test_anchor_queue_and_stale_response(self):
        anchor = np.array([-.6, .1, 1.2, .3, 0, 0, 0, .14], dtype=np.float32)
        absolute = np.tile([-.5, .15, 1.3, 0, 0, -.1, 1], (50, 1)).astype(np.float32)
        normalized = normalize(encode_chunk(absolute, anchor), self.stats['action'])
        queue = ActionQueue(self.settings)
        self.assertIsNone(queue.pop())
        queue.request('episode16-request0')
        self.assertIsNone(queue.pop())  # caller keeps stepping physics on an empty queue
        with self.assertRaises(ValueError):
            queue.receive('episode15-request0', normalized, anchor, self.stats)
        queue.receive('episode16-request0', normalized, anchor, self.stats)
        with self.assertRaises(RuntimeError):
            queue.request('premature')
        for _ in range(10):
            np.testing.assert_allclose(queue.pop(), absolute[0], atol=1e-6)
        self.assertIsNone(queue.pop())
        queue.request('episode16-request1')

    def test_success_requires_lift_release_and_dwell(self):
        cfg = json.loads(Path('artifacts/M3/episode_016/config.json').read_text())
        task = cfg['task']
        outcome = Outcome(task, self.settings)
        base = task['bin_position']
        placed = {'block_state': [base[0], base[1], base[2]+.04, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  'contacts': [False, False], 'command_open': True}
        outcome.update(placed, 3.)
        self.assertFalse(outcome.success)  # a block already in the bin is not a pick/place
        lifted = {**placed, 'block_state': [0, 0, 1.1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  'contacts': [True, True], 'command_open': False}
        outcome.update(lifted, .1)
        outcome.update(placed, 1.)
        self.assertFalse(outcome.success)
        outcome.update(placed, 1.)
        self.assertTrue(outcome.success)


if __name__ == '__main__':
    unittest.main()
