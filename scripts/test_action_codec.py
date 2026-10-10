#!/usr/bin/env python3
"""Simulator-independent tests for the frozen M3 action convention."""
import unittest
import numpy as np
from airvla_isaac.policy.action_codec import decode_chunk, encode_chunk, episode_chunk, normalize, unnormalize, wrap_yaw


class CodecChecks(unittest.TestCase):
    def test_one_anchor_for_entire_chunk(self):
        anchor = np.array([1., 2., 3., 0., 0., 0., 0., .14])
        actions = np.array([[1.2, 2., 3., 0., 0., 0., 0.], [1.5, 2., 3., 0., 0., 0., 1.]])
        relative = encode_chunk(actions, anchor)
        np.testing.assert_allclose(relative[:, 0], [.2, .5], atol=1e-7)
        np.testing.assert_array_equal(relative[:, 6], [0., 1.])
        np.testing.assert_allclose(decode_chunk(relative, anchor), actions, atol=1e-6)

    def test_yaw_crosses_pi_and_uses_state_index_three(self):
        anchor = np.array([0., 0., 0., np.pi-.01, 99., 88., 77., .08])
        action = np.array([[0., 0., 0., 0., 0., -np.pi+.02, 1.]])
        relative = encode_chunk(action, anchor)
        self.assertAlmostEqual(relative[0, 5], .03, places=6)
        self.assertAlmostEqual(float(wrap_yaw(decode_chunk(relative, anchor)[0, 5]-action[0, 5])), 0., places=6)
        self.assertEqual(float(wrap_yaw(np.pi)), -np.pi)

    def test_episode_tail_padding_stays_zero_through_normalization(self):
        actions = np.ones((3, 7), dtype=np.float32)
        actions[:, 3:5] = 0.
        absolute, valid = episode_chunk(actions, 2, 50)
        self.assertEqual(valid.sum(), 1)
        anchor = np.arange(8, dtype=np.float32)
        relative = encode_chunk(absolute, anchor, valid)
        stats = {'mean': [2.]*7, 'std': [.5]*7}
        norm = normalize(relative, stats, valid)
        restored = decode_chunk(unnormalize(norm, stats, valid), anchor, valid)
        np.testing.assert_array_equal(norm[1:], 0.)
        np.testing.assert_array_equal(restored[1:], 0.)
        np.testing.assert_allclose(restored[0], actions[2], atol=1e-6)

    def test_random_roundtrip(self):
        rng = np.random.default_rng(48)
        for _ in range(100):
            anchor = rng.uniform(-3, 3, 8).astype(np.float32)
            action = rng.uniform(-3, 3, (50, 7)).astype(np.float32)
            action[:, 3:5] = 0.
            action[:, 6] = rng.integers(0, 2, 50)
            recovered = decode_chunk(encode_chunk(action, anchor), anchor)
            diff = recovered-action
            diff[:, 5] = wrap_yaw(diff[:, 5])
            self.assertLess(np.max(np.abs(diff)), 1e-6)


if __name__ == '__main__':
    unittest.main()
