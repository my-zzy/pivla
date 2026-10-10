#!/usr/bin/env python3
"""Check statistics use relative future chunks and never read held-out episodes."""
import json
from pathlib import Path
import tempfile
import unittest

from airvla_isaac.data.convert_to_lerobot import compute_normalization


class StatisticsChecks(unittest.TestCase):
    def test_training_only_valid_chunk_distribution(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            episode = root/'episode_000'
            episode.mkdir()
            rows = [{'state': [float(i),0.,0.,0.,0.,0.,0.,.14],
                     'action': [float(i+1),0.,0.,0.,0.,0.,float(i==1)]} for i in range(3)]
            (episode/'samples.jsonl').write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
            manifest = {'schema': {'chunk_size': 50, 'normalization_std_floor': 1e-6},
                        'episodes': [{'episode_id':0,'split':'train'},
                                     {'episode_id':1,'split':'validation'}]}
            # The held-out episode deliberately has no file: even reading it is a bug.
            stats = compute_normalization(root,manifest)
            self.assertEqual(stats['source_episode_ids'],[0])
            self.assertEqual(stats['observation.state']['count'],3)
            self.assertEqual(stats['action']['count'],6)
            self.assertAlmostEqual(stats['action']['mean'][0],10/6)
            self.assertAlmostEqual(stats['action']['mean'][6],2/6)
            self.assertEqual(stats['action']['std'][3:5],[1.,1.])


if __name__ == '__main__':
    unittest.main()
