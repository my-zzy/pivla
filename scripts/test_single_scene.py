#!/usr/bin/env python3
import unittest
from m5_prepare_single_scene import prepare

class SingleSceneTests(unittest.TestCase):
    def test_same_layout_distinct_resets_and_no_faults(self):
        plan=prepare();items=plan['episodes'];base=items[0]['config']
        self.assertEqual([i['split'] for i in items],['train']*6+['validation']*2)
        positions=[]
        for item in items:
            cfg=item['config'];positions.append(tuple(cfg['drone']['initial_position']))
            self.assertEqual(cfg['task'],base['task'])
            self.assertEqual(cfg['drone']['initial_yaw'],0.)
            self.assertEqual(cfg['drone']['reset_position_jitter_m'],0.)
            self.assertIsNone(item['fault'])
        self.assertEqual(len(set(positions)),8)
        self.assertGreater(base['task']['block_color'][2],base['task']['block_color'][0])

if __name__=='__main__':unittest.main()
