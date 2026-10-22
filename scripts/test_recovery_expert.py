import copy,json,math,unittest
from pathlib import Path
from airvla_isaac.data.recovery_expert import RecoveryExpert
from airvla_isaac.data.recovery_plan import recovery_plan

class Tests(unittest.TestCase):
 def setUp(self):
  self.cfg=json.loads(Path('artifacts/M3/episode_000/config.json').read_text())
  p=[*self.cfg['task']['block_position'][:2],1.4]
  self.obs={'drone_state':p+[1,0,0,0,0,0,0,0,0,0],'block_state':self.cfg['task']['block_position']+[1,0,0,0,0,0,0,0,0,0],'contacts':[False,False],'aperture_m':.14}
 def test_no_eight_second_wait(self):
  e=RecoveryExpert(self.cfg)
  for k in range(8):e.update(k*.1,self.obs)
  self.assertEqual(e.stage,'descend')
 def test_missed_closure_reopens(self):
  e=RecoveryExpert(self.cfg);e.stage='grasp'
  _,opened=e.update(2.1,self.obs)
  self.assertTrue(opened);self.assertEqual(e.stage,'recover')
 def test_lost_contact_recovers(self):
  e=RecoveryExpert(self.cfg);e.stage='lift'
  for k in range(4):e.update(k*.1,self.obs)
  self.assertEqual(e.stage,'recover');self.assertEqual(e.retries,1)
 def test_live_block_alignment(self):
  e=RecoveryExpert(self.cfg);e.update(0,self.obs)
  o=copy.deepcopy(self.obs);o['block_state'][0]+=.1
  target,_=e.update(.1,o)
  self.assertGreater(target[0],self.obs['drone_state'][0])
 def test_retry_bound(self):
  e=RecoveryExpert(self.cfg)
  for k in range(3):e.recover(k,self.obs,'test')
  with self.assertRaises(RuntimeError):e.recover(4,self.obs,'test')
 def test_heading_must_settle_before_descent(self):
  e=RecoveryExpert(self.cfg)
  self.obs['drone_state'][3:7]=[math.cos(.15/2),0,0,math.sin(.15/2)]
  for k in range(10):e.update(k*.1,self.obs)
  self.assertEqual(e.stage,'approach')
  self.obs['drone_state'][3:7]=[1,0,0,0]
  for k in range(10,18):e.update(k*.1,self.obs)
  self.assertEqual(e.stage,'descend')
 def test_lift_does_not_follow_attached_payload_drift(self):
  e=RecoveryExpert(self.cfg);e.update(0,self.obs)
  self.obs['contacts']=[True,True]
  e.transition('lift',0,self.obs,'test grasp')
  anchor=e.lift_xy.copy()
  self.obs['block_state'][0]+=.2
  self.obs['block_state'][1]-=.2
  target,_=e.update(.1,self.obs)
  self.assertAlmostEqual(target[0],anchor[0])
  self.assertAlmostEqual(target[1],anchor[1])
  e.transition('lift',.2,self.obs,'test retry grasp')
  self.assertNotEqual(float(e.lift_xy[0]),float(anchor[0]))
 def test_collection_preserves_every_original_training_reset(self):
  manifest=json.loads(Path('artifacts/M3/collection_manifest.json').read_text())
  before=copy.deepcopy(manifest);plan=recovery_plan(manifest,'collect')
  originals=[x for x in plan if x['case']=='original']
  expected=[x for x in manifest['episodes'] if x['split']=='train']
  self.assertEqual(len(originals),16)
  self.assertEqual([x['config'] for x in originals],[x['config'] for x in expected])
  self.assertTrue(all(x['source_episode_id']<16 and x['split']=='train' and not x['fault'] for x in plan))
  yaws=[x['config']['drone']['initial_yaw'] for x in plan]
  self.assertLessEqual(min(yaws),-.15);self.assertGreaterEqual(max(yaws),.15)
  self.assertEqual(manifest,before)
 def test_pilot_contains_nonzero_yaw_recovery_and_low_approaches(self):
  manifest=json.loads(Path('artifacts/M3/collection_manifest.json').read_text())
  plan=recovery_plan(manifest,'pilot')
  self.assertEqual(len(plan),10)
  faults=[x for x in plan if x['fault']]
  self.assertEqual(len(faults),4)
  self.assertEqual({x['fault'] for x in faults},{'missed_closure','lost_contact'})
  self.assertEqual({x['config']['drone']['initial_yaw'] for x in faults},{-.15,.15})
  self.assertEqual(sum(x['case']=='near_grasp' for x in plan),2)
if __name__=='__main__':unittest.main()
