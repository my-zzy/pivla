import copy,json,unittest
from pathlib import Path
from airvla_isaac.data.recovery_expert import RecoveryExpert

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
if __name__=='__main__':unittest.main()
