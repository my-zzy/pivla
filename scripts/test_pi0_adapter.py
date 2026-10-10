#!/usr/bin/env python3
"""Offline boundary checks; tokenizer stub is never used for model training."""
import unittest
import json
from pathlib import Path
from PIL import Image

import numpy as np
import torch
from torch.utils.data import default_collate

from airvla_isaac.data.chunk_dataset import ChunkDataset
from airvla_isaac.policy.pi0_adapter import CAMERA_MAP, Pi0BatchAdapter, masked_flow_loss, policy_config, remap_checkpoint_state
from lerobot.policies.pi0.modeling_pi0 import PI0Policy
from airvla_isaac.policy.action_codec import episode_chunk


class TokenizerStub:
    def __call__(self, texts, **kwargs):
        self.texts, self.kwargs = texts, kwargs
        return {'input_ids':torch.ones(len(texts),48,dtype=torch.long),
                'attention_mask':torch.ones(len(texts),48,dtype=torch.long)}


class AdapterChecks(unittest.TestCase):
    def test_live_matches_recorded_preprocessing(self):
        dataset = ChunkDataset('datasets/airvla_m3')
        raw_dir = Path('artifacts/M3/episode_000')
        raw = json.loads((raw_dir/'samples.jsonl').read_text().splitlines()[0])
        images = {}
        for name in CAMERA_MAP:
            with Image.open(raw_dir/raw['camera']['cameras'][name]['path']) as im:
                images[name] = np.asarray(im.convert('RGB'))
        adapter = Pi0BatchAdapter('datasets/airvla_m3', 'unused', device='cpu', tokenizer=TokenizerStub())
        live = adapter.from_observation(raw['state'], images)
        recorded = adapter(default_collate([dataset[0]]))
        for key in live:
            torch.testing.assert_close(live[key], recorded[key], atol=0, rtol=0)

    def test_reload_preserves_a_trained_embedding(self):
        # Exercise the actual upstream remapper without constructing its 4B model.
        remapper=type('Remapper',(),{'_fix_pytorch_state_dict_keys':PI0Policy._fix_pytorch_state_dict_keys})()
        key='model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight'
        head='model.paligemma_with_expert.paligemma.lm_head.weight'
        state={key:torch.ones(2,2),head:torch.zeros(2,2)}
        fixed=remap_checkpoint_state(remapper,state,None)
        torch.testing.assert_close(fixed[key],state[key])
        reconstructed=remap_checkpoint_state(remapper,{head:state[head]},None)
        torch.testing.assert_close(reconstructed[key],state[head])

    def test_actual_dataset_mapping_and_decode(self):
        ds=ChunkDataset('datasets/airvla_m3')
        rows=default_collate([ds[0],ds[516]])
        tokenizer=TokenizerStub()
        adapter=Pi0BatchAdapter('datasets/airvla_m3','unused',device='cpu',tokenizer=tokenizer)
        batch=adapter(rows)
        torch.testing.assert_close(batch['action'],rows['action'],rtol=0,atol=0)
        torch.testing.assert_close(batch['observation.state'],rows['observation.state'],rtol=0,atol=0)
        for source,target in CAMERA_MAP.items():
            torch.testing.assert_close(batch[target],rows['observation.images.'+source],rtol=0,atol=0)
        self.assertEqual(tokenizer.texts,[s+'\n' for s in rows['task']])
        self.assertEqual(tokenizer.kwargs['padding_side'],'right')
        decoded=adapter.decode(batch['action'],rows['anchor_state'],rows['action_is_pad'])
        actions=np.array(ds.dataset.hf_dataset.select_columns(['action']).with_format('numpy')[:]['action'])[:517]
        for i,index in enumerate((0,516)):
            expected,valid=episode_chunk(actions,index,50)
            np.testing.assert_allclose(decoded[i,valid],expected[valid],atol=1e-6)
            np.testing.assert_array_equal(decoded[i,~valid],0.)
        self.assertNotIn('anchor_state',batch)
        self.assertNotIn('episode_index',batch)

    def test_padding_has_no_loss_or_gradient(self):
        losses=torch.ones(2,3,32,requires_grad=True)
        mask=torch.tensor([[False,False,True],[False,True,True]])
        with torch.no_grad():
            losses[:,:,7:]=1e6
            losses[mask]=1e6
        result=masked_flow_loss(losses,mask)
        self.assertEqual(result.item(),1.)
        result.backward()
        self.assertTrue(torch.all(losses.grad[mask]==0))
        self.assertTrue(torch.all(losses.grad[:,:,7:]==0))
        self.assertAlmostEqual(losses.grad.sum().item(),1.,places=6)

    def test_config_disables_second_conversion(self):
        cfg=policy_config('checkpoints/pi0_base')
        self.assertFalse(cfg.use_relative_actions)
        self.assertTrue(all(v.value=='IDENTITY' for v in cfg.normalization_mapping.values()))
        self.assertEqual(cfg.input_features['observation.state'].shape,(8,))
        self.assertEqual(cfg.output_features['action'].shape,(7,))
        self.assertEqual(list(cfg.image_features),list(CAMERA_MAP.values()))
        self.assertFalse(cfg.push_to_hub)


if __name__=='__main__':
    unittest.main()
