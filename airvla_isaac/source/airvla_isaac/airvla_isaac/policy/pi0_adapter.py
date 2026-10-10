"""Project-local M3/π0 boundary; shared LeRobot sources remain unchanged.

ChunkDataset owns relative conversion and numeric normalization. This adapter
only maps cameras, tokenizes text, moves tensors, and decodes predictions.
"""
import json
from pathlib import Path

import numpy as np
import torch
from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.pi0.configuration_pi0 import PI0Config
from lerobot.policies.pi0.modeling_pi0 import PI0Policy
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
from safetensors.torch import load_file
from transformers import AutoTokenizer

from .action_codec import decode_chunk, normalize, unnormalize


CAMERA_MAP = {
    'front': 'observation.images.base_0_rgb',
    'down': 'observation.images.left_wrist_0_rgb',
    'external': 'observation.images.right_wrist_0_rgb',
}


def policy_config(base, device='cpu', expert_only=True):
    cfg = PI0Config.from_pretrained(str(base), local_files_only=True)
    cfg.input_features = {name: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256))
                          for name in CAMERA_MAP.values()}
    cfg.input_features['observation.state'] = PolicyFeature(type=FeatureType.STATE, shape=(8,))
    cfg.output_features = {'action': PolicyFeature(type=FeatureType.ACTION, shape=(7,))}
    cfg.use_relative_actions = False
    cfg.normalization_mapping = {name: NormalizationMode.IDENTITY for name in ('VISUAL', 'STATE', 'ACTION')}
    cfg.dtype = 'bfloat16'
    cfg.device = device
    cfg.train_expert_only = expert_only
    cfg.freeze_vision_encoder = expert_only
    cfg.gradient_checkpointing = True
    cfg.compile_model = False
    cfg.push_to_hub = False
    return cfg


def remap_checkpoint_state(policy, state, config):
    embedding = 'model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight'
    saved_embedding = state.get(embedding, state.get(embedding.removeprefix('model.')))
    fixed = policy._fix_pytorch_state_dict_keys(state, config)
    fixed = {k if k.startswith('model.') else 'model.'+k: v for k,v in fixed.items()}
    # Base weights omit this embedding; upstream reconstructs it from lm_head.
    # A saved fine-tuned embedding must take precedence if it is already present.
    if saved_embedding is not None:
        fixed[embedding] = saved_embedding
    return fixed


def load_policy_strict(path, config):
    """Fail on missing/incompatible weights; upstream currently catches these errors."""
    weights = Path(path)/'model.safetensors'
    if not weights.is_file():
        raise FileNotFoundError(f'Complete pretrained weights required: {weights}')
    policy = PI0Policy(config)
    state = load_file(str(weights), device='cpu')
    state = remap_checkpoint_state(policy, state, config)
    policy.load_state_dict(state, strict=True)
    return policy


def masked_flow_loss(losses, action_is_pad, action_dim=7):
    """Exclude unavailable time steps AND model-only padded action dimensions."""
    losses = losses[:, :, :action_dim]
    valid = ~action_is_pad.to(device=losses.device, dtype=torch.bool)
    if valid.shape != losses.shape[:2] or not valid.any():
        raise ValueError('Expected at least one valid action and matching [B,H] padding mask')
    return losses.masked_select(valid[:, :, None].expand_as(losses)).mean()


def training_loss(policy, batch):
    images, image_masks = policy._preprocess_images(batch)
    state, actions = policy.prepare_state(batch), policy.prepare_action(batch)
    noise = policy.model.sample_noise(actions.shape, actions.device)
    time = policy.model.sample_time(actions.shape[0], actions.device)
    losses = policy.model(images, image_masks, batch[OBS_LANGUAGE_TOKENS],
                          batch[OBS_LANGUAGE_ATTENTION_MASK], state, actions, noise, time)
    return masked_flow_loss(losses, batch['action_is_pad'])


class Pi0BatchAdapter:
    def __init__(self, dataset_root, tokenizer_path, device='cuda', tokenizer=None):
        root = Path(dataset_root)
        self.schema = json.loads((root/'schema.json').read_text())
        self.stats = json.loads((root/'normalization.json').read_text())
        self.device = device
        # Injection is used only by offline unit tests; production uses the approved local tokenizer.
        self.tokenizer = tokenizer if tokenizer is not None else AutoTokenizer.from_pretrained(
            str(tokenizer_path), local_files_only=True)
        self.tokenizer_path = str(tokenizer_path)

    def __call__(self, rows):
        if rows['observation.state'].ndim != 2 or rows['observation.state'].shape[-1] != 8:
            raise ValueError('Pass a collated ChunkDataset batch with state [B,8]')
        batch = {'observation.state': rows['observation.state'].to(self.device)}
        for source, target in CAMERA_MAP.items():
            batch[target] = rows['observation.images.'+source].to(self.device)
        for name in ('action', 'action_is_pad'):
            if name in rows:
                batch[name] = rows[name].to(self.device)
        encoded = self.tokenizer([text+'\n' for text in rows['task']], max_length=48,
            truncation=True, padding='max_length', padding_side='right', return_tensors='pt')
        batch[OBS_LANGUAGE_TOKENS] = encoded['input_ids'].to(self.device)
        batch[OBS_LANGUAGE_ATTENTION_MASK] = encoded['attention_mask'].to(self.device, dtype=torch.bool)
        return batch

    def decode(self, predictions, anchors, action_is_pad=None):
        pred = predictions.detach().float().cpu().numpy()
        anchors = anchors.detach().float().cpu().numpy()
        masks = None if action_is_pad is None else ~action_is_pad.cpu().numpy().astype(bool)
        return np.stack([decode_chunk(unnormalize(p, self.stats['action'],
            None if masks is None else masks[i]), anchors[i], None if masks is None else masks[i])
            for i,p in enumerate(pred)])

    def from_observation(self, state, images):
        """Live counterpart of ChunkDataset: raw state and three HWC uint8 images."""
        state = np.asarray(state, dtype=np.float32)
        if state.shape != (8,) or not np.isfinite(state).all():
            raise ValueError('Live state must contain eight finite values')
        rows = {'observation.state': torch.from_numpy(normalize(state, self.stats['observation.state']))[None],
                'task': [self.schema['task']]}
        for name in CAMERA_MAP:
            rgb = np.asarray(images[name])
            if rgb.shape != (256, 256, 3) or rgb.dtype != np.uint8:
                raise ValueError(f'Expected HWC uint8 RGB image for {name}')
            rows['observation.images.'+name] = torch.from_numpy(rgb.copy()).permute(2, 0, 1)[None].float()/255.
        return self(rows)

    def save_contract(self, destination):
        dest = Path(destination)
        for name, value in [('schema.json',self.schema), ('normalization.json',self.stats),
                            ('adapter.json', {'camera_map':CAMERA_MAP, 'tokenizer':'tokenizer',
                             'numeric_preprocessing':'ChunkDataset: single anchor, then training-only normalization',
                             'image_preprocessing':'PI0Policy: resize with padding to 224 and map [0,1] to [-1,1]',
                             'action_loss':'mean over valid times and first seven action dimensions'})]:
            (dest/name).write_text(json.dumps(value,indent=2)+'\n')
        self.tokenizer.save_pretrained(dest/'tokenizer')
