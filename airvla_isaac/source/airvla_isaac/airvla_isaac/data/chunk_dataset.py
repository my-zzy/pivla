"""LeRobot loader adapter for the frozen M3 chunk convention.

M4 must use these statistics/conventions exactly once in its training processor.
The simulator's privileged object/contact fields are never model observations.
"""
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from ..policy.action_codec import encode_chunk, normalize


class ChunkDataset(Dataset):
    def __init__(self, root, split='train', normalize_features=True):
        root = Path(root)
        self.schema = json.loads((root/'schema.json').read_text())
        self.stats = json.loads((root/'normalization.json').read_text())
        self.normalize_features = normalize_features
        self.dataset = LeRobotDataset(self.schema['repo_id']+'_'+split, root=root/split,
            delta_timestamps={'action': [k/self.schema['fps'] for k in range(self.schema['chunk_size'])]},
            video_backend='pyav')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        row = self.dataset[index]
        anchor = row['observation.state'].numpy()
        valid = ~row['action_is_pad'].numpy()
        relative = encode_chunk(row['action'].numpy(), anchor, valid)
        state = anchor.copy()
        if self.normalize_features:
            relative = normalize(relative, self.stats['action'], valid)
            state = normalize(state, self.stats['observation.state'])
        return {**{k:row[k] for k in row if k.startswith('observation.images.')},
                'observation.state': torch.from_numpy(state), 'action': torch.from_numpy(relative),
                'action_is_pad': torch.from_numpy(~valid), 'anchor_state': torch.from_numpy(anchor.copy()),
                'task': row['task'], 'episode_index': row['episode_index'], 'frame_index': row['frame_index']}
