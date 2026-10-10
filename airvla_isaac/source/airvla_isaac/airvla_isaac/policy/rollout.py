"""Deployment numeric contract and a bounded, non-overlapping action queue."""
from collections import deque
import numpy as np
from .action_codec import decode_chunk, encode_chunk, normalize, unnormalize


def decode_prediction(prediction, anchor, stats):
    return decode_chunk(unnormalize(prediction, stats['action']), anchor)


def replay_chunk(absolute, anchor, stats):
    encoded = normalize(encode_chunk(absolute, anchor), stats['action'])
    return decode_prediction(encoded, anchor, stats)


def checked_action(action, settings):
    action = np.asarray(action, dtype=np.float32)
    if action.shape != (7,) or not np.isfinite(action).all():
        raise ValueError('Action must contain seven finite values')
    lo, hi = settings['safe_z_m']
    if not lo <= action[2] <= hi or np.linalg.norm(action[:2]) > settings['safe_xy_radius_m']:
        raise ValueError(f'Predicted target outside test volume: {action.tolist()}')
    return action


class ActionQueue:
    def __init__(self, settings):
        self.settings = settings
        self.actions = deque()
        self.pending = None

    def request(self, request_id):
        if self.pending is not None or self.actions:
            raise RuntimeError('Naive policy requests only when its queue is empty')
        self.pending = request_id

    def receive(self, request_id, prediction, anchor, stats):
        if self.pending != request_id:
            raise ValueError('Stale or unexpected inference response')
        prediction = np.asarray(prediction)
        if prediction.shape != (self.settings['action_horizon'], 7) or not np.isfinite(prediction).all():
            raise ValueError('Invalid predicted chunk')
        # Decode once using the state at observation time, never the arrival-time state.
        absolute = decode_prediction(prediction, anchor, stats)
        self.actions.extend(checked_action(a, self.settings) for a in absolute[:self.settings['execute_actions']])
        self.pending = None

    def pop(self):
        return self.actions.popleft() if self.actions else None
