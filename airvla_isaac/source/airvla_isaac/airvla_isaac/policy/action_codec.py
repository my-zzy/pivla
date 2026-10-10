"""World-frame chunk deltas with one observed-pose anchor, before normalization."""
import numpy as np


def wrap_yaw(angle):
    return (np.asarray(angle) + np.pi) % (2*np.pi) - np.pi


def state_from_raw(drone_state, aperture):
    s = np.asarray(drone_state, dtype=np.float64)
    w, x, y, z = s[3:7]
    yaw = np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))
    return np.array([*s[:3], wrap_yaw(yaw), *s[7:10], aperture], dtype=np.float32)


def encode_chunk(absolute, anchor, valid=None):
    action = np.array(absolute, dtype=np.float64, copy=True)
    anchor = np.asarray(anchor)
    if action.ndim != 2 or action.shape[1] != 7 or anchor.shape != (8,):
        raise ValueError('Expected action [H,7] and observation state [8]')
    action[:, :3] -= anchor[:3]
    action[:, 3:5] = 0.
    action[:, 5] = wrap_yaw(action[:, 5] - anchor[3])
    if valid is not None:
        action[~np.asarray(valid, dtype=bool)] = 0.
    return action.astype(np.float32)


def decode_chunk(relative, anchor, valid=None):
    action = np.array(relative, dtype=np.float64, copy=True)
    action[:, :3] += np.asarray(anchor)[:3]
    action[:, 3:5] = 0.
    action[:, 5] = wrap_yaw(action[:, 5] + anchor[3])
    if valid is not None:
        action[~np.asarray(valid, dtype=bool)] = 0.
    return action.astype(np.float32)


def episode_chunk(actions, index, horizon):
    if not 0 <= index < len(actions):
        raise IndexError(index)
    count = min(horizon, len(actions)-index)
    out = np.zeros((horizon, 7), dtype=np.float32)
    out[:count] = actions[index:index+count]
    return out, np.arange(horizon) < count


def normalize(values, stats, valid=None):
    out = (np.asarray(values)-np.asarray(stats['mean']))/np.asarray(stats['std'])
    if valid is not None:
        out[~np.asarray(valid, dtype=bool)] = 0.
    return out.astype(np.float32)


def unnormalize(values, stats, valid=None):
    out = np.asarray(values)*np.asarray(stats['std'])+np.asarray(stats['mean'])
    if valid is not None:
        out[~np.asarray(valid, dtype=bool)] = 0.
    return out.astype(np.float32)
