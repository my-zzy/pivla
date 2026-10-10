"""Three RGB sensors, with body-mounted front/down cameras and a fixed observer."""
import torch

import isaaclab.sim as sim_utils
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.utils.math import quat_apply, quat_mul


def create_cameras(config):
    cfg = config['cameras']
    cameras = {}
    sim_utils.create_prim('/World/Sensors', 'Xform')
    for name in ('front', 'down', 'external'):
        mount = cfg[name]
        # World-parented sensor prims avoid stale USD/Fabric parent transforms
        # in the pinned Lab camera view. Fixed body extrinsics are composed from
        # the actual PhysX pose immediately before each capture.
        parent = '/World/Sensors'
        cameras[name] = Camera(CameraCfg(
            prim_path=f'{parent}/{name}_camera',
            update_period=0.,  # Explicitly captured only after rendering, at camera_hz.
            height=cfg['height'], width=cfg['width'], data_types=['rgb'],
            update_latest_camera_pose=True,
            offset=CameraCfg.OffsetCfg(
                pos=tuple(mount['position']),
                rot=tuple(mount.get('quaternion', (1., 0., 0., 0.))), convention='world',
            ),
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=cfg['focal_length_mm'], horizontal_aperture=cfg['horizontal_aperture_mm'],
                clipping_range=tuple(cfg['clipping_range_m']), focus_distance=3.,
            ),
        ))
    return cameras


def aim_external(cameras, config, device):
    cfg = config['cameras']['external']
    cameras['external'].set_world_poses_from_view(
        torch.tensor([cfg['position']], device=device),
        torch.tensor([cfg['look_at']], device=device),
    )


def update_body_mounts(cameras, config, state):
    """Compose fixed camera extrinsics with the current physical drone pose.

    Only the virtual sensor prims are positioned. The drone remains entirely
    physics-driven. This avoids the pinned view's USD/Fabric parent mismatch.
    """
    for name in ('front', 'down'):
        mount = config['cameras'][name]
        offset = torch.tensor([mount['position']], device=state.device)
        rotation = torch.tensor([mount['quaternion']], device=state.device)
        cameras[name].set_world_poses(
            state[:, :3] + quat_apply(state[:, 3:7], offset),
            quat_mul(state[:, 3:7], rotation), convention='world',
        )
