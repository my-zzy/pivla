"""Raw, timestamped M2 evidence; LeRobot conversion is a separate M3 step."""
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from ..sim.flight_controller import rotation_matrix


class EpisodeRecorder:
    def __init__(self, output, config):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.flight = (self.output / 'flight.jsonl').open('w')
        self.cameras = (self.output / 'cameras.jsonl').open('w')
        self.frame_count = 0
        self.started = time.perf_counter()
        self.max_mount_position_error_m = 0.
        self.max_mount_rotation_error = 0.

    def record(self, env, obs, stage):
        row = {'time_s': env.time, 'stage': stage, **obs,
               'target_position_w': env.target_position[0].cpu().tolist(),
               'target_yaw': float(env.target_yaw[0]),
               'force_b_n': env.force[0, 0].cpu().tolist(),
               'torque_b_nm': env.torque[0, 0].cpu().tolist(),
               'action_time_s': ((env.step_count-1)//(self.config['physics_hz']//self.config['action_hz']))/self.config['action_hz'],
               'control_time_s': ((env.step_count-1)//env.control_interval)*env.control_interval*env.dt}
        self.flight.write(json.dumps(row, allow_nan=False)+'\n')
        return row

    def capture(self, env, stage):
        frames = env.capture()
        width, height = self.config['cameras']['width'], self.config['cameras']['height']
        mosaic = Image.new('RGB', (3*width, height+40), '#172132')
        draw = ImageDraw.Draw(mosaic)
        record = {'frame': self.frame_count, 'capture_time_s': env.time, 'available_time_s': env.time,
                  'available_wall_elapsed_s': time.perf_counter()-self.started, 'stage': stage, 'cameras': {}}
        state = env.state()[0].cpu()
        body_r = rotation_matrix(state[None, 3:7])[0].numpy()
        for index, (name, frame) in enumerate(frames.items()):
            rgb = frame.pop('rgb')
            if rgb.shape != (height, width, 3) or str(rgb.dtype) != 'uint8' or rgb.std() < 5.:
                raise RuntimeError(f'Invalid RGB frame for {name}')
            mount = self.config['cameras'][name]
            expected_position = np.array(mount['position'])
            if name != 'external':
                expected_position = state[:3].numpy() + body_r @ expected_position
                expected_r = body_r @ rotation_matrix(torch.tensor([mount['quaternion']]))[0].numpy()
                actual_r = rotation_matrix(torch.tensor([frame['quaternion_world_wxyz']]))[0].numpy()
                self.max_mount_rotation_error = max(self.max_mount_rotation_error, float(np.max(np.abs(expected_r-actual_r))))
            self.max_mount_position_error_m = max(self.max_mount_position_error_m,
                float(np.linalg.norm(np.array(frame['position_w'])-expected_position)))
            if self.max_mount_position_error_m > 1e-4 or self.max_mount_rotation_error > 1e-4:
                raise RuntimeError('Recorded camera mount differs from the physical drone transform')
            directory = self.output / 'frames' / name
            directory.mkdir(parents=True, exist_ok=True)
            relative = f'frames/{name}/{self.frame_count:04d}.png'
            image = Image.fromarray(rgb)
            image.save(self.output / relative)
            mosaic.paste(image, (index*width, 40))
            draw.text((index*width+8, 5), f'{name.upper()}  t={env.time:.1f}s\n{stage}', fill='white')
            record['cameras'][name] = {'path': relative, **frame}
        mosaic.save(self.output / 'frames' / f'{self.frame_count:04d}.png')
        mosaic.save(self.output / 'camera_preview.png')
        self.cameras.write(json.dumps(record, allow_nan=False)+'\n')
        self.frame_count += 1

    def close(self):
        self.flight.close()
        self.cameras.close()
