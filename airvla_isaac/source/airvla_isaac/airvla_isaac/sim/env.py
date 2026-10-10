"""Single-drone environment with explicit physics/control/render scheduling.

Root-state writes are confined to reset(). During flight only a body-frame
thrust and torque act on a gravity-enabled PhysX rigid body. This standalone
environment lives in the external Isaac Lab package; M0's Gym task is retained
as a historical installation smoke test.
"""
import math

import torch

import isaaclab.sim as sim_utils

from .cameras import aim_external, create_cameras, update_body_mounts
from .drone_cfg import create_drone
from .flight_controller import FlightController
from .scene_cfg import create_room


class HoverEnv:
    def __init__(self, config, device='cuda:0', enable_cameras=True):
        self.cfg = config
        self.dt = 1. / config['physics_hz']
        for rate in ('controller_hz', 'camera_hz'):
            if config['physics_hz'] % config[rate]:
                raise ValueError(f'{rate} must divide physics_hz')
        self.control_interval = config['physics_hz'] // config['controller_hz']
        self.camera_interval = config['physics_hz'] // config['camera_hz']
        self.viewport_interval = max(1, config['physics_hz'] // 50)
        self.sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(
            # GUI clock is 50 Hz; explicit sensor capture stays at camera_hz.
            dt=self.dt, render_interval=self.viewport_interval, device=device,
            # The headless rendering experience disables USD physics writeback.
            # Fabric propagates the physical drone pose to rendered geometry.
            gravity=(0., 0., -config['gravity']), use_fabric=True,
        ))
        self.device = self.sim.device
        self.create_scene(config)
        self.drone = self.create_drone(config)
        self.cameras = create_cameras(config) if enable_cameras else {}
        self.controller = FlightController(config, self.device)
        self.target_position = torch.tensor([config['target']['position']], device=self.device)
        self.target_yaw = torch.tensor([config['target']['yaw']], device=self.device)
        self.sim.reset()
        if self.sim.has_gui():
            observer = config['cameras']['external']
            self.sim.set_camera_view(eye=observer['position'], target=observer['look_at'])
        if self.cameras:
            aim_external(self.cameras, config, self.device)
        self.reset(config['seed'])

    def create_scene(self, config):
        create_room(config)

    def create_drone(self, config):
        return create_drone(config)

    def reset(self, seed):
        # Local CPU generator: repeatable jitter, no modification of global random state.
        generator = torch.Generator(device='cpu').manual_seed(seed)
        state = self.drone.data.default_root_state.clone()
        jitter = (2 * torch.rand((1, 3), generator=generator) - 1)
        state[:, :3] = torch.tensor(self.cfg['drone']['initial_position'], device=self.device)
        state[:, :3] += jitter.to(self.device) * self.cfg['drone']['reset_position_jitter_m']
        yaw = self.cfg['drone']['initial_yaw']
        state[:, 3:7] = torch.tensor([math.cos(yaw/2), 0., 0., math.sin(yaw/2)], device=self.device)
        state[:, 7:] = 0
        self.drone.reset()
        self.drone.write_root_pose_to_sim(state[:, :7])
        self.drone.write_root_velocity_to_sim(state[:, 7:])
        self.drone.update(0.)
        self.step_count = 0
        self.physics_step_origin = self.sim.current_time_step_index
        self.force = torch.zeros((1, 1, 3), device=self.device)
        self.torque = torch.zeros_like(self.force)
        self.sim.forward()
        for camera in self.cameras.values():
            camera.reset()
        if self.cameras:
            update_body_mounts(self.cameras, self.cfg, state)
        return self.state()

    def state(self):
        return self.drone.data.root_state_w.clone()

    def step(self, apply_control=True):
        if self.step_count % self.control_interval == 0:
            if apply_control:
                self.force, self.torque = self.controller.compute(self.state(), self.target_position, self.target_yaw)
            else:
                self.force.zero_()
                self.torque.zero_()
            # Set (not add): the wrench persists between control ticks and is expressed in body coordinates.
            self.drone.permanent_wrench_composer.set_forces_and_torques(
                forces=self.force, torques=self.torque, is_global=False)
        self.drone.write_data_to_sim()
        # Always advance exactly one physics tick. In this Isaac Sim version,
        # step(render=True) advances a whole rendering interval,
        # not one 0.005 s tick, leaving the controller inactive between updates.
        before = self.sim.current_time_step_index
        self.sim.step(render=False)
        if self.sim.current_time_step_index - before != 1:
            raise RuntimeError('Expected exactly one physics tick per environment step')
        # Refresh the desktop viewport separately; recorded cameras remain at 5 Hz.
        render_gui = self.sim.has_gui() and (self.step_count + 1) % self.viewport_interval == 0
        if render_gui:
            self.render()
        self.drone.update(self.dt)
        self.step_count += 1
        return self.state()

    def capture(self):
        update_body_mounts(self.cameras, self.cfg, self.state())
        # Rendering does not advance physics; flush transforms and complete the RGB render.
        self.render()
        self.render()
        frames = {}
        for name, camera in self.cameras.items():
            camera.update(1. / self.cfg['camera_hz'], force_recompute=True)
            data = camera.data
            frames[name] = {
                'rgb': data.output['rgb'][0, ..., :3].cpu().numpy().copy(),
                'position_w': data.pos_w[0].cpu().tolist(),
                'quaternion_world_wxyz': data.quat_w_world[0].cpu().tolist(),
                'intrinsic_matrix': data.intrinsic_matrices[0].cpu().tolist(),
            }
        return frames

    def warmup_rendering(self):
        for _ in range(8):
            self.render()

    def render(self):
        """Refresh the GUI/sensors without advancing the physical simulation."""
        before = self.sim.current_time_step_index
        self.sim.render()
        if self.sim.current_time_step_index != before:
            raise RuntimeError('Rendering unexpectedly advanced physics')

    @property
    def physics_steps(self):
        return self.sim.current_time_step_index - self.physics_step_origin

    @property
    def time(self):
        return self.step_count * self.dt
