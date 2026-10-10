"""Position/yaw geometric PD controller with a body-frame resultant wrench.

World is right-handed Z-up; body is X-forward/Y-left/Z-up. Quaternions
are wxyz, mapping body to world. No pose or velocity writes occur here.
"""
import math

import torch
import torch.nn.functional as F


def rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    """Convert (..., 4) unit wxyz quaternions to (..., 3, 3) rotations."""
    w, x, y, z = F.normalize(q, dim=-1).unbind(-1)
    return torch.stack((
        1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w),
        2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w),
        2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y),
    ), dim=-1).reshape(*q.shape[:-1], 3, 3)


class FlightController:
    def __init__(self, config, device):
        self.cfg = config['controller']
        self.mass = config['drone']['mass_kg']
        self.gravity = config['gravity']
        self.inertia = torch.tensor(config['drone']['inertia_kg_m2'], device=device)
        self.kp = torch.tensor(self.cfg['position_kp'], device=device)
        self.kd = torch.tensor(self.cfg['velocity_kd'], device=device)
        self.kr = torch.tensor(self.cfg['attitude_kp'], device=device)
        self.kw = torch.tensor(self.cfg['angular_velocity_kd'], device=device)
        self.torque_limit = torch.tensor(self.cfg['max_torque_nm'], device=device)

    def compute(self, state, target_position, target_yaw):
        """State: [position_W(3), quaternion_WB(4), velocity_W(3), omega_W(3)].

        Recompute at 100 Hz. Desired angular-rate feedforward is zero for the
        hover/slow-setpoint baseline. Torque includes omega x J omega.
        """
        r = rotation_matrix(state[:, 3:7])
        accel = self.kp * (target_position - state[:, :3]) - self.kd * state[:, 7:10]
        accel[:, 2] += self.gravity
        accel[:, 2].clamp_(min=0.5, max=self.cfg['max_thrust_n'] / self.mass)
        horizontal = accel[:, :2].norm(dim=-1, keepdim=True).clamp_min(1e-6)
        limit = accel[:, 2:3] * math.tan(self.cfg['max_tilt_rad'])
        accel[:, :2] *= (limit / horizontal).clamp(max=1.0)
        desired_z = F.normalize(accel, dim=-1)
        heading = torch.stack((torch.cos(target_yaw), torch.sin(target_yaw),
                               torch.zeros_like(target_yaw)), dim=-1)
        desired_y = F.normalize(torch.cross(desired_z, heading, dim=-1), dim=-1)
        desired_x = torch.cross(desired_y, desired_z, dim=-1)
        desired_r = torch.stack((desired_x, desired_y, desired_z), dim=-1)
        skew = 0.5 * (desired_r.transpose(-1, -2) @ r - r.transpose(-1, -2) @ desired_r)
        attitude_error = torch.stack((skew[:, 2, 1], skew[:, 0, 2], skew[:, 1, 0]), dim=-1)
        omega = (r.transpose(-1, -2) @ state[:, 10:13, None]).squeeze(-1)
        torque = -self.kr * attitude_error - self.kw * omega + torch.cross(omega, self.inertia * omega, dim=-1)
        torque = torch.clamp(torque, -self.torque_limit, self.torque_limit)
        thrust = (self.mass * accel * r[:, :, 2]).sum(-1).clamp(0, self.cfg['max_thrust_n'])
        force = torch.zeros_like(torque)
        force[:, 2] = thrust
        return force[:, None, :].contiguous(), torque[:, None, :].contiguous()
