#!/usr/bin/env python3
"""CPU checks for wrench signs, equilibrium and saturation; no Sim startup."""
import importlib.util
import math
from pathlib import Path
import unittest

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
module_path = ROOT / 'airvla_isaac/source/airvla_isaac/airvla_isaac/sim/flight_controller.py'
spec = importlib.util.spec_from_file_location('flight_controller', module_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.config = yaml.safe_load((ROOT / 'configs/sim.yaml').read_text())
        self.controller = module.FlightController(self.config, 'cpu')
        self.state = torch.zeros((1, 13))
        self.state[:, 3] = 1.
        self.target = torch.zeros((1, 3))
        self.yaw = torch.zeros(1)

    def test_hover_equilibrium(self):
        force, torque = self.controller.compute(self.state, self.target, self.yaw)
        torch.testing.assert_close(force.flatten(), torch.tensor([0., 0., 9.81]))
        torch.testing.assert_close(torque, torch.zeros_like(torque))

    def test_translation_and_yaw_restoring_signs(self):
        self.target[0, 0] = 1.
        _, torque = self.controller.compute(self.state, self.target, self.yaw)
        self.assertGreater(torque[0, 0, 1].item(), 0.)  # Pitch + rotates thrust toward +X.
        self.target.zero_()
        self.target[0, 1] = 1.
        _, torque = self.controller.compute(self.state, self.target, self.yaw)
        self.assertLess(torque[0, 0, 0].item(), 0.)  # Roll - rotates thrust toward +Y.
        self.target.zero_()
        self.state[:, 3:7] = torch.tensor([math.cos(.1), 0., 0., math.sin(.1)])
        _, torque = self.controller.compute(self.state, self.target, self.yaw)
        self.assertLess(torque[0, 0, 2].item(), 0.)

    def test_actuator_limits(self):
        self.target[:] = torch.tensor([100., -100., 100.])
        self.state[:, 10:13] = torch.tensor([100., 100., 100.])
        force, torque = self.controller.compute(self.state, self.target, self.yaw)
        self.assertTrue(torch.isfinite(force).all() and torch.isfinite(torque).all())
        self.assertLessEqual(force[0, 0, 2].item(), self.config['controller']['max_thrust_n'])
        self.assertTrue((torque.abs() <= self.controller.torque_limit).all())

    def test_body_world_rotation(self):
        q = torch.tensor([[math.sqrt(.5), 0., math.sqrt(.5), 0.]])
        r = module.rotation_matrix(q)[0]
        torch.testing.assert_close(r @ torch.tensor([0., 0., 1.]), torch.tensor([1., 0., 0.]), atol=1e-6, rtol=0.)


if __name__ == '__main__':
    unittest.main()
