"""The M3 scene/reset implementation, shared by collection and evaluation."""
import copy
import numpy as np
from pxr import Gf
import isaaclab.sim as sim_utils
from .manipulation_env import ManipulationEnv


class ScenarioEnv(ManipulationEnv):
    def reset(self, seed):
        state = super().reset(seed)
        block_state = self.block.data.default_root_state.clone()
        block_state[:, :3] = block_state.new_tensor(self.cfg['task']['block_position'])
        self.block.write_root_pose_to_sim(block_state[:, :7])
        self.block.write_root_velocity_to_sim(block_state[:, 7:])
        self.block.update(0.)
        self.sim.forward()
        return state

    def configure(self, config):
        delta = np.array(config['task']['bin_position'])-self.cfg['task']['bin_position']
        stage = sim_utils.get_current_stage()
        for name in ('Base', 'Stand', 'Left', 'Right', 'Front', 'Back'):
            attr = stage.GetPrimAtPath('/World/Bin'+name).GetAttribute('xformOp:translate')
            attr.Set(Gf.Vec3d(*(np.array(attr.Get())+delta)))
        self.cfg = copy.deepcopy(config)
        self.reset(config['seed'])
