"""M2 physical grasping scene, retaining M1's force-driven flight scheduling."""
import copy

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg

from .env import HoverEnv
from .flight_controller import FlightController
from .gripper_controller import GripperController
from .scene_cfg import box


class ManipulationEnv(HoverEnv):
    def __init__(self, config, device='cuda:0', enable_cameras=True):
        super().__init__(config, device, enable_cameras)
        # Account for installed hardware in empty mass, never the unknown payload.
        control_config = copy.deepcopy(config)
        control_config['drone']['mass_kg'] += 2 * config['gripper']['finger_mass_kg']
        self.controller = FlightController(control_config, self.device)

    def create_scene(self, config):
        super().create_scene(config)
        task = config['task']
        x, y, _ = task['block_position']
        top = task['table_top_m']
        box('/World/TableTop', (.65, .65, .06), (x, y, top-.03), (.40, .25, .13))
        for i, (dx, dy) in enumerate([(-.25, -.25), (-.25, .25), (.25, -.25), (.25, .25)]):
            box(f'/World/TableLeg{i}', (.05, .05, top-.06), (x+dx, y+dy, (top-.06)/2), (.2, .22, .25))
        bx, by, bz = task['bin_position']
        sx, sy = task['bin_inner_size_m']
        h, w = task['bin_wall_height_m'], .025
        box('/World/BinBase', (sx+2*w, sy+2*w, .05), (bx, by, bz-.025), (.12, .38, .55))
        box('/World/BinStand', (.2, .2, bz-.05), (bx, by, (bz-.05)/2), (.2, .22, .25))
        for name, size, pos in [
            ('Left', (sx+2*w, w, h), (bx, by+(sy+w)/2, bz+h/2)),
            ('Right', (sx+2*w, w, h), (bx, by-(sy+w)/2, bz+h/2)),
            ('Front', (w, sy, h), (bx+(sx+w)/2, by, bz+h/2)),
            ('Back', (w, sy, h), (bx-(sx+w)/2, by, bz+h/2))]:
            box('/World/Bin'+name, size, pos, (.15, .48, .7))
        self.block = RigidObject(RigidObjectCfg(
            prim_path='/World/Block',
            spawn=sim_utils.CuboidCfg(
                size=tuple(task['block_size_m']),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False, linear_damping=0., angular_damping=0.,
                    solver_position_iteration_count=16, solver_velocity_iteration_count=4),
                mass_props=sim_utils.MassPropertiesCfg(mass=task['block_mass_kg']),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=.002, rest_offset=0.),
                physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1., dynamic_friction=1., restitution=0.),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(task.get('block_color', (.95, .30, .03))))),
            init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(task['block_position']))))

    def create_drone(self, config):
        drone = super().create_drone(config)
        self.gripper = GripperController(config)
        return drone

    def reset(self, seed):
        state = super().reset(seed)
        self.gripper.reset(state)
        block_state = self.block.data.default_root_state.clone()
        self.block.reset()
        self.block.write_root_pose_to_sim(block_state[:, :7])
        self.block.write_root_velocity_to_sim(block_state[:, 7:])
        self.block.update(0.)
        self.target_position[:] = self.target_position.new_tensor(self.cfg['target']['position'])
        self.sim.forward()
        return state

    def step(self, apply_control=True):
        state = super().step(apply_control)
        self.block.update(self.dt)
        self.gripper.update(self.dt)
        return state

    def observation(self):
        return {'drone_state': self.state()[0].cpu().tolist(),
                'block_state': self.block.data.root_state_w[0].cpu().tolist(),
                **self.gripper.measurements()}
