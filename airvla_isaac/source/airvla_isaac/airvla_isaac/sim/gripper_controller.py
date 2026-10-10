"""Two independently driven physical fingers; no grasp attachment or pose tracking.

Prismatic USD joints constrain each finger to the drone. Force-limited spring
drives close the fingers; Coulomb friction alone holds the payload.
"""
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.utils.math import quat_apply
from pxr import Gf, Sdf, UsdPhysics

from .scene_cfg import box


class GripperController:
    def __init__(self, config):
        self.cfg = config['gripper']
        self.fingers, self.sensors, self.drives = [], [], []
        self.command_open = True
        cfg = self.cfg
        # Fixed mount is collision geometry belonging to the original 1 kg body.
        box('/World/Drone/GripperStem', (.035, .035, .14), (0, 0, -.12), (.3, .32, .35))
        box('/World/Drone/GripperRail', (.10, .20, .025), (0, 0, -.20), (.5, .52, .55))
        stage = sim_utils.get_current_stage()
        for side, sign in [('Left', 1), ('Right', -1)]:
            path = '/World/Finger' + side
            offset = (0., sign * (cfg['closed_center_y_m'] + cfg['travel_m']), cfg['center_z_m'])
            position = [a + b for a, b in zip(config['drone']['initial_position'], offset)]
            finger = RigidObject(RigidObjectCfg(
                prim_path=path,
                spawn=sim_utils.CuboidCfg(
                    size=tuple(cfg['finger_size_m']),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        disable_gravity=False, linear_damping=0., angular_damping=0.,
                        solver_position_iteration_count=16, solver_velocity_iteration_count=4),
                    mass_props=sim_utils.MassPropertiesCfg(mass=cfg['finger_mass_kg']),
                    collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=.002, rest_offset=0.),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=cfg['friction'], dynamic_friction=cfg['friction'], restitution=0.),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.8, .82, .85)),
                    activate_contact_sensors=True),
                init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(position))))
            joint = UsdPhysics.PrismaticJoint.Define(stage, '/World/GripperJoint' + side)
            joint.CreateBody0Rel().SetTargets([Sdf.Path('/World/Drone')])
            joint.CreateBody1Rel().SetTargets([Sdf.Path(path)])
            joint.CreateLocalPos0Attr(Gf.Vec3f(0., sign * cfg['closed_center_y_m'], cfg['center_z_m']))
            joint.CreateLocalPos1Attr(Gf.Vec3f(0.))
            joint.CreateAxisAttr('Y')
            joint.CreateLowerLimitAttr(0. if sign == 1 else -cfg['travel_m'])
            joint.CreateUpperLimitAttr(cfg['travel_m'] if sign == 1 else 0.)
            joint.CreateCollisionEnabledAttr(False)
            drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), 'linear')
            drive.CreateTypeAttr('force')
            drive.CreateStiffnessAttr(cfg['stiffness_n_m'])
            drive.CreateDampingAttr(cfg['damping_ns_m'])
            drive.CreateMaxForceAttr(cfg['max_force_n'])
            drive.CreateTargetPositionAttr(sign * cfg['travel_m'])
            drive.CreateTargetVelocityAttr(0.)
            sensor = ContactSensor(ContactSensorCfg(
                prim_path=path, update_period=0., filter_prim_paths_expr=['/World/Block']))
            self.fingers.append(finger)
            self.drives.append(drive)
            self.sensors.append(sensor)

    def command(self, opened):
        if opened != self.command_open:
            for sign, drive in zip((1, -1), self.drives):
                drive.GetTargetPositionAttr().Set(sign * self.cfg['travel_m'] if opened else 0.)
            self.command_open = opened

    def reset(self, root_state):
        self.command(False)
        self.command(True)
        for sign, finger, sensor in zip((1, -1), self.fingers, self.sensors):
            state = root_state.clone()
            offset = torch.tensor([[0., sign * (self.cfg['closed_center_y_m'] + self.cfg['travel_m']),
                                    self.cfg['center_z_m']]], device=state.device)
            state[:, :3] += quat_apply(state[:, 3:7], offset)
            finger.reset()
            finger.write_root_pose_to_sim(state[:, :7])
            finger.write_root_velocity_to_sim(state[:, 7:])
            finger.update(0.)
            sensor.reset()

    def update(self, dt):
        for finger, sensor in zip(self.fingers, self.sensors):
            finger.update(dt)
            sensor.update(dt, force_recompute=True)

    def measurements(self):
        centers = [f.data.root_pos_w[0] for f in self.fingers]
        aperture = float(torch.linalg.vector_norm(centers[0] - centers[1])) - self.cfg['finger_size_m'][1]
        forces = [s.data.force_matrix_w[0, 0, 0].cpu().tolist() for s in self.sensors]
        contact = [sum(x*x for x in force)**.5 > self.cfg['contact_threshold_n'] for force in forces]
        return {'aperture_m': aperture, 'contact_force_w_n': forces, 'contacts': contact,
                'command_open': self.command_open}
