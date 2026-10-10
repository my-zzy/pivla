"""One dynamic rigid body with explicit inertia and simple compound collisions."""
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from pxr import Gf, UsdPhysics

from .scene_cfg import box


def create_drone(config):
    drone = config['drone']
    cfg = RigidObjectCfg(
        prim_path='/World/Drone',
        spawn=sim_utils.CuboidCfg(
            size=tuple(drone['body_size_m']),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False, linear_damping=0., angular_damping=0.,
                max_linear_velocity=20., max_angular_velocity=30.,
                solver_position_iteration_count=8, solver_velocity_iteration_count=2,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=drone['mass_kg']),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.08, .12, .18)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(drone['initial_position'])),
    )
    robot = RigidObject(cfg)
    # Child geometry has no independent RigidBodyAPI: it belongs to the same body.
    box('/World/Drone/ArmX', (.68, .045, .035), (0, 0, .015), (.2, .25, .3))
    box('/World/Drone/ArmY', (.045, .68, .035), (0, 0, .015), (.2, .25, .3))
    for name, pos in [('Front', (.31, 0, .035)), ('Rear', (-.31, 0, .035)),
                      ('Left', (0, .31, .035)), ('Right', (0, -.31, .035))]:
        cfg = sim_utils.CylinderCfg(
            radius=.09, height=.018,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(.98, .3, .04) if name == 'Front' else (.1, .45, .85)),
        )
        cfg.func('/World/Drone/Rotor' + name, cfg, translation=pos)
    prim = sim_utils.get_current_stage().GetPrimAtPath('/World/Drone')
    mass = UsdPhysics.MassAPI.Apply(prim)
    mass.CreateMassAttr(drone['mass_kg'])
    mass.CreateCenterOfMassAttr(Gf.Vec3f(0, 0, 0))
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*drone['inertia_kg_m2']))
    mass.CreatePrincipalAxesAttr(Gf.Quatf(1, 0, 0, 0))
    return robot
