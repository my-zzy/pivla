"""Procedural room and colored orientation landmarks; no remote assets."""
import isaaclab.sim as sim_utils


def box(path, size, position, color, collision=True):
    cfg = sim_utils.CuboidCfg(
        size=tuple(size),
        collision_props=sim_utils.CollisionPropertiesCfg() if collision else None,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(color), roughness=0.8),
    )
    return cfg.func(path, cfg, translation=tuple(position))


def create_room(config):
    x, y, z = config['room_size']
    box('/World/Floor', (x, y, .10), (0, 0, -.05), (.30, .34, .39))
    for name, size, pos in (
        ('East', (.1, y, z), (x/2, 0, z/2)),
        ('West', (.1, y, z), (-x/2, 0, z/2)),
        ('North', (x, .1, z), (0, y/2, z/2)),
        ('South', (x, .1, z), (0, -y/2, z/2)),
    ):
        box('/World/' + name, size, pos, (.62, .66, .72))
    # Thin non-colliding markings identify position and orientation in the down view.
    for i in range(-5, 6):
        box(f'/World/GridX_{i+5}', (.012, y, .002), (i*.5, 0, .002), (.65, .68, .72), False)
        box(f'/World/GridY_{i+5}', (x, .012, .002), (0, i*.5, .002), (.65, .68, .72), False)
    box('/World/Pad', (.65, .65, .008), (0, 0, .007), (.04, .55, .35), False)
    box('/World/PadArrow', (.28, .07, .004), (.18, 0, .013), (1., .72, .05), False)
    # The front camera faces +X: orange post on world +Y, blue on -Y.
    box('/World/GateLeft', (.10, .10, 1.6), (2., .65, .8), (.95, .24, .04))
    box('/World/GateRight', (.10, .10, 1.6), (2., -.65, .8), (.04, .35, .95))
    box('/World/GateTop', (.10, 1.4, .10), (2., 0, 1.6), (.90, .90, .94))
    light = sim_utils.DomeLightCfg(intensity=1800., color=(.85, .90, 1.))
    light.func('/World/DomeLight', light)
    light = sim_utils.DistantLightCfg(intensity=2200., color=(1., .92, .82))
    light.func('/World/Sun', light, orientation=(.9238795, .3826834, 0., 0.))
