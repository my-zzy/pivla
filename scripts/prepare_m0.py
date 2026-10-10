#!/usr/bin/env python3
"""Generate an external template and bounded tutorial copies without pip/install."""

from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
LAB = Path('/home/user/IsaacLab')
PROJECT = ROOT / 'airvla_isaac'
OUT = ROOT / 'artifacts/M0'


def main():
    if PROJECT.exists():
        raise SystemExit(f'Refusing to regenerate existing project: {PROJECT}')
    OUT.mkdir(parents=True, exist_ok=True)
    spec = {'external': True, 'path': str(ROOT), 'name': 'airvla_isaac',
            'workflows': [{'name': 'direct', 'type': 'single-agent'}], 'rl_libraries': []}
    (OUT / 'template_spec.json').write_text(json.dumps(spec, indent=2) + '\n')
    sys.path.insert(0, str(LAB / 'tools/template'))
    import generator
    # Generate files without invoking Git hooks or changing any source repository.
    generator._setup_git_repo = lambda path: print('Git initialization deferred:', path)
    generator.generate(spec)
    tutorials = OUT / 'tutorials'
    tutorials.mkdir(exist_ok=True)
    for name, relpath, steps in (
        ('empty', '00_sim/create_empty.py', 120),
        ('camera', '04_sensors/run_usd_camera.py', 20),
    ):
        source = LAB / 'scripts/tutorials' / relpath
        text = source.read_text()
        loop = 'while simulation_app.is_running():'
        assert text.count(loop) == 1, f'Unexpected tutorial structure: {source}'
        text = text.replace(loop, f'for _m0_step in range({steps}):')
        text = text.replace('    main()\n', f'    main()\n    print("M0 {name}: completed {steps} steps", flush=True)\n')
        if name == 'camera':
            # A procedural floor avoids an unrelated remote asset dependency.
            text = text.replace('cfg = sim_utils.GroundPlaneCfg()',
                                'cfg = sim_utils.CuboidCfg(size=(10.0, 10.0, 0.1), '
                                'collision_props=sim_utils.CollisionPropertiesCfg(), '
                                'visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.4, 0.4)))')
            text = text.replace('    simulation_app.close()', '    rep.BackendDispatch.wait_until_done()\n    simulation_app.close(skip_cleanup=True)')
        else:
            text = text.replace('    simulation_app.close()', '    simulation_app.close(skip_cleanup=True)')
        (tutorials / source.name).write_text(text)
    print('Prepared bounded tutorial copies; original scripts are unchanged.')


if __name__ == '__main__':
    main()
