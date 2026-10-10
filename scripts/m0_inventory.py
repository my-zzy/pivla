#!/usr/bin/env python3
"""Read-only package/source inventory; writes evidence only under this project."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
ENVS = {name: Path('/home/user/miniconda3/envs') / name for name in ('isaaclab', 'lerobot')}
REPOS = {'isaaclab': Path('/home/user/IsaacLab'), 'lerobot': Path('/home/user/lerobot'),
         'isaaclab_shared_editable': Path('/home/user/workspace/objsplat_ws/.cache/IsaacLab')}


def command(argv):
    proc = subprocess.run(argv, text=True, capture_output=True, timeout=60)
    return {'exit_code': proc.returncode, 'stdout': proc.stdout.strip(), 'stderr': proc.stderr.strip()}


def metadata_fingerprint(prefix):
    paths = list((prefix / 'conda-meta').glob('*'))
    for site in prefix.glob('lib/python*/site-packages'):
        for pattern in ('*.dist-info/*', '*.egg-info/*', '*.pth', '*.egg-link'):
            paths.extend(site.glob(pattern))
    digest = hashlib.sha256()
    count = 0
    for path in sorted(set(paths)):
        if path.is_file():
            digest.update(str(path.relative_to(prefix)).encode())
            digest.update(path.read_bytes())
            count += 1
    return {'sha256': digest.hexdigest(), 'files': count}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', choices=('before', 'after'), required=True)
    args = parser.parse_args()
    probe = '''import json, sys
from importlib.metadata import distributions
packages = sorted(({'name': d.metadata['Name'], 'version': d.version} for d in distributions()), key=lambda d: d['name'].lower())
print(json.dumps({'python': sys.version, 'executable': sys.executable, 'packages': packages}))
'''
    report = {'platform': platform.platform(), 'os_release': Path('/etc/os-release').read_text(),
              'disk': dict(zip(('total', 'used', 'free'), shutil.disk_usage(ROOT))), 'environments': {}, 'sources': {}}
    for name, prefix in ENVS.items():
        info = command([str(prefix / 'bin/python'), '-B', '-c', probe])
        if info['exit_code']:
            raise RuntimeError(info)
        report['environments'][name] = {**json.loads(info['stdout']), 'metadata': metadata_fingerprint(prefix)}
    for name, repo in REPOS.items():
        report['sources'][name] = {key: command(['git', '--no-optional-locks', '-C', str(repo), *argv])
                                  for key, argv in {'commit': ['rev-parse', 'HEAD'], 'version': ['describe', '--tags', '--always'],
                                                    'status': ['status', '--porcelain']}.items()}
    report['gpu'] = command(['nvidia-smi', '--query-gpu=name,memory.total,memory.used,driver_version', '--format=csv,noheader'])
    output = ROOT / 'artifacts/M0' / f'inventory_{args.label}.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(output)
    for name, info in report['environments'].items():
        print(name, info['executable'], info['metadata'])
    if args.label == 'after':
        before = json.loads((output.parent / 'inventory_before.json').read_text())
        unchanged = all(before['environments'][n] == report['environments'][n] for n in ENVS)
        unchanged &= before['sources'] == report['sources']
        print('Package metadata and source status unchanged:', unchanged)
        if not unchanged:
            raise SystemExit(1)


if __name__ == '__main__':
    main()
