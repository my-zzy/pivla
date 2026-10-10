#!/usr/bin/env python3
"""Offline LeRobot GPU worker; only RGB, drone state, and task enter inference."""
import argparse
import json
from pathlib import Path
import time
import traceback

import numpy as np
import torch
import yaml
from airvla_isaac.policy.pi0_adapter import Pi0BatchAdapter, load_policy_strict, policy_config


def atomic_json(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, allow_nan=False)+'\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ipc', type=Path, required=True)
    args = parser.parse_args()
    settings = yaml.safe_load(Path('configs/eval.yaml').read_text())
    checkpoint = Path(settings['checkpoint'])
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    adapter = Pi0BatchAdapter(checkpoint, checkpoint/'tokenizer')
    cfg = policy_config(checkpoint)
    cfg.num_inference_steps = settings['inference_steps']
    policy = load_policy_strict(checkpoint, cfg).to('cuda').eval()
    policy.config.device = 'cuda'
    # Warm up independently before starting the simulator; no dataset targets are used.
    dummy = {'state': np.zeros(8, dtype=np.float32), **{k: np.zeros((256, 256, 3), dtype=np.uint8)
             for k in ('front', 'down', 'external')}}

    def predict(data, seed):
        batch = adapter.from_observation(data['state'], data)
        torch.manual_seed(seed)
        with torch.inference_mode():
            return policy.predict_action_chunk(batch)[0].float().cpu().numpy()

    predict(dummy, settings['seed'])
    atomic_json(args.ipc/'ready.json', {'checkpoint': str(checkpoint), 'status': 'ready'})
    seen = set()
    while not (args.ipc/'stop').exists():
        requests = sorted(args.ipc.glob('request_*.npz'))
        for request in requests:
            if request.name in seen:
                continue
            seen.add(request.name)
            request_id = request.stem.removeprefix('request_')
            response = args.ipc/f'response_{request_id}.json'
            try:
                with np.load(request, allow_pickle=False) as data:
                    torch.cuda.synchronize()
                    start = time.monotonic()
                    predicted = predict(data, int(data['seed']))
                    torch.cuda.synchronize()
                    duration = time.monotonic()-start
                    atomic_json(response, {'request_id': request_id, 'prediction': predicted.tolist(),
                        'inference_wall_s': duration, 'peak_allocated_gib': torch.cuda.max_memory_allocated()/2**30})
            except Exception as exc:
                atomic_json(response, {'request_id': request_id, 'error': f'{type(exc).__name__}: {exc}'})
                traceback.print_exc()
                return
        time.sleep(.005)


if __name__ == '__main__':
    main()
