#!/usr/bin/env python3
"""Check a tiny GPU operation with an existing interpreter; no package changes."""
import json
from pathlib import Path
import sys

import torch

assert torch.cuda.is_available(), 'CUDA is unavailable'
value = torch.arange(16, device='cuda', dtype=torch.float32).sum().item()
assert value == 120
torch.cuda.synchronize()
report = {'python': sys.executable, 'torch': torch.__version__, 'cuda_runtime': torch.version.cuda,
          'device': torch.cuda.get_device_name(0), 'sum': value, 'status': 'passed'}
name = Path(sys.prefix).name
path = Path(__file__).resolve().parents[1] / 'artifacts/M0' / f'cuda_{name}.json'
path.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report))
