#!/usr/bin/env python3
"""Plot a completed M4 run using the existing Isaac Lab plotting environment."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

if importlib.util.find_spec('matplotlib') is None:
    plotting_python='/home/user/miniconda3/envs/isaaclab/bin/python'
    if sys.executable==plotting_python:
        raise RuntimeError('No Matplotlib in the existing plotting environment')
    os.execv(plotting_python,[plotting_python,'-B',str(Path(__file__).resolve()),*sys.argv[1:]])

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run',type=Path,default=Path('artifacts/M4/debug_run'))
parser.add_argument('--previous-run',type=Path,help='Prepend the initial run to a refinement loss plot')
args=parser.parse_args()
root=args.run
history=[json.loads(line) for line in (root/'loss.jsonl').read_text().splitlines()]
boundary=None
if args.previous_run:
    previous=[json.loads(line) for line in (args.previous_run/'loss.jsonl').read_text().splitlines()]
    boundary=previous[-1]['step']
    history=previous+[{**r,'step':r['step']+boundary} for r in history]
fig,ax=plt.subplots(figsize=(9,4))
ax.plot([r['step'] for r in history],[r['loss'] for r in history],alpha=.5,label='Masked flow-matching loss')
if len(history)>=25:
    median=np.median(np.lib.stride_tricks.sliding_window_view(np.array([r['loss'] for r in history]),25),axis=-1)
    ax.plot([r['step'] for r in history[24:]],median,label='25-step median')
if boundary is not None: ax.axvline(boundary,color='gray',linestyle='--',label='Lower-LR refinement')
ax.set(xlabel='Optimizer step',ylabel='Loss (log scale)',yscale='log',title='M4 training loss (not task success)')
ax.grid(alpha=.2);ax.legend();fig.tight_layout();fig.savefig(root/'loss.png',dpi=140);plt.close(fig)
rows=json.loads((root/'evaluation_rows.json').read_text())
data=np.load(root/'final_predictions.npz')
episodes=sorted({r['episode'] for r in rows})
fig,axes=plt.subplots(len(episodes),3,figsize=(15,4*len(episodes)),squeeze=False)
for row_id,episode in enumerate(episodes):
    indices=[i for i,r in enumerate(rows) if r['episode']==episode]
    t=np.array([rows[i]['frame']/10 for i in indices])
    predicted=data['predictions'][indices,0]
    target=data['targets'][indices,0]
    for dim,name in enumerate(('x','y','z')):
        axes[row_id,0].plot(t,target[:,dim],label=f'{name} expert')
        axes[row_id,0].plot(t,predicted[:,dim],'--',label=f'{name} predicted')
    for dim,column,name in ((5,1,'Yaw (rad)'),(6,2,'Gripper (0=open, 1=closed)')):
        axes[row_id,column].plot(t,target[:,dim],label='Expert')
        axes[row_id,column].plot(t,predicted[:,dim],'--',label='Predicted')
        axes[row_id,column].set_ylabel(name)
    axes[row_id,0].set_ylabel(f'Episode {episode}: position (m)')
    for ax in axes[row_id]:
        ax.set_xlabel('Simulation time (s)');ax.grid(alpha=.2);ax.legend(fontsize=7)
fig.suptitle('M4 offline predictions: first action at each evaluated observation; full chunks assessed in metrics')
fig.tight_layout();fig.savefig(root/'predictions.png',dpi=140);plt.close(fig)
print(root/'loss.png',root/'predictions.png')
