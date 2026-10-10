#!/usr/bin/env python3
"""Local π0 overfit experiment with the audited M3 adapter and masked loss.

Use --probe first. Checkpoint and tokenizer downloads must already be complete.
No Hub upload or model download occurs here.
"""
import argparse
import gc
import json
from pathlib import Path
import random
import time
import traceback

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, default_collate
import yaml

from airvla_isaac.data.chunk_dataset import ChunkDataset
from airvla_isaac.policy.action_codec import wrap_yaw
from airvla_isaac.policy.pi0_adapter import Pi0BatchAdapter, load_policy_strict, policy_config, training_loss


def save(path, data):
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def evaluate(policy, adapter, dataset, indices, seed):
    policy.eval()
    predictions, targets, masks, anchors, records = [], [], [], [], []
    for index in indices:
        rows=default_collate([dataset[index]])
        batch=adapter(rows)
        # A fixed seed per row makes pre/post/reload predictions comparable.
        torch.manual_seed(seed+index)
        with torch.inference_mode():
            prediction=policy.predict_action_chunk(batch)
        predictions.append(adapter.decode(prediction,rows['anchor_state'],rows['action_is_pad'])[0])
        targets.append(adapter.decode(rows['action'],rows['anchor_state'],rows['action_is_pad'])[0])
        masks.append(~rows['action_is_pad'][0].numpy())
        anchors.append(rows['anchor_state'][0].numpy())
        records.append({'index':index,'episode':int(rows['episode_index'][0]),'frame':int(rows['frame_index'][0])})
    predictions,targets,valid=np.array(predictions),np.array(targets),np.array(masks)
    error=predictions-targets
    position_rmse=float(np.sqrt(np.mean(np.sum(error[:,:,:3][valid]**2,axis=-1))))
    yaw_rmse=float(np.sqrt(np.mean(wrap_yaw(error[:,:,5][valid])**2)))
    correct=(predictions[:,:,6]>=.5)==(targets[:,:,6]>=.5)
    metrics={'position_rmse_m':position_rmse,'yaw_rmse_rad':yaw_rmse,
             'gripper_accuracy':float(correct[valid].mean()),'evaluated_chunks':len(indices),
             'valid_future_actions':int(valid.sum())}
    return metrics, {'predictions':predictions,'targets':targets,'valid':valid,'anchors':np.array(anchors)},records


def run(args, settings, out):
    seed=settings['seed']
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32=True
    dataset=ChunkDataset(settings['dataset'],'train')
    membership=np.asarray(dataset.dataset.hf_dataset.select_columns(['episode_index']).with_format('numpy')[:]['episode_index'])
    episodes=settings['train_episode_ids']
    indices=np.flatnonzero(np.isin(membership,episodes)).tolist() if episodes is not None else list(range(len(dataset)))
    if not indices:
        raise ValueError('No rows selected')
    eval_indices=indices[::settings['evaluation_stride']]
    # Include both sides of every gripper transition and the last row of each episode.
    numeric=dataset.dataset.hf_dataset.select_columns(['action']).with_format('numpy')[:]['action']
    eval_indices=sorted(set(eval_indices+[i for i in indices if i==len(dataset)-1
        or (i+1<len(dataset) and membership[i]!=membership[i+1])
        or (i and membership[i]==membership[i-1] and numeric[i,6]!=numeric[i-1,6])
        or (i+1<len(dataset) and membership[i]==membership[i+1] and numeric[i,6]!=numeric[i+1,6])]))
    adapter=Pi0BatchAdapter(settings['dataset'],settings['tokenizer'])
    cfg=policy_config(settings['base_checkpoint'],device='cpu',expert_only=settings['expert_only'])
    cfg.gradient_checkpointing=settings['gradient_checkpointing']
    cfg.dtype=settings['dtype']
    cfg.num_inference_steps=settings['inference_steps']
    print('Loading pretrained weights with strict key checks',flush=True)
    policy=load_policy_strict(settings['base_checkpoint'],cfg).to('cuda')
    policy.config.device='cuda'
    trainable=[p for p in policy.parameters() if p.requires_grad]
    save(out/'run_config.json',{'settings':settings,'training_rows':len(indices),'evaluation_indices':eval_indices,
        'parameters':sum(p.numel() for p in policy.parameters()),'trainable_parameters':sum(p.numel() for p in trainable),
        'pretrained_weights_loaded_strictly':True})
    loader=DataLoader(Subset(dataset,indices),batch_size=settings['batch_size'],shuffle=True,num_workers=0,
                      generator=torch.Generator().manual_seed(seed))
    optimizer=torch.optim.AdamW(trainable,lr=settings['learning_rate'],betas=(.9,.95),
                               eps=1e-8,weight_decay=settings['weight_decay'])
    initial=None
    if not args.probe:
        initial,arrays,records=evaluate(policy,adapter,dataset,eval_indices,seed)
        save(out/'initial_metrics.json',initial)
        np.savez_compressed(out/'initial_predictions.npz',**arrays)
        save(out/'evaluation_rows.json',records)
        print('Initial predictions:',initial,flush=True)
    policy.train()
    steps=1 if args.probe else settings['steps']
    stream=iter(loader)
    torch.cuda.reset_peak_memory_stats()
    start=time.monotonic()
    with (out/'loss.jsonl').open('w') as log:
        for step in range(1,steps+1):
            try: rows=next(stream)
            except StopIteration:
                stream=iter(loader); rows=next(stream)
            batch=adapter(rows)
            optimizer.zero_grad(set_to_none=True)
            loss=training_loss(policy,batch)
            if not torch.isfinite(loss): raise RuntimeError(f'Nonfinite loss at step {step}')
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(trainable,settings['grad_clip_norm'],error_if_nonfinite=True)
            optimizer.step()
            record={'step':step,'loss':float(loss.detach()),'grad_norm':float(norm),
                    'elapsed_s':time.monotonic()-start,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30}
            log.write(json.dumps(record)+'\n'); log.flush()
            if step==1 or step%10==0 or step==steps: print(record,flush=True)
    if args.probe:
        save(out/'result.json',{'status':'passed','scope':'One real pretrained forward/backward/optimizer step',**record})
        return
    del optimizer,trainable,batch,loss
    gc.collect(); torch.cuda.empty_cache()
    final,arrays,records=evaluate(policy,adapter,dataset,eval_indices,seed)
    save(out/'final_metrics.json',final)
    np.savez_compressed(out/'final_predictions.npz',**arrays)
    checkpoint=out/'checkpoint'
    policy.save_pretrained(checkpoint)
    adapter.save_contract(checkpoint)
    # Verify a freshly reconstructed model and saved preprocessing on a fixed batch.
    reload_indices=eval_indices[:2]
    _,expected,_=evaluate(policy,adapter,dataset,reload_indices,seed)
    del policy
    gc.collect(); torch.cuda.empty_cache()
    restored_cfg=policy_config(checkpoint,device='cpu',expert_only=settings['expert_only'])
    restored=load_policy_strict(checkpoint,restored_cfg).to('cuda')
    restored_adapter=Pi0BatchAdapter(checkpoint,checkpoint/'tokenizer')
    _,actual,_=evaluate(restored,restored_adapter,dataset,reload_indices,seed)
    reload_error=float(np.abs(actual['predictions']-expected['predictions']).max())
    limits=settings.get('acceptance')
    checks={'finite_training':True,'checkpoint_reload':reload_error<=(limits or {}).get('reload_max_abs_error',1e-6)}
    if limits:
        checks.update(position_fit=final['position_rmse_m']<=limits['position_rmse_m'],
                      yaw_fit=final['yaw_rmse_rad']<=limits['yaw_rmse_rad'],
                      gripper_fit=final['gripper_accuracy']>=limits['gripper_accuracy'])
    result={'status':'passed' if all(checks.values()) else 'failed','checks':checks,
            'initial':initial,'final':final,'reload_max_abs_error':reload_error,'checkpoint':str(checkpoint),
            'peak_allocated_gib':record['peak_allocated_gib'],'training_elapsed_s':record['elapsed_s']}
    save(out/'result.json',result)
    print(json.dumps(result,indent=2),flush=True)
    if not all(checks.values()): raise RuntimeError('Training completed but acceptance checks failed; see result.json')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('configs/train_debug.yaml'))
    parser.add_argument('--probe',action='store_true')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--checkpoint',type=Path,help='Continue from saved weights with a fresh optimizer')
    parser.add_argument('--steps',type=int)
    parser.add_argument('--learning-rate',type=float)
    args=parser.parse_args()
    settings=yaml.safe_load(args.config.read_text())
    if args.checkpoint is not None: settings['base_checkpoint']=str(args.checkpoint)
    if args.steps is not None: settings['steps']=args.steps
    if args.learning_rate is not None: settings['learning_rate']=args.learning_rate
    out=args.output or Path(settings['output']+('_probe' if args.probe else ''))
    out.mkdir(parents=True,exist_ok=False)
    save(out/'status.json',{'status':'running'})
    try:
        run(args,settings,out)
        save(out/'status.json',{'status':'passed'})
    except BaseException as exc:
        save(out/'status.json',{'status':'failed','error':f'{type(exc).__name__}: {exc}'})
        traceback.print_exc()
        raise SystemExit(1)
