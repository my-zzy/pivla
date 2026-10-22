#!/usr/bin/env python3
"""Local π0 training with the audited M3 adapter and masked loss.

Use --probe first. Checkpoint and tokenizer downloads must already be complete.
No Hub upload or model download occurs here.
"""
import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import time
import traceback

if __name__ == '__main__':
    print('Starting π0 trainer: loading Python libraries…', flush=True)

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, default_collate
import yaml
from training_ui import TrainingDisplay

from airvla_isaac.data.chunk_dataset import ChunkDataset
from airvla_isaac.policy.action_codec import wrap_yaw
from airvla_isaac.policy.pi0_adapter import Pi0BatchAdapter, load_policy_strict, policy_config, training_loss


def save(path, data):
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def evaluate(policy, adapter, dataset, indices, seed, progress=None):
    # Monitoring must not reset the training noise stream or leave the model in eval mode.
    was_training = policy.training
    devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
    try:
        with torch.random.fork_rng(devices=devices):
            return _evaluate(policy, adapter, dataset, indices, seed, progress)
    finally:
        policy.train(was_training)


def _evaluate(policy, adapter, dataset, indices, seed, progress=None):
    policy.eval()
    predictions, targets, masks, anchors, records = [], [], [], [], []
    for done, index in enumerate(indices, 1):
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
        if progress is not None:
            progress(done, len(indices))
    predictions,targets,valid=np.array(predictions),np.array(targets),np.array(masks)
    error=predictions-targets
    position_rmse=float(np.sqrt(np.mean(np.sum(error[:,:,:3][valid]**2,axis=-1))))
    yaw_rmse=float(np.sqrt(np.mean(wrap_yaw(error[:,:,5][valid])**2)))
    correct=(predictions[:,:,6]>=.5)==(targets[:,:,6]>=.5)
    metrics={'position_rmse_m':position_rmse,'yaw_rmse_rad':yaw_rmse,
             'gripper_accuracy':float(correct[valid].mean()),'evaluated_chunks':len(indices),
             'valid_future_actions':int(valid.sum())}
    return metrics, {'predictions':predictions,'targets':targets,'valid':valid,'anchors':np.array(anchors)},records


def evaluation_indices(dataset, indices, stride):
    numeric = dataset.dataset.hf_dataset.select_columns(['episode_index', 'action']).with_format('numpy')[:]
    membership, actions = np.asarray(numeric['episode_index']), np.asarray(numeric['action'])
    selected = set(indices[::stride])
    for i in indices:
        first = i == 0 or membership[i] != membership[i-1]
        last = i == len(dataset)-1 or membership[i] != membership[i+1]
        if first or last or (not first and actions[i, 6] != actions[i-1, 6]) or (not last and actions[i, 6] != actions[i+1, 6]):
            selected.add(i)
    return sorted(selected)


def learning_rate_at(step, settings):
    peak = settings['learning_rate']
    if settings.get('lr_schedule', 'constant') == 'constant':
        return peak
    if settings['lr_schedule'] != 'cosine':
        raise ValueError('lr_schedule must be constant or cosine')
    warmup = min(settings.get('warmup_steps', 0), settings['steps'])
    if warmup and step <= warmup:
        return peak*step/warmup
    progress = min(1., max(0., (step-warmup)/max(1, settings['steps']-warmup)))
    floor = settings.get('min_lr_ratio', .1)
    return peak*(floor+(1-floor)*.5*(1+math.cos(math.pi*progress)))


def validate_settings(settings, args):
    for name in ('steps', 'batch_size', 'evaluation_stride', 'inference_steps'):
        if settings[name] < 1:
            raise ValueError(f'{name} must be positive')
    for name in ('warmup_steps', 'eval_every', 'checkpoint_every'):
        if settings.get(name, 0) < 0:
            raise ValueError(f'{name} cannot be negative')
    if settings.get('validation_stride', 1) < 1 or not 0 <= settings.get('min_lr_ratio', .1) <= 1:
        raise ValueError('Invalid validation stride or minimum learning rate')
    if settings['learning_rate'] <= 0 or args.probe_steps < 1:
        raise ValueError('Learning rate and probe steps must be positive')
    learning_rate_at(1, settings)


def check_dataset_coverage(settings):
    """Require a current passed coverage report when a configuration declares one."""
    if not settings.get('dataset_coverage_report'):
        return
    report=json.loads(Path(settings['dataset_coverage_report']).read_text())
    root=Path(settings['dataset']).resolve()
    repo_id=json.loads((root/'schema.json').read_text())['repo_id']
    digest=hashlib.sha256((root/'normalization.json').read_bytes()).hexdigest()
    if (report.get('status')!='passed' or not report.get('checks') or
        not all(report['checks'].values()) or report.get('dataset_repo_id')!=repo_id or
        report.get('normalization_sha256')!=digest):
        raise ValueError('Dataset coverage report is failed, stale, or belongs to another dataset')


def run(args, settings, out, ui):
    ui.phase('Preparing datasets, tokenizer, and CUDA')
    check_dataset_coverage(settings)
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
    eval_indices=evaluation_indices(dataset, indices, settings['evaluation_stride'])
    validation = None
    validation_indices = []
    if settings.get('validation_split'):
        if settings['validation_split'] != 'validation':
            raise ValueError('Monitoring requires the separate validation split')
        validation = ChunkDataset(settings['dataset'], 'validation')
        if validation.schema != dataset.schema or validation.stats != dataset.stats:
            raise ValueError('Train/validation preprocessing contracts differ')
        validation_indices = evaluation_indices(validation, list(range(len(validation))), settings['validation_stride'])
    adapter=Pi0BatchAdapter(settings['dataset'],settings['tokenizer'])
    cfg=policy_config(settings['base_checkpoint'],device='cpu',expert_only=settings['expert_only'])
    cfg.gradient_checkpointing=settings['gradient_checkpointing']
    cfg.dtype=settings['dtype']
    cfg.num_inference_steps=settings['inference_steps']
    ui.phase('Loading pretrained weights with strict key checks')
    policy=load_policy_strict(settings['base_checkpoint'],cfg).to('cuda')
    policy.config.device='cuda'
    trainable=[p for p in policy.parameters() if p.requires_grad]
    save(out/'run_config.json',{'settings':settings,'training_rows':len(indices),'evaluation_indices':eval_indices,
        'training_episode_ids':sorted(set(membership[indices].tolist())),
        'validation_rows':len(validation) if validation is not None else 0,
        'validation_indices':validation_indices,
        'parameters':sum(p.numel() for p in policy.parameters()),'trainable_parameters':sum(p.numel() for p in trainable),
        'pretrained_weights_loaded_strictly':True,
        'probe_steps':args.probe_steps if args.probe else None})
    loader=DataLoader(Subset(dataset,indices),batch_size=settings['batch_size'],shuffle=True,num_workers=0,
                      generator=torch.Generator().manual_seed(seed))
    optimizer=torch.optim.AdamW(trainable,lr=settings['learning_rate'],betas=(.9,.95),
                               eps=1e-8,weight_decay=settings['weight_decay'])
    def monitor(step):
        ui.phase(f'Evaluating training split · step {step:,}')
        metrics, arrays, records = evaluate(policy, adapter, dataset, eval_indices, seed, ui.advance)
        val_metrics = None
        if validation is not None:
            ui.phase(f'Evaluating validation split · step {step:,}')
            val_metrics, val_arrays, val_records = evaluate(policy, adapter, validation, validation_indices, seed+100000, ui.advance)
            save(out/f'validation_step_{step:06d}.json', val_metrics)
            np.savez_compressed(out/f'validation_step_{step:06d}.npz', **val_arrays)
            save(out/'validation_rows.json', val_records)
        with (out/'evaluation.jsonl').open('a') as handle:
            handle.write(json.dumps({'step':step,'train':metrics,'validation':val_metrics}, allow_nan=False)+'\n')
        ui.evaluated(step, metrics, val_metrics)
        return metrics, arrays, records, val_metrics

    def checkpoint_at(path, step):
        ui.phase(f'Saving checkpoint · step {step:,} · {path.name}')
        policy.save_pretrained(path)
        adapter.save_contract(path)
        save(path/'training_metadata.json', {'step':step, 'settings':settings,
            'optimizer_state_saved':False, 'restart_note':'--checkpoint starts a new optimizer and learning-rate schedule'})
        save(out/'latest_checkpoint.json', {'step':step,'checkpoint':str(path)})

    initial=None
    if not args.probe:
        initial,arrays,records,_=monitor(0)
        save(out/'initial_metrics.json',initial)
        np.savez_compressed(out/'initial_predictions.npz',**arrays)
        save(out/'evaluation_rows.json',records)
    policy.train()
    ui.phase('Training')
    steps=args.probe_steps if args.probe else settings['steps']
    stream=iter(loader)
    torch.cuda.reset_peak_memory_stats()
    start=time.monotonic()
    step_times=[]
    with (out/'loss.jsonl').open('w') as log:
        for step in range(1,steps+1):
            torch.cuda.synchronize()
            step_started=time.monotonic()
            try: rows=next(stream)
            except StopIteration:
                stream=iter(loader); rows=next(stream)
            batch=adapter(rows)
            optimizer.zero_grad(set_to_none=True)
            lr=learning_rate_at(step, settings)
            for group in optimizer.param_groups:
                group['lr']=lr
            loss=training_loss(policy,batch)
            if not torch.isfinite(loss): raise RuntimeError(f'Nonfinite loss at step {step}')
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(trainable,settings['grad_clip_norm'],error_if_nonfinite=True)
            optimizer.step()
            torch.cuda.synchronize()
            step_times.append(time.monotonic()-step_started)
            record={'step':step,'loss':float(loss.detach()),'grad_norm':float(norm),
                    'learning_rate':lr,'step_s':step_times[-1], 'batch_rows':len(rows['observation.state']),
                    'elapsed_s':time.monotonic()-start,'training_elapsed_s':sum(step_times),
                    'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
                    'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30}
            log.write(json.dumps(record)+'\n'); log.flush()
            ui.update(record)
            if not args.probe and step < steps:
                if settings.get('eval_every', 0) and step % settings['eval_every'] == 0:
                    monitor(step)
                    ui.phase('Training')
                if settings.get('checkpoint_every', 0) and step % settings['checkpoint_every'] == 0:
                    checkpoint_at(out/'checkpoints'/f'step_{step:06d}', step)
                    ui.phase('Training')
    if args.probe:
        warmup = min(5, max(0, steps-1))
        measured = float(np.median(step_times[warmup:]))
        result={'status':'passed','scope':'Real forward/backward/optimizer throughput and memory probe',**record,
                'probe_steps':steps,'excluded_warmup_steps':warmup,'median_step_s':measured,
                'estimated_training_only_minutes':measured*settings['steps']/60,
                'batch_size':settings['batch_size'],'training_rows':len(indices)}
        save(out/'result.json',result)
        ui.note = f'Probe passed · median {measured:.3f} s/step · result saved to {out / "result.json"}'
        return
    del optimizer,trainable,batch,loss
    gc.collect(); torch.cuda.empty_cache()
    final,arrays,records,final_validation=monitor(steps)
    save(out/'final_metrics.json',final)
    np.savez_compressed(out/'final_predictions.npz',**arrays)
    checkpoint=out/'checkpoint'
    checkpoint_at(checkpoint, steps)
    # Verify a freshly reconstructed model and saved preprocessing on a fixed batch.
    ui.phase('Verifying checkpoint reload')
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
            'validation':final_validation,
            'scope':'Offline training/validation and reload checks; task success requires M5 closed-loop evaluation',
            'peak_allocated_gib':record['peak_allocated_gib'],'training_elapsed_s':sum(step_times)}
    save(out/'result.json',result)
    ui.note = f'Checkpoint reload error {reload_error:.2e} · results saved to {out / "result.json"}'
    if not all(checks.values()): raise RuntimeError('Training completed but acceptance checks failed; see result.json')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('configs/train_debug.yaml'))
    parser.add_argument('--probe',action='store_true')
    parser.add_argument('--probe-steps',type=int,default=1,help='Optimizer steps for a disposable memory/throughput probe')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--checkpoint',type=Path,help='Continue from saved weights with a fresh optimizer')
    parser.add_argument('--steps',type=int)
    parser.add_argument('--learning-rate',type=float)
    parser.add_argument('--plain',action='store_true',help='Print readable log lines instead of a live terminal panel')
    args=parser.parse_args()
    settings=yaml.safe_load(args.config.read_text())
    if args.checkpoint is not None: settings['base_checkpoint']=str(args.checkpoint)
    if args.steps is not None: settings['steps']=args.steps
    if args.learning_rate is not None: settings['learning_rate']=args.learning_rate
    validate_settings(settings, args)
    out=args.output or Path(settings['output']+('_probe' if args.probe else ''))
    out.mkdir(parents=True,exist_ok=False)
    save(out/'status.json',{'status':'running'})
    try:
        with TrainingDisplay(args.probe_steps if args.probe else settings['steps'],
                             settings['batch_size'], out, plain=args.plain) as ui:
            run(args,settings,out,ui)
            ui.phase('Finished · passed', keep_note=True)
        save(out/'status.json',{'status':'passed'})
    except BaseException as exc:
        save(out/'status.json',{'status':'failed','error':f'{type(exc).__name__}: {exc}'})
        traceback.print_exc()
        raise SystemExit(1)
