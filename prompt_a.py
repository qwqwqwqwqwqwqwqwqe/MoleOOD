"""Shared node-feature prompt: source preparation and label-free TTA.

Run `python prompt_a.py --help`. Graph dependencies are loaded lazily.
"""
import argparse
import copy
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def freeze(model):
    model.eval()
    model.requires_grad_(False)


def load_backbone(model, path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    source = checkpoint.get('main', checkpoint)
    target = model.state_dict()
    report = {'expanded': [], 'initialized_aux': []}
    for name, value in target.items():
        if name not in source:
            if not name.startswith('aux_predictor.'):
                raise ValueError(f'Missing backbone weight: {name}')
            report['initialized_aux'].append(name)
            continue
        weight = source[name]
        if weight.shape != value.shape:
            if (name.endswith('node_embeddings.0.linears.0.weight')
                    and weight.ndim == 2 and value.shape == (weight.shape[0], 40)
                    and weight.shape[1] == 39):
                expanded = torch.zeros_like(value)
                expanded[:, :39] = weight
                weight = expanded
                report['expanded'].append(name)
            else:
                raise ValueError(f'Incompatible weight {name}: {weight.shape} vs {value.shape}')
        target[name] = weight
    unexpected = set(source) - set(target)
    if unexpected:
        raise ValueError(f'Unexpected model weights: {sorted(unexpected)}')
    model.load_state_dict(target, strict=True)
    freeze(model)
    return report


def mask_graph(data, rate, generator):
    """Per-molecule masking; retain one visible atom and skip singleton graphs."""
    masked = data.clone()
    indices = []
    for start, end in zip(data.ptr[:-1].tolist(), data.ptr[1:].tolist()):
        size = end - start
        if size < 2:
            continue
        count = min(size - 1, max(1, int(size * rate)))
        indices.append(torch.randperm(size, generator=generator)[:count] + start)
    if not indices:
        return masked, None, None
    idx = torch.cat(indices).to(data.x.device)
    labels = data.x[idx, 0].clone()
    masked.x[idx, 0] = 39
    return masked, idx, labels


def reconstruction(model, masked, idx, labels, prompt):
    if idx is None:
        return prompt.sum() * 0
    logits = model(masked, return_node_feats=True, graph_prompt=prompt)
    losses = F.cross_entropy(logits[idx], labels, reduction='none')
    # Equal graph weights rather than letting larger molecules dominate.
    graph_ids = masked.batch[idx]
    return torch.stack([losses[graph_ids == g].mean() for g in graph_ids.unique()]).mean()


def adapt_batch(model, graph, prompt, optimizer, generator, args, anchor):
    """Consumes no property labels or domain IDs. Teacher fixed within this batch."""
    graph = graph.clone()
    for key in ('y', 'domain_id'):
        if hasattr(graph, key):
            delattr(graph, key)
    freeze(model)
    before = prompt.detach().clone()
    with torch.no_grad():
        teacher = model(graph, graph_prompt=prompt).softmax(-1)
    loss_value = 0.0
    for _ in range(args.steps):
        masked, idx, labels = mask_graph(graph, args.mask_rate, generator)
        optimizer.zero_grad(set_to_none=True)
        rec = reconstruction(model, masked, idx, labels, prompt)
        logits = model(masked, graph_prompt=prompt)
        cons = F.kl_div(logits.log_softmax(-1), teacher, reduction='batchmean')
        # Source anchor, NOT the zero-gradient per-step starting-point penalty.
        loss = rec + args.cons_weight * cons + args.anchor_weight * (prompt - anchor).square().mean()
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite adaptation loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_([prompt], 1.0, error_if_nonfinite=True)
        optimizer.step()
        # Hard batch trust region works even when only taking one gradient step.
        with torch.no_grad():
            delta = prompt - before
            norm = delta.norm()
            if norm > args.max_step_norm:
                prompt.copy_(before + delta * (args.max_step_norm / norm))
        loss_value = float(loss.detach())
    with torch.no_grad():
        probabilities = model(graph, graph_prompt=prompt).softmax(-1)
    return probabilities, loss_value


def collate_graphs(items):
    from torch_geometric.data import Batch
    clean, subs = [], []
    for item in items:
        item = item.clone()
        subs.append(item.subs)
        del item.subs
        clean.append(item)
    result = Batch.from_data_list(clean)
    result.subs = subs
    return result


def loader(dataset, batch_size, shuffle=False, seed=0):
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                      collate_fn=collate_graphs, num_workers=0,
                                      generator=torch.Generator().manual_seed(seed))


def identities(dataset):
    from rdkit import Chem
    result = []
    for item in dataset.data_infos:
        mol = Chem.MolFromSmiles(item['smiles'])
        if mol is None or not mol.GetNumAtoms():
            raise ValueError(f'Invalid or empty molecule: {item["smiles"]}')
        result.append(Chem.MolToSmiles(mol, canonical=True))
    return result


def source_split(dataset, seed):
    ids = identities(dataset)
    groups = sorted(set(ids))
    random.Random(seed).shuffle(groups)
    if len(groups) < 2:
        raise ValueError('Source preparation needs at least two unique molecules')
    validation = set(groups[:max(1, int(len(groups) * .1))])
    train = [i for i, identity in enumerate(ids) if identity not in validation]
    val = [i for i, identity in enumerate(ids) if identity in validation]
    return torch.utils.data.Subset(dataset, train), torch.utils.data.Subset(dataset, val), ids


def source_epoch(model, batches, prompt, args, stage, optimizer=None):
    # Validation uses fixed masks, independent of training generator consumption.
    generator = torch.Generator().manual_seed(args.seed + 101)
    total, size = 0., 0
    for graph in batches:
        graph = graph.to(args.device)
        masked, idx, labels = mask_graph(graph, args.mask_rate, generator)
        with torch.set_grad_enabled(optimizer is not None):
            rec = reconstruction(model, masked, idx, labels, prompt)
            if stage == 'aux':
                if idx is None:
                    continue
                loss = rec
            else:
                cls = F.cross_entropy(model(graph, graph_prompt=prompt), graph.y.view(-1))
                loss = cls + args.source_aux_weight * rec
                if optimizer is not None:
                    loss = loss + args.anchor_weight * prompt.square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite source loss')
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                parameters = model.aux_predictor.parameters() if stage == 'aux' else [prompt]
                torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
                optimizer.step()
        total += float(loss.detach()) * graph.num_graphs
        size += graph.num_graphs
    if not size:
        raise ValueError('No valid source samples for this stage')
    return total / size


def prepare(model, dataset, args):
    train, val, ids = source_split(dataset, args.seed)
    prompt = torch.nn.Parameter(torch.zeros(args.emb_dim, device=args.device))
    frozen_state = {k: v.detach().clone() for k, v in model.state_dict().items()
                    if not k.startswith('aux_predictor.')}
    history = []
    for stage, epochs in [('aux', args.aux_epochs), ('prompt', args.prompt_epochs)]:
        freeze(model)
        prompt.requires_grad_(stage == 'prompt')
        if stage == 'aux':
            model.aux_predictor.requires_grad_(True)
            parameters = model.aux_predictor.parameters()
        else:
            parameters = [prompt]
        optimizer = torch.optim.Adam(parameters, lr=args.source_lr)
        best_score, best_state = float('inf'), None
        for epoch in range(epochs):
            train_loss = source_epoch(model, loader(train, args.batch_size, True, args.seed + epoch),
                                      prompt, args, stage, optimizer)
            val_loss = source_epoch(model, loader(val, args.batch_size), prompt, args, stage)
            row = dict(stage=stage, epoch=epoch + 1, train_loss=train_loss, val_loss=val_loss)
            history.append(row)
            print(json.dumps(row), flush=True)
            if val_loss < best_score:
                best_score = val_loss
                best_state = (copy.deepcopy(model.aux_predictor.state_dict()), prompt.detach().clone())
        model.aux_predictor.load_state_dict(best_state[0])
        with torch.no_grad():
            prompt.copy_(best_state[1])
    freeze(model)
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in frozen_state.items()), 'Source backbone changed'
    return prompt, {'history': history, 'source_ids': sorted(set(ids)),
                    'train_indices': train.indices, 'validation_indices': val.indices}


def score(labels, probs):
    from sklearn.metrics import roc_auc_score, accuracy_score
    return {'auc': float(roc_auc_score(labels, probs[:, 1])) if len(np.unique(labels)) == 2 else None,
            'accuracy': float(accuracy_score(labels, probs.argmax(-1)))}


def evaluate_modes(model, dataset, initial_prompt, args):
    results = {}
    initial_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    for mode in ('moleood', 'source_frozen', 'episodic', 'continual'):
        prompt = torch.nn.Parameter(initial_prompt.detach().clone())
        optimizer = torch.optim.Adam([prompt], lr=args.tta_lr)
        generator = torch.Generator().manual_seed(args.seed + 202)
        ys, probabilities, trace = [], [], []
        for step, graph in enumerate(loader(dataset, args.batch_size)):
            graph = graph.to(args.device)
            ys.extend(graph.y.detach().cpu().view(-1).tolist())
            graph = graph.clone()
            del graph.y
            if hasattr(graph, 'domain_id'):
                del graph.domain_id
            if mode in ('episodic', 'continual'):
                if mode == 'episodic':
                    with torch.no_grad():
                        prompt.copy_(initial_prompt)
                    optimizer = torch.optim.Adam([prompt], lr=args.tta_lr)
                probs, loss = adapt_batch(model, graph, prompt, optimizer, generator, args, initial_prompt)
            else:
                with torch.no_grad():
                    probs = model(graph, graph_prompt=None if mode == 'moleood' else prompt).softmax(-1)
                loss = None
            probabilities.append(probs.cpu().numpy())
            trace.append(dict(batch=step, loss=loss,
                              prompt_norm=0. if mode == 'moleood' else float(prompt.detach().norm())))
        probs = np.concatenate(probabilities)
        results[mode] = score(ys, probs)
        torch.save(torch.zeros_like(prompt).cpu() if mode == 'moleood' else prompt.detach().cpu(),
                   args.output / f'{mode}_final_prompt.pt')
        np.savez(args.output / f'{mode}_predictions.npz', labels=np.array(ys), probabilities=probs)
        (args.output / f'{mode}_trace.json').write_text(json.dumps(trace, indent=2), encoding='utf-8')
        assert all(torch.equal(v, model.state_dict()[k]) for k, v in initial_state.items()), 'Frozen state changed'
        print(mode, results[mode], flush=True)
    baseline = results['moleood']['auc']
    for metrics in results.values():
        metrics['auc_gain_pp'] = (100 * (metrics['auc'] - baseline)
                                  if baseline is not None and metrics['auc'] is not None else None)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['prepare', 'evaluate', 'all'], default='all')
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--train-path')
    parser.add_argument('--test-path')
    parser.add_argument('--prepared-path', help='Prepared checkpoint required for evaluate')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--emb-dim', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--aux-epochs', type=int, default=5)
    parser.add_argument('--prompt-epochs', type=int, default=5)
    parser.add_argument('--source-lr', type=float, default=1e-3)
    parser.add_argument('--tta-lr', type=float, default=1e-3)
    parser.add_argument('--steps', type=int, default=1)
    parser.add_argument('--mask-rate', type=float, default=.2)
    parser.add_argument('--source-aux-weight', type=float, default=.5)
    parser.add_argument('--cons-weight', type=float, default=1.)
    parser.add_argument('--anchor-weight', type=float, default=1e-3)
    parser.add_argument('--max-step-norm', type=float, default=.05)
    args = parser.parse_args()
    if not 0 < args.mask_rate < 1 or min(args.aux_epochs, args.prompt_epochs, args.steps, args.batch_size) < 1:
        parser.error('Mask rate must be in (0,1); epochs, steps and batch size must be positive')
    if args.max_step_norm <= 0:
        parser.error('--max-step-norm must be positive')
    if args.stage in ('all', 'prepare') and not args.train_path:
        parser.error('--train-path required for source preparation')
    if args.stage in ('all', 'evaluate') and not args.test_path:
        parser.error('--test-path required for evaluation')
    if args.stage == 'evaluate' and not args.prepared_path:
        parser.error('--prepared-path required for evaluate')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    from models.Framework import Framework
    from models.dataset import LBAPDatasetWithSub
    model = Framework(args.emb_dim, args.emb_dim, 2).to(args.device)
    report = load_backbone(model, args.model_path)
    args.output.mkdir(parents=True, exist_ok=True)
    fingerprint = hashlib.sha256(Path(args.model_path).read_bytes()).hexdigest()
    print('Checkpoint compatibility:', report)
    if args.stage in ('all', 'prepare'):
        source = LBAPDatasetWithSub(args.train_path, 'train')
        prompt, metadata = prepare(model, source, args)
        torch.save({'model': model.state_dict(), 'prompt': prompt.detach(), 'metadata': metadata,
                    'backbone_sha256': fingerprint}, args.output / 'prepared.pt')
    else:
        state = torch.load(args.prepared_path, map_location=args.device, weights_only=True)
        if state['backbone_sha256'] != fingerprint:
            raise ValueError('Prepared state belongs to a different backbone')
        model.load_state_dict(state['model'], strict=True)
        freeze(model)
        prompt, metadata = state['prompt'], state['metadata']
    results = None
    if args.stage in ('all', 'evaluate'):
        target = LBAPDatasetWithSub(args.test_path, 'ood_test')
        if set(identities(target)) & set(metadata['source_ids']):
            raise ValueError('Source/target molecule overlap; investigate data before evaluation')
        results = evaluate_modes(model, target, prompt, args)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    (args.output / 'results.json').write_text(json.dumps(dict(config=config, compatibility=report,
        backbone_sha256=fingerprint, results=results, source_history=metadata['history']), indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
