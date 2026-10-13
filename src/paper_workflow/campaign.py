"""Frozen paper runs, portable model construction, and campaign execution."""
from __future__ import annotations

import argparse
from dataclasses import asdict, fields, replace
from functools import partial
import json
import math
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]


def load_campaign(root: Path = ROOT) -> list[dict]:
    manifest = json.loads((root / 'configs/campaign.json').read_text())
    runs = [json.loads((root / item['config']).read_text()) for item in manifest['runs']]
    if len(runs) != manifest['num_runs'] or len({r['run_id'] for r in runs}) != len(runs):
        raise ValueError('Campaign contains missing or duplicate run definitions.')
    return runs


def build_runtime(config: dict, *, device: str = 'cpu'):
    import torch
    from tnqs.adapters.config.spec_codec import specification_from_dict
    from tnqs.adapters.torch.builders import TorchRuntimeBuilder
    from tnqs.application.execution import ExecutionOptions
    specification = specification_from_dict(config['resolved_specification'])
    dtype = getattr(torch, config['runtime']['parameter_dtype'])
    return TorchRuntimeBuilder().build(
        specification, device=device, dtype=dtype, seed=config['runtime']['seed'],
        execution=ExecutionOptions(mode='portable', model_backend='pytorch'),
    )


def json_finite(value):
    """Keep optional diagnostics JSON-standard instead of encoding NaN."""
    if isinstance(value, dict):
        return {k: json_finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_finite(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(json_finite(value), indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def run_one(config: dict, output: Path, *, device='auto', steps=None, samples=None) -> Path:
    import torch
    from tnqs.application.training import GroundStateTrainingOptions
    from tnqs.adapters.torch.training import train_ground_state
    run_id = config['run_id']
    pilot = steps is not None or samples is not None
    if pilot:
        output = output / f"pilot_steps_{steps or config['training']['num_steps']}_samples_{samples or config['training']['num_samples']}"
    result_path = output / f'{run_id}.json'
    if result_path.exists():
        saved = json.loads(result_path.read_text())
        if saved.get('complete') and saved.get('config') == config:
            print(f'Skipping completed run {run_id}', flush=True)
            return result_path
        raise FileExistsError(f'Existing result does not match this run: {result_path}')
    accepted = {field.name for field in fields(GroundStateTrainingOptions)}
    unexpected = set(config['training']) - accepted
    if unexpected:
        raise ValueError(f'Unsupported frozen training fields: {sorted(unexpected)}')
    options = GroundStateTrainingOptions(**config['training'])
    if steps is not None:
        if steps < 1 or steps > options.num_steps:
            raise ValueError('Pilot steps must lie within the recorded training horizon.')
        options = replace(options, num_steps=steps)
    if samples is not None:
        if samples < 1:
            raise ValueError('Pilot sample count must be positive.')
        options = replace(options, num_samples=samples, final_num_samples=samples)
    runtime = build_runtime(config, device=device)
    scheduler = partial(torch.optim.lr_scheduler.CosineAnnealingLR,
                        T_max=config['scheduler']['T_max'], eta_min=config['scheduler']['eta_min'])
    started = time.perf_counter()

    def report_progress(step, report):
        if step == 0 or (step+1) % 100 == 0 or step+1 == options.num_steps:
            print(f"{run_id}: {step+1}/{options.num_steps}, energy/site={report.energy_density:.9g}", flush=True)

    history = train_ground_state(runtime, options, scheduler_factory=scheduler,
                                 callback=report_progress)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / f'{run_id}.pt'
    temporary = checkpoint_path.with_suffix('.pt.tmp')
    torch.save({'schema_version': 1, 'run_id': run_id,
                'resolved_specification': config['resolved_specification'],
                'wavefunction_state_dict': runtime.wavefunction.state_dict()}, temporary)
    temporary.replace(checkpoint_path)
    parameter = next(runtime.wavefunction.parameters())
    atomic_json(result_path, {
        'schema_version': 1, 'run_id': run_id, 'config': config,
        'complete': True, 'pilot': pilot, 'history': [r._asdict() for r in history],
        'runtime': {'device': str(parameter.device), 'dtype': str(parameter.dtype),
                    'torch_version': str(torch.__version__),
                    'training_seconds': time.perf_counter()-started},
        'checkpoint': checkpoint_path.name,
    })
    return result_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, default=ROOT/'build/raw')
    parser.add_argument('--device', choices=('auto','cpu','cuda','mps'), default='auto')
    parser.add_argument('--run-id', action='append', help='Select one or more exact frozen runs.')
    parser.add_argument('--task-index', type=int, help='Select one zero-based campaign index for cluster arrays.')
    parser.add_argument('--list', action='store_true', help='List runs without training.')
    parser.add_argument('--steps', type=int, help='Short pilot; saves separately from production.')
    parser.add_argument('--samples', type=int, help='Pilot sample override; saves separately from production.')
    args = parser.parse_args(argv)
    runs = load_campaign(args.root)
    if args.list:
        for index, config in enumerate(runs):
            print(f"{index:3d} {config['run_id']}")
        return
    if args.task_index is not None:
        if args.run_id or not 0 <= args.task_index < len(runs):
            parser.error('Use either --run-id or a --task-index in 0..539.')
        runs = [runs[args.task_index]]
    elif args.run_id:
        requested = set(args.run_id)
        selected = [r for r in runs if r['run_id'] in requested]
        if len(selected) != len(requested):
            parser.error('One or more run IDs are absent from the frozen campaign.')
        runs = selected
    for config in runs:
        run_one(config, args.output, device=args.device, steps=args.steps, samples=args.samples)


if __name__ == '__main__':
    main()
