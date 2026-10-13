"""Compile training histories and calculate the paper's accuracy/cost summaries."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import polars as pl

from paper_workflow.campaign import ROOT, atomic_json, build_runtime, load_campaign


def measure_designs(configs: list[dict], *, batch_size=8) -> list[dict]:
    from tnqs.adapters.torch.flop_inspector import TorchFlopInspector
    from tnqs.adapters.torch.parameter_inspector import TorchParameterInspector
    from tnqs.domain.specifications import LogPsiOperation
    representatives = {}
    for config in configs:
        identity = config['identity']
        key = (identity['hamiltonian'], identity['size'], identity['architecture'])
        representatives.setdefault(key, config)
    results = []
    for (hamiltonian, size, alias), config in sorted(representatives.items()):
        runtime = build_runtime(config, device='cpu')
        counts = TorchParameterInspector().inspect(runtime)
        audit = TorchFlopInspector().inspect(runtime, LogPsiOperation(batch_size=batch_size))
        if audit.total % batch_size:
            raise ValueError('Measured FLOPs are not divisible by the representative batch size.')
        if counts.trainable != config['provenance']['trainable_parameters']:
            raise ValueError(f'Extracted parameter count differs for {(hamiltonian,size,alias)}')
        results.append({
            'hamiltonian': hamiltonian, 'size': size, 'arch_alias': alias,
            'trainable_parameters': counts.trainable,
            'api_flops_per_sample': audit.total//batch_size,
            'api_flops_b8': audit.total, 'batch_size': batch_size,
            'supported_operator_flops': audit.by_operator,
            'unsupported_operator_calls': audit.unsupported_operator_calls,
        })
    return results


def compile_histories(configs: list[dict], raw_dir: Path, *, allow_partial=False) -> dict[str, pl.DataFrame]:
    from paper_workflow.style import ARCH_ALIAS
    rows = []
    missing = []
    for config in configs:
        path = raw_dir / f"{config['run_id']}.json"
        if not path.is_file():
            missing.append(config['run_id'])
            continue
        result = json.loads(path.read_text())
        if not result.get('complete') or result.get('pilot') or result.get('config') != config:
            raise ValueError(f'Expected a completed production result with its frozen configuration: {path}')
        history = result['history']
        if len(history) != config['training']['num_steps']:
            raise ValueError(f'Incomplete training history: {path}')
        identity = config['identity']
        spec = config['resolved_specification']['wavefunction']
        family = spec['backbone']
        mode = spec['mode']
        if family == 'transformer' and mode == 'hierarchical':
            style_mode = 'hiermixer' if identity['architecture']=='HTF+M' else 'hierrnn'
        else:
            style_mode = mode
        style = ARCH_ALIAS[f'{family}_{style_mode}']
        energies = [r['energy_density'] for r in history]
        reference = config['reference_energy_density']
        final = float(np.mean(energies[-1000:]))
        rows.append({
            'architecture': family, 'hamiltonian': identity['hamiltonian'],
            'size': identity['size'], 'mode': mode, 'arch_alias': identity['architecture'],
            'marker': style.marker, 'colour': style.colour.tolist(),
            'sweep_parameter': {'name': 'h' if identity['hamiltonian']=='tfim' else 'delta',
                                'value': identity['phase']},
            'seed': identity['seed'], 'reference_energy': reference, 'energies': energies,
            'final_energy': final, 'signed_error': final-reference,
            'absolute_error': abs(final-reference),
            'trainable_parameters': config['provenance']['trainable_parameters'],
            'requested_parameters': config['provenance']['trainable_parameters'],
            'rationale': 'Frozen resolved paper architecture',
        })
    if missing and not allow_partial:
        raise FileNotFoundError(f'{len(missing)} production results are missing; first: {missing[0]}')
    if not rows:
        raise ValueError('No completed production histories found.')
    frame = pl.DataFrame(rows)
    return {name: frame.filter(pl.col('hamiltonian')==name) for name in ('tfim','rydberg')
            if frame.filter(pl.col('hamiltonian')==name).height}


def add_counts(frame: pl.DataFrame, counts: list[dict]) -> pl.DataFrame:
    columns = ('hamiltonian','size','arch_alias','api_flops_per_sample','api_flops_b8')
    table = pl.DataFrame([{key: row[key] for key in columns} for row in counts])
    existing = [c for c in ('api_flops_per_sample','api_flops_b8') if c in frame.columns]
    joined = frame.drop(existing).join(table, on=['hamiltonian','size','arch_alias'], how='left')
    if joined.get_column('api_flops_per_sample').null_count():
        raise ValueError('A paper run has no corresponding resolved design FLOP count.')
    return joined.with_columns(pl.col('sweep_parameter').struct.field('value').alias('phase'))


def summarize(frame: pl.DataFrame, *, iterations=15000, window=1000) -> list[dict]:
    """Absolute error in the trailing energy-density mean, pooled over phases/seeds."""
    grouped = {}
    for row in frame.to_dicts():
        history = row['energies']
        if len(history) < iterations:
            raise ValueError('History ends before the paper comparison iteration.')
        error = abs(float(np.mean(history[max(0,iterations-window):iterations]))-row['reference_energy'])
        key = (row['size'], row['arch_alias'])
        grouped.setdefault(key, []).append((error,row))
    results = []
    for (size, alias), items in sorted(grouped.items()):
        errors = np.asarray([item[0] for item in items])
        rows = [item[1] for item in items]
        costs = {row['api_flops_per_sample'] for row in rows}
        if len(costs) != 1:
            raise ValueError('A design has inconsistent measured forward costs.')
        q25, q75 = np.quantile(errors,[.25,.75])
        results.append({
            'size': size, 'arch_alias': alias, 'flops_per_forward_pass': costs.pop(),
            'mean_absolute_error': float(errors.mean()), 'median_absolute_error': float(np.median(errors)),
            'q25_absolute_error': float(q25), 'q75_absolute_error': float(q75),
            'run_absolute_errors': errors.tolist(), 'runs': len(rows),
            'phases': len({r['sweep_parameter']['value'] for r in rows}),
            'seeds': len({r['seed'] for r in rows}),
            'training_iterations': iterations, 'trailing_window': window,
        })
    return results


def write_flop_table(counts: list[dict], path: Path) -> None:
    order = ('M','HM','RNN','PRNN','HRNN','TF','PTF','HTF+M','HTF+R')
    lines = [
        '% Generated from all 36 frozen paper designs.',
        r'\begin{table*}[htbp]',
        r'\caption{Trainable parameters and PyTorch-dispatcher-supported forward FLOPs per sample.',
        r'Only supported matrix-product operations are credited; MACs are FLOPs/2.}',
        r'\label{tab:full-flops}', r'\centering', r'\begin{tabular}{lllrrr}',
        r'Hamiltonian & Size & Architecture & Parameters & FLOPs & MACs \\', r'\hline',
    ]
    for row in sorted(counts, key=lambda r:(r['hamiltonian'],int(r['size'].split('x')[0]),order.index(r['arch_alias']))):
        size = row['size'].replace('x',r'\times')
        alias = row['arch_alias'].replace('+',r'{+}')
        flops = row['api_flops_per_sample']
        lines.append(f"{row['hamiltonian'].upper()} & ${size}$ & $\\mathrm{{{alias}}}$ & "
                     f"{row['trainable_parameters']:,} & {flops:,} & {flops/2:,.0f} " + r'\\')
    lines.extend([r'\end{tabular}',r'\end{table*}',''])
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text('\n'.join(lines))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--raw',type=Path,default=ROOT/'build/raw')
    parser.add_argument('--input',type=Path,help='Existing TFIM/Rydberg paper parquet directory; skips history compilation.')
    parser.add_argument('--output',type=Path,default=ROOT/'build/processed')
    parser.add_argument('--allow-partial',action='store_true',help='Analyze completed production runs before a campaign finishes.')
    args = parser.parse_args(argv)
    configs = load_campaign(args.root)
    counts = measure_designs(configs)
    if args.input:
        frames = {name:pl.read_parquet(args.input/f'{name}_data.parquet') for name in ('tfim','rydberg')}
    else:
        frames = compile_histories(configs,args.raw,allow_partial=args.allow_partial)
    args.output.mkdir(parents=True,exist_ok=True)
    atomic_json(args.output/'forward_costs.json',{'convention':'PyTorch dispatcher-supported FLOPs; B=8 divided by 8','designs':counts})
    for name,frame in frames.items():
        frame = add_counts(frame,counts)
        frame.write_parquet(args.output/f'{name}_data.parquet')
        points = summarize(frame)
        atomic_json(args.output/f'{name}_summary.json',{'points':points})
        scalar = [{k:v for k,v in row.items() if k!='run_absolute_errors'} for row in points]
        pl.DataFrame(scalar).write_csv(args.output/f'{name}_summary.csv')
    write_flop_table(counts,args.output/'full_flops_api.tex')
    print(f'Analyzed {sum(frame.height for frame in frames.values())} runs and {len(counts)} designs into {args.output}')


if __name__ == '__main__':
    main()
