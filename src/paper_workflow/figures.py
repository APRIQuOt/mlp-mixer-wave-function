"""Render the current paper's figures from analyzed TFIM/Rydberg histories."""
from __future__ import annotations
import argparse
from pathlib import Path
import shutil

import polars as pl

from paper_workflow.campaign import ROOT


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=ROOT/'build/processed')
    parser.add_argument('--output',type=Path,default=ROOT/'build/paper')
    parser.add_argument('--diagram-only',action='store_true')
    args = parser.parse_args(argv)
    from paper_workflow import plots
    plots.configure_output(args.output)
    if not args.diagram_only:
        if not shutil.which('pdflatex'):
            raise RuntimeError('Paper PGF/PDF plots require pdflatex on PATH; see README.md.')
        frames = []
        for name in ('tfim','rydberg'):
            frame = pl.read_parquet(args.input/f'{name}_data.parquet')
            if 'api_flops_per_sample' not in frame.columns:
                raise ValueError('Run the analysis step first to attach measured FLOP counts.')
            frame = frame.with_columns(pl.col('sweep_parameter').struct.field('value').alias('phase'))
            frames.append(frame)
        plots.generate_paper_assets(*frames)
        source_table = args.input/'full_flops_api.tex'
        if source_table.is_file():
            target = args.output/'tables/full_flops_api.tex'
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source_table,target)
    from paper_workflow import diagram
    diagram.OUTPUT = args.output/'images/diagrams'
    diagram.main()
    static_asset = ROOT/'assets/architecture.pdf'
    target = args.output/'images/diagrams/architecture.pdf'
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(static_asset,target)
    print(f'Paper assets written to {args.output}')


if __name__=='__main__':
    main()
