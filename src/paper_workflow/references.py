"""Independent sparse diagonalization for the small-system references."""
from __future__ import annotations
import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh

from paper_workflow.campaign import ROOT, atomic_json, load_campaign


def exact_energy(specification: dict) -> dict:
    lattice = specification['wavefunction']['lattice']
    lx, ly = lattice['Lx'], lattice['Ly']
    n = lx*ly
    if n > 16:
        raise ValueError('This reference generator is limited to at most 16 sites.')
    h = specification['hamiltonian']
    dimension = 1 << n
    indices = np.arange(dimension,dtype=np.int64)
    bits = ((indices[:,None] >> np.arange(n)) & 1).astype(float)
    diagonal = np.zeros(dimension,dtype=np.float64)
    if h['kind']=='tfim':
        spins = 2*bits-1
        for x in range(lx):
            for y in range(ly):
                i = x*ly+y
                neighbors = []
                if x+1 < lx:
                    neighbors.append((x+1)*ly+y)
                elif h['periodic'] and lx>2:
                    neighbors.append(y)
                if y+1 < ly:
                    neighbors.append(x*ly+y+1)
                elif h['periodic'] and ly>2:
                    neighbors.append(x*ly)
                for j in neighbors:
                    diagonal -= h['J']*spins[:,i]*spins[:,j]
        off_diagonal = -h['h']
    elif h['kind']=='rydberg':
        diagonal -= h['delta']*bits.sum(axis=1)
        c6 = h['C_6'] if h.get('C_6') is not None else h['Omega']*(h.get('R_b') or 1.2)**6
        for i,j in combinations(range(n),2):
            distance2 = (i//ly-j//ly)**2 + (i%ly-j%ly)**2
            cutoff = h.get('interaction_cutoff')
            if cutoff is None or np.sqrt(distance2)<=cutoff:
                diagonal += c6/distance2**3*bits[:,i]*bits[:,j]
        off_diagonal = -h['Omega']/2
    else:
        raise ValueError('Only TFIM and Rydberg reference Hamiltonians are supported.')
    matrix = coo_matrix((
        np.r_[diagonal,np.full(dimension*n,off_diagonal)],
        (np.r_[indices,np.tile(indices,n)],
         np.r_[indices,np.concatenate([indices^(1<<j) for j in range(n)])]),
    ),shape=(dimension,dimension)).tocsr()
    values,vectors = eigsh(matrix,k=1,which='SA',tol=1e-11,
                          v0=np.random.default_rng(829).normal(size=dimension))
    energy = float(values[0])
    residual = float(np.linalg.norm(matrix@vectors[:,0]-energy*vectors[:,0])/n)
    if residual>=1e-8:
        raise ArithmeticError(f'Exact reference residual is too large: {residual}')
    return {'energy_total':energy,'energy_density':energy/n,
            'residual_per_site':residual,'num_states':dimension}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--output',type=Path,default=ROOT/'build/references/exact_4x4.json')
    parser.add_argument('--hamiltonian',choices=('tfim','rydberg'))
    parser.add_argument('--phase',type=float)
    args = parser.parse_args(argv)
    representatives = {}
    for config in load_campaign(args.root):
        identity = config['identity']
        if identity['size']!='4x4':
            continue
        if args.hamiltonian and identity['hamiltonian']!=args.hamiltonian:
            continue
        if args.phase is not None and identity['phase']!=args.phase:
            continue
        representatives.setdefault((identity['hamiltonian'],identity['phase']),config)
    if not representatives:
        parser.error('No matching 4x4 reference in the paper campaign.')
    results = []
    for (hamiltonian,phase),config in sorted(representatives.items()):
        result = exact_energy(config['resolved_specification'])
        result.update(hamiltonian=hamiltonian,phase=phase,
                      supplied_reference=config['reference_energy_density'],
                      reference_minus_exact=config['reference_energy_density']-result['energy_density'])
        results.append(result)
        print(f"{hamiltonian} phase={phase}: exact energy/site={result['energy_density']:.12g}",flush=True)
    atomic_json(args.output,{'method':'float64 sparse exact diagonalization','references':results})


if __name__=='__main__':
    main()
