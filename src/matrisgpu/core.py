"""Independent FIRE states, one shared model evaluation per active batch."""
from dataclasses import dataclass, field
from itertools import islice

import numpy as np
import torch
from ase import Atoms, units
from ase.calculators.singlepoint import SinglePointCalculator
from ase.filters import FrechetCellFilter
from ase.optimize import FIRE
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor


def _atoms(value):
    if isinstance(value, Structure):
        value = AseAtomsAdaptor.get_atoms(value)
    if not isinstance(value, Atoms):
        raise TypeError("Expected ase.Atoms or pymatgen.Structure")
    value = value.copy()
    if not len(value) or not all(value.pbc):
        raise ValueError("Only nonempty, fully periodic structures are supported; do not silently change PBC")
    if (not np.isfinite(value.positions).all() or not np.isfinite(value.cell.array).all()
            or abs(np.linalg.det(value.cell.array)) < 1e-12):
        raise ValueError("Expected finite positions and a nonsingular cell")
    return value


def _single(value):
    return isinstance(value, (Atoms, Structure))


@dataclass
class Trajectory:
    """README-compatible trajectory attributes; stress uses ASE Voigt units."""
    energies: list = field(default_factory=list)
    forces: list = field(default_factory=list)
    stresses: list = field(default_factory=list)
    magmoms: list = field(default_factory=list)
    atom_positions: list = field(default_factory=list)
    cells: list = field(default_factory=list)

    def record(self, atoms, keep):
        if not keep:
            for values in vars(self).values():
                values.clear()
        self.energies.append(float(atoms.get_potential_energy()))
        self.forces.append(atoms.get_forces().copy())
        self.stresses.append(atoms.get_stress().copy())
        self.magmoms.append(atoms.get_magnetic_moments().copy())
        self.atom_positions.append(atoms.positions.copy())
        self.cells.append(atoms.cell.array.copy())

    def __len__(self):
        return len(self.energies)


class BatchCalculator:
    """Batch E/F/S/m predictions. Unlike an ASE Calculator, accepts many atoms."""

    def __init__(self, model="matris_10m_oam", task="efsm", device="cuda", batch_size=32):
        if task != "efsm":
            raise ValueError("This release supports the validated task='efsm' only")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if torch.device(device).type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        if isinstance(model, str):
            from matris.model.model import MatRIS
            model = MatRIS.load(model_name=model, device=device)
        self.model, self.task, self.device, self.batch_size = model, task, device, batch_size

    def _predict(self, atoms):
        good, graphs, result = [], [], [None] * len(atoms)
        for i, a in enumerate(atoms):
            try:
                graph = self.model.graph_converter(AseAtomsAdaptor.get_structure(a))
            except AssertionError as exc:
                if "Number of directed indices" not in str(exc):
                    raise
                result[i] = {"status": "failed", "error": str(exc), "failure_stage": "graph_conversion"}
                continue
            graphs.append(graph.to(self.device))
            good.append(i)
        if graphs:
            # Forces/stress require autograd. Never use inference_mode/no_grad here.
            with torch.enable_grad():
                prediction = self.model(graphs, task=self.task, is_training=False)
            for j, i in enumerate(good):
                p = {k: prediction[k][j].detach().cpu().numpy().copy() for k in "efsm"}
                n = len(atoms[i])
                if (p['e'].size != 1 or p['f'].shape != (n, 3)
                        or p['s'].shape not in [(3, 3), (6,)] or p['m'].shape != (n,)):
                    raise ValueError("Unexpected MatRIS E/F/S/m shapes")
                if not all(np.isfinite(v).all() for v in p.values()):
                    raise FloatingPointError("Non-finite MatRIS prediction")
                factor = n if self.model.is_intensive else 1
                result[i] = dict(status="success", energy=float(p['e'].reshape(-1)[0])*factor,
                                 forces=p['f'], stress=p['s']*units.GPa, magmoms=p['m'])
        return result

    def ipredict(self, atoms):
        """Yield input-ordered results, buffering at most batch_size structures."""
        iterator = iter([atoms] if _single(atoms) else atoms)
        index = 0
        while chunk := list(islice(iterator, self.batch_size)):
            for result in self._predict([_atoms(a) for a in chunk]):
                yield dict(index=index, **result)
                index += 1

    def predict(self, atoms):
        """Single input -> dict; iterable -> list of dicts. Stress is eV/Å³."""
        result = list(self.ipredict(atoms))
        return result[0] if _single(atoms) else result


class StructOptimizer:
    """MatRIS-style interface with a separate FIRE state for each structure."""

    def __init__(self, model="matris_10m_oam", task="efsm", optimizer="FIRE",
                 device="cuda", batch_size=32):
        if optimizer != "FIRE":
            raise ValueError("Only the validated FIRE optimizer is supported")
        self.calculator = BatchCalculator(model, task, device, batch_size)

    def irelax(self, atoms, fmax=0.05, steps=500, relax_cell=True,
               ase_filter="FrechetCellFilter", mask=None, verbose=False,
               keep_trajectory=True, assign_magmoms=True):
        """Yield completion-ordered results with input index; consume input lazily."""
        if not np.isfinite(fmax) or fmax <= 0:
            raise ValueError("fmax must be positive and finite")
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
            raise ValueError("steps must be a nonnegative integer")
        if ase_filter != "FrechetCellFilter":
            raise ValueError("Only FrechetCellFilter is supported")
        iterator = enumerate([atoms] if _single(atoms) else atoms)
        active, exhausted = [], False
        while active or not exhausted:
            while not exhausted and len(active) < self.calculator.batch_size:
                try:
                    index, item = next(iterator)
                except StopIteration:
                    exhausted = True
                    break
                a = _atoms(item)
                filt = FrechetCellFilter(a, mask=mask) if relax_cell else a
                opt = FIRE(filt, logfile=None)
                opt.fmax = fmax
                active.append((index, a, filt, opt, Trajectory()))
            if not active:
                break
            predictions = self.calculator._predict([s[1] for s in active])
            remaining = []
            for state, p in zip(active, predictions):
                index, a, filt, opt, trajectory = state
                if p['status'] == 'failed':
                    yield dict(index=index, **p, converged=False, steps=opt.nsteps,
                               final_structure=AseAtomsAdaptor.get_structure(a), trajectory=trajectory)
                    continue
                a.calc = SinglePointCalculator(a, **{k:p[k] for k in ['energy','forces','stress','magmoms']})
                trajectory.record(a, keep_trajectory)
                forces = filt.get_forces()
                maximum = float(np.linalg.norm(forces, axis=1).max())
                if not np.isfinite(maximum):
                    raise FloatingPointError("Non-finite cell-filter forces")
                converged = bool(opt.converged(forces))
                if converged or opt.nsteps >= steps:
                    final = AseAtomsAdaptor.get_structure(a)
                    if assign_magmoms:
                        final.add_site_property('magmom', p['magmoms'].tolist())
                    result = dict(index=index, status='success' if converged else 'not_converged',
                                  converged=converged, steps=opt.nsteps, filter_fmax=maximum,
                                  final_structure=final, trajectory=trajectory,
                                  **{k:p[k] for k in ['energy','forces','stress','magmoms']})
                    if verbose:
                        print(f"index={index} {result['status']} steps={opt.nsteps} fmax={maximum:.6g}")
                    yield result
                else:
                    opt.step(forces)
                    opt.nsteps += 1
                    remaining.append(state)
            active = remaining

    def relax(self, atoms, **kwargs):
        """README-style single result, or input-ordered results for an iterable."""
        single = _single(atoms)
        results = sorted(self.irelax(atoms, **kwargs), key=lambda r:r['index'])
        return results[0] if single else results
