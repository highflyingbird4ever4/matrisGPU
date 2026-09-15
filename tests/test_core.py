import numpy as np
import pytest
import torch
from ase import Atoms, units
from ase.calculators.lj import LennardJones
from ase.filters import FrechetCellFilter
from ase.optimize import FIRE
from pymatgen.io.ase import AseAtomsAdaptor
from matrisgpu import BatchCalculator, StructOptimizer


class Graph:
    def __init__(self, structure): self.structure = structure
    def to(self, device): return self


class Model:
    is_intensive = True
    def __init__(self): self.batches = []
    def graph_converter(self, structure):
        if structure[0].specie.symbol == 'He':
            raise AssertionError('Number of directed indices (1) != 2 * number of undirected indices (2)!')
        return Graph(structure)
    def __call__(self, graphs, **kwargs):
        self.batches.append(len(graphs))
        out = {k:[] for k in 'efsm'}
        for graph in graphs:
            a = AseAtomsAdaptor.get_atoms(graph.structure)
            a.calc = LennardJones()
            values = dict(e=a.get_potential_energy()/len(a), f=a.get_forces(),
                          s=a.get_stress(voigt=False)/units.GPa, m=np.zeros(len(a)))
            for k,v in values.items(): out[k].append(torch.as_tensor(v))
        return out


def sample(distance=1.2):
    return Atoms('Ar2', positions=[[1,1,1],[1+distance,1,1]], cell=[5,5,12], pbc=True)


@pytest.mark.parametrize('relax_cell',[False,True])
def test_fire_native_parity_and_candidate_failure(relax_cell):
    inputs = [sample(1.05), Atoms('He',cell=[5,5,12],pbc=True), sample(1.25)]
    snapshots = [a.positions.copy() for a in inputs]
    model = Model()
    result = StructOptimizer(model=model,device='cpu',batch_size=2).relax(
        inputs,fmax=.01,steps=500,relax_cell=relax_cell,mask=[1,1,0,0,0,1])
    assert [r['index'] for r in result] == [0,1,2]
    assert result[1]['status']=='failed'
    for i in [0,2]:
        ref = inputs[i].copy();ref.calc=LennardJones()
        target=FrechetCellFilter(ref,mask=[1,1,0,0,0,1]) if relax_cell else ref
        opt=FIRE(target,logfile=None);opt.run(fmax=.01,steps=500)
        np.testing.assert_allclose(ref.positions,result[i]['final_structure'].cart_coords,atol=1e-12,rtol=0)
        np.testing.assert_allclose(ref.cell.array,result[i]['final_structure'].lattice.matrix,atol=1e-12,rtol=0)
        assert opt.nsteps==result[i]['steps']
        assert len(result[i]['trajectory'])==opt.nsteps+1
    for a,b in zip(inputs,snapshots): np.testing.assert_array_equal(a.positions,b)
    assert max(model.batches)==2


def test_singlepoint_units_and_streaming():
    model=Model();calc=BatchCalculator(model=model,device='cpu',batch_size=2)
    a=sample();a.calc=LennardJones()
    r=calc.predict(a)
    assert r['energy']==pytest.approx(a.get_potential_energy())
    np.testing.assert_allclose(r['stress'],a.get_stress(voigt=False),atol=1e-14)
    consumed=[]
    def source():
        for i in range(10):
            consumed.append(i);yield sample()
    iterator=calc.ipredict(source());next(iterator)
    assert len(consumed)==2


def test_zero_steps_and_final_only():
    r=StructOptimizer(model=Model(),device='cpu').relax(sample(),steps=0,fmax=1e-8)
    assert r['steps']==0 and not r['converged'] and r['status']=='not_converged'
    r=StructOptimizer(model=Model(),device='cpu').relax(sample(),steps=3,fmax=1e-8,keep_trajectory=False)
    assert r['steps']==3 and len(r['trajectory'])==1


@pytest.mark.parametrize('error',[RuntimeError('out of memory'),AssertionError('unrelated')])
def test_unexpected_errors_not_swallowed(error):
    class Broken(Model):
        def graph_converter(self, structure):raise error
    with pytest.raises(type(error),match=str(error)):
        BatchCalculator(model=Broken(),device='cpu').predict(sample())


def test_reject_unsupported_inputs():
    a=sample();a.pbc=[True,True,False]
    with pytest.raises(ValueError,match='periodic'):
        BatchCalculator(model=Model(),device='cpu').predict(a)
    with pytest.raises(ValueError):StructOptimizer(model=Model(),optimizer='BFGS')
    with pytest.raises(ValueError):BatchCalculator(model=Model(),batch_size=0)


def test_refill_is_lazy():
    consumed=[]
    def source():
        for i in range(100):
            consumed.append(i);yield sample()
    iterator=StructOptimizer(model=Model(),device='cpu',batch_size=3).irelax(source(),steps=0)
    assert next(iterator)['index']==0
    assert len(consumed)==3
