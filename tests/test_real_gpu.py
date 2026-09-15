"""Opt-in short check against the actual upstream model and README optimizer."""
import os
import numpy as np
if __name__ != '__main__':
    import pytest
    pytestmark=pytest.mark.skipif(os.environ.get('MATRIS_GPU_SMOKE')!='1',reason='requires cached model and GPU')


def test_upstream_prediction_and_short_relaxation():
    from ase.build import bulk
    from matris.applications.base import MatRISCalculator
    from matris.applications.relax import StructOptimizer as NativeOptimizer
    from matrisgpu import BatchCalculator, StructOptimizer
    inputs=[bulk('Cu',a=3.7,cubic=True),bulk('Fe',a=2.9,cubic=True)]
    inputs[0].positions[0,0]+=.02
    calc=MatRISCalculator(model='matris_10m_oam',task='efsm',device='cuda')
    batch=BatchCalculator(model=calc.model,device='cuda',batch_size=2)
    results=batch.predict(inputs)
    for a,r in zip(inputs,results):
        a=a.copy();a.calc=calc
        np.testing.assert_allclose(r['energy'],a.get_potential_energy(),atol=1e-4,rtol=0)
        np.testing.assert_allclose(r['forces'],a.get_forces(),atol=2.5e-4,rtol=0)
        np.testing.assert_allclose(r['stress'],a.get_stress(voigt=False),atol=1e-5,rtol=0)
        np.testing.assert_allclose(r['magmoms'],a.get_magnetic_moments(),atol=1e-4,rtol=0)
    opt=StructOptimizer(model=calc.model,device='cuda',batch_size=2)
    got=opt.relax(inputs,steps=3,fmax=1e-6)
    native=NativeOptimizer(model='matris_10m_oam',task='efsm',optimizer='FIRE',device='cuda')
    for a,r in zip(inputs,got):
        expected=native.relax(atoms=a.copy(),steps=3,fmax=1e-6,verbose=False)
        np.testing.assert_allclose(r['final_structure'].cart_coords,expected['final_structure'].cart_coords,atol=1e-4,rtol=0)
        np.testing.assert_allclose(r['final_structure'].lattice.matrix,expected['final_structure'].lattice.matrix,atol=1e-4,rtol=0)


if __name__ == '__main__':
    test_upstream_prediction_and_short_relaxation()
    print('PASS: actual MatRIS batch prediction and two three-step FIRE trajectories match upstream tolerances')
