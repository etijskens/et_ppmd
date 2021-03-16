#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for C++ module et_ppmd.corecpp.
"""


import et_ppmdcommon as cmn
import et_ppmd as md
import et_ppmd.forces as lj
import numpy as np

# create an alias for the binary extension cpp module
cpp = md.corecpp

__plot = True
__plot = False

def test_computeForces():
    """"""
    box = md.Box(0., 0., 5.*md.hcp.uc_centered_a, 0.5*md.hcp.uc_centered_b)
    atoms = md.MD(box,cutoff=2.5)
    if __plot:
        cmn.figure()
        cmn.plotBox(box)
        cmn.plotAtoms(atoms.x, atoms.y, radius=atoms.radius)
        cmn.plt.show()
    atoms.buildVerletLists()
    print(atoms.vl)
    cpp.computeForces( atoms.x, atoms.y, atoms.vl.vl_array
                     , atoms.ax, atoms.ay
                     )
    print(f'ax={atoms.ax}')
    print(f'ay={atoms.ay}')
    r = md.hcp.uc_centered_a
    r01sq = r**2
    r02sq = (2*r) ** 2
    fx01 = lj.force_factor(r01sq)*r
    fx02 = lj.force_factor(r02sq)*2*r
    expected = np.array([fx01, fx01, fx01, fx01, 0.00]) \
             + np.array([fx02, fx02, fx02, 0.00, 0.00]) \
             - np.array([0.00, fx01, fx01, fx01, fx01]) \
             - np.array([0.00, 0.00, fx02, fx02, fx02])
    assert np.all(atoms.ax == expected)
    assert np.all(atoms.ay == np.zeros((atoms.n_atoms,), dtype=float))


#===============================================================================
# The code below is for debugging a particular test in eclipse/pydev.
# (normally all tests are run with pytest)
#===============================================================================
if __name__ == "__main__":
    the_test_you_want_to_debug = test_computeForces

    print(f"__main__ running {the_test_you_want_to_debug} ...")
    the_test_you_want_to_debug()
    print('-*# finished #*-')
#===============================================================================
