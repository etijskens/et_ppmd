# -*- coding: utf-8 -*-

"""Tests for et_ppmd package."""

import et_ppmdcommon as cmn
import et_ppmd as md
import et_ppmd.forces as lj
import numpy as np

__plot = True
__plot = False

def test_MDCtor():
    box = md.Box(0., 0., 5.*md.hcp.uc_centered_a, 3.*md.hcp.uc_centered_b)
    atoms = md.MD(box)
    cmn.figure()
    cmn.plotBox(box)
    cmn.plotAtoms(atoms.x, atoms.y, radius=atoms.radius)
    if __plot:
        cmn.plt.show()


def test_computeEnergy():
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
    energy = atoms.computeEnergy()
    r01sq = md.hcp.uc_centered_a**2
    r02sq = (2*md.hcp.uc_centered_a) ** 2
    expected = 4*lj.potential(r01sq) + 3*lj.potential(r02sq)
    print(energy)
    print(expected)
    assert energy == expected


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
    energy = atoms.computeForces()
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


# ==============================================================================
# The code below is for debugging a particular test in eclipse/pydev.
# (otherwise all tests are normally run with pytest)
# Make sure that you run this code with the project directory as CWD, and
# that the source directory is on the path
# ==============================================================================
if __name__ == "__main__":
    the_test_you_want_to_debug = test_computeForces

    print("__main__ running", the_test_you_want_to_debug)
    the_test_you_want_to_debug()
    print('-*# finished #*-')
    
# eof