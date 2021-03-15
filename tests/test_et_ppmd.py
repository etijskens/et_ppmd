# -*- coding: utf-8 -*-

"""Tests for et_ppmd package."""

import et_ppmdcommon as cmn
import et_ppmd as md

def test_AtomsCtor():
    box = md.Box(0., 0., 5.*md.hcp.uc_centered_a, 3.*md.hcp.uc_centered_b)
    atoms = md.Atoms(box)
    fig = cmn.figure()
    cmn.plotBox(box)
    cmn.plotAtoms(atoms.x, atoms.y, radius=atoms.radius)
    cmn.plt.show()


# ==============================================================================
# The code below is for debugging a particular test in eclipse/pydev.
# (otherwise all tests are normally run with pytest)
# Make sure that you run this code with the project directory as CWD, and
# that the source directory is on the path
# ==============================================================================
if __name__ == "__main__":
    the_test_you_want_to_debug = test_Vlj_cutoff

    print("__main__ running", the_test_you_want_to_debug)
    the_test_you_want_to_debug()
    print('-*# finished #*-')
    
# eof