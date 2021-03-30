#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `et_ppmd` package."""

import pytest
import numpy as np
import et_ppmd.grid


def test_build():
    grid = et_ppmd.grid.Grid(cell_size=1.0, wx=2., )
    assert grid.cl.shape[0] == 2 # number of cells in x-direction
    assert grid.cl.shape[1] == 2 # number of cells in y-direction

    rx = np.array([0.5, 1.5, 0.5, 1.5,])
    ry = np.array([0.5, 0.5, 1.5, 1.5,])
    for build_method in [grid.build_simple, grid.build]:
        build_method(rx, ry, linearise=False)
        print(f'{build_method}\n{grid}')

        # all cells contain 1 atom
        assert np.all(grid.cl[:,:,0] == 1)
        # cell (0,0) contains atom 0
        assert np.all(grid.cl[0,0,1] == 0)
        # cell (1,0) contains atom 1
        assert np.all(grid.cl[1,0,1] == 1)
        # cell (0,1) contains atom 2
        assert np.all(grid.cl[0,1,1] == 2)
        # cell (1,1) contains atom 3
        assert np.all(grid.cl[1,1,1] == 3)

        grid.clear()

def test_build_random():
    max_atoms_per_cell = 40
    grid1 = et_ppmd.grid.Grid(cell_size=0.5, wx=1.0, max_atoms_per_cell=max_atoms_per_cell)
    grid2 = et_ppmd.grid.Grid(cell_size=0.5, wx=1.0, max_atoms_per_cell=max_atoms_per_cell)
    grid1.clear(full=True) # full clear, otherwise the test will fail because the entries in
    grid2.clear(full=True) # the cell lists that are not use.

    rx = np.random.random((100,))
    ry = np.random.random((100,))

    grid1.build_simple(rx, ry, linearise=False)
    print(grid1)

    grid2.build(rx, ry, linearise=False)
    print(grid2)

    # all entries should be equal since the order of atoms is the same.
    assert np.all(grid1.cl == grid2.cl)


def test_linearised():
    n_atoms = 100
    if n_atoms == 4: # a small test that is easy to debug
        rx = np.array([0.25, 0.75, 0.25, 0.75,])
        ry = np.array([0.25, 0.25, 0.75, 0.75,])
        max_atoms_per_cell = 2
    else:
        rx = np.random.random((n_atoms,))
        ry = np.random.random((n_atoms,))
        # estimate the maximum number of atoms per cell, this may accidentally be too small
        # because we are using random positions. The average number of atoms per cell is
        # the n_atoms/4 (there are 4 cells)
        max_atoms_per_cell = int((n_atoms/4)*1.5)

    grid1 = et_ppmd.grid.Grid(cell_size=0.5, wx=1.0, max_atoms_per_cell=max_atoms_per_cell)
    grid2 = et_ppmd.grid.Grid(cell_size=0.5, wx=1.0, max_atoms_per_cell=max_atoms_per_cell)

    grid1.build_simple(rx, ry, linearise=False)
    print(grid1)

    grid2.build(rx, ry, linearise=False)
    print(grid2)
    grid2.linearise()
    print(grid2)

    for l in range(grid1.n):
        for k in range(grid1.m):
            assert np.all(grid1.cell_list(k,l) == grid2.cell_list(k,l))


# ==============================================================================
# The code below is for debugging a particular test in eclipse/pydev.
# (normally all tests are run with pytest)
# ==============================================================================
if __name__ == "__main__":
    the_test_you_want_to_debug = test_linearised

    the_test_you_want_to_debug()
    print("-*# finished #*-")
# ==============================================================================
