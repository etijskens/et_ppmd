#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `et_ppmd` package."""

# import pytest

import et_ppmd
import et_ppmd.verlet as verlet
import et_ppmd.grid as grid

import numpy as np

def vl2set(vl):
    """Convert VerletList object into set of pairs."""
    pairs = set()
    for i in range(vl.n_atoms):
        vli = vl.verlet_list(i)
        n_pairs_i = len(vli)
        for j in vli:
            pair = (i,j) if i<j else (j,i)
            # print(pair)
            pairs.add(pair)
    return pairs


def test_build_simple_1():
    """"""
    x = np.array([0.0, 1, 2, 3, 4])
    n_atoms = len(x)
    y = np.zeros((n_atoms,), dtype=float)
    vl = verlet.VerletList(cutoff=2.5, max_neighbours=4)
    vl.build_simple(x, y, keep_2d=True)
    print(vl)
    assert     vl.has((0, 1))
    assert     vl.has((0, 2))
    assert not vl.has((0, 3))
    assert not vl.has((0, 4))
    assert not vl.has((1, 0))
    assert     vl.has((1, 2))
    assert     vl.has((1, 3))
    assert not vl.has((1, 4))
    assert not vl.has((2, 0))
    assert not vl.has((2, 1))
    assert     vl.has((2, 3))
    assert     vl.has((2, 4))
    assert not vl.has((3, 0))
    assert not vl.has((3, 1))
    assert not vl.has((3, 2))
    assert     vl.has((3, 4))
    assert not vl.has((4, 0))
    assert not vl.has((4, 1))
    assert not vl.has((4, 2))
    assert not vl.has((4, 3))
    assert np.all(vl.vl_size == np.array([2,2,2,1,0]))
    assert np.all(vl.vl_list == np.array([1,2,2,3,3,4,4]))

def test_build_simple_2():
    """"""
    x = np.array([0.0, 1, 2, 3, 4])
    n_atoms = len(x)
    y = np.zeros((n_atoms,), dtype=float)
    vl = verlet.VerletList(cutoff=2.0, max_neighbours=4)
    vl.build_simple(x, y, keep_2d=True)
    print(vl)
    assert     vl.has((0, 1))
    assert     vl.has((0, 2))
    assert not vl.has((0, 3))
    assert not vl.has((0, 4))
    assert not vl.has((1, 0))
    assert     vl.has((1, 2))
    assert     vl.has((1, 3))
    assert not vl.has((1, 4))
    assert not vl.has((2, 0))
    assert not vl.has((2, 1))
    assert     vl.has((2, 3))
    assert     vl.has((2, 4))
    assert not vl.has((3, 0))
    assert not vl.has((3, 1))
    assert not vl.has((3, 2))
    assert     vl.has((3, 4))
    assert not vl.has((4, 0))
    assert not vl.has((4, 1))
    assert not vl.has((4, 2))
    assert not vl.has((4, 3))
    assert np.all(vl.vl_size == np.array([2,2,2,1,0]))
    assert np.all(vl.vl_list == np.array([1,2,2,3,3,4,4]))

def test_vl2pairs():
    x = np.array([0.0, 1, 2, 3, 4])
    n_atoms = len(x)
    y = np.zeros((n_atoms,), dtype=float)
    vl = verlet.VerletList(cutoff=2.0, max_neighbours=4)
    vl.build_simple(x, y, keep_2d=True)
    print(vl)
    print(vl2set(vl))

def test_build_simple_2b():
    """"""
    x = np.array([0.0, 1, 2, 3, 4])
    n_atoms = len(x)
    y = np.zeros((n_atoms,), dtype=float)
    vl = verlet.VerletList(cutoff=2.0, max_neighbours=4)
    vl.build_simple(x, y, keep_2d=True)
    # print(vl)
    pairs = vl2set(vl)
    expected = {(0,1),(0,2),(1,2),(1,3),(2,3),(2,4),(3,4)}
    assert pairs == expected

def test_neighbours():
    """"""
    x = np.array([0.0, 1, 2, 3, 4])
    n_atoms = len(x)
    y = np.zeros((n_atoms,), dtype=float)
    vl = verlet.VerletList(cutoff=2.0, max_neighbours=4)
    vl.build_simple(x,y)
    print(vl.vl_size[0])
    assert vl.vl_size[0] == 2
    vl0 = vl.verlet_list(0)
    print(vl0)
    assert vl0[0] == 1
    assert vl0[1] == 2

def test_build_1():
    """Verify VerletList.build against VerletList.build_simple."""

    cutoff = 2.5
    x = np.array([0.0, 1, 2, 3, 4])
    n_atoms = len(x)
    y = np.zeros((n_atoms,), dtype=float)
    vl = verlet.VerletList(cutoff=cutoff, max_neighbours=4)
    vl.build(x,y)
    print(vl)
    pairs = vl2set(vl)

    vlsimple = verlet.VerletList(cutoff=cutoff, max_neighbours=4)
    vlsimple.build_simple(x,y)
    print(vlsimple)
    expected = vl2set(vlsimple)
    assert pairs == expected
    assert np.all(vl.vl_size == np.array([2,2,2,1,0]))
    assert np.all(vl.vl_list == np.array([1,2,2,3,3,4,4]))

def test_build_2():
    """Verify VerletList.build against VerletList.build_simple."""
    cutoff = 5.0
    max_neighbours = 50
    n_squares = 3
    box = et_ppmd.Box(0., 0., n_squares*5.*et_ppmd.hcp.uc_centered_a, n_squares*3.*et_ppmd.hcp.uc_centered_b)
    x,y = box.generateAtoms(r=et_ppmd.hcp.radius)
    vl = verlet.VerletList(cutoff=cutoff, max_neighbours=max_neighbours)
    vl.build(x,y)
    print(vl)
    pairs = vl2set(vl)

    vlsimple = verlet.VerletList(cutoff=cutoff, max_neighbours=max_neighbours)
    vlsimple.build_simple(x,y)
    print(vlsimple)
    expected = vl2set(vlsimple)
    assert pairs == expected

def test_build_3():
    """Verify VerletList.build_grid against VerletList.build_simple."""
    cutoff = 5.0
    max_neighbours = 100
    n_squares = 3
    box = et_ppmd.Box(0., 0., n_squares*5.*et_ppmd.hcp.uc_centered_a, n_squares*3.*et_ppmd.hcp.uc_centered_b)
    x,y = box.generateAtoms(r=et_ppmd.hcp.radius)

    # compute grid
    the_grid = grid.Grid(cell_size=cutoff, wx=box.xur - box.xll, wy=box.yur - box.yll, max_atoms_per_cell=100)
    the_grid.build(x, y)

    # build grid-based verlet list
    vl = verlet.VerletList(cutoff=cutoff, max_neighbours=max_neighbours)
    vl.build_grid(x, y, the_grid)
    print(vl)
    pairs = vl2set(vl)

    vlsimple = verlet.VerletList(cutoff=cutoff, max_neighbours=max_neighbours)
    vlsimple.build_simple(x,y)
    print(vlsimple)
    expected = vl2set(vlsimple)
    assert pairs == expected

# ==============================================================================
# The code below is for debugging a particular test in eclipse/pydev.
# (normally all tests are run with pytest)
# ==============================================================================
if __name__ == "__main__":
    the_test_you_want_to_debug = test_build_3
    print(f'__main__ running {the_test_you_want_to_debug}')

    the_test_you_want_to_debug()
    print("-*# finished #*-")
# ==============================================================================
