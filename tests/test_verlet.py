#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `et_ppmd` package."""

# import pytest

import et_ppmd.verlet as verlet

import numpy as np

def test_simple_1():
    """"""
    x = np.array([0.0, 1, 2, 3, 4])
    n_atoms = len(x)
    y = np.zeros((n_atoms,), dtype=float)
    vl = verlet.VerletList(cutoff=2.5, max_contacts=4)
    vl.build_simple(x,y)
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

def test_simple_2():
    """"""
    x = np.array([0.0, 1, 2, 3, 4])
    n_atoms = len(x)
    y = np.zeros((n_atoms,), dtype=float)
    vl = verlet.VerletList(cutoff=2.0, max_contacts=4)
    vl.build_simple(x,y)
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


# ==============================================================================
# The code below is for debugging a particular test in eclipse/pydev.
# (normally all tests are run with pytest)
# ==============================================================================
if __name__ == "__main__":
    the_test_you_want_to_debug = test_simple_2

    the_test_you_want_to_debug()
    print("-*# finished #*-")
# ==============================================================================
