#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `et_ppmd` package."""

import et_ppmd.forces as lj
R0 = lj.R0
import numpy as np
import pytest


def test_potential_minimum():
    """potential has a minimum at r = R0"""
    r0 = R0
    vr0 = lj.potential(r0**2)
    d = 0.1
    for i in range(5):
        rLeft  = r0 - d
        rRight = r0 + d
        vLeft  = lj.potential(rLeft**2)
        vRight = lj.potential(rRight**2)
        print(f'{d}: {vLeft} < {vr0} < {vRight}')
        assert vLeft  > vr0
        assert vRight > vr0
        d *= 0.1

def test_potential_zero():
    """potential has a zero at r = 1.0"""
    r =  1.0
    vr = lj.potential(r**2)
    assert vr == 0.0

def test_potential_increasing_right_of_R0():
    """potential increases to the right of R0, and remains negative."""
    ri = rprev = R0
    vprev = lj.potential(ri**2)
    for i in range(10):
        ri = 2*ri
        vi = lj.potential(ri**2)
        print(f'{i} {ri}: {vi}')
        assert vprev < vi
        assert vi < 0.0
        vprev = vi
        rprev = ri

def test_potential_decreasing_left_of_R0():
    """potential increases to the right of R0, and remains negative."""
    ri = rprev = R0
    vprev = lj.potential(ri**2)
    for i in range(10):
        ri = 0.9*ri
        vi = lj.potential(ri**2)
        print(f'{i} {ri}: {vi}')
        assert vi > vprev
        if ri > 1:
            assert vi < 0.0
        else:
            assert vi > 0.0
        vprev = vi
        rprev = ri

def test_potential_cutoff():
    """not actually a test, just to show the magnitude of the interaction at cut-off."""
    for i in range(1,11):
        rc = i*R0
        vrc = lj.potential(rc**2)
        print(f'{i} {rc}: `{vrc} {"cut-off" if i==3 else ""}')


def random_unit_vector(n=1):
    """"""
    theta = (2.0*np.py)*np.random.random(n)
    xij = np.cos(theta)
    yij = np.sin(theta)


def test_zero_force_R0():
    """verify that the force magnitude is zero at R0."""
    rij2 = R0**2
    fij = lj.force_factor(rij2)
    # account for round-off error R0**6 is not exactly 5, although R0 is defined as 2**1/6
    assert fij == pytest.approx(0.0, 5e-16)


def test_force_is_derivative_of_potential():
    n = 1000
    # Generate n random numbers in ]0,5*R0]
    # np.random.random generate numbers in [
    rij = (5*R0)*(1.0 - np.random.random(n))
    fij = lj.force_factor(rij**2)*rij
    d = 1e-10
    rij0 = rij - d
    vij0 = lj.potential(rij0**2)
    rij1 = rij + d
    vij1 = lj.potential(rij1**2)
    dvij = (vij1 - vij0)/(2*d)
    for i in range(n):
        print(f'{i} {fij[i]} == {dvij[i]} {np.abs(fij[i]-dvij[i])}')
        assert fij[i] == pytest.approx(dvij[i],1e-4)

def test_force_direction():
    """test that atoms nearer than R0 are repelled and
    that atomsw farther than R0 are attracted
    """
    xij = 0.9 * R0
    yij = 0
    fijx,fijy = lj.force(xij,yij)
    assert fijx < 0
    xij = 1.1 * R0
    yij = 0
    fijx,fijy = lj.force(xij,yij)
    assert fijx > 0

# ==============================================================================
# The code below is for debugging a particular test in eclipse/pydev.
# (normally all tests are run with pytest)
# ==============================================================================
if __name__ == "__main__":
    the_test_you_want_to_debug = test_potential_cutoff

    the_test_you_want_to_debug()
    print("-*# finished #*-")
# ==============================================================================
