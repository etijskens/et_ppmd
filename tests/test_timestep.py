#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `et_ppmd` package."""

import pytest
import numpy as np

import et_ppmd.timestep


def test_VelocityVerlet_no_motion():
    rx = np.zeros((2,), dtype=float)
    ry = np.zeros((2,), dtype=float)
    vx = np.zeros((2,), dtype=float)
    vy = np.zeros((2,), dtype=float)
    ax = np.zeros((2,), dtype=float)
    ay = np.zeros((2,), dtype=float)
    vv = et_ppmd.timestep.VelocityVerlet(rx,ry,vx,vy,ax,ay)
    dt = 1.0
    for t in range(10):
        vv.step_12(dt)
        assert np.all(vv.rx==0.0)
        assert np.all(vv.ry==0.0)
        assert np.all(vv.vx==0.0)
        assert np.all(vv.vy==0.0)
        assert np.all(vv.ax==0.0)
        assert np.all(vv.ay==0.0)
        # Step 3: no forces
        vv.step_4(dt)
        assert np.all(vv.rx == 0.0)
        assert np.all(vv.ry == 0.0)
        assert np.all(vv.vx == 0.0)
        assert np.all(vv.vy == 0.0)
        assert np.all(vv.ax == 0.0)
        assert np.all(vv.ay == 0.0)

def test_VelocityVerlet_constant_velocity():
    rx = np.zeros((2,), dtype=float)
    ry = np.zeros((2,), dtype=float)
    vx = np.array([1,0], dtype=float)
    vy = np.array([0,1], dtype=float)
    ax = np.zeros((2,), dtype=float)
    ay = np.zeros((2,), dtype=float)
    vv = et_ppmd.timestep.VelocityVerlet(rx, ry, vx, vy, ax, ay)
    dt = 1.0
    for it in range(10):
        vv.step_12(dt)
        assert vv.rx[0] == 1.0*dt*(it+1)
        assert vv.rx[1] == 0.0
        assert vv.ry[0] == 0.0
        assert vv.ry[1] == 1.0*dt*(it+1)

        assert vv.vx[0] == 1.0
        assert vv.vx[1] == 0.0
        assert vv.vy[0] == 0.0
        assert vv.vy[1] == 1.0

        assert np.all(vv.ax == 0.0)
        assert np.all(vv.ay == 0.0)

        # Step 3: no forces

        vv.step_4(dt)
        assert vv.rx[0] == 1.0*dt*(it+1)
        assert vv.rx[1] == 0.0
        assert vv.ry[0] == 0.0
        assert vv.ry[1] == 1.0*dt*(it+1)

        assert vv.vx[0] == 1.0
        assert vv.vx[1] == 0.0
        assert vv.vy[0] == 0.0
        assert vv.vy[1] == 1.0

        assert vv.vy[0] == 0.0
        assert vv.vy[1] == 1.0
        assert np.all(vv.ax == 0.0)
        assert np.all(vv.ay == 0.0)

def test_VelocityVerlet_constant_acceleration():
    """analytical solution:
    a = cst
    v = a*t
    r = 0.5*a*t**2
    """
    a = 1.0
    rx = np.zeros((2,), dtype=float)
    ry = np.zeros((2,), dtype=float)
    vx = np.zeros((2,), dtype=float)
    vy = np.zeros((2,), dtype=float)
    ax = np.array([a,0], dtype=float)
    ay = np.array([0,a], dtype=float)
    vv = et_ppmd.timestep.VelocityVerlet(rx, ry, vx, vy, ax, ay)
    dt = 1e-4
    nsteps = 10000
    for it in range(nsteps):
        vv.step_12(dt)
        t = (it+1)*dt
        print(f't={t}')
        r = 0.5 * a * t**2
        print(f'r={r}')
        assert vv.rx[0] == pytest.approx(r, 1e-8)
        assert vv.rx[1] == 0.0
        assert vv.ry[0] == 0.0
        assert vv.ry[1] == pytest.approx(r, 1e-8)

        # Step 3: no forces

        vv.step_4(dt)
        v = a*t
        print(f'v={v}\n')
        assert vv.vx[0] == pytest.approx(v, 1e-8)
        assert vv.vx[1] == 0.0
        assert vv.vy[0] == 0.0
        assert vv.vy[1] == pytest.approx(v, 1e-8)

    # test the current time:
    assert vv.t == pytest.approx(dt*nsteps,1e-8)

def test_VelocityVerlet_constant_acceleration_2():
    """analytical solution:
    a = cst
    v = a*t
    r = 0.5*a*t**2
    """
    a = 1.0
    for impl in ['f90','cpp']:
        print(f'testing {impl} implementation')
        ry = np.zeros((2,), dtype=float)
        vx = np.zeros((2,), dtype=float)
        rx = np.zeros((2,), dtype=float)
        vy = np.zeros((2,), dtype=float)
        ax = np.array([a,0], dtype=float)
        ay = np.array([0,a], dtype=float)
        vv = et_ppmd.timestep.VelocityVerlet(rx, ry, vx, vy, ax, ay,impl=impl)
        dt = 1e-4
        nsteps = 10000
        for it in range(nsteps):
            vv.step_12(dt)
            t = (it+1)*dt
            print(f't={t}')
            r = 0.5 * a * t**2
            print(f'r={r}')
            assert vv.rx[0] == pytest.approx(r, 1e-8)
            assert vv.rx[1] == 0.0
            assert vv.ry[0] == 0.0
            assert vv.ry[1] == pytest.approx(r, 1e-8)

            # Step 3: no forces

            vv.step_4(dt)
            v = a*t
            print(f'v={v}\n')
            assert vv.vx[0] == pytest.approx(a * t, 1e-8)
            assert vv.vx[1] == 0.0
            assert vv.vy[0] == 0.0
            assert vv.vy[1] == pytest.approx(a * t, 1e-8)

        # test the current time:
        assert vv.t == pytest.approx(dt*nsteps,1e-8)

# ==============================================================================
# The code below is for debugging a particular test in eclipse/pydev.
# (normally all tests are run with pytest)
# ==============================================================================
if __name__ == "__main__":
    the_test_you_want_to_debug = test_VelocityVerlet_constant_acceleration_2

    the_test_you_want_to_debug()
    print("-*# finished #*-")
# ==============================================================================
