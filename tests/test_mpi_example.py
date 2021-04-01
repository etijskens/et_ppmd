#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for mpi_example.py

Check https://pytest-mpi.readthedocs.io/en/latest/usage.html for how to test
mpi4py code

to run the test code::

    (.venv)> mpirun -np 2 python -m pytest tests/test_mpi_example.py --with-mpi

The output looks like this. Note that some output is duplicated because there
are two processes being run::

    ============================= test session starts ==============================
    platform darwin -- Python 3.8.5, pytest-4.6.11, py-1.10.0, pluggy-0.13.1
    ============================= test session starts ==============================
    platform darwin -- Python 3.8.5, pytest-4.6.11, py-1.10.0, pluggy-0.13.1
    rootdir: /Users/etijskens/software/dev/workspace/et_ppmd
    plugins: mpi-0.5
    rootdir: /Users/etijskens/software/dev/workspace/et_ppmd
    plugins: mpi-0.5
    collected 1 item
    collected 1 item

    tests/test_mpi_example.py
    tests/test_mpi_example.py ..                                              [100%]                                              [100%]

    =============================== MPI Information ================================
    =============================== MPI Information ================================
    rank: 1
    size: 2
    rank: 0
    size: 2
    MPI version: 3.1
    MPI library version: Open MPI v4.0.5, package: Open MPI brew@Catalina Distribution, ident: 4.0.5, repo rev: v4.0.5, Aug 26, 2020
    MPI version: 3.1
    MPI library version: Open MPI v4.0.5, package: Open MPI brew@Catalina Distribution, ident: 4.0.5, repo rev: v4.0.5, Aug 26, 2020
    MPI vendor: Open MPI 4.0.5
    mpi4py rc:
     initialize: True
    MPI vendor: Open MPI 4.0.5
    mpi4py rc:
     threads: True
     thread_level: multiple
     initialize: True
     threads: True
     thread_level: multiple
     finalize: None
     fast_reduce: True
     recv_mprobe: True
     errors: exception
    mpi4py config:
     finalize: None
     fast_reduce: True
     recv_mprobe: True
     errors: exception
    mpi4py config:
     mpicc: /usr/local/bin/mpicc
     mpicxx: /usr/local/bin/mpicxx
     mpifort: /usr/local/bin/mpifort
     mpif90: /usr/local/bin/mpif90
     mpif77: /usr/local/bin/mpif77
     mpicc: /usr/local/bin/mpicc
     mpicxx: /usr/local/bin/mpicxx
     mpifort: /usr/local/bin/mpifort
     mpif90: /usr/local/bin/mpif90
     mpif77: /usr/local/bin/mpif77

    =========================== 1 passed in 0.98 seconds ===========================
    =========================== 1 passed in 0.98 seconds ===========================
"""

import pytest

import et_ppmd.mpi_example as mpix


@pytest.mark.mpi(min_size=2)
def test_mpi_message():
    rank = mpix.mpi_rank
    size = mpix.mpi_size
    assert rank<=size
    assert size>=2
    msg = 'test'
    s = mpix.mpi_message(msg)
    print(s)
    assert s == f'rank {rank}/{size}: {msg}'

# ==============================================================================
# The code below is for debugging a particular test in eclipse/pydev.
# (normally all tests are run with pytest)
# check https://stackoverflow.com/questions/57519129/how-to-run-python-script-with-mpi4py-using-mpiexec-from-within-pycharm
# to debug mpi4py code in pycharm
# ==============================================================================
if __name__ == "__main__":
    the_test_you_want_to_debug = test_mpi_message

    the_test_you_want_to_debug()
    print("-*# finished #*-")
# ==============================================================================
