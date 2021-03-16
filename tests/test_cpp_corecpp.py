#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for C++ module et_ppmd.corecpp.
"""


import numpy as np
import et_ppmd
# create an alias for the binary extension cpp module
cpp = et_ppmd.corecpp


def test_cpp_add():
    x = np.array([0,1,2,3,4],dtype=float)
    shape = x.shape
    y = np.ones (shape,dtype=float)
    z = np.zeros(shape,dtype=float)
    expected_z = x + y
    result = cpp.add(x,y,z)
    assert (z == expected_z).all()


#===============================================================================
# The code below is for debugging a particular test in eclipse/pydev.
# (normally all tests are run with pytest)
#===============================================================================
if __name__ == "__main__":
    the_test_you_want_to_debug = test_cpp_add

    print(f"__main__ running {the_test_you_want_to_debug} ...")
    the_test_you_want_to_debug()
    print('-*# finished #*-')
#===============================================================================
