#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `et_ppmd` package."""

import pytest

import et_ppmd 


def test_greet():
    expected = "Hello John!"
    greeting = greet("John")
    assert greeting==expected


# ==============================================================================
# The code below is for debugging a particular test in eclipse/pydev.
# (normally all tests are run with pytest)
# ==============================================================================
if __name__ == "__main__":
    the_test_you_want_to_debug = test_

    the_test_you_want_to_debug()
    print("-*# finished #*-")
# ==============================================================================
