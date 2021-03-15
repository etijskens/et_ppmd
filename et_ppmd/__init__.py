# -*- coding: utf-8 -*-

"""
Package et_ppmd
=======================================

Top-level package for et_ppmd.
"""

__version__ = "0.0.0"

from et_ppmd.forces import R0
from et_ppmdcommon import Box, ClosestPacking2D
import numpy as np

hcp = ClosestPacking2D(r=R0)

class Atoms:
    def __init__(self, box, r=R0, noise=None, rc=3*R0):
        """

        box should be aligned with grid -> box size = multiples of rc
        :param box:
        :param r:
        :param noise:
        :param rc:
        """
        self.box = box
        self.radius = 0.5*r
        self.rc = rc
        self.x, self.y = self.box.generateAtoms(r=r, noise=noise)



