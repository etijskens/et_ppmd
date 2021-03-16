# -*- coding: utf-8 -*-

"""
Package et_ppmd
=======================================

Top-level package for et_ppmd.
"""

__version__ = "0.0.0"

from et_ppmd.forces import R0
from et_ppmd.verlet import VerletList
from et_ppmdcommon import Box, ClosestPacking2D
import numpy as np

hcp = ClosestPacking2D(r=R0)

class MD:
    def __init__(self, box, interatomic_distance=R0, noise=None, cutoff=5*R0):
        """

        :param Box box: box containing the atoms
        :param the interatomic_distance: interatomic distance (without noise)
        :param noise: add noise to the atom positions, specified as a fraction of the interatomic distance.
        :param cutoff:
        """
        self.box = box
        self.interatomic_distance = interatomic_distance
        self.radius = 0.5*interatomic_distance
        self.cutoff = cutoff
        self.x, self.y = self.box.generateAtoms(r=interatomic_distance, noise=noise)
        self.vl = VerletList(cutoff, max_contacts=50)
        shape = self.x.shape
        self.vx = np.empty(shape, dtype=float)
        self.vy = np.empty(shape, dtype=float)
        self.ax = np.empty(shape, dtype=float)
        self.ay = np.empty(shape, dtype=float)
        self.n_atoms = shape[0]

    def buildVerletLists(self):
        self.vl.build(self.x,self.y)

    def computeEnergy(self):
        """"""
        energy = 0.0
        n_atoms = len(self.x)
        for i in range(n_atoms):
            neighbours = self.vl.neighbours(i)
            xi = self.x[i]
            yi = self.y[i]
            for j in neighbours:
                rij2 = (self.x[j] - xi)**2 + (self.y[j] - yi)**2
                energy += forces.potential(rij2)
        return energy

    def computeForces(self):
        """"""
        n_atoms = len(self.x)

        self.ax[:] = 0.0
        self.ax[:] = 0.0
        for i in range(n_atoms):
            neighbours = self.vl.neighbours(i)
            for j in neighbours:
                xij = self.x[j] - self.x[i]
                yij = self.y[j] - self.y[i]
                rij2 = xij**2 + yij**2
                ff = forces.force_factor(rij2)
                fx = ff*xij
                fy = ff*yij
                self.ax[i] += fx
                self.ay[i] += fy
                # add the opposite force to atom j
                self.ax[j] -= fx
                self.ay[j] -= fy
