# -*- coding: utf-8 -*-

"""
Module et_ppmd.verlet 
=================================================================

A module

"""

import numpy as np

class VerletList:
    def __init__(self, cutoff, max_contacts=50):
        """Verlet List class of all atoms.

        The data structure is a 2D integer numpy array. There is one row = one
        Verlet list per atom. Each row starts with the number of neighbours,
        followed by the atom indices of the indices. Thus:

        * vl_array[i,:] is the Verlet list of atom i.
        * vl_array[i,0] = ni is the number of atoms in Verlet list of atom i.
        * vl_array[i,1:ni+1] contains the indices of the ni neighbouring atoms.

        vl_array has row major order, so row entries are contiguous. Consequently,
        the Verlet list entries are also contiguous. This is important for
        performance.

        :param float cutoff: cutoff distance
        :param integer max_contacts: maximum number of contacts per atom
        """
        self.cutoff = cutoff
        self.max_contacts = max_contacts
        self.vl_array = None
        self.debug = False

    def allocate(self, n_atoms):
        """Allocate and initialize the data structure.

        `If the datastructure was already allocated, and large enough
        to accommodate the atoms and the contacts, it is only reinitialized.

        :param n_atoms: number of atoms
        """
        n,m = self.vl_array.shape if not self.vl_array is None else (0,0)
        if n<n_atoms or m<self.max_contacts:
            # the array is too small
            self.vl_array = np.empty((n_atoms, self.max_contacts+1), dtype=int, order='C')
            # row-major storage order ('C') is important for performance in the build function
        # initialize all lists as empty (the content of the lists is not relevant)
        self.vl_array[:,0] = 0

    def actual_max_neighbours(self):
        return np.max(self.vl_n_neighbours)

    def n_atoms(self):
        return len(self.vl_n_neighbours)

    def __str__(self):
        s = ""
        n = len(self.vl_n_neighbours)
        nb0 = 0
        for i in range(n):
            n_neighbours_i = self.vl_n_neighbours[i]
            if n_neighbours_i > 0:
                s += f"[{i}] {self.vl_neighbours[nb0:nb0+n_neighbours_i]}\n"
                nb0 += n_neighbours_i
            else:
                s += f"[{i}] []\n"

        return s

    def add(self,i,j):
        """Add pair (i,j) to the Verlet list.

        (for the 2D vl_array).
        """

        # increase the count of atom i
        self.vl_array[i,0] += 1
        n = self.vl_array[i,0]
        if n > self.max_contacts:
            raise RuntimeError(f"The maximum number of contatx for this Verlet List is {self.max_contacts}.\n"
                               f"Increase max_contacts.")
        # store j in the Verlet list of atom i
        self.vl_array[i,n] = j


    def has(self, ij):
        """Test if the verlet list of atom i contains atom j, ij = (i,j). """
        i,j = ij
        return j in self.neighbours(i)


    def build(self, x, y, keep_vl2d=False ):
        """Build the Verlet list from the positions.

        Brute force approach, but using array arithmetic, rather
        than pairwise computations.

        :param x: x-coordinates of atoms
        :param y: y-coordinates of atoms
        """
        n_atoms = len(x)
        self.allocate(n_atoms)
        rc2 = self.cutoff**2

        xij = np.empty((n_atoms,),dtype=float)
        yij = np.empty((n_atoms,),dtype=float)
        ri2 = np.empty((n_atoms,),dtype=float)
        for i in range(n_atoms-1):
            xij[i+1:] = x[i+1:] - x[i]
            yij[i+1:] = y[i+1:] - y[i]
            if self.debug:
                ri2 = 0
            ri2[i+1:] = xij[i+1:]**2 + yij[i+1:]**2
            for j in range(i+1,n_atoms):
                if ri2[j] <= rc2:
                    self.add(i,j)
        self.linearize(keep_vl2d)

    def build_simple(self, x, y, keep_vl2d=False ):
        """Build the Verlet list from the positions.

        Brute force approach, in the simplest way.

        :param x: x-coordinates of atoms
        :param y: y-coordinates of atoms
        """
        n_atoms = len(x)
        self.allocate(n_atoms)
        rc2 = self.cutoff**2
        for i in range(n_atoms):
            for j in range(i+1,n_atoms):
                rij2 = (x[j] - x[i])**2 + (y[j] - y[i])**2
                if rij2 <= rc2:
                    self.add(i, j)
        self.linearize(keep_vl2d=keep_vl2d)

    def n_neighbours(self,i):
        """Return the number of neighbours of atom i."""
        return self.vl_n_neighbours[i]

    def neighbours(self,i):
        """Return the neighbours of atom i.

        :return: view of a numpy array
        """
        nb0 = np.sum(self.vl_n_neighbours[:i])
        return self.vl_neighbours[nb0:nb0+self.vl_n_neighbours[i]]

    def linearize(self,keep_vl2d=False):
        """Linearize the Verlet list.

        :param bool keep_vl2d: keep self.vl_array or not. True is used for testing purposes.
        """
        n_atoms = self.vl_array.shape[0]
        n_pairs = np.sum(self.vl_array[:,0])
        self.vl_n_neighbours = np.empty(n_atoms, dtype=int)
        self.vl_neighbours   = np.empty(n_pairs, dtype=int)
        nb0 = 0
        for i in range(n_atoms):
            n_neighbours_i = self.vl_array[i,0]
            self.vl_n_neighbours[i] =n_neighbours_i
            if n_neighbours_i > 0:
                self.vl_neighbours[nb0:nb0+n_neighbours_i] = self.vl_array[i,1:1+n_neighbours_i]
                nb0 += n_neighbours_i

        if not keep_vl2d:
            self.vl_array = None # garbage collection takes care of it.