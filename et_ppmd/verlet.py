# -*- coding: utf-8 -*-

"""
Module et_ppmd.verlet 
=================================================================

A module

"""

import numpy as np

class VerletList:
    def __init__(self, cutoff, max_neighbours=50):
        """Verlet List class of all atoms.

        :param float cutoff: cutoff distance
        :param integer max_neighours: maximum size of an atom's Verlet list.
            If this number is too small build|build_simple|build_grid will
            raise IndexError.

        The initial data structure is a 2D integer numpy array. There is one
        row for each atom. Each row starts with the number of neighbours,
        followed by the atom indices of the neighbours. Thus:

        * vl[i,:] is the Verlet list of atom i.
        * vl[i,0] = ni is the number of atoms in Verlet list of atom i.
        * vl[i,1:ni+1] contains the indices of the ni neighbouring atoms.

        This 2D data structure is linearised to facilitate passing the same
        linear data structure to Fortran and C++ in an efficient way. The
        linearised data structure consist of:

        *   vl_list : 1D numpy array containing all Verlet lists, one after the other,
                      i.e. vl(0), vl(1), ..., vl(n_atoms-1), with n_atoms the total
                      number of atoms.
        *   cl_size : 1D numpy array containing the number of atoms in the individual
                      Verlet lists, i.e. length of each Verlet list.
        *   cl_offset : 1D numpy array containing the starting position of all the
                      Verlet lists in the cl_list array.
        """
        self.cutoff = cutoff
        self.max_neighbours = max_neighbours
        self.vl = None
        self.debug = False

    def allocate_2d(self, n_atoms):
        """allocate and initialize the 2D data structure. 

        :param n_atoms: number of atoms to accomodate

        If the datastructure was already allocate_2dd, and large enough
        to accommodate the atoms and the contacts, it is only reinitialized.
        """
        self.n_atoms = n_atoms
        n,m = self.vl.shape if not self.vl is None else (0,0)
        if n<n_atoms or m<self.max_neighbours:
            # the array is too small, reallocate
            self.vl = np.empty((n_atoms, self.max_neighbours+1), dtype=int)


        # initialize all lists as empty (the content of the lists is not relevant)
        self.vl[:,0] = 0
        # garbage collection of the linear data structure
        self.vl_list = None
        self.vl_size = None
        self.vl_offset = None


    def __str__(self):
        s = "verlet lists:\n"
        max_neighbours = 0
        for i in range(self.n_atoms):
            vli = self.verlet_list(i)
            s += f'({i}): {vli}\n'
            size = len(vli)
            if size > max_neighbours:
                max_neighbours = size
        s += f'max elements: {max_neighbours}/{self.max_neighbours}, linearised={self.linearised()}\n'
        return s
    

    def add(self,i,j):
        """Add pair (i,j) to the Verlet list.

        (for the 2D vl).
        """

        # increase the count of atom i
        self.vl[i,0] += 1
        n = self.vl[i,0]
        if n > self.max_neighbours:
            raise IndexError(f"The maximum number of neighbours for this VerletList object is {self.max_neighbours}.\n"
                               f"Increase max_neighbours in the constructor.")
        # store j in the Verlet list of atom i
        self.vl[i,n] = j


    def build(self, x, y, keep_2d=False ):
        """Build the Verlet list from the positions.

        Brute force approach, but using array arithmetic, rather
        than pairwise computations.
        This algorithm has complexity O(N), but is significantly faster than
        build_simple

        :param x: x-coordinates of atoms
        :param y: y-coordinates of atoms
        :param bool keep_2d: if True the 2D Verlet list data structure is not deleted after linearisation.
        """
        self.allocate_2d(len(x))
        rc2 = self.cutoff**2

        xij = np.empty((self.n_atoms,),dtype=float)
        yij = np.empty((self.n_atoms,),dtype=float)
        ri2 = np.empty((self.n_atoms,),dtype=float)
        for i in range(self.n_atoms-1):
            xij[i+1:] = x[i+1:] - x[i]
            yij[i+1:] = y[i+1:] - y[i]
            if self.debug:
                ri2 = 0
            ri2[i+1:] = xij[i+1:]**2 + yij[i+1:]**2
            for j in range(i+1,self.n_atoms):
                if ri2[j] <= rc2:
                    self.add(i,j)
        self.linearise(keep_2d)

    def build_simple(self, x, y, keep_2d=False ):
        """Build the Verlet list from the positions.

        Brute force approach, in the simplest way.
        This algorithm has complexity O(N).

        :param x: x-coordinates of atoms
        :param y: y-coordinates of atoms
        """
        self.allocate_2d(len(x))
        rc2 = self.cutoff**2
        for i in range(self.n_atoms):
            for j in range(i+1,self.n_atoms):
                rij2 = (x[j] - x[i])**2 + (y[j] - y[i])**2
                if rij2 <= rc2:
                    self.add(i, j)
        self.linearise(keep_2d=keep_2d)


    def build_grid(self, x, y, grid, keep_2d=False ):
        """Build Verlet lists using a grid.

        This algorithm has complexity O(N).
        """
        if not grid.linearised():
            raise ValueError("The grid list must be built and linearised first.")

        self.allocate_2d(len(x))
        rc2 = self.cutoff**2
        # loop over all cells
        for l in range(grid.n):
            for k in range(grid.m):
                ckl = grid.cell_list(k,l)
                n_atoms_in_ckl = len(ckl)
                # loop over all atom pairs in ckl
                for ia in range(n_atoms_in_ckl):
                    i = ckl[ia]
                    for j in ckl[ia+1:]:
                        rij2 = (x[j] - x[i])**2 + (y[j] - y[i])**2
                        if rij2 <= rc2:
                            self.add(i, j)
                # loop over neighbouring cells. If the cell does not exist an IndexError is raised
                for kl2 in ((k+1,l), (k-1,l+1), (k,l+1), (k+1,l+1)):
                    try:
                        ckl2 = grid.cell_list(*kl2)
                    except IndexError:
                        pass # Cell kl2 does not exist
                    else: # The else clause is executed only when the try clause does not raise an error
                        # loop over all atom pairs i,j with i in ckl and j in ckl2
                        for i in ckl:
                            for j in ckl2:
                                rij2 = (x[j] - x[i]) ** 2 + (y[j] - y[i]) ** 2
                                if rij2 <= rc2:
                                    self.add(i, j)

        self.linearise(keep_2d=keep_2d)


    def linearised(self):
        return not self.vl_list is None


    def linearise(self,keep_2d=False):
        """linearise the Verlet list.

        :param bool keep_2d: keep self.vl or not. True is used for testing purposes.
        """
        self.vl_size   = np.empty(self.n_atoms, dtype=int)
        self.vl_offset = np.empty(self.n_atoms, dtype=int)
        n_neighbours_total = np.sum(self.vl[:,0])
        self.vl_list = np.empty(n_neighbours_total, dtype=int)
        offset = 0
        for i in range(self.n_atoms):
            n_neighbours_i = self.vl[i,0]
            self.vl_size[i] = n_neighbours_i
            self.vl_offset[i] = offset
            self.vl_list[offset:offset+n_neighbours_i] = self.vl[i,1:1+n_neighbours_i]
            offset += n_neighbours_i

        if not keep_2d:
            self.vl = None # garbage collection takes care of it.


    def verlet_list(self,i):
        """Return the Verlet list of atom i.

        :return: view of a numpy array
        """
        if self.linearised():
            offset = self.vl_offset[i]
            size   = self.vl_size[i]
            vli = self.vl_list[offset:offset+size]
        else:
            size = self.vl[i,0]
            vli = self.vl[i,1:1+size]
        return vli


    def has(self, ij):
        """Test if the verlet list of atom i contains atom j, ij = (i,j). """
        i,j = ij
        return j in self.verlet_list(i)


