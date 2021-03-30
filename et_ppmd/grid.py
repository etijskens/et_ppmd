# -*- coding: utf-8 -*-

"""
Module et_ppmd.grid 
=================================================================

A module

"""

import numpy as np

class Grid:
    def __init__(self, cell_size, wx, wy=None, max_atoms_per_cell=5):
        """Construct a Grid object over a rectangular domain of lenght wx
        and height wy.

        :param float cell_size: the width of the square cell
        :param float wx: the width of the domain in the x-direction
        :param float wy: the width of the domain in the y-direction. If not specified,
            it is the same as wx.

        The grid is aligned with the coordinate axes. The axes coincide with the cell
        boundaries. An atom i with position (rxi,ryi) belongs to cell (k,l) iff:

            k*cell_size <= rxi < (k+1)*cell_size

        and
            l*cell_size <= rxi < (l+1)*cell_size

        Note, that we have a <= sign on the left inequalitty and a < on the right inequality.
        This guarantees that an atom can only belong to a single cell.
        Consequently, the cell of atom i is found as:

            k(i) = floor(rxi/cell_size)
            l(i) = floor(rxi/cell_size)

        Data structure:

        Similar to the Verlet list approach we use a numpy array for the cell lists.
        There is one Verlet list for each atom i, hence we used a 2D numpy array with
        one row per atom, containing first the number of neighours nn and then the nn
        indices af the atoms in the verlet list of atom i. Here, we use a similar
        approach. Since an atom is identified by a single index and a cell by two
        indices, we now need at 3D numpy array, with a row of atom indices per cell
        (k,l):

            cl(k,l,0) contains the number of atoms in cell (k,l)

            cl(k,l,1..n) are the indices of the atoms in cell (k,l)

        To facilitate passing the cell lists to Fortran routines, we will need to
        linearise the cl data structure. The linearised data structure should allow
        easy lookup of the cell list for a cell given its cell indices (k,l).
        This was different for the Verlet list. Our linearised data structure for
        the verlet list does not allow to easily obtain the Verlet list of atom i,
        because we do not know where it starts. We only know how long it is. The only
        way to know where it starts, is to sum the lengths of all Verlet lists of
        atoms before i. This presented no problem for computating the interactions
        because we always need all interactions and thus we start at i=0, loop over
        Verlet list of atom 0, next, we move to atom 1, whose Verlet list come right
        after that of atom 0, and so on. The starting point of each Verlet list is
        easily computed on the fly.

        For the cell lists the situation is different. E.g., to build the Verlet list
        of atom i, which is in cell (k,l), we need to access the cell lists (k+1,l),
        (k-1,l+1), (k,l+1) and (k+1,l+1). The problem is easily solved by adding an
        extra array with the starting position of each verlet list. So we have:

            cl_list : 1D numpy array containing all cell list, one after the other,
                      i.e. cl(0,0), cl(1,0), ..., cl(m,0), cl(0,1), cl(1,1), ...,
                      cl(m,1), ..., cl(0,n), cl(1,n), ..., cl(m,n), with m the number
                      of cells in the x-direction, and m the number of cells in the
                      y-direction.
            cl_size : 1D numpy array containing the number of atoms in the cells, i.e.
                      the length of each cell list.
                      This is a linearisation of a 2D matrix, rows being stored one
                      after the other. Thus cell (k,l) is stored at position (k+l*m).
            cl_offset : 1D numpy array containing the starting position of all the
                      cell lists in the cl_list array.
                      This too is a linearisation of a 2D matrix, rows being stored
                      one after the other. Thus cell (k,l) is stored at position
                      (k+l*m).

        So, the cell list of cell (k,l) starts at cl_list[ cl_offset[k+l*m] ] and ends
        cl_list[ cl_offset[k+l*m] + cl_size[k+l*m]-1 ].
        """
        self.cell_size = cell_size
        self.wx = wx
        if wy:
            self.wy = wy
        else: # wy not specified (None)
            self.wy = wx

        self.m = int(np.ceil(self.wx / cell_size)) # ceil() returns a float
        self.n = int(np.ceil(self.wy / cell_size))
        self.max_atoms_per_cell = max_atoms_per_cell

        self.clear()


    def clear(self, full=False):
        """allocate empty 3D data structure.

        :param bool zero: explicitly assign -1 to all cell list entries. Otherwise, only the
            cell counts are zeroed.
        """
        self.cl = np.empty((self.m, self.n, 1+self.max_atoms_per_cell), dtype=int)
        self.cl[:,:,0] = 0
        if full:
            self.cl[:,:,1:] = -1

    def __str__(self):
        s = ''
        for l in range(self.n):
            for k in range(self.m):
                s += f'({k},{l}) {self.cell_list(k,l)}\n'
        s += f'max elements: {self.max_elements()}/{self.max_atoms_per_cell}, linearised={self.linearised()}'
        return s


    def add(self, k, l, i):
        """Add atom i to the cell list of cell (k,l)."""
        # increment number of atoms in cell list
        self.cl[k,l,0] += 1
        # store the atom in the list at position self.cl[k, l, 0]
        self.cl[ k, l, self.cl[k, l, 0] ] = i


    def build_simple(self, rx, ry, linearise=True):
        """Build cell lists in a straightforward approach.

        :param np.array rx: x-component of atom positions
        :param np.array ry: y-component of atom positions
        """
        # loop over atoms
        self.n_atoms =  rx.shape[0] # remember n_atoms for linearization
        for i in range(self.n_atoms):
            k = int(np.floor(rx[i] / self.cell_size)) # floor() returns a float
            l = int(np.floor(ry[i] / self.cell_size))
            self.add(k,l,i)

        if linearise:
            self.linearise()


    def build(self, rx, ry, linearise=True):
        """Build cell lists using array operations.

        :param np.array rx: x-component of atom positions
        :param np.array ry: y-component of atom positions
        """
        inv_cell_size = 1/self.cell_size
        k = np.floor(rx * inv_cell_size).astype(int) # floor() returns a float
        l = np.floor(ry * inv_cell_size).astype(int)
        self.n_atoms =  rx.shape[0] # remember n_atoms for linearization
        i = np.arange(self.n_atoms)
        for i in range(self.n_atoms):
            self.add(k[i],l[i],i)

        if linearise:
            self.linearise()


    def max_elements(self):
        """Return the length of the longest cell list"""
        if self.linearised():
            pass
        else:
            return np.max(self.cl[:,:,0])


    def linearised(self):
        """Has this object's data structure been linearised?"""
        return self.cl is None
    
    
    def linearise(self):
        """linearise self.cl

            cl_list : 1D numpy array containing all cell list, one after the other,
                      i.e. cl(0,0), cl(1,0), ..., cl(m,0), cl(0,1), cl(1,1), ...,
                      cl(m,1), ..., cl(0,n), cl(1,n), ..., cl(m,n), with m the number
                      of cells in the x-direction, and m the number of cells in the
                      y-direction.
            cl_size : 1D numpy array containing the number of atoms in the cells, i.e.
                      the length of each cell list.
                      This is a linearisation of a 2D matrix, rows being stored one
                      after the other. Thus cell (k,l) is stored at position (k+l*m).
            cl_offset : 1D numpy array containing the starting position of all the
                      cell lists in the cl_list array.
                      This too is a linearisation of a 2D matrix, rows being stored
                      one after the other. Thus cell (k,l) is stored at position
                      (k+l*m).
        """
        # since every atom belongs to one cell, the length of cl_list is n_atoms

        self.cl_list   = np.empty((self.n_atoms, ), dtype=int)
        self.cl_size   = np.empty((self.m * self.n), dtype=int)
        self.cl_offset = np.empty((self.m * self.n), dtype=int)

        offset = 0
        for l in range(self.n):
            for k in range(self.m):
                n_atoms_in_cell = self.cl[k,l,0]
                # copy the entire cell list at once
                # the cell list is self.cl[k,l,1:1+n_atoms_in_cell]
                self.cl_list[offset : offset+n_atoms_in_cell] = self.cl[k, l, 1 : 1+n_atoms_in_cell]
                # store the current offset for the current cell list
                self.cl_offset[k + l*self.m] = offset
                # store the length of the current cell list
                self.cl_size[k + l*self.m] = n_atoms_in_cell
                # move the offset
                offset += n_atoms_in_cell

        # after linearizing we can delete self.cl
        self.cl = None # Garbage collected


    def cell_list(self, k, l):
        """Obtain the cell list of cell (k,l)."""
        if self.linearised():
            offset  = self.cl_offset[k+l*self.m]
            n_atoms_in_list = self.cl_size[k+l*self.m]
            ckl = self.cl_list[offset:offset+n_atoms_in_list]
        else:
            n_atoms_in_list = self.cl[k,l,0]
            ckl = self.cl[k,l,1:1+n_atoms_in_list]

        # print(f'{(k,l)}: {ckl}')
        return ckl