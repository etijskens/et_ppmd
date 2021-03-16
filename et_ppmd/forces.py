# -*- coding: utf-8 -*-

"""
Module et_ppmd.forces 
=================================================================

Lennard-Jones potential and forces
"""

# some constants
# equilibrium distance of (coefficientless) Lennard-Jones potential : V(r) = 1/r**12 - 1*r**6
R0 = pow(2.,1/6) # ~ 1.12246

def potential(r2):
    """Compute the Lennard-Jones potential

    :param float|np.array r2: squared distance between atoms.
    :returns: a float.
    """
    rm6 = 1./(r2*r2*r2)
    vlj = (rm6 - 1.0)*rm6
    return vlj


def force_factor(rij2):
    """Lennard-Jones force magnitudefd exerted by atom j on atom i.

    :param float|np.array rij2: squared interatomic distance from atom i to atom j
    :return: fij
    """
    rm2 = 1.0/rij2
    rm6 = (rm2*rm2*rm2)
    f = (1.0 - 2.0*rm6 )*rm6*rm2*6.0
    return f


def force(xij,yij):
    """Lennard-Jones force exerted by atom j on atom i.

    :param float|np.array xij: x-coordinates of vector from atom i to atom j
    :param float|np.array yij: y-coordinates of vector from atom i to atom j
    :return: Fxij, Fyij
    """
    rm2 = 1./(xij**2 + yij**2)
    rm6 = (rm2*rm2*rm2)
    f = (1.0 - 2.0*rm6 )*rm6*rm2*6.0
    return f*xij,f*yij
