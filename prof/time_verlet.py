import et_ppmd
import et_ppmd.verlet as verlet
from et_stopwatch import Stopwatch

import matplotlib.pyplot as plt

if __name__ == '__main__':
    cutoff = 5.0
    # this cutoff generates ~40 contacts per atom in the interior of the box,
    # hence we set
    max_contacts = 50

    # some lists to store our results
    n_at = [] # number of atoms
    t_bs = [] # timings for VerletList.build_simple
    t_b  = [] # timings for VerletList.build
    for n_squares in range(1,16):
        # a single square contains on average 30 atoms ( 6 rows of each 5 atoms)
        box = et_ppmd.Box(0., 0., n_squares*5.*et_ppmd.hcp.uc_centered_a, n_squares*3.*et_ppmd.hcp.uc_centered_b)
        x,y = box.generateAtoms(r=et_ppmd.hcp.radius)
        n_atoms = len(x)
        # store n_atoms in the result array
        n_at.append(n_atoms)
        vl = verlet.VerletList(cutoff=cutoff, max_contacts=max_contacts)
        # time building the Verlet list
        with Stopwatch(message=f"VerletList.build_simple for {n_atoms} atoms") as sw_build_simple:
            vl.build_simple(x,y)

        # store the timing in the result array
        t_bs.append(sw_build_simple.time)

        # time building the Verlet list
        with Stopwatch(message=f"VerletList.build for {n_atoms} atoms") as sw_build:
            vl.build(x,y)

        # store the timing in the result array
        t_b.append(sw_build.time)

    print(n_at)
    print(t_bs)
    print(t_b)

    # make a plot
    fig, ax = plt.subplots()
    ax.plot(n_at, t_bs,'-*', label='VerletList.build_simple')
    ax.plot(n_at, t_b ,'-*', label='VerletList.build')
    legend = ax.legend(loc='upper left')
    ax.set_xlabel('#atoms')
    ax.set_ylabel('seconds')

    plt.show()

    print(f'actual max contacts: {vl.actual_max_contacts()}')
    print('-*# done #*-')


