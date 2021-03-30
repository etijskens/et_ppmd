import et_ppmd
import et_ppmd.verlet as verlet
import et_ppmd.grid as grid
from et_stopwatch import Stopwatch

import matplotlib.pyplot as plt

if __name__ == '__main__':
    cutoff = 5.0
    max_neighbours = 60

    # some lists to store our results
    n_at = [] # number of atoms
    t_bs = [] # timings for VerletList.build_simple
    t_b  = [] # timings for VerletList.build
    t_bg = [] # timings for VerletList.build_grid
    for n_squares in range(1,16):
        # a single square contains on average 30 atoms ( 6 rows of each 5 atoms)
        box = et_ppmd.Box(0., 0., n_squares*5.*et_ppmd.hcp.uc_centered_a, n_squares*3.*et_ppmd.hcp.uc_centered_b)
        x,y = box.generateAtoms(r=et_ppmd.hcp.radius)
        n_atoms = len(x)
        # store n_atoms in the result array
        n_at.append(n_atoms)
        vl = verlet.VerletList(cutoff=cutoff, max_neighbours=max_neighbours)
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

        the_grid = grid.Grid(wx=box.xur-box.xll, wy=box.yur-box.yll,cell_size=cutoff,max_atoms_per_cell=30)
        # time building the Verlet list
        with Stopwatch(message=f"VerletList.build_grid for {n_atoms} atoms") as sw_build:
            the_grid.build(x,y)
            vl.build_grid(x, y, the_grid)

        # store the timing in the result array
        t_bg.append(sw_build.time)

    print(n_at)
    print(t_bs)
    print(t_b)
    print(t_bb)

    # make a plot
    fig, ax = plt.subplots()
    ax.plot(n_at, t_bs,'-*', label='VerletList.build_simple')
    ax.plot(n_at, t_b ,'-*', label='VerletList.build')
    ax.plot(n_at, t_bg,'-*', label='VerletList.build_grid')
    legend = ax.legend(loc='upper left')
    ax.set_xlabel('#atoms')
    ax.set_ylabel('seconds')

    plt.show()

    print('-*# done #*-')


