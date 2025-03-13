from time import perf_counter

import numpy as np
import numba
from numba import cuda
from math import sin

@cuda.jit(device=True)
def compute_forces(lattice, lattice_vel, forces, x, y):
    rows, columns = lattice.shape

    this_theta = lattice[x, y]
    this_force = 0.0
    for dx, dy in (-1,0), (1,0), (0,-1), (0,1):
        x2 = (x + dx)%columns
        y2 = (y + dy)%rows
        that_theta = lattice[x2, y2]
        delta_theta = this_theta - that_theta
        this_force += -sin(delta_theta)
    forces[x, y] = this_force

@cuda.jit(device=True)
def update(lattice, lattice_vel, forces, x, y):
    dt = 0.001
    lattice_vel[x, y] += forces[x, y]*dt
    lattice[x, y] += lattice_vel[x, y]*dt

@cuda.jit
def run_simulation(lattice, lattice_vel, forces, steps):
    i, j = cuda.grid(2)
    grid = cuda.cg.this_grid()

    for step in range(steps):
        compute_forces(lattice, lattice_vel, forces, i, j)
        grid.sync()
        update(lattice, lattice_vel, forces, i, j)
        grid.sync()

@numba.njit
def get_energy(lattice):
    ...

def main():
    # Setup system size
    threads_per_block = (8, 8)
    blocks = (16, 16)
    rows = blocks[0]*threads_per_block[0]
    cols = blocks[1]*threads_per_block[1]
    print(f"{rows} x {cols}")

    # Set variables on host and device
    lattice = np.random.random(size = (rows, cols))-0.5
    lattice = np.array(lattice, dtype=np.float32)
    d_lattice = cuda.to_device(lattice)
    lattice_vel = np.zeros(lattice.shape, dtype=np.float32)
    d_lattice_vel = cuda.to_device(lattice_vel)
    forces = np.zeros(lattice.shape, dtype = np.float32)
    d_forces = cuda.to_device(forces)

    # Run simulation
    times_run = []
    times_copy = []
    time_blocks = 16
    for i in range(time_blocks):
        tic = perf_counter()
        run_simulation[blocks, threads_per_block](d_lattice, d_lattice_vel, d_forces, 2048)
        toc = perf_counter()
        times_run.append(toc - tic)
        tic = perf_counter()
        lattice = d_lattice.copy_to_host()
        toc = perf_counter()
        times_copy.append(toc - tic)

        print(f'{i}, theta_0={float(lattice[0,0])}, {times_run[-1]=:0.2e}, {times_copy[-1]=:0.2e}')

    print(f"Compile, wallclock time: {times_run[0]=:0.2f}")
    print(f"Run, wallclock time: {np.mean(times_run[1:])=:0.2e} +- {np.std(times_run[1:])=:0.2e}")
    print(f"Copy, wallclock time: {np.mean(times_copy[1:])=:0.2e} +- {np.std(times_copy[1:])=:0.2e}")
    print(f"{time_blocks*np.mean(times_run[1:])=:0.5f} s")

if __name__ == '__main__':
    main()
