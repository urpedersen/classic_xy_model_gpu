""" Implementation of the classic XY model on GPU's. By Ulf R. Pedersen. """

import math

import numba
import numpy as np
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states
from numba.cuda.random import xoroshiro128p_normal_float32


@cuda.jit(device=True)
def compute_forces(lattice, forces, tiles, xx, yy):
    """ Device function for computing forces """
    rows, columns = lattice.shape

    for tx in range(tiles[0]):
        x1 = xx + tx
        for ty in range(tiles[1]):
            y1 = yy + ty
            this_theta = lattice[x1, y1]
            this_force = 0.0
            for dx, dy in (-1, 0), (1, 0), (0, -1), (0, 1):
                x2 = (x1 + dx) % rows
                y2 = (y1 + dy) % columns
                that_theta = lattice[x2, y2]
                delta_theta = this_theta - that_theta
                this_force += -math.sin(delta_theta)
            forces[x1, y1] = this_force


@cuda.jit(device=True)
def update_nvt(lattice, lattice_vel, forces, betas, tiles, rng_states, xx, yy):
    """ Device function for NVT Langevin, Leap-frog, REF: https://arxiv.org/pdf/1303.7011.pdf Sec. 2.C. """
    # Parameters
    temperature = numba.float32(0.4)
    dt = numba.float32(0.01)
    alpha = numba.float32(0.4)
    mass = numba.float32(1.0)

    # Helper variables
    rows, columns = lattice.shape
    two = numba.float32(2.0)
    one_half = numba.float32(0.5)

    for tx in range(tiles[0]):
        x = xx + tx
        for ty in range(tiles[1]):
            y = yy + ty
            idx = x + y * rows

            # Eq. (16) in https://arxiv.org/pdf/1303.7011.pdf#
            random_number = xoroshiro128p_normal_float32(rng_states, idx)
            beta_new = math.sqrt(two * alpha * temperature * dt) * random_number
            numerator = two * mass - alpha * dt
            denominator = two * mass + alpha * dt
            a = numerator / denominator
            b_over_m = two / denominator
            lattice_vel[x, y] = a * lattice_vel[x, y] + b_over_m * forces[x, y] * dt + b_over_m * one_half * (
                    beta_new + betas[x, y])
            betas[x, y] = beta_new
            lattice[x, y] += lattice_vel[x, y] * dt


@cuda.jit
def run_simulation(lattice, lattice_vel, forces, betas, tiles, rng_states, steps):
    """ Kernel than run simulation for one spin step on the device """
    x, y = cuda.grid(2)
    xx, yy = x * tiles[0], y * tiles[1]
    grid = cuda.cg.this_grid()

    for step in range(steps):
        compute_forces(lattice, forces, tiles, xx, yy)
        grid.sync()
        update_nvt(lattice, lattice_vel, forces, betas, tiles, rng_states, xx, yy)
        grid.sync()


def show(grid):
    """ Graphical representation of the simulation """
    from math import pi
    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(grid % (2 * pi) - pi, cmap='hsv', vmin=-pi, vmax=pi)
    plt.colorbar()
    plt.show()


def energy(lattice):
    delta_x = lattice - np.roll(lattice, shift=1, axis=0)
    energy_x = -np.cos(delta_x)
    delta_y = lattice - np.roll(lattice, shift=1, axis=1)
    energy_y = -np.cos(delta_y)
    energy_per_spin = np.sum(energy_x + energy_y) / lattice.size
    return energy_per_spin


def main():
    # Print info for this GPU
    cuda.detect()
    device = cuda.get_current_device()
    print(f"Device name: {device.name}")
    cc = device.compute_capability
    print(f"Compute capability: {cc[0]}.{cc[1]}")
    print(f"Multiprocessor count: {device.MULTIPROCESSOR_COUNT}")
    print(f"Max threads per block: {device.MAX_THREADS_PER_BLOCK}")
    print(f"Max block dimensions: {device.MAX_BLOCK_DIM_X}, {device.MAX_BLOCK_DIM_Y}, {device.MAX_BLOCK_DIM_Z}")
    print(f"Max grid dimensions: {device.MAX_GRID_DIM_X}, {device.MAX_GRID_DIM_Y}, {device.MAX_GRID_DIM_Z}")
    print("")

    # For timing of device code
    start = cuda.event()
    end = cuda.event()
    start_block = cuda.event()
    end_block = cuda.event()
    start.record()

    # Setup system size
    threads_per_block = (8, 8)
    blocks = (16, 16)  # (24, 24)
    tiles = (8, 8)  # (24, 24)
    rows = tiles[0] * blocks[0] * threads_per_block[0]
    cols = tiles[1] * blocks[1] * threads_per_block[1]
    N = rows * cols
    print(f"Lattice size: {rows} x {cols} = {N}")
    print(f"with {tiles = }, {blocks = }, and {threads_per_block = } ")

    # Set variables on host and copy to device memory
    lattice = (np.random.random((rows, cols)) - 0.5) * 2 * np.pi
    lattice = np.array(lattice, dtype=np.float32)
    d_lattice = cuda.to_device(lattice)
    lattice_vel = np.zeros(lattice.shape, dtype=np.float32)
    d_lattice_vel = cuda.to_device(lattice_vel)
    forces = np.zeros(lattice.shape, dtype=np.float32)
    d_forces = cuda.to_device(forces)
    rng_states = create_xoroshiro128p_states(N, seed=2025)
    betas = np.zeros(lattice.shape, dtype=np.float32)
    d_betas = cuda.to_device(betas)

    # Run simulation on device
    wallclock_times = []
    num_time_blocks = 16
    steps_per_time_block = 128
    print(f'{num_time_blocks=}, {steps_per_time_block=}')
    print(f'Total number of time steps: {num_time_blocks * steps_per_time_block}')
    for i in range(num_time_blocks):
        params = d_lattice, d_lattice_vel, d_forces, d_betas, tiles, rng_states, steps_per_time_block
        start_block.record()
        run_simulation[blocks, threads_per_block](*params)
        end_block.record()
        end_block.synchronize()
        wallclock_times.append(cuda.event_elapsed_time(start_block, end_block))
        lattice = d_lattice.copy_to_host()
        enr = energy(lattice)
        theta0 = float(lattice[0, 0])
        wct = wallclock_times[-1]
        print(f'{i:>4}: energy = {enr:.3f}, theta[0,0] = {theta0:.3f}. Wall-clock time: {wct:0.1f} ms')
    end.record()
    end.synchronize()
    total_wallclock_time = cuda.event_elapsed_time(start, end)
    print(f'{total_wallclock_time = :0.1f} ms')

    print(f"First, wallclock time (compile): {wallclock_times[0] = :0.2f} ms")
    print(f"Other avg. wallclock time: {np.mean(wallclock_times[1:]):0.1f} ms +- {np.std(wallclock_times[1:]):0.1f} ms")
    delta_t = np.mean(wallclock_times[1:]) / 1000  # Seconds
    steps_per_second = steps_per_time_block / delta_t
    print(f"{steps_per_second = :0.2e}")
    spin_updates_per_second = steps_per_second * rows * cols
    print(f"{spin_updates_per_second = :0.2e}")

    show(lattice)


if __name__ == '__main__':
    main()
