# The classic XY-model implemented in GPU's
by Ulf R. Pedersen

## Workaround for Numba CUDA error

The error

    numba.cuda.cudadrv.driver.LinkerError: libcudadevrt.a not found

can be fixed with something like

    ln -s /usr/lib/x86_64-linux-gnu/libcudadevrt.a .
