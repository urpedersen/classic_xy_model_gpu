

## Workaround

Error: `numba.cuda.cudadrv.driver.LinkerError: libcudadevrt.a not found`.
Fix: `ln -s /usr/lib/x86_64-linux-gnu/libcudadevrt.a .`