import dask
from dask.distributed import Client, LocalCluster

import ctypes


if __name__ == '__main__':

    def trim_memory() -> int:
        libc = ctypes.CDLL("libc.so.6")
        return libc.malloc_trim(0)
    # trim_memory()
    c = Client('inproc://127.0.0.1:8786', timeout=5)
    c.run(trim_memory)
