from numba import njit, prange, config
import numpy as np
import time

# Parallelized function using Numba
@njit(parallel=True)
def parallel_sum(arr):
    total = 0
    for i in prange(len(arr)):
        total += arr[i]
    return total

# Main execution
if __name__ == "__main__":
    arr = np.arange(2_000_000_000, dtype=np.float64)

    # Sequential version
    start = time.time()
    total = np.sum(arr)  # Using NumPy for sequential sum
    print(f"Sequential Sum: {total}, Time: {time.time() - start} seconds")

    # Parallelized version
    start = time.time()
    total = parallel_sum(arr)  # Using the parallelized Numba function
    print(f"Parallel Sum: {total}, Time: {time.time() - start} seconds")

print(f"Numba is using {config.NUMBA_NUM_THREADS} threads")
