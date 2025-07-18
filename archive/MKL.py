import numpy as np
print("NumPy configuration:")
np.show_config()

# Check for MKL-specific functions
try:
    import mkl
    print(f"MKL version: {mkl.get_version_string()}")
    print(f"MKL threads: {mkl.get_max_threads()}")
except ImportError:
    print("MKL service not available")

