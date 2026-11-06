try:
    # raise ImportError
    import cupy as np
    from cupy.lib.stride_tricks import sliding_window_view
except ImportError:
    print("Failed to use GPU acceleration with cupy, falling back to numpy CPU")
    import numpy as np
    from numpy.lib.stride_tricks import sliding_window_view