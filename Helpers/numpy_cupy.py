try:
    # raise ImportError
    import cupy as np
    from cupy.lib.stride_tricks import sliding_window_view
    # Fuses a chain of elementwise operations into a single GPU kernel (one pass over memory)
    fuse = np.fuse
except ImportError:
    print("Failed to use GPU acceleration with cupy, falling back to numpy CPU")
    import numpy as np
    from numpy.lib.stride_tricks import sliding_window_view

    def fuse(*args, **kwargs):
        """No-op stand-in for cupy.fuse on the CPU. Works as @fuse and @fuse()."""
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return lambda f: f
