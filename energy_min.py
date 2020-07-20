import numpy as np

class Energy:
    @staticmethod
    def gradient(points):
        ...

    @staticmethod
    def energy(points):
        ...

    # Useful functions
    @staticmethod
    def calc_diffs(axis):
        """
        :param axis     (N,)
        :returns diffs  (N, N) where diffs[i, j] = (axis[i] - axis[j]) mod 1
        """
        N = axis.shape[0]
        diffs = axis.reshape((N, 1)) - axis
        return diffs

    @staticmethod
    def set_diag(points, val):
        N = points.shape[0]
        diag_indices = (np.arange(N), np.arange(N))
        points[diag_indices] = val
