"""
From https://arxiv.org/abs/1902.00441
Energy to minimize: E = sum_{m, n}  (1 - log(2 sin(π|x_m - x_n|))) * (1 - log(2 sin(π|y_m - y_n|)))
where x_i is the x coordinate of the ith point and y_i is the y coordinate of the yth point.
"""
import numpy as np
from energy_min import Energy


class LogSinEnergy(Energy):
    @staticmethod
    def gradient(points):
        """
        TODO: decide if we want to use a machine learning framework for this
        because it might be easier/they will have a lot more tools
        for optimization, e.g. auto calculated gradients, rather than us having to do
        it manually

        :param points    (N, 2)
        :return gradient (N, 2)
        """
        x, y = points.T
        dummy = 0.5

        x_diffs, y_diffs = Energy.calc_diffs(x), Energy.calc_diffs(y)
        x_diffs_abs, y_diffs_abs = np.abs(x_diffs), np.abs(y_diffs)

        # Set dummy values for the diagonal so that we don't get division by 0 error
        Energy.set_diag(x_diffs_abs, dummy)
        Energy.set_diag(y_diffs_abs, dummy)

        cot_x_diffs = -1 / np.tan(np.pi * x_diffs_abs) * np.pi * ((x_diffs > 0) * 2 - 1)
        cot_y_diffs = -1 / np.tan(np.pi * y_diffs_abs) * np.pi * ((y_diffs > 0) * 2 - 1)
        x_inds = (1 - np.log(2 * np.sin(np.pi * y_diffs_abs))) * cot_x_diffs
        y_inds = (1 - np.log(2 * np.sin(np.pi * x_diffs_abs))) * cot_y_diffs

        # Set diagonal entries to 0
        Energy.set_diag(x_inds, 0)
        Energy.set_diag(y_inds, 0)

        x_grad = np.sum(x_inds, axis=1)
        y_grad = np.sum(y_inds, axis=1)
        return np.array([x_grad, y_grad]).T

    @staticmethod
    def energy(points):
        x, y = points.T
        N = points.shape[0]
        diag_indices = (np.arange(N), np.arange(N))
        dummy = 0.5

        x_diffs, y_diffs = Energy.calc_diffs(x), Energy.calc_diffs(y)
        x_diffs_abs, y_diffs_abs = np.abs(x_diffs), np.abs(y_diffs)

        # Set dummy values for the diagonal so that we don't get division by 0 error
        x_diffs_abs[diag_indices] = dummy
        y_diffs_abs[diag_indices] = dummy

        es = (1 - np.log(2 * np.sin(np.pi * x_diffs_abs))) * \
             (1 - np.log(2 * np.sin(np.pi * y_diffs_abs)))
        es[diag_indices] = 0
        return np.sum(es)
