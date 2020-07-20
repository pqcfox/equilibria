import numpy as np
from energy_min import Energy


def mod1(arr):
    arr = np.mod(arr, 1)
    arr[arr > 0.5] -= 1
    return arr


class InvSquareEnergy(Energy):
    @staticmethod
    def gradient(points):
        x, y = points.T
        x_diffs, y_diffs = mod1(Energy.calc_diffs(x)), mod1(Energy.calc_diffs(y))
        r_diffs = (x_diffs**2 + y_diffs**2)**1.5
        Energy.set_diag(r_diffs, np.inf)
        force_mags = 1. / r_diffs
        forces_x = np.sum(x_diffs * force_mags, axis=1)
        forces_y = np.sum(y_diffs * force_mags, axis=1)

        grad = -np.array([forces_x, forces_y]).T
        return grad

    @staticmethod
    def energy(points):
        x, y = points.T
        x_diffs, y_diffs = mod1(Energy.calc_diffs(x)), mod1(Energy.calc_diffs(y))
        r_diffs = np.sqrt(x_diffs**2 + y_diffs**2)
        Energy.set_diag(r_diffs, np.inf)
        energies = np.sum(1. / r_diffs)
        return energies


class InvDistanceEnergy(Energy):
    @staticmethod
    def gradient(points):
        x, y = points.T
        x_diffs, y_diffs = mod1(Energy.calc_diffs(x)), mod1(Energy.calc_diffs(y))
        r_diffs = x_diffs**2 + y_diffs**2
        Energy.set_diag(r_diffs, np.inf)
        force_mags = 1. / r_diffs
        forces_x = np.sum(x_diffs * force_mags, axis=1)
        forces_y = np.sum(y_diffs * force_mags, axis=1)

        grad = -np.array([forces_x, forces_y]).T
        return grad

    @staticmethod
    def energy(points):
        x, y = points.T
        x_diffs, y_diffs = mod1(Energy.calc_diffs(x)), mod1(Energy.calc_diffs(y))
        r_diffs = x_diffs**2 + y_diffs**2
        Energy.set_diag(r_diffs, 1)
        energies = np.sum(-0.5 * np.log(r_diffs))
        return energies
