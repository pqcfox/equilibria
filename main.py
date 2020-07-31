from matplotlib import pyplot as plt
import numpy as np
from log_sin_energy import LogSinEnergy
from inv_distance_energy import InvSquareEnergy, InvDistanceEnergy
from queen_solver import get_queen_solver_points, random_permutation_points
from utils import (calc_discrepancy, generate_points,
                   generate_points_gradmethod, gen_von_der_corupt,
                   display_points, display_energy_plot)


def main():
    start_points = gen_von_der_corupt(1000)
    points, energy = generate_points_gradmethod(start_points=start_points, init_lr=1e-8, decay_period=25, energy_sample_rate=25)
    # points = random_permutation_points(1000)
    # points = generate_points(1000)
    plt.figure(1)
    display_points(points)
    plt.figure(2)
    display_energy_plot(energy)
    plt.show()


if __name__ == "__main__":
    main()
