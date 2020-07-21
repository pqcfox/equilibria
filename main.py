from matplotlib import pyplot as plt
import numpy as np
import tqdm
from log_sin_energy import LogSinEnergy
from inv_distance_energy import InvSquareEnergy, InvDistanceEnergy
from queen_solver import get_queen_solver_points, random_permutation_points


def calc_discrepancy(points):
    n = len(points)
    added_points = [(0, 0)] + points + [(1, 1)]
    sorted_points = sorted(added_points, key=lambda p: p[0])
    zetas = []
    for i in range(n + 1):
        partial_points = sorted_points[:(i + 1)] + [sorted_points[-1]]
        zeta_i = sorted([p[1] for p in partial_points])
        zetas.append(zeta_i)

    values = []
    for i in range(n + 1):
        for k in range(i + 1):
            left = k/n - sorted_points[i][0] * zetas[i][k]
            right = sorted_points[i + 1][0] * zetas[i][k + 1] - k/n
            values.append(left)
            values.append(right)
    d = max(values)
    return d


def generate_points(n, dimension=2):
    return np.random.random((n, dimension))


def generate_points_gradmethod(start_points=None, energy_method=LogSinEnergy,
                               init_lr=0.000005, num_iter=300,
                               decay_period=50, decay_rate=0.75,
                               energy_sample_rate=50):
    if start_points is None:
        points = generate_points(100)  # Random initialization
    else:
        points = start_points
    d = calc_discrepancy(list(points))
    print(f'Original discrepancy: {d}')

    grad = np.mean(np.abs(energy_method.gradient(points)))
    lr = init_lr
    print("Approx magnitude of initial change", grad * lr)  # To sanity check the learning rate, should be < 1
    print("Initial Energy", energy_method.energy(points))

    energy_samples = []

    for i in tqdm.tqdm(range(num_iter)):
        grad = energy_method.gradient(points)
        points += -lr * grad
        points = np.mod(points, 1)
        if (i + 1) % decay_period == 0:  # Decay learning rate
            lr *= decay_rate
        if i % energy_sample_rate == 0 or i + 1 == num_iter:
            energy_samples.append((i, energy_method.energy(points)))

    print("Final energy", energy_method.energy(points))
    return points, energy_samples


def gen_von_der_corupt(n=100):
    byte_reverse = np.zeros(n)
    for i in range(1, n+1):
        x = bin(i)[2:]
        byte_reverse[i - 1] = int(''.join(reversed(x)), 2) / (2 ** len(x))
    points = np.array([np.arange(n) / n, byte_reverse]).T
    return points


def display_points(points):
    d = calc_discrepancy(list(points))
    print(f'Discrepancy: {d}')
    x, y = points.T
    plt.scatter(x, y)


def display_energy_plot(energies):
    x, y = [list(x) for x in zip(*energies)]
    plt.plot(x, y)


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