import itertools
import numpy as np
from log_sin_energy import LogSinEnergy
from utils import generate_points, generate_points_gradmethod

def gen_bad_minima(num_points, population_size=50, top_n=10, generations=5, sigma=0.05):
    population = [generate_points(num_points) for _ in range(population_size)]
    best_points = []
    best_energies = []
    for _ in range(generations):
        fitness = []
        for points in population:
            _, energies = generate_points_gradmethod(start_points=points)
            min_energy = energies[-1][-1]
            fitness.append(min_energy)

        sorted_pairs = sorted(zip(population, fitness), key=lambda pair: pair[1])
        best_pair = sorted_pairs[-1]
        best_points.append(best_pair[0])
        best_energies.append(best_pair[1])

        best = [pair[0] for pair in sorted_pairs[-top_n:]]
        pool = itertools.cycle(population)
        new_population = []
        for _ in range(population_size):
            noise = np.random.normal(scale=sigma, size=points.shape)
            new_population.append(points + noise)
        population = new_population



if __name__ == '__main__':
    gen_bad_minima(10)
