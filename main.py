import random


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


def generate_points(n):
    points = []
    for _ in range(n):
        p = (random.random(), random.random())
        points.append(p)
    return points


if __name__ == "__main__":
    points = generate_points(100)
    d = calc_discrepancy(points)
    print(f'Discrepancy of 100 random points: {d}')
