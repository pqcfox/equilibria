import numpy as np


def solve_queens_problem(allowed_places, start_row=0):
    N = allowed_places.shape[0]
    if start_row >= N:
        return []

    indices = np.where(allowed_places[start_row])[0]
    np.random.shuffle(indices)
    for i in indices:
        allowed_places_copy = np.copy(allowed_places)
        # Try placing queen at (start_row, i)
        # Diagonal 1: x-y == start_row-i
        const_diff = start_row - i
        diag1 = np.arange(N - abs(const_diff)) + max(0, const_diff), \
                np.arange(N - abs(const_diff)) - min(0, const_diff)
        allowed_places_copy[diag1] = False
        # Diagonal 2: x+y == start_row+i
        const_sum = start_row + i
        diag2 = np.arange(N - abs(const_sum - (N - 1))) + max(0, const_sum - (N-1)), \
                -np.arange(N - abs(const_sum - (N - 1))) + min(const_sum, N-1)
        allowed_places_copy[diag2] = False
        # Column
        allowed_places_copy[:, i] = False
        # Row is unecessary technically since we won't ever try again
        poss_result = solve_queens_problem(allowed_places_copy, start_row + 1)

        if poss_result is not None:
            poss_result.append(i)
            return poss_result
    return None


def get_queen_solver_points(N):
    board = np.ones((N, N), dtype=bool)
    result = np.array(solve_queens_problem(board))
    return np.array([np.arange(N) / N, result / N]).T


def random_permutation_points(N):
    perm = np.arange(N) / N
    np.random.shuffle(perm)
    return np.array([np.arange(N) / N, perm]).T


if __name__ == "__main__":
    board = np.ones((20, 20), dtype=bool)
    print(solve_queens_problem(board))
