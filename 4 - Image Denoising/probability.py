import numpy as np


def get_neighbours(position, grid):
    neighbours = []

    row_pos = position[0]
    col_pos = position[1]

    if row_pos > 0:
        neighbours.append(grid[row_pos-1][col_pos])
    if row_pos < grid.shape[0]-1:
        neighbours.append(grid[row_pos+1][col_pos])
    if col_pos > 0:
        neighbours.append(grid[row_pos][col_pos-1])
    if col_pos < grid.shape[1]-1:
        neighbours.append(grid[row_pos][col_pos+1])

    return neighbours


def get_sigmoid(value):
    return 1/(1 + np.exp(-1*value))


def get_probability(position, grid, params):
    bias_param = params[0]
    weight_param = params[1]

    value_for_sigmoid = 0
    for neighbour in get_neighbours(position, grid):
        value_for_sigmoid += weight_param * neighbour
    value_for_sigmoid += bias_param

    sigmoid_value = get_sigmoid(2 * value_for_sigmoid)

    return np.array([1-sigmoid_value, sigmoid_value])


def main():
    grid = np.ones(shape=(30, 30)) * -1
    bias_params = np.zeros(shape=(30, 30))
    print get_probability((1, 1), grid, (bias_params[1][1], 0.3))


if __name__ == '__main__':
    main()
