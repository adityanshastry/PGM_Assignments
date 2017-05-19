import probability
import numpy as np
import pickle
import image_utils
import pylab as plt


def gibbs_sampler(t_max, grid, params):
    grids = []
    bias_param = params[0]
    # print bias_param.shape
    weight_param = params[1]

    for time_step in xrange(t_max):
        for row in xrange(grid.shape[0]):
            for col in xrange(grid.shape[1]):
                grid[row][col] = np.random.choice(a=[-1, 1], p=probability.get_probability((row, col), grid, (
                    bias_param[row][col], weight_param)))
        grids.append(grid.copy())

    return grids


def question_3():
    grid_structure = (30, 30)
    grid = np.ones(shape=grid_structure) * -1
    bias_param = np.zeros(shape=grid_structure)
    weight_params = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    # weight_params = [0.5]

    for weight_param in weight_params:
        print weight_param
        sampled_grid = gibbs_sampler(100, grid.copy(), (bias_param.copy(), weight_param))[-1]
        print np.sum(sampled_grid)
        image_location = '../data/sampled_images/image_w_' + str(weight_param) + '.png'
        image_utils.save_grid_image_to_location(image_utils.get_image_from_sample(sampled_grid),
                                                image_location)


def question_5():
    grid_structure = (30, 30)
    markov_chains = 100
    t_max = 100
    grid = np.ones(shape=grid_structure) * -1
    bias_param = np.zeros(shape=grid_structure)
    weight_params = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    sampled_grids = {}

    for weight_param in weight_params:
        print weight_param
        sampled_grids[weight_param] = np.zeros(shape=(t_max, grid_structure[0], grid_structure[1]))
        for markov_chain in xrange(markov_chains):
            sampled_grids[weight_param] += gibbs_sampler(t_max, grid.copy(), (bias_param.copy(), weight_param))
        sampled_grids[weight_param] /= markov_chains
        sampled_grids[weight_param] = np.reshape(sampled_grids[weight_param], (t_max, grid_structure[0] * grid_structure[1]))
        sampled_grids[weight_param] = np.average(sampled_grids[weight_param], axis=1)
        # print sampled_grids[weight_param]

    pickle.dump(sampled_grids, open("../data/expected_y_values.p", "w"))


def question_5_plot():
    expectd_values = pickle.load(open("../data/expected_y_values.p", "r"))

    plt.xlabel('Number of iterations')
    plt.ylabel('Average Sample')
    plt.title('Expected sample values')

    for weight_param in expectd_values:
        plt.plot(range(1, len(expectd_values[weight_param])+1), expectd_values[weight_param], label=str(weight_param))

    plt.legend()
    plt.show()


def main():
    # question_3()
    # question_5()
    question_5_plot()


if __name__ == '__main__':
    main()
