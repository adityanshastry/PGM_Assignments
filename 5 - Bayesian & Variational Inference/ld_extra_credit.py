import langevin_dynamics
import utils
import numpy as np
from collections import defaultdict
from sklearn.metrics import accuracy_score


def question_9_1():
    train_data = np.loadtxt("../data/normal/X_train.csv")
    train_labels = np.loadtxt("../data/normal/Y_train.csv")

    step_sizes = [0.1, 0.05, 0.01, 0.005, 0.001]
    t_max_values = [10, 100, 1000, 10000, 100000]

    for step_size in step_sizes:
        print step_size
        for t_max in t_max_values:
            print t_max
            sampled_weights, running_average = langevin_dynamics.get_sampled_weights(train_data, train_labels,
                                                                                     step_size, t_max)
            print sampled_weights.shape
            np.savetxt(
                "../data/extra_credit/results/langevin_dynamics/run_length/weights_" + str(step_size) + "_" + str(
                    t_max) + ".csv", sampled_weights)

            utils.plot_results(sampled_weights, "Iteration Step", "Sampled Z values",
                               "Sampled weight vs Iteration steps",
                               "../data/extra_credit/results/langevin_dynamics/run_length/weights_plot_" + str(
                                   step_size) + "_" + str(
                                   t_max) + ".png")
            utils.plot_results(running_average, "Iteration Step", "Running Average",
                               "Running average vs Iteration steps",
                               "../data/extra_credit/results/langevin_dynamics/run_length/running_average_" + str(
                                   step_size) + "_" + str(
                                   t_max) + ".png")

    pass


def question_9_2():
    test_data = np.loadtxt("../data/normal/X_test.csv")
    test_labels = np.loadtxt("../data/normal/Y_test.csv")

    step_sizes = [0.1, 0.05, 0.01, 0.005, 0.001]
    t_max_values = [10, 100, 1000, 10000, 100000]

    for step_size in step_sizes:
        for t_max in t_max_values:
            sampled_weights = np.loadtxt(
                "../data/extra_credit/results/langevin_dynamics/run_length/weights_" + str(step_size) + "_" + str(
                    t_max) + ".csv")
            predictions = utils.get_predictions(test_data, 1, sampled_weights)
            print 'step_size: ', step_size, ', t_max: ', t_max, ', error_rate', 1 - accuracy_score(test_labels,
                                                                                                   predictions)


def question_11_1():
    train_data = np.loadtxt("../data/extra_credit/X_forextracreditonly.csv")
    train_labels = np.loadtxt("../data/extra_credit/Y_forextracreditonly.csv")

    minibatch_sizes = [10, 50, 100, 250, 500, 750, 1000]
    step_sizes = [0.1, 0.05, 0.01, 0.005, 0.001]

    for step_size in step_sizes:
        print 'Step size: ', step_size
        for minibatch_size in minibatch_sizes:
            print 'Minibatch size: ', minibatch_size
            minibatch_indices = np.random.randint(0, train_data.shape[0], minibatch_size)
            sampled_z, running_average = langevin_dynamics.get_sampled_weights(train_data[minibatch_indices],
                                                                               train_labels[minibatch_indices],
                                                                               step_size, 10000)
            np.savetxt(
                "../data/extra_credit/results/langevin_dynamics/mini_batch/weights_" + str(minibatch_size) + "_" + str(
                    step_size), sampled_z)
    pass


def question_11_2():
    test_data = np.loadtxt("../data/normal/X_test.csv")
    test_labels = np.loadtxt("../data/normal/Y_test.csv")
    minibatch_sizes = [10, 50, 100, 250, 500, 750, 1000]
    step_sizes = [0.1, 0.05, 0.01, 0.005, 0.001]

    error_rates = defaultdict(float)

    for step_size in step_sizes:
        print step_size
        error_rates[step_size] = []
        for minibatch_size in minibatch_sizes:
            weights = np.loadtxt(
                "../data/extra_credit/results/langevin_dynamics/mini_batch/weights_" + str(minibatch_size) + "_" + str(
                    step_size))
            predictions = utils.get_predictions(test_data, 1, weights)
            error_rates[step_size].append(1 - accuracy_score(test_labels, predictions))

    utils.plot_graph(minibatch_sizes, "Minibatch Sizes", error_rates, "Error Rates",
                     "../data/extra_credit/results/langevin_dynamics/mini_batch/q_11.png")

    pass


def main():
    # question_9_1()
    question_9_2()
    # question_11_1()
    # question_11_2()


if __name__ == '__main__':
    main()
