import numpy as np
from sklearn.metrics import accuracy_score
import variational_inference
import utils
from collections import defaultdict


def question_12_1():
    train_data = np.loadtxt("../data/normal/X_train.csv")
    train_labels = np.loadtxt("../data/normal/Y_train.csv")

    step_sizes = [0.1, 0.05, 0.01, 0.005, 0.001]
    t_max_values = [10, 100, 1000, 10000, 100000]

    for step_size in step_sizes:
        print step_size
        for t_max in t_max_values:
            print t_max
            sampled_weights = variational_inference.get_sampled_weights(train_data, train_labels, step_size, t_max)
            print sampled_weights.shape
            np.savetxt(
                "../data/extra_credit/results/variational_inference/run_length/weights_" + str(step_size) + "_" + str(
                    t_max) + ".csv", sampled_weights)

            utils.plot_results(sampled_weights, "Iteration Step", "Sampled W values",
                               "Sampled weight vs Iteration steps",
                               "../data/extra_credit/results/variational_inference/run_length/weights_plot_" + str(
                                   step_size) + "_" + str(
                                   t_max) + ".png")

    pass


def question_12_2():
    test_data = np.loadtxt("../data/normal/X_test.csv")
    test_labels = np.loadtxt("../data/normal/Y_test.csv")
    step_sizes = [0.1, 0.05, 0.01, 0.005, 0.001]
    t_max_values = [10, 100, 1000, 10000, 100000]
    target_value = 1
    variance = 0.5

    for step_size in step_sizes:
        for t_max in t_max_values:
            weight = np.loadtxt(
                "../data/extra_credit/results/variational_inference/run_length/weights_" + str(step_size) + "_" + str(
                    t_max) + ".csv")[-1]
            z_values = np.random.normal(weight, np.sqrt(variance), size=(1000, weight.shape[0]))
            predictions = utils.get_predictions(test_data, target_value, z_values)
            print 'step_size: ', step_size, ', t_max: ', t_max, ', error_rate', 1 - accuracy_score(test_labels,
                                                                                                   predictions)


def question_13_1():
    train_data = np.loadtxt("../data/extra_credit/X_forextracreditonly.csv")
    train_labels = np.loadtxt("../data/extra_credit/Y_forextracreditonly.csv")

    minibatch_sizes = [10, 50, 100, 250, 500, 750, 1000]
    step_sizes = [0.1, 0.05, 0.01, 0.005, 0.001]
    t_max = 10000
    variance = 0.5

    for step_size in step_sizes:
        print 'Step size: ', step_size
        for minibatch_size in minibatch_sizes:
            print 'Minibatch size: ', minibatch_size
            minibatch_indices = np.random.randint(0, train_data.shape[0], minibatch_size)
            weight = variational_inference.get_final_weight(train_data[minibatch_indices],
                                                            train_labels[minibatch_indices], step_size, t_max)
            sampled_z = np.random.normal(weight, np.sqrt(variance), size=(1000, weight.shape[0]))
            np.savetxt("../data/extra_credit/results/variational_inference/mini_batch/weights_" + str(
                minibatch_size) + "_" + str(step_size), sampled_z)
    pass


def question_13_2():
    test_data = np.loadtxt("../data/normal/X_test.csv")
    test_labels = np.loadtxt("../data/normal/Y_test.csv")

    minibatch_sizes = [10, 50, 100, 250, 500, 750, 1000]
    step_sizes = [0.1, 0.05, 0.01, 0.005, 0.001]

    error_rates = defaultdict(float)

    for step_size in step_sizes:
        print step_size
        error_rates[step_size] = []
        for minibatch_size in minibatch_sizes:
            sampled_z = np.loadtxt("../data/extra_credit/results/variational_inference/mini_batch/weights_" + str(
                minibatch_size) + "_" + str(step_size))
            predictions = utils.get_predictions(test_data, 1, sampled_z)
            error_rates[step_size].append(1 - accuracy_score(test_labels, predictions))
            print 1 - accuracy_score(test_labels, predictions)

    utils.plot_graph(minibatch_sizes, "Minibatch Sizes", error_rates, "Error Rates",
                     "../data/extra_credit/results/variational_inference/mini_batch/q_13.png")

    pass


def main():
    # question_12_1()
    question_12_2()
    # question_13_1()
    # question_13_2()


if __name__ == '__main__':
    main()
