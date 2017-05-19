import autograd.numpy as np
from autograd import grad
from sklearn.metrics import accuracy_score
import utils


def get_posterior(z, data, labels):
    log_likelihood = -1 * np.logaddexp(0, -1 * labels * np.dot(data, z)).sum()
    log_prior = np.inner(z, z) * -0.5

    return log_likelihood + log_prior

    pass


def get_sampled_weights(data, labels, step_size, t_max):
    z = np.random.normal(0, 1, size=data.shape[1])
    z_values = []
    cumulative_sum = []
    for time_step in xrange(t_max):
        # print time_step
        gradient = grad(get_posterior)
        gradient_value = gradient(z, data, labels)
        noise = np.random.normal(0, 1, size=data.shape[1])
        z += step_size * 0.5 * gradient_value + np.sqrt(step_size) * noise
        z_values.append(z.copy())
        if not len(cumulative_sum):
            cumulative_sum.append(z.copy())
        else:
            cumulative_sum.append(cumulative_sum[-1] + z.copy())

    for time_step in xrange(1, t_max + 1):
        cumulative_sum[time_step - 1] /= time_step

    return np.array(z_values), np.array(cumulative_sum)


def question_4():
    train_data = np.loadtxt("../data/normal/X_train.csv")
    train_labels = np.loadtxt("../data/normal/Y_train.csv")
    step_size = 0.01
    t_max = 10000
    sampled_z, running_average = get_sampled_weights(train_data, train_labels, step_size, t_max)

    # np.savetxt("../data/normal/ld_sampled_weights.csv", sampled_z)
    #
    # utils.plot_results(sampled_z, "Iteration Step", "Sampled Z values", "Sampled weight vs Iteration steps",
    #                    "../data/normal/sampled_z_plot_2.png")
    # utils.plot_results(running_average, "Iteration Step", "Running Average", "Running average vs Iteration steps",
    #                    "../data/normal/running_average_plot_2.png")


def question_5():
    test_data = np.loadtxt("../data/normal/X_test.csv")
    test_labels = np.loadtxt("../data/normal/Y_test.csv")
    train_data = np.loadtxt("../data/normal/X_train.csv")
    train_labels = np.loadtxt("../data/normal/Y_train.csv")
    step_size = 0.01

    t_max_values = [10, 100, 1000, 10000]
    for t_max in t_max_values:
        print t_max
        for i in xrange(5):
            sampled_weights = get_sampled_weights(train_data, train_labels, step_size, t_max)[0]
            predictions = utils.get_predictions(test_data, 1, sampled_weights)
            print 't_max: ', t_max, ', inference: ', i+1, ', error_rate: ', 1 - accuracy_score(test_labels, predictions)


def main():
    # question_4()
    question_5()


if __name__ == '__main__':
    main()
