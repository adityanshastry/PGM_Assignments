import autograd.numpy as np
from autograd import grad
import utils
from sklearn.metrics import accuracy_score


def get_mean_evidence(w, variance, data, labels):
    z = w + np.sqrt(variance) * np.random.normal(0, 1, size=data.shape)
    log_prior = np.sum(-np.sum((z - w) * (z - w), axis=1)) / data.shape[0]
    log_likelihood = np.sum(-np.logaddexp(np.zeros(shape=data.shape[0]), -labels * np.sum(z * data, axis=1)))

    return log_prior + log_likelihood


def get_sampled_weights(data, labels, step_size, t_max):
    w = np.random.normal(0, 1, size=data.shape[1])
    mean_evidence_gradient = grad(get_mean_evidence)
    sampled_w = []
    for time_step in xrange(t_max):
        w += step_size * mean_evidence_gradient(w, 0.5, data, labels)
        sampled_w.append(w.copy())
    return np.array(sampled_w)


def get_final_weight(data, labels, step_size, t_max):
    w = np.random.normal(0, 1, size=data.shape[1])
    mean_evidence_gradient = grad(get_mean_evidence)
    for time_step in xrange(t_max):
        w += step_size * mean_evidence_gradient(w, 0.5, data, labels)
    return w


def question_7():
    train_data = np.loadtxt("../data/normal/X_train.csv")
    train_labels = np.loadtxt("../data/normal/Y_train.csv")

    sampled_weights = get_sampled_weights(train_data, train_labels, 0.02, 10000)
    # np.savetxt("../data/normal/vi_sampled_weights.csv", sampled_weights)

    # sampled_weights = np.loadtxt("../data/normal/vi_sampled_weights.csv")
    utils.plot_results(sampled_weights, "Iteration Step", "Sampled W values", "Sampled weight vs Iteration steps",
                       "../data/normal/sampled_w_plot_2.png")
    pass


def question_8_1():
    train_data = np.loadtxt("../data/normal/X_train.csv")
    train_labels = np.loadtxt("../data/normal/Y_train.csv")

    t_max_values = [10, 100, 1000, 10000]

    for t_max in t_max_values:
        print t_max
        sampled_weights = []
        for i in xrange(5):
            sampled_weights.append(get_final_weight(train_data, train_labels, 0.02, t_max))
        np.savetxt("../data/normal/results/vi_sampled_weights_" + str(t_max) + ".csv", np.array(sampled_weights))


def question_8_2():
    test_data = np.loadtxt("../data/normal/X_test.csv")
    test_labels = np.loadtxt("../data/normal/Y_test.csv")
    t_max_values = [10, 100, 1000, 10000]
    target_value = 1
    variance = 0.5

    for t_max in t_max_values:
        weights = np.loadtxt("../data/normal/results/vi_sampled_weights_" + str(t_max) + ".csv")
        for weight_index, weight in enumerate(weights):
            z_values = np.random.normal(weight, np.sqrt(variance), size=(1000, weight.shape[0]))
            predictions = utils.get_predictions(test_data, target_value, z_values)
            print 't_max: ', t_max, ', inference: ', weight_index + 1, ', error_rate: ', 1 - accuracy_score(test_labels,
                                                                                                            predictions)


def main():
    # question_7()
    # question_8_1()
    question_8_2()


if __name__ == '__main__':
    main()
