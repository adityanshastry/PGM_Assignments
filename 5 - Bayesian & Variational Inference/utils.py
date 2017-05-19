import numpy as np
import matplotlib.pyplot as plt


def plot_results(results, x_label, y_label, title, fig_path):
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    for dimension in xrange(results.shape[1]):
        plt.plot(range(1, results.shape[0]+1), results[:, dimension], label="dimension_" + str(dimension+1))

    plt.legend(loc='upper right')
    plt.savefig(fig_path)
    plt.close()
    # plt.show()
    pass


def plot_graph(x_data, x_label, y_data, y_label, save_path):
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    for key in y_data:
        plt.plot(x_data, y_data[key], label="step_size_" + str(key))

    plt.legend(loc='upper right')
    plt.savefig(save_path)
    plt.close()
    # plt.show()
    pass


def get_predictions(data, target_value, weights):

    threshold = np.log(0.5)
    predictions = []

    for value in data:
        probability = 0
        for weight in weights:
            probability -= np.logaddexp(0, -1*target_value*np.dot(weight, value))
        probability /= weights.shape[0]
        if probability > threshold:
            predictions.append(target_value)
        else:
            predictions.append(-1*target_value)

    return np.array(predictions)
