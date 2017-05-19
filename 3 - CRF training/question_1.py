from __future__ import division
import Message_Passing
import numpy as np
import matplotlib.pyplot as plt


def plot_graph(x_values, y_values, x_label, y_label, title):
    plt.plot(x_values, y_values, marker='o')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()


def plot_training_time_graph(training_set_length, total_training_images, execution_times_file):
    training_set_axis_values = range(training_set_length, total_training_images + 1, training_set_length)
    with open(execution_times_file, "r") as execution_times_obj:
        file_execution_times = [float(execution_time) for execution_time in execution_times_obj.readlines()[0].split(',')]

    execution_times = []

    for index, file_execution_time in enumerate(file_execution_times):
        if index == 0:
            execution_times.append(file_execution_time)
        else:
            execution_times.append(file_execution_time - file_execution_times[index-1])

    print execution_times

    # plot_graph(training_set_axis_values, execution_times, 'Training Set Size', 'Training Time (in seconds)',
    #            'Training Time vs Training Set Size')


def plot_test_error_rate(feature_params_file, transition_params_file, character_labels, training_set_size,
                         total_training_images, test_words_file, test_image_files):
    training_set_axis_values = range(training_set_size, total_training_images + 1, training_set_size)
    error_rates = []

    for training_set in training_set_axis_values:

        feature_params = np.loadtxt(feature_params_file.format(training_set), delimiter=',')
        transition_params = np.loadtxt(transition_params_file.format(training_set), delimiter=',')

        with open(test_words_file, "r") as test_words_file_obj:
            test_words = [test_word.strip() for test_word in test_words_file_obj.readlines()]

        count = 0
        total_count = 0
        for test_image_index in xrange(1, len(test_words) + 1):
            image = np.loadtxt(test_image_files.format(test_image_index))
            forward_log_partition, message_values = Message_Passing.get_log_partition(image, feature_params,
                                                                                      transition_params,
                                                                                      character_labels)
            univariate_marginals = Message_Passing.get_univariate_marginals(image, feature_params, character_labels,
                                                                            message_values, forward_log_partition)
            for character in xrange(len(image)):
                if character_labels[np.argmax(univariate_marginals[character])] == test_words[test_image_index - 1][
                    character]:
                    count += 1
            total_count += len(image)

        error_rate = (1 - (count / total_count)) * 100
        # print error_rate
        error_rates.append(error_rate)

    plot_graph(training_set_axis_values, error_rates, 'Number of Training Images', 'Error Rate (in %)',
               'Error Rate vs Training Set Size')


def get_average_log_likelihood(feature_params, transition_params, images, character_labels, test_words):
    log_likelihood = 0

    for image_index, image in enumerate(images):
        log_partition, message_values = Message_Passing.get_log_partition(image, feature_params, transition_params,
                                                                          character_labels)

        feature_potential = 0
        transition_potential = 0

        for character_index, image_character in enumerate(image):
            true_character_label = test_words[image_index][character_index]

            feature_vector = Message_Passing.get_feature_vector(character_labels, true_character_label)
            feature_potential += np.sum(
                feature_params[np.newaxis] * (feature_vector[np.newaxis].T * image_character[np.newaxis]))

            if character_index < len(image) - 1:
                transition_vector = Message_Passing.get_transition_vector(character_labels, (
                    test_words[image_index][character_index], test_words[image_index][character_index + 1]))
                transition_potential += np.sum(transition_params * transition_vector)

        log_likelihood += (feature_potential + transition_potential - log_partition)

    return (1 / len(images)) * log_likelihood


def plot_test_average_log_likelihood(test_images_file, test_words_file, feature_params_file, transition_params_file,
                                     training_set_length, total_training_images, character_labels):
    training_set_axis_values = range(training_set_length, total_training_images + 1, training_set_length)
    average_log_likelihood_values = []

    with open(test_words_file, "r") as test_words_file_obj:
        test_words = [test_word.strip() for test_word in test_words_file_obj.readlines()]

    images = []
    for test_image_index in xrange(1, len(test_words) + 1):
        images.append(np.loadtxt(test_images_file.format(test_image_index)))

    for training_set in training_set_axis_values:
        feature_params = np.loadtxt(feature_params_file.format(training_set), delimiter=',')
        transition_params = np.loadtxt(transition_params_file.format(training_set), delimiter=',')

        average_log_likelihood_value = get_average_log_likelihood(feature_params, transition_params, images,
                                                                  character_labels, test_words)
        print training_set, ': ', average_log_likelihood_value

        average_log_likelihood_values.append(average_log_likelihood_value)

    plot_graph(training_set_axis_values, average_log_likelihood_values, 'Number of Training Images',
               'Average Log-Likelihood', 'Average Test Images Log-Likelihood vs Training set size')


def main():
    feature_params_file = '../Data/parameters/feature_param_{}.txt'
    transition_params_file = '../Data/parameters/transition_param_{}.txt'
    test_words_file = '../Data/test_words.txt'
    test_image_files = '../Data/test_images/test_img{}.txt'
    execution_times_file = '../Data/execution_times.txt'

    character_labels = 'etainoshrd'
    training_set_length = 50

    plot_training_time_graph(training_set_length, 400, execution_times_file)
    # plot_test_error_rate(feature_params_file, transition_params_file, character_labels, training_set_length, 400,
    #                      test_words_file, test_image_files)
    # plot_test_average_log_likelihood(test_image_files, test_words_file, feature_params_file, transition_params_file,
    #                                  training_set_length, 400, character_labels)


if __name__ == '__main__':
    main()
