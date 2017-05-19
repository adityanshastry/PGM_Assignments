from __future__ import division

import numpy as np
from scipy import optimize

import Message_Passing
import time


def average_log_likelihood_objective(parameters, images, character_labels, train_words):
    log_likelihood = 0

    jacobian = np.zeros(shape=(len(character_labels) * 321 + len(character_labels) * len(character_labels)))

    feature_params = np.reshape(parameters[:len(character_labels) * 321], (len(character_labels), 321))
    transition_params = np.reshape(parameters[len(character_labels) * 321:],
                                   (len(character_labels), len(character_labels)))

    for image_index, image in enumerate(images):
        log_partition, message_values = Message_Passing.get_log_partition(image, feature_params, transition_params,
                                                                          character_labels)
        univariate_marginals = Message_Passing.get_univariate_marginals(image, feature_params, character_labels,
                                                                        message_values, log_partition)
        pairwise_marginals = Message_Passing.get_pairwise_marginals(image, feature_params, transition_params,
                                                                    character_labels, message_values, log_partition)

        feature_jacobian = np.zeros(shape=(len(character_labels), 321))
        transition_jacobian = np.zeros(shape=(len(character_labels), len(character_labels)))
        feature_potential = 0
        transition_potential = 0

        for character_index, image_character in enumerate(image):
            true_character_label = train_words[image_index][character_index]

            feature_vector = Message_Passing.get_feature_vector(character_labels, true_character_label)
            feature_jacobian += (
                feature_vector - univariate_marginals[character_index])[np.newaxis].T * image_character[np.newaxis]

            feature_potential += np.sum(feature_params[np.newaxis] * (feature_vector[np.newaxis].T * image_character[np.newaxis]))

            if character_index < len(image) - 1:
                transition_vector = Message_Passing.get_transition_vector(character_labels, (
                    train_words[image_index][character_index], train_words[image_index][character_index + 1]))
                transition_jacobian += (transition_vector - pairwise_marginals[character_index])
                transition_potential += np.sum(transition_params * transition_vector)

        log_likelihood += (feature_potential + transition_potential - log_partition)
        jacobian += np.concatenate((feature_jacobian.flatten(), transition_jacobian.flatten()))

    return -(1/len(images)) * log_likelihood, -(1/len(images)) * jacobian


def train_crf(image_files_path, feature_params_file_name, transition_params_file_name, character_labels,
              train_words_file_name, training_set_length):
    with open(train_words_file_name, "r") as train_words_file_obj:
        train_words = [train_word.strip() for train_word in train_words_file_obj.readlines()]

    execution_times = []

    images = []

    parameters = np.zeros(shape=(len(character_labels) * 321 + len(character_labels) * len(character_labels)))

    for image_index in xrange(1, 401):
        images.append(np.loadtxt(image_files_path.format(image_index)))
        if image_index % training_set_length == 0:
            print 'CRF-Model: ', image_index

            start_time = time.time()
            optimizer = optimize.minimize(fun=average_log_likelihood_objective, x0=np.array(parameters),
                                          args=(images[:image_index], character_labels, train_words[:image_index]),
                                          jac=True, method='BFGS')

            print time.time() - start_time
            execution_times.append(time.time() - start_time)

            parameters = optimizer['x']

            # print time.time() - start_time
            # feature_params = np.reshape(np.array(parameters[:len(character_labels) * 321]),
            #                             (len(character_labels), 321))
            # transition_params = np.reshape(np.array(parameters[len(character_labels) * 321:]),
            #                                (len(character_labels), len(character_labels)))
            # np.savetxt(feature_params_file_name.format(image_index), feature_params, delimiter=',')
            # np.savetxt(transition_params_file_name.format(image_index), transition_params, delimiter=',')

    return execution_times


def main():
    character_labels = 'etainoshrd'
    image_files_path = '../Data/images/train_img{}.txt'
    feature_params_file_name = '../Data/parameters_2/feature_param_{}.txt'
    transition_params_file_name = '../Data/parameters_2/transition_param_{}.txt'
    train_words_file_name = '../Data/train_words.txt'
    training_set_length = 50

    print train_crf(image_files_path, feature_params_file_name, transition_params_file_name, character_labels,
                    train_words_file_name, training_set_length)


if __name__ == '__main__':
    main()
