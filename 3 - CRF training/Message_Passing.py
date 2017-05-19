from __future__ import division

import numpy as np

import utils


def get_feature_vector(character_labels, true_character_label):
    feature_vector = np.zeros(shape=(len(character_labels)))
    feature_vector[character_labels.index(true_character_label)] += 1

    return feature_vector


def get_transition_vector(character_labels, true_character_labels):
    transition_vector = np.zeros(shape=(len(character_labels), len(character_labels)))

    for label_index in xrange(0, len(true_character_labels) - 1):
        row_index = character_labels.index(true_character_labels[label_index])
        column_index = character_labels.index(true_character_labels[label_index + 1])
        transition_vector[row_index][column_index] += 1

    return transition_vector


def compute_message_values(image, feature_params, transition_params, character_labels):
    forward_message_values = np.zeros(shape=(len(image) - 1, len(character_labels)))
    reverse_message_values = np.zeros(shape=(len(image) - 1, len(character_labels)))

    for source in xrange(0, len(image) - 1):
        for target_index, character_label_target in enumerate(character_labels):
            local_message_value = 0
            for source_index, character_label_source in enumerate(character_labels):
                feature_potential = np.sum(
                    feature_params * (
                        get_feature_vector(character_labels, character_label_source)[np.newaxis].T * image[source][
                            np.newaxis]))
                transition_potential = np.sum(transition_params * get_transition_vector(character_labels, (
                    character_label_source, character_label_target
                )))
                neighbour_message_value = 0
                if source > 0:
                    neighbour_message_value = forward_message_values[source - 1][source_index]
                local_message_value += np.exp(feature_potential + transition_potential + neighbour_message_value)
            forward_message_values[source][target_index] = np.log(local_message_value)

    # To-Do - compute reverse order messages
    for source in xrange(len(image) - 1, 0, -1):
        for target_index, character_label_target in enumerate(character_labels):
            local_message_value = 0
            for source_index, character_label_source in enumerate(character_labels):
                feature_potential = np.sum(
                    feature_params * (
                        get_feature_vector(character_labels, character_label_source)[np.newaxis].T * image[source][
                            np.newaxis]))
                transition_potential = np.sum(transition_params * get_transition_vector(character_labels, (
                    character_label_source, character_label_target
                )))
                neighbour_message_value = 0
                if source < len(image) - 1:
                    neighbour_message_value = reverse_message_values[source][source_index]
                local_message_value += np.exp(feature_potential + transition_potential + neighbour_message_value)
            reverse_message_values[source - 1][target_index] = np.log(local_message_value)

    return np.array([forward_message_values, reverse_message_values])


def get_log_partition(image, feature_params, transition_params, character_labels):
    message_values = compute_message_values(image, feature_params, transition_params, character_labels)

    forward_message_values = message_values[0]
    forward_log_partition = 0

    for character_index, character_label in enumerate(character_labels):
        forward_log_partition += np.exp(
            np.sum(
                feature_params * (
                    get_feature_vector(character_labels, character_label)[np.newaxis].T * image[-1][
                        np.newaxis])) + \
            forward_message_values[-1][character_index])

    return np.log(forward_log_partition), message_values


def get_univariate_marginals(image, feature_params, character_labels, message_values, log_partition):
    univariate_marginals = []

    forward_message_values = message_values[0]
    reverse_message_values = message_values[1]

    for source in xrange(len(image)):
        local_univariate_marginals = []
        for character_index, character_label in enumerate(character_labels):
            marginal = np.sum(
                feature_params * (
                    get_feature_vector(character_labels, character_label)[np.newaxis].T * image[source][
                        np.newaxis]))
            if source < len(image) - 1:
                marginal += reverse_message_values[source][character_index]
            if source > 0:
                marginal += forward_message_values[source - 1][character_index]
            local_univariate_marginals.append(np.exp(marginal - log_partition))
        univariate_marginals.append(local_univariate_marginals)

    return np.array(univariate_marginals)


def get_pairwise_marginals(image, feature_params, transition_params, character_labels, message_values, log_partition):
    pairwise_marginals = []

    forward_message_values = message_values[0]
    reverse_message_values = message_values[1]

    for source in xrange(len(image) - 1):
        local_pairwise_marginals = []
        for character_index_1, character_label_1 in enumerate(character_labels):
            marginal_1_message = 0
            marginal_2_message = 0
            marginal_1_feature = np.sum(
                feature_params * (
                    get_feature_vector(character_labels, character_label_1)[np.newaxis].T * image[source][
                        np.newaxis]))
            if source > 0:
                marginal_1_message = forward_message_values[source - 1][character_index_1]
            for character_index_2, character_label_2 in enumerate(character_labels):
                marginal_2_feature = np.sum(
                    feature_params * (
                        get_feature_vector(character_labels, character_label_2)[np.newaxis].T * image[source][
                            np.newaxis]))
                marginal_12_transition = np.sum(
                    transition_params * get_transition_vector(character_labels, (character_label_1, character_label_2)))
                if source + 1 < len(image) - 1:
                    marginal_2_message = reverse_message_values[source + 1][character_index_2]
                local_pairwise_marginals.append(np.exp(
                    marginal_1_feature + marginal_1_message + marginal_2_feature + marginal_2_message + marginal_12_transition - log_partition))
        pairwise_marginals.append(
            np.reshape(np.array(local_pairwise_marginals), (len(character_labels), len(character_labels))))

    return np.array(pairwise_marginals)
