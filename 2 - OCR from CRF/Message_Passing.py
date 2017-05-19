from __future__ import division
import utils
import math
import numpy as np
import json


def get_message_dict_key(source, target):
    return str(source) + '->' + str(target)


def get_message_values(message_values, source, target):
    return message_values[get_message_dict_key(source, target)]


def get_feature_potential(feature_params, character_labels, image_character, true_character_label):
    feature_potential = 0

    for c, character_label_c in enumerate(character_labels):
        for f, feature_param in enumerate(feature_params[c]):
            feature_potential += feature_param * utils.indicator(true_character_label, character_label_c) * \
                                 image_character[f]

    return feature_potential


def get_transition_potential(transition_params, character_labels, true_character_labels):
    transition_potential = 0

    for c_1, character_label_c_1 in enumerate(character_labels):
        for c_2, character_label_c_2 in enumerate(character_labels):
            transition_potential += transition_params[c_1][c_2] * utils.indicator(true_character_labels[0],
                                                                                  character_label_c_1) * utils.indicator(
                true_character_labels[1], character_label_c_2)

    return transition_potential


def get_neighbour_for_message(source, target):
    if source > target:
        return source + 1
    return source - 1


def get_neighbours(source):
    return [source - 1, source + 1]


def compute_message_values(image, feature_params, transition_params, character_labels):
    message_values = {}

    for source in xrange(1, len(image)):
        message_key = str(source) + '->' + str(source + 1)
        # print 'message: ', message_key
        if message_key not in message_values:
            message_values[message_key] = {}
        neighbour_message_key = str(source - 1) + '->' + str(source)
        # print 'neighbour: ', neighbour_message_key
        for character_label_target in character_labels:
            local_message_value = 0
            for character_label_source in character_labels:
                feature_potential = get_feature_potential(feature_params, character_labels, image[source - 1],
                                                          character_label_source)
                transition_potential = get_transition_potential(transition_params, character_labels,
                                                                (character_label_source, character_label_target
                                                                 ))
                neighbour_message_value = 0
                if neighbour_message_key in message_values:
                    neighbour_message_value = message_values[neighbour_message_key][character_label_source]
                local_message_value += np.exp(feature_potential + transition_potential + neighbour_message_value)
            message_values[message_key][character_label_target] = np.log(local_message_value)
            # print message_values

    for source in xrange(len(image), 1, -1):
        message_key = str(source) + '->' + str(source - 1)
        # print 'message: ', message_key
        if message_key not in message_values:
            message_values[message_key] = {}
        neighbour_message_key = str(source + 1) + '->' + str(source)
        # print 'neighbour: ', neighbour_message_key
        for character_label_target in character_labels:
            local_message_value = 0
            for character_label_source in character_labels:
                feature_potential = get_feature_potential(feature_params, character_labels, image[source - 1],
                                                          character_label_source)
                transition_potential = get_transition_potential(transition_params, character_labels,
                                                                (character_label_target, character_label_source
                                                                 ))
                neighbour_message_value = 0
                if neighbour_message_key in message_values:
                    neighbour_message_value = message_values[neighbour_message_key][character_label_source]
                local_message_value += np.exp(feature_potential + transition_potential + neighbour_message_value)
            message_values[message_key][character_label_target] = np.log(local_message_value)

    return message_values


def get_marginal_probabilities(image, feature_params, character_labels, message_values):
    marginal_probabilities = []

    for source in xrange(1, len(image) + 1):
        local_marginal_probabilities = []
        for character_label_source in character_labels:
            feature_potential = np.exp(get_feature_potential(feature_params, character_labels, image[source - 1],
                                                             character_label_source))
            message_from_neighbours = 1
            for neighbour in get_neighbours(source):
                neighbour_message_key = str(neighbour) + '->' + str(source)
                if neighbour_message_key in message_values:
                    message_from_neighbours *= np.exp(message_values[neighbour_message_key][character_label_source])

            local_marginal_probabilities.append(feature_potential * message_from_neighbours)
        local_marginal_probabilities = np.array(local_marginal_probabilities)
        local_marginal_probabilities /= np.sum(local_marginal_probabilities)
        marginal_probabilities.append(local_marginal_probabilities)

    return marginal_probabilities


def get_pairwise_marginal_probabilities(image, feature_params, transition_params, character_labels, message_values):

    pairwise_marginal_probabilities = []

    for source in xrange(1, len(image)):
        local_pairwise_probabilities = []
        for character_label_1 in character_labels:
            feature_potential_1 = get_feature_potential(feature_params, character_labels, image[source - 1],
                                                        character_label_1)
            neighbour_message_1_key = str(get_neighbour_for_message(source, source + 1)) + '->' + str(source)
            neighbour_message_1_value = 1
            if neighbour_message_1_key in message_values:
                neighbour_message_1_value = np.exp(message_values[neighbour_message_1_key][character_label_1])

            for character_label_2 in character_labels:
                feature_potential_2 = get_feature_potential(feature_params, character_labels, image[source],
                                                            character_label_2)
                transition_potential = get_transition_potential(transition_params, character_labels,
                                                                (character_label_1, character_label_2
                                                                 ))
                neighbour_message_2_key = str(get_neighbour_for_message(source + 1, source)) + '->' + str(source + 1)
                neighbour_message_2_value = 1
                if neighbour_message_2_key in message_values:
                    neighbour_message_2_value = np.exp(message_values[neighbour_message_2_key][character_label_2])

                local_pairwise_probabilities.append(np.exp(
                    feature_potential_1 + feature_potential_2 + transition_potential) * neighbour_message_1_value *
                                                    neighbour_message_2_value)
        local_pairwise_probabilities = np.array(local_pairwise_probabilities)
        local_pairwise_probabilities /= np.sum(local_pairwise_probabilities)
        pairwise_marginal_probabilities.append(local_pairwise_probabilities)

    return pairwise_marginal_probabilities
