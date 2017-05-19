import numpy as np

import utils


# calculate the node potential for each character label for each character of an image
def get_node_potential(image, feature_params, character_labels):
    image_node_potentials = []

    for image_character in image:
        node_potential_for_position = []
        for character_label in character_labels:
            local_node_potential = 0
            for c, character_label_c in enumerate(character_labels):
                for f, feature_param in enumerate(feature_params[c]):
                    local_node_potential += feature_param * utils.indicator(character_label, character_label_c) * \
                                            image_character[f]
            node_potential_for_position.append(local_node_potential)
        image_node_potentials.append(node_potential_for_position)

    return image_node_potentials


# calculate the node potential for only the true label characters of an image
def get_node_potential_for_true_labels(image, feature_params, character_labels, true_labels):
    node_potential_true_labels = []
    node_potentials = get_node_potential(image, feature_params, character_labels)

    for i, node_potential in enumerate(node_potentials):
        node_potential_true_labels.append(node_potential[character_labels.index(true_labels[i])])

    return node_potential_true_labels


# calculate the negative energy for an image with true labels provided
def get_negative_energy(image, feature_params, transition_params, character_labels, true_labels):
    node_potential_true_labels = get_node_potential_for_true_labels(image, feature_params, character_labels,
                                                                    true_labels)
    sum_node_potentials = 0
    sum_transition_params = 0

    for i in xrange(len(image)):
        sum_node_potentials += node_potential_true_labels[i]

    for i in xrange(len(image) - 1):
        for c_1, c_1_label in enumerate(character_labels):
            for c_2, c_2_label in enumerate(character_labels):
                sum_transition_params += transition_params[c_1][c_2] * utils.indicator(true_labels[i], c_1_label) * \
                                         utils.indicator(true_labels[i + 1], c_2_label)

    return sum_node_potentials + sum_transition_params


# calculate the log partition function for the given images
def get_log_partition_function(image, feature_params, transition_params, character_labels):
    partition_function_value = []

    for label_int in xrange(len(character_labels) ** len(image)):
        label_indices = str(label_int).zfill(len(image))
        label_string = []
        for label_index in label_indices:
            label_string.append(character_labels[int(label_index)])
        label_string = ''.join(label_string)
        # print label_string
        partition_function_value.append(np.exp(
            get_negative_energy(image, feature_params, transition_params, character_labels,
                                label_string)))

    return partition_function_value
