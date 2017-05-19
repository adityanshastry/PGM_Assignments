from __future__ import division

import numpy as np

import utils
import Exhaustive_Inference
import json


def part_1(number_of_images):
    feature_params_file = '../data/feature-params.txt'
    character_labels = 'etainoshrd'
    test_image_file = '../data/images/test_img{}.txt'

    feature_params = utils.input_file_to_matrix(feature_params_file)

    node_potentials = []
    for i in xrange(1, number_of_images + 1):
        node_potentials.append(
            Exhaustive_Inference.get_node_potential(utils.get_image_file(test_image_file, i), feature_params,
                                                    character_labels))
    return node_potentials


def part_2(number_of_images):
    feature_params_file = '../data/feature-params.txt'
    transition_params_file = '../data/transition-params.txt'
    character_labels = 'etainoshrd'
    test_image_file = '../data/images/test_img{}.txt'
    test_words_file = '../data/test_words.txt'

    feature_params = utils.input_file_to_matrix(feature_params_file)
    transition_params = utils.input_file_to_matrix(transition_params_file)

    with open(test_words_file, "r") as test_words_file_obj:
        test_words = test_words_file_obj.readlines()

    # calculating the negative energy for the first 3 test images
    negative_energy_values = []
    for i in xrange(1, number_of_images + 1):
        negative_energy_values.append(
            Exhaustive_Inference.get_negative_energy(utils.get_image_file(test_image_file, i), feature_params,
                                                     transition_params,
                                                     character_labels, test_words[i - 1]))
    return negative_energy_values


def part_3_4(number_of_images):
    feature_params_file = '../data/feature-params.txt'
    transition_params_file = '../data/transition-params.txt'
    character_labels = 'etainoshrd'
    test_image_file = '../data/images/test_img{}.txt'

    feature_params = utils.input_file_to_matrix(feature_params_file)
    transition_params = utils.input_file_to_matrix(transition_params_file)

    # calculating the negative energy for the first 3 test images
    joint_sequence_probabilities = []
    most_likely_sequence_probabilities = []
    most_likely_sequences = []
    log_partition_values = []
    for i in xrange(1, number_of_images + 1):
        print i
        image = utils.get_image_file(test_image_file, i)
        joint_sequence_potentials = Exhaustive_Inference.get_log_partition_function(
            image, feature_params,
            transition_params,
            character_labels)
        partition_value = np.sum(joint_sequence_potentials)
        joint_sequence_probability = joint_sequence_potentials / partition_value
        joint_sequence_probabilities.append(joint_sequence_probability)
        most_likely_sequence_index = np.argmax(joint_sequence_probability)
        most_likely_sequence_probability = joint_sequence_probability[most_likely_sequence_index]
        most_likely_sequence = []
        for label_index in str(most_likely_sequence_index).zfill(len(image)):
            most_likely_sequence.append(character_labels[int(label_index)])
        most_likely_sequence = ''.join(most_likely_sequence)

        log_partition_values.append(np.log(partition_value))
        most_likely_sequence_probabilities.append(most_likely_sequence_probability)
        most_likely_sequences.append(most_likely_sequence)

    return log_partition_values, most_likely_sequences, most_likely_sequence_probabilities


def part_5():
    feature_params_file = '../data/feature-params.txt'
    transition_params_file = '../data/transition-params.txt'
    character_labels = 'etainoshrd'
    test_image_file = '../data/images/test_img{}.txt'
    image = utils.get_image_file(test_image_file, 1)

    feature_params = utils.input_file_to_matrix(feature_params_file)
    transition_params = utils.input_file_to_matrix(transition_params_file)

    # calculating the negative energy for the first 3 test images
    joint_sequence_potential = Exhaustive_Inference.get_log_partition_function(
        image, feature_params,
        transition_params,
        character_labels)

    potentials_at_positions = {}
    joint_sequence_potential_length = len(joint_sequence_potential)
    zfill_length = int(np.log10(joint_sequence_potential_length))

    for i in xrange(1, zfill_length+1):
        potentials_at_positions[str(i)] = {}

    for i in xrange(joint_sequence_potential_length):
        label_indices = str(i).zfill(zfill_length)
        for j, label_index in enumerate(label_indices):
            if character_labels[int(label_index)] not in potentials_at_positions[str(j+1)]:
                potentials_at_positions[str(j+1)][character_labels[int(label_index)]] = 0
            potentials_at_positions[str(j+1)][character_labels[int(label_index)]] += joint_sequence_potential[i]

    for position in potentials_at_positions:
        sum_potentials = sum(potentials_at_positions[position].values())
        for character_label in potentials_at_positions[position]:
            potentials_at_positions[position][character_label] /= sum_potentials

    return potentials_at_positions


def main():
    # print part_1(1)
    # print part_2(3)
    # print 'part 3, 4'
    # print part_3_4(3)
    print part_5()
    return


if __name__ == '__main__':
    main()
