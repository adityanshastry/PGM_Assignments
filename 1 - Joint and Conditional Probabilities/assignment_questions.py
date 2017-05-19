from __future__ import division

import json

import numpy

import probability_computation
import utils


def question_5a():

    print 'Problem 5.a'

    input_data = utils.input_file_to_matrix('../Data_Files/data-train-1.txt')
    given_variable_values_1 = [2.0, 2.0, 4.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    given_variable_values_2 = [2.0, 2.0, 4.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0]
    target_variable_probabilities = []
    bayesian_network = json.load(open('../Data_Files/heart_disease_bayesian_network.json'))
    variable_order = open('../Data_Files/heart_disease_bayesian_network_order.txt').read().split(';')

    ch_1 = probability_computation.get_joint_probability(input_data, bayesian_network,
                                                         variable_order,
                                                         given_variable_values_1)
    ch_2 = probability_computation.get_joint_probability(input_data, bayesian_network,
                                                         variable_order,
                                                         given_variable_values_2)

    target_variable_probabilities.append(ch_1 / (ch_1 + ch_2))
    target_variable_probabilities.append(ch_2 / (ch_1 + ch_2))

    print target_variable_probabilities


def question_5b():

    print 'Problem 5.b'

    input_data = utils.input_file_to_matrix('../Data_Files/data-train-1.txt')
    given_variable_values_11 = [2.0, 1.0, 1.0, 1.0, 2.0, 1.0, 2.0, 2.0, 1.0]
    given_variable_values_21 = [2.0, 2.0, 1.0, 1.0, 2.0, 1.0, 2.0, 2.0, 1.0]
    given_variable_values_12 = [2.0, 1.0, 1.0, 2.0, 2.0, 1.0, 2.0, 2.0, 1.0]
    given_variable_values_22 = [2.0, 2.0, 1.0, 2.0, 2.0, 1.0, 2.0, 2.0, 1.0]

    target_variable_probabilities = []
    bayesian_network = json.load(open('../Data_Files/heart_disease_bayesian_network.json'))
    variable_order = open('../Data_Files/heart_disease_bayesian_network_order.txt').read().split(';')

    bp_11 = probability_computation.get_joint_probability(input_data, bayesian_network,
                                                          variable_order,
                                                          given_variable_values_11)
    bp_21 = probability_computation.get_joint_probability(input_data, bayesian_network,
                                                          variable_order,
                                                          given_variable_values_21)
    bp_12 = probability_computation.get_joint_probability(input_data, bayesian_network,
                                                          variable_order,
                                                          given_variable_values_12)
    bp_22 = probability_computation.get_joint_probability(input_data, bayesian_network,
                                                          variable_order,
                                                          given_variable_values_22)

    target_variable_probabilities.append((bp_11 + bp_21) / (bp_11 + bp_21 + bp_12 + bp_22))
    target_variable_probabilities.append((bp_12 + bp_22) / (bp_11 + bp_21 + bp_12 + bp_22))

    print target_variable_probabilities


def question_6c():

    print 'Problem 6.c'

    train_data_path = '../Data_Files/data-train-{}.txt'
    test_data_path = '../Data_Files/data-test-{}.txt'
    train_data = []
    test_data = []
    test_accuracies = []
    pseudocount = (1, 2)  # tuple: 0 -> pseudocount, 1 -> vocabulary (Number of possible values of the variable)
    bayesian_network = json.load(open('../Data_Files/heart_disease_bayesian_network.json'))
    variable_order = open('../Data_Files/heart_disease_bayesian_network_order.txt').read().split(';')

    for i in xrange(1, 6):
        train_data.append(utils.input_file_to_matrix(train_data_path.format(i)))
        test_data.append(utils.input_file_to_matrix(test_data_path.format(i)))

    for i, current_test_data in enumerate(test_data):
        count = 0
        for instance in current_test_data:
            given_variable_values_1 = instance[:8] + [1]
            given_variable_values_2 = instance[:8] + [2]
            hd_1 = probability_computation.get_joint_probability(train_data[i], bayesian_network, variable_order,
                                                                 given_variable_values_1)
            hd_2 = probability_computation.get_joint_probability(train_data[i], bayesian_network, variable_order,
                                                                 given_variable_values_2)

            if ((hd_1 >= hd_2) and instance[8] == 1) or ((hd_1 <= hd_2) and instance[8] == 2):
                count += 1

        test_accuracies.append(count / len(test_data[i]))

    print test_accuracies

    accuracy_mean = numpy.mean(test_accuracies)
    accuracy_std = numpy.std(test_accuracies)

    print 'Mean of Accuracy: ', accuracy_mean
    print 'Standard Deviation of Accuracy: ', accuracy_std

    return


def question_7d():

    print 'Problem 7.d'

    train_data_path = '../Data_Files/data-train-{}.txt'
    test_data_path = '../Data_Files/data-test-{}.txt'
    train_data = []
    test_data = []
    test_accuracies = []
    pseudocount = (1, 2)  # tuple: 0 -> pseudocount, 1 -> vocabulary (Number of possible values of the variable)
    bayesian_network = json.load(open('../Data_Files/heart_disease_bayesian_network_modelled.json'))
    variable_order = open('../Data_Files/heart_disease_bayesian_network_order.txt').read().split(';')

    for i in xrange(1, 6):
        train_data.append(utils.input_file_to_matrix(train_data_path.format(i)))
        test_data.append(utils.input_file_to_matrix(test_data_path.format(i)))

    for i, current_test_data in enumerate(test_data):
        count = 0
        for instance in current_test_data:
            given_variable_values_1 = instance[:8] + [1]
            given_variable_values_2 = instance[:8] + [2]
            hd_1 = probability_computation.get_joint_probability(train_data[i], bayesian_network, variable_order,
                                                                 given_variable_values_1)
            hd_2 = probability_computation.get_joint_probability(train_data[i], bayesian_network, variable_order,
                                                                 given_variable_values_2)

            if ((hd_1 >= hd_2) and instance[8] == 1) or ((hd_1 <= hd_2) and instance[8] == 2):
                count += 1

        test_accuracies.append(count / len(test_data[i]))

    print test_accuracies

    accuracy_mean = numpy.mean(test_accuracies)
    accuracy_std = numpy.std(test_accuracies)

    print 'Mean of Accuracy: ', accuracy_mean
    print 'Standard Deviation of Accuracy: ', accuracy_std

    return


def main():
    # question_5a()
    # question_5b()
    question_6c()
    question_7d()
    return


if __name__ == '__main__':
    main()
