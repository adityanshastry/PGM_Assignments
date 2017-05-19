from __future__ import division
import getopt
import sys
import math
import utils


# Returns the conditional probability, given the input data, the target variables, parents,
# and the values of the required variables,
def get_conditional_probability(input_data, target_variable_index, given_variable_values,
                                target_variable_parent_indices, pseudocount=(0, 0)):
    input_data_with_parents = input_data

    if not len(target_variable_parent_indices) == 0:
        input_data_with_parents = utils.get_data_with_parents(input_data, given_variable_values,
                                                              target_variable_parent_indices)

    total_values = len(input_data_with_parents)
    target_value_count = 0
    for instance in input_data_with_parents:
        if not math.isnan(given_variable_values[target_variable_index]) and (
                    instance[target_variable_index] == given_variable_values[target_variable_index]) or \
                math.isnan(given_variable_values[target_variable_index]):
            target_value_count += 1

    return (target_value_count + pseudocount[0]) / (total_values + pseudocount[1])


# Returns the joint probability of variables in a bayesian graph
def get_joint_probability(input_data, bayesian_network, variable_order, variable_values, pseudocount=(0, 0)):
    joint_probability = 1

    for variable in bayesian_network:
        variable_index = variable_order.index(variable)
        variable_parent_indices = [variable_order.index(variable_parent) for variable_parent in
                                   bayesian_network[variable]]
        joint_probability *= get_conditional_probability(input_data, variable_index, variable_values, variable_parent_indices,
                                                   pseudocount)

    return joint_probability


# Used to compute the conditional probability separately for question 4.
def main(argv):
    correct_input_format = 'probability_computation.py -i <input_data_file_path> -t <target_variable_index> ' \
                           '-v <target_variable_values> -p <target_variable_parent_indices>'

    input_data_file = ''
    target_variable_index = 0
    given_variable_values = []
    target_variable_parent_indices = []

    try:
        opts, args = getopt.getopt(argv, "hi:t:v:p:")
    except getopt.GetoptError:
        print correct_input_format
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print correct_input_format
            sys.exit(1)
        elif opt == "-i":
            input_data_file = arg
        elif opt == "-t":
            target_variable_index = int(arg)
        elif opt == '-v':
            given_variable_values = [float(i) for i in arg.split(',')]
        elif opt == '-p':
            target_variable_parent_indices = [int(i) for i in arg.split(',')]

    print get_conditional_probability(utils.input_file_to_matrix(input_data_file), target_variable_index,
                                      given_variable_values, target_variable_parent_indices)


if __name__ == '__main__':
    main(sys.argv[1:])
