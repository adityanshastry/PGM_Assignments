import csv


# filters the data to only those instances having the given values of a variable's parents
def get_data_with_parents(input_data, given_variable_values, target_variable_parent_indices):

    input_data_with_parents = []
    for instance in input_data:
        comparison_count = 0
        for parent_index in target_variable_parent_indices:
            if instance[parent_index] == given_variable_values[parent_index]:
                comparison_count += 1
        if comparison_count == len(target_variable_parent_indices):
            input_data_with_parents.append(instance)

    return input_data_with_parents


# returns a 2D array equivalent of a file
def input_file_to_matrix(input_file_path):
    with open(input_file_path, 'rb') as f:
        reader = csv.reader(f)
        return [[int(val) for val in row] for row in list(reader)]

