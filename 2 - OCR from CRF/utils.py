import csv


def input_file_to_matrix(input_file_path):
    with open(input_file_path, 'rb') as f:
        reader = csv.reader(f)
        return [[float(val) for val in row[0].split()] for row in list(reader)]


def indicator(value_1, value_2):
    if value_1 == value_2:
        return 1
    return 0


def get_image_file(image_files_path, image_index):
    return input_file_to_matrix(image_files_path.format(image_index))
