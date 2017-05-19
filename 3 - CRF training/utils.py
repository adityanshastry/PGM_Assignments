import numpy as np
import csv


def get_matrix_from_file(input_file_path):
    with open(input_file_path, 'rb') as f:
        reader = csv.reader(f)
        return [[float(val) for val in row[0].split()] for row in list(reader)]


def indicator(value_1, value_2):
    if value_1 == value_2:
        return 1
    return 0


def get_image_file(image_files_path, image_index):
    return get_matrix_from_file(image_files_path.format(image_index))


def input_file_to_matrix(input_file_path):
    with open(input_file_path, 'rb') as f:
        reader = csv.reader(f)
        return [[float(val) for val in row[0].split()] for row in list(reader)]


def get_feature_transition_labels(training_words_file_path, character_labels):

    with open(training_words_file_path, "r") as train_words_file_obj:
        train_words = [train_word.strip() for train_word in train_words_file_obj.readlines()]

    feature_labels = []
    transition_labels = []

    for train_word in train_words:
        train_word_feature_label = []
        train_word_transition_label = []

        for character_index, character in enumerate(train_word):
            local_feature_label = np.zeros(shape=len(character_labels))
            local_transition_label = np.zeros(shape=(len(character_labels), len(character_labels)))

            local_feature_label[character_labels.index(character)] = 1

            if character_index < len(train_word) - 1:
                local_transition_label[character_labels.index(character)][character_labels.index(train_word[character_index+1])] = 1

            train_word_feature_label.append(local_feature_label)
            train_word_transition_label.append(local_transition_label)

        feature_labels.append(train_word_feature_label)
        transition_labels.append(train_word_transition_label)

    return feature_labels, transition_labels


def merge_jacobian_parameters(feature_parameters, transition_parameters):
    return np.concatenate((feature_parameters.flatten(), transition_parameters.flatten()))
