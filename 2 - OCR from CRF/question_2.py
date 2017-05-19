from __future__ import division
import utils
import Message_Passing
import numpy as np
import json


# compute all the messages
def part_1_1(number_of_images):
    feature_params_file = '../data/feature-params.txt'
    transition_params_file = '../data/transition-params.txt'
    character_labels = 'etainoshrd'
    test_image_file = '../data/images/test_img{}.txt'
    messages_json_file = '../data/messages.json'

    feature_params = utils.input_file_to_matrix(feature_params_file)
    transition_params = utils.input_file_to_matrix(transition_params_file)

    message_values = {}

    for image_index in xrange(1, number_of_images+1):
        print 'image: ', image_index
        image = utils.get_image_file(test_image_file, image_index)
        message_values[str(image_index)] = Message_Passing.compute_message_values(image, feature_params,
                                                                                  transition_params, character_labels)

    with open(messages_json_file, 'w') as fp:
        json.dump(message_values, fp, indent=4)


# get the required message values for question 2.1
def part_1_2(image_index):
    messages_json_file = '../data/messages.json'
    with open(messages_json_file, 'r') as fp:
        message_values = json.load(fp)

    print Message_Passing.get_message_values(message_values[str(image_index)], 1, 2)
    print Message_Passing.get_message_values(message_values[str(image_index)], 2, 1)
    print Message_Passing.get_message_values(message_values[str(image_index)], 2, 3)
    print Message_Passing.get_message_values(message_values[str(image_index)], 3, 2)


# compute individual marginal probabilities
def part_2_1(image_index):
    feature_params_file = '../data/feature-params.txt'
    character_labels = 'etainoshrd'
    test_image_file = '../data/images/test_img{}.txt'
    messages_json_file = '../data/messages.json'

    feature_params = utils.input_file_to_matrix(feature_params_file)
    with open(messages_json_file, 'r') as fp:
        message_values = json.load(fp)

    image = utils.get_image_file(test_image_file, image_index)

    marginal_probabilities = Message_Passing.get_marginal_probabilities(image, feature_params,
                                                                        character_labels, message_values[str(image_index)])
    print marginal_probabilities


# compute pairwise marginal probabilities, for characters ['t', 'a', 'h']
def part_2_2(image_index):
    feature_params_file = '../data/feature-params.txt'
    transition_params_file = '../data/transition-params.txt'
    character_labels = 'etainoshrd'
    test_image_file = '../data/images/test_img{}.txt'
    messages_json_file = '../data/messages.json'

    feature_params = utils.input_file_to_matrix(feature_params_file)
    transition_params = utils.input_file_to_matrix(transition_params_file)
    with open(messages_json_file, 'r') as fp:
        message_values = json.load(fp)

    image = utils.get_image_file(test_image_file, image_index)

    pairwise_marginal_probabilities = Message_Passing.get_pairwise_marginal_probabilities(image, feature_params,
                                                                                          transition_params,
                                                                                          character_labels,
                                                                                          message_values)
    print pairwise_marginal_probabilities


def part_3(number_of_images):
    feature_params_file = '../data/feature-params.txt'
    character_labels = 'etainoshrd'
    test_image_file = '../data/images/test_img{}.txt'
    messages_json_file = '../data/messages.json'
    test_words_file = '../data/test_words.txt'

    feature_params = utils.input_file_to_matrix(feature_params_file)
    with open(messages_json_file, 'r') as fp:
        message_values = json.load(fp)
    with open(test_words_file, "r") as test_words_file_obj:
        test_words = test_words_file_obj.readlines()

    accuracy_count = 0
    total_count = 0
    for image_index in xrange(1, number_of_images + 1):
        image = utils.get_image_file(test_image_file, image_index)
        total_count += len(image)
        marginal_probabilities = Message_Passing.get_marginal_probabilities(image, feature_params,
                                                                            character_labels,
                                                                            message_values[str(image_index)])
        most_probable_sequence = []
        for probability_list in marginal_probabilities:
            most_probable_sequence.append(character_labels[np.argmax(probability_list)])

        for i, label in enumerate(most_probable_sequence):
            # print label, ': ', test_words[image_index-1][i]
            if label == test_words[image_index-1][i]:
                accuracy_count += 1

        if image_index <= 5:
            print 'Image: ', image_index
            print 'Predicted Sequence: ', ''.join(most_probable_sequence)
            print 'Actual Sequence: ', test_words[image_index-1]

    accuracy = accuracy_count / total_count
    print 'Character level accuracy: ', accuracy


def main():
    # part_1_1(200)
    # part_1_2(1)
    # part_2_1(1)
    # part_2_2(1)
    part_3(200)


if __name__ == '__main__':
    main()
