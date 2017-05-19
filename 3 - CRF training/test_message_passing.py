from __future__ import division
import Message_Passing
import numpy as np

'''
This code was written to test the individual functionalities like marginals, log partitions etc.

'''


def main():
    feature_params = np.loadtxt('../Data/parameters/feature_param_100.txt', delimiter=',')
    transition_params = np.loadtxt('../Data/parameters/transition_param_100.txt', delimiter=',')
    character_labels = 'etainoshrd'

    with open('../Data/train_words.txt', "r") as train_words_file_obj:
        train_words = [train_word.strip() for train_word in train_words_file_obj.readlines()]
    with open('../Data/test_words.txt', "r") as test_words_file_obj:
        test_words = [test_word.strip() for test_word in test_words_file_obj.readlines()]

    count = 0
    total_count = 0
    for i in xrange(1, 201):
        print 'Image: ', i, ' -> ', test_words[i-1]
        image = np.loadtxt('../Data/test_images/test_img{}.txt'.format(i))
        forward_log_partition, message_values = Message_Passing.get_log_partition(image, feature_params,
                                                                                  transition_params, character_labels)
        univariate_marginals = Message_Passing.get_univariate_marginals(image, feature_params, character_labels,
                                                                        message_values, forward_log_partition)
        for character in xrange(len(image)):
            if character_labels[np.argmax(univariate_marginals[character])] == test_words[i - 1][character]:
                count += 1
        total_count += len(image)

    print count/total_count


if __name__ == '__main__':
    main()
