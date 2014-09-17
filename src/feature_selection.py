import math
import utils


def generate_feature_test(feature_value):
    """
    Returns a function closure that will evaluate an example based on its Feature Type:
        Type can be either NOMINAL or CONTINUOUS
    :param feature_value: feature value
    """
    test_function = None
    if feature_value.type == 'NOMINAL':
        test_function = lambda example: True if example == feature_value else False
    else:
        pass
        #TODO
        #test_function = lambda example:
    return test_function


def get_best_feature_index(examples, feature_indices):
    """

    Gets the index of the feature with the highest gain ratio.

    :param examples: numpy array of Examples
    :param feature_indices: the indices of features on each Example's Schema
    :return: int index of the best feature
    """
    gain_ratios = [get_gain_ratio(examples, index, get_entropy(examples, index)) for index in feature_indices]
    max_index, max_value = max(enumerate(gain_ratios), key=lambda p: p[1])
    return max_index


def get_gain_ratio(examples, feature_index, entropy):
    """

    :param examples: numpy array of Examples
    :param feature_index: the index of the feature on each Example's Schema
    @type entropy: float
    :return: gain ratio
    """
    gain = get_information_gain(examples, entropy) / entropy
    return gain/entropy


def get_information_gain(examples, entropy_of_set):
    """

    :param examples: numpy array of Examples
    @type entropy_of_set: float
    :return: float
    """
    gain = entropy_of_set
    class_dict = utils.get_class_label_values(examples)
    for class_label, count in class_dict.items():
        gain -= count/len(examples) * get_entropy(class_dict, len(examples))
    return gain


def get_entropy(examples, feature_index):
    """

    @type example:
    :param feature_index:
    :return:
    """
    entropy = 0
    feature_dict = utils.get_example_values_for_feature(examples, feature_index)
    for feature_index, count in feature_dict.items():
        proportion = count/len(examples)
        entropy -= proportion * math.log(proportion, 2)
    return entropy
