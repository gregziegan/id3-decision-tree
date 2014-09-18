import math
import utils


def get_feature_threshold(examples, class_label):
    sorted_by_x = sorted(examples)
    pass


def test_all_feature_values(example, node):
    for feature_value_index in range(len(node.feature_values)):
        if example[node.feature_index] == node.feature_values[feature_value_index]:
            return feature_value_index

    raise Exception("Example does not match any feature value.")

"""
def generate_feature_test(feature_values, feature_type):
    test_function = None
    if feature_type == 'NOMINAL':
        test_function = test_all_feature_values()
    else:
        pass
        #TODO
        #test_function = lambda example:
    return test_function
"""

def get_best_feature_index(examples, schema, feature_indices):
    """

    Gets the index of the feature with the highest gain ratio.

    :param examples: numpy array of Examples
    :param feature_indices: the indices of features on each Example's Schema
    :return: int index of the best feature
    """
    gain_ratios = [-1 for i in range(len(schema))]
    for index in feature_indices:
        entropy_of_feature_set = get_entropy(examples, schema, index)
        gain_ratios[index] = get_gain_ratio(examples, schema, index, entropy_of_feature_set)
    max_index, max_value = max(enumerate(gain_ratios), key=lambda p: p[1])
    return max_index


def get_gain_ratio(examples, schema, feature_index, entropy):
    """

    :param examples: list of Examples
    :param feature_index: the index of the feature on each Example's Schema
    @type entropy: float
    :return: gain ratio
    """
    if entropy == 0:
        return 1
    gain = get_information_gain(examples, feature_index, schema, entropy) / entropy
    return gain/entropy


def get_information_gain(examples, feature_index, schema, entropy_of_set):
    """

    :param examples: list of examples
    @type entropy_of_set: float
    :return: float
    """
    gain = entropy_of_set
    feature_counts = utils.get_example_values_for_feature(examples, schema, feature_index)
    for feature_value, count in feature_counts.items():
        if count == 0:
            continue
        gain -= count/len(examples) * get_entropy(utils.subset(examples, feature_index, feature_value), schema, feature_index)
    return gain


def get_entropy(examples, schema, feature_index):
    """

    Calculates the entropy
    :param examples: list of examples
    :param feature_index: the feature to check the proportion of compared to the entire set
    :return:
    """
    entropy = 0
    feature_counts = utils.get_example_values_for_feature(examples, schema, feature_index)
    for count in feature_counts.values():
        #if count == 0:
        #    continue
        proportion = float(count)/float(len(examples))
        #print 'count: {}, num_of_ex: {}, proportion: {}'.format(count, len(examples), proportion)
        entropy -= proportion * math.log(proportion, 2)
    return entropy
