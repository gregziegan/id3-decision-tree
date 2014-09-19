import math
import utils
from _collections import defaultdict


def get_best_threshold(examples):
    """
    Partitioning algorithm for CONTINUOUS data sets
    :param examples: list of examples
    :param class_label: label to determine partition
    :return: test value
    """

    pass


def test_all_feature_values(example, node, is_nominal):

    if is_nominal:
        for feature_value_index in range(len(node.feature_values)):
            if example[node.feature_index] == node.feature_values[feature_value_index]:
                return feature_value_index
    else:
        if example[node.feature_index] <= node.feature_values[0]:
            return 0
        else:
            return 1

    raise Exception("Example does not match any feature value.")


def get_continuous_feature_values(examples, feature_index):
    unique_continuous_values = []
    for example in sorted(examples, key=lambda e: e[feature_index]):
        current_class_label = example[-1]
        current_feature_value = example[feature_index]
        if not unique_continuous_values:
            unique_continuous_values.append((current_feature_value, [current_class_label]))
        if current_feature_value != unique_continuous_values[-1][0]:
            unique_continuous_values.append((current_feature_value, [current_class_label]))
        elif current_class_label not in unique_continuous_values[-1][1]:
            unique_continuous_values[-1][1].append(current_class_label)

    feature_values = []
    for i in range(len(unique_continuous_values) - 1):
        current_labels = set(unique_continuous_values[i][1])
        next_labels = set(unique_continuous_values[i+1][1])
        if has_partition_point(current_labels, next_labels):
            feature_values.append((unique_continuous_values[i+1][0] + unique_continuous_values[i][0]) / 2)

    return feature_values


def has_partition_point(labels1, labels2):
    for l1 in labels1:
        for l2 in labels2:
            if l1 != l2:
                return True
    return False




def get_best_feature_index_and_value(examples, schema, feature_indices):
    """

    Gets the index of the feature with the highest gain ratio.

    :param examples: list of Examples
    :param feature_indices: the indices of features on each Example's Schema
    :return: int index of the best feature
    """
    gain_ratios = [-1 for i in range(len(schema))]
    for index in feature_indices:
        entropy_of_feature_set = None
        is_nominal = schema[index].type == 'NOMINAL'
        if is_nominal:
            entropy_of_feature_set = get_entropy(examples, schema, index, is_nominal)
            gain_ratios[index] = get_gain_ratio(examples, schema, index, entropy_of_feature_set, is_nominal)
        else:
            continuous_feature_values = get_continuous_feature_values(examples, index)
            continuous_gain_ratios = []
            for feature_value in continuous_feature_values:
                entropy_of_feature_set = get_entropy(examples, feature_value, index, is_nominal)
                continuous_gain_ratios.append(get_gain_ratio(examples, feature_value, index, entropy_of_feature_set, is_nominal))
            max_continuous_index, max_continuous_value = max(enumerate(continuous_gain_ratios), key=lambda p: p[1])
            gain_ratios[index] = max_continuous_value
    max_index, max_value = max(enumerate(gain_ratios), key=lambda p: p[1])

    feature_values = None
    feature_type = schema[max_index].type  # NOMINAL or CONTINUOUS
    test_threshold = None
    if not feature_type == 'NOMINAL':
        continuous_feature_values = get_continuous_feature_values(examples, max_index)
        test_threshold = continuous_feature_values[max_continuous_index]

    return max_index, test_threshold


def get_gain_ratio(examples, schema, feature_index, entropy, is_nominal):
    """

    :param examples: list of Examples
    :param feature_index: the index of the feature on each Example's Schema
    @type entropy: float
    :return: gain ratio
    """
    if entropy == 0:
        return 1
    gain = get_information_gain(examples, feature_index, schema, entropy, is_nominal) / entropy
    return gain/entropy


def get_information_gain(examples, feature_index, schema, entropy_of_set, is_nominal):
    """

    :param examples: list of examples
    @type entropy_of_set: float
    :return: float
    """
    gain = entropy_of_set
    feature = utils.get_example_values_for_feature(examples, schema, feature_index, is_nominal)
    feature_counts = feature
    for feature_value, count in feature_counts.items():
        if count == 0:
            continue
        gain -= count/len(examples) * get_entropy(utils.subset(examples, feature_index, feature_value, is_nominal), schema, feature_index, is_nominal)
    return gain


def get_entropy(examples, schema, feature_index, is_nominal):
    """

    Calculates the entropy
    :param examples: list of examples
    :param feature_index: the feature to check the proportion of compared to the entire set
    :return:
    """
    entropy = 0
    feature_counts = utils.get_example_values_for_feature(examples, schema, feature_index, is_nominal)
    for count in feature_counts.values():
        if count == 0:
           continue
        proportion = float(count)/float(len(examples))
        #print 'count: {}, num_of_ex: {}, proportion: {}'.format(count, len(examples), proportion)
        entropy -= proportion * math.log(proportion, 2)
    return entropy
