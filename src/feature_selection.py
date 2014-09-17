import math
import utils


def generate_feature_test(feature_value):
    """
    Returns a function closure that will evaluate an example based on its Feature Type:
        Type can be either NOMINAL or CONTINUOUS
    :param feature_value:
    """
    test_function = None
    if feature_value.type == 'NOMINAL':
        test_function = lambda example: True if example == feature_value else False
    else:
        pass
        #TODO
        #test_function = lambda example:
    return test_function


def get_best_feature(examples, features):
    gain_ratios = [get_gain_ratio(examples, feature, get_entropy(examples, feature)) for feature in features]
    max_index, max_value = max(enumerate(gain_ratios), key=lambda p: p[1])
    return max_value


def get_gain_ratio(examples, feature, entropy):
    gain = get_information_gain(examples, feature, entropy) / entropy
    return gain/entropy


def get_information_gain(examples, feature, entropy_of_set):
    gain = entropy_of_set
    class_dict = utils.get_class_label_values(examples)
    for class_label, count in class_dict.items():
        gain -= count/len(examples) * get_entropy(class_dict, len(examples))
    return gain


def get_entropy(examples, feature):
    entropy = 0
    feature_dict = utils.get_example_values_for_feature(examples, feature)
    for feature, count in feature_dict.items():
        proportion = count/len(examples)
        entropy -= proportion * math.log(proportion, 2)
    return entropy
