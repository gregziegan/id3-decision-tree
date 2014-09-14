import math
import utils


def get_information_gain(examples, feature, entropy_of_set):
    gain = entropy_of_set
    class_dict = utils.get_cl
    for class_label, count in class_dict.items():
        gain -= count/len(examples) * get_entropy(feature_dict, len(examples))
    return gain


def get_entropy(example_count):
    entropy = 0
    feature_dict = utils.get_example_values_for_feature(examples, feature)
    for feature, count in feature_dict.items():
        proportion = count/example_count
        entropy -= proportion * math.log(proportion, 2)
    return entropy
