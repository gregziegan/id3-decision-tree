import math
import utils


def info_gain(examples, feature, entropy_of_set):
    gain = entropy_of_set
    for feature, count in utils.get_example_values_for_feature(examples, feature).items():
        gain -= count_/len(examples) * get_entropy(sub)
    return gain


def get_entropy(examples, target_feature):
    """

    :param examples: list of example data
    :param target_feature: index of a feature in the schema
    :return: entropy value
    """
    result = 0
    target_examples = summarize_examples(examples, target_feature)
    for example in target_examples:
        proportion = example/len(examples)
        result -= proportion * math.log(proportion, 2)
    return result


def summarize_examples(examples, target_attribute):
    pass