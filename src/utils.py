from collections import defaultdict

def generate_feature_test(best_feature):
    pass


def get_best_feature(examples):
    pass


def get_example_values_for_feature(examples, feature_index):

    """
    :param examples: Example data
    :param feature_index: The index you of the feature you want
    :return: A dict containing the possible feature values mapped to the count associated with each.
    """
    feature_values = defaultdict(int)

    for feature_value in examples.schema[feature_index].tup:
        feature_values[feature_value] = 0

    for i in range(0, len(examples)):
        example_value = examples[i][feature_index]
        if example_value in feature_values:
            feature_values[example_value] += 1
    return feature_values



