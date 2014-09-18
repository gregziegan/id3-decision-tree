from src.feature_selection import get_information_gain, get_entropy, generate_feature_test, get_best_feature_index
from src import mldata as md

example_set = None

def setup_module():
    global example_set
    example_set = md.parse_c45("voting")


def test_info_gain():
    info_gain = get_information_gain(example_set[:len(example_set)/5], example_set.schema, .5)
    assert type(info_gain) == float


def test_entropy():
    entropy = get_entropy(example_set[:len(example_set)/5], example_set.schema, 1)
    assert type(entropy) == float
