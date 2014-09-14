from src.mldata import *
from src.utils import *

import logging
logging.basicConfig(filename='testing.log', level=logging.DEBUG)

example_data = None


def setup_module():
    global example_data
    example_data =parse_c45("voting")


def test_get_feature_values_from_example():
    t = get_example_values_for_feature(example_data, 1)
    print(t)


def test_get_class_label_values():
    t = get_class_label_values(example_data)
    print(t)
