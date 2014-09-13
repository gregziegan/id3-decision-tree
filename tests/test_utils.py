from src.mldata import *
from src.utils import *

import logging
logging.basicConfig(filename='testing.log', level=logging.DEBUG)

example_data = None


def setup_module():
    global example_data
    example_data =parse_c45("voting")


def test_get_feature_values_from_example():
    logging.debug(get_example_values_for_feature())
