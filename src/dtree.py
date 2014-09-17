import mldata as md
import numpy as np
import argparse
from tree_generation import DecisionTree


def main(problem_name, max_depth=0):
    example_set = md.parse_c45(problem_name, '../data')
    data_array = np.array(example_set.to_float())
    np.random.shuffle(data_array)
    training_set = data_array[:4 * len(data_array)/5]
    validation_set = data_array[4 * len(data_array)/5:]
    dtree = DecisionTree(training_set, example_set.schema, max_depth=max_depth)
    for example in validation_set:
        classify(dtree, example)
    print dtree.root


def classify(tree, example):
    node = tree.root
    while not node.label:
        for child in node.get_children():
            if child.feature_test(example[node.feature_index]):
                node = child
    print(node.label)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='An implementation of the ID3 algorithm.')
    parser.add_argument('problem_name', metavar='P', help='The name of the problem set to process.')
    parser.add_argument('--max-depth', type=int, help='Limit the tree to a certain depth.')
    #TODO check if depth > 0
    args = parser.parse_args()
    max_depth = args.max_depth if args.max_depth else 0
    main(args.problem_name, max_depth)