import mldata as md
import numpy as np
import random
import argparse
from tree_generation import DecisionTree, print_tree


def main(problem_name, max_depth=0):
    example_set = md.parse_c45(problem_name, '../data')
    random.shuffle(example_set)
    training_set = example_set[:4 * len(example_set)/5]
    validation_set = example_set[4 * len(example_set)/5:]
    feature_indices = [i for i in range(1, len(example_set.schema.features[1:-1]))]
    dtree = DecisionTree(training_set, example_set.schema, feature_indices, max_depth=max_depth)
    #for example in validation_set:
    #    classify(dtree, example)
    #for child in dtree.root.get_children():
    #    print child
    print_tree(dtree)

def classify(tree, example):
    node = tree.root
    while not node.label:
        for child in node.get_children():
            print(child.feature_index)
            if child.feature_test(example[node.feature_index]):
                node = child
    print("Assigned label: {}\tActual Class label: {}\n".format(node.label, example[-1]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='An implementation of the ID3 algorithm.')
    parser.add_argument('problem_name', metavar='P', help='The name of the problem set to process.')
    parser.add_argument('--max-depth', type=int, help='Limit the tree to a certain depth.')
    #TODO check if depth > 0
    args = parser.parse_args()
    max_depth = args.max_depth if args.max_depth else 0
    main(args.problem_name, max_depth)