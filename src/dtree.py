import mldata as md
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
    accuracy = dtree.get_accuracy(validation_set)
    print "Accuracy: {}".format(accuracy)
    tree_size, tree_depth = dtree.get_size_and_depth()
    print "Size: {}".format(tree_size)
    print "Maximum Depth: {}".format(tree_depth)
    print_tree(dtree)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='An implementation of the ID3 algorithm.')
    parser.add_argument('problem_name', metavar='P', help='The name of the problem set to process.')
    parser.add_argument('--max-depth', type=int, help='Limit the tree to a certain depth.')
    #TODO check if depth > 0
    args = parser.parse_args()
    max_depth = args.max_depth if args.max_depth else 0
    main(args.problem_name, max_depth)