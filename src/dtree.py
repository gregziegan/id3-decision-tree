import mldata as md
import random
import argparse
from tree_generation import DecisionTree, print_tree
from feature_selection import test_all_feature_values

def main(problem_name, max_depth=0):
    example_set = md.parse_c45(problem_name, '../data')
    random.shuffle(example_set)
    training_set = example_set[:4 * len(example_set)/5]
    validation_set = example_set[4 * len(example_set)/5:]
    feature_indices = [i for i in range(1, len(example_set.schema.features[1:-1]))]
    dtree = DecisionTree(training_set, example_set.schema, feature_indices, max_depth=max_depth)
    for example in validation_set:
        classify(dtree, example)
    #for child in dtree.root.get_children():
    #    print child
    print_tree(dtree)


def get_accuracy(tree, examples):
    actual_class_label = {True: 0, False: 0}
    predicted_class_label = {True: 0, False: 0}
    for example in examples:
        classification = classify(tree, example)
        actual_class_label[example[-1]] += 1
        predicted_class_label[classification] += 1



def classify(tree, example):
    node = tree.root
    while node.label is None:
        print node.label
        print node.feature_values
        child_index = test_all_feature_values(example, node)
        node = node.get_children()[child_index]
        print("Assigned label: {}\tActual Class label: {}\n".format(node.label, example[-1]))
    return node.label


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='An implementation of the ID3 algorithm.')
    parser.add_argument('problem_name', metavar='P', help='The name of the problem set to process.')
    parser.add_argument('--max-depth', type=int, help='Limit the tree to a certain depth.')
    #TODO check if depth > 0
    args = parser.parse_args()
    max_depth = args.max_depth if args.max_depth else 0
    main(args.problem_name, max_depth)