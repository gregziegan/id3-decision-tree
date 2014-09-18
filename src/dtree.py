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
    accuracy = get_accuracy(dtree, validation_set)
    print "Accuracy: {}".format(accuracy)
    tree_size, tree_depth = get_tree_size_and_depth(dtree.root)
    print "Size: {}".format(tree_size)
    print "Maximum Depth: {}".format(tree_depth)
    print_tree(dtree)


def get_accuracy(tree, examples):
    accuracy = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
    for example in examples:
        predicted_label = classify(tree, example)
        actual_label = example[-1]
        if predicted_label and actual_label:
            accuracy['TP'] += 1
        elif predicted_label and not actual_label:
            accuracy['FP'] += 1
        elif not predicted_label and actual_label:
            accuracy['FN'] += 1
        else:
            accuracy['TN'] += 1
    final_accuracy = (float(accuracy['TP'] + accuracy['TN'])) / (float(accuracy['TP'] + accuracy['FP'] + accuracy['FN'] + accuracy['TN']))
    return final_accuracy


def get_tree_size_and_depth(node):
    tree_size = 0
    max_depth = -1
    parents = [node]
    while parents:
        for n in parents:
            new_parents = []
            for child in n.get_children():
                if(child.depth > max_depth):
                    max_depth = child.depth
                tree_size += 1
                new_parents.append(child)
            parents = new_parents
    return tree_size, max_depth


def classify(tree, example):
    node = tree.root
    while node.label is None:
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