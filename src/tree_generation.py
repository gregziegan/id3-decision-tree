# coding: utf-8
import datetime
import pydot

from feature_selection import get_best_feature_index, generate_feature_test
import utils


class DecisionTreeNode(object):
    def __init__(self, feature_test=None, feature_index=None, label=None):
        if feature_test and label:
            raise Exception("Node cannot contain a feature test and a label.")

        self.feature_index = feature_index
        self.feature_test = feature_test
        self._children = []
        self.label = label

    def get_children(self):
        return self._children

    def add_child(self, new_child):
        self._children.append(new_child)


class DecisionTree(object):

    def __init__(self, examples, features, max_depth):
        self.max_depth = max_depth
        self.root = self._generate_tree(examples, features, 0)

    def _generate_tree(self, examples, feature_indices, depth):
        """
        Generates a Decision Tree using the ID3 algorithm.

        :param examples: numpy array of `Example`s
        :param feature_indices:
       :param depth:
        :param parent:
       :return:
        """
        root = DecisionTreeNode()

        if len(examples) == 0:
            raise Exception("No examples provided. ID3 failed.")

        if self.max_depth > 0 and depth == self.max_depth:
            leaf = DecisionTreeNode(label=utils.most_common_value(examples))
            root.add_child(leaf)
            return root

        if utils.is_homogeneous(examples, positive=True): # test if all examples for class label are positive
            root.label = True
            return root
        elif utils.is_homogeneous(examples, positive=False):
            root.label = False
            return root

        if not feature_indices:
            root.label = utils.most_common_value(examples)
            return root

        feature_index = get_best_feature_index(examples, feature_indices)
        for feature_value in examples[0].schema[feature_index].tup[2]:
            feature_test = generate_feature_test(feature_value)
            node = DecisionTreeNode(feature_test=feature_test, feature_index=feature_index, parent=root)
            examples_matching_feature_value = utils.subset(examples, feature_index, feature_value)
            if not examples_matching_feature_value:
                leaf = DecisionTreeNode(feature_index=feature_index, label=utils.most_common_value(examples))
                node.add_child(leaf)
            else:
                features_without_best_classifier = [index for index in feature_indices if index != feature_index]
                self._generate_tree(
                    examples=examples_matching_feature_value,
                    feature_indices=features_without_best_classifier,
                    depth=depth + 1
                )

        return root


def print_tree(decision_tree):
    graph = pydot.Dot(graph_type='graph# the idea here is not to cover how to represent the hierarchical data')
    graph.write_png('decision_tree_{}.png'.format(str(datetime.datetime.now())))
