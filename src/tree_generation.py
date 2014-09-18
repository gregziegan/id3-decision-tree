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

    def __repr__(self):
        return '<DTNode: [label: {}, feature_index: {}]>'.format(self.label, self.feature_index)


class DecisionTree(object):

    def __init__(self, examples, schema, features, max_depth):
        self.max_depth = max_depth
        self.schema = schema
        self.root = self._generate_tree(examples, features, 0, None)

    def _generate_tree(self, examples, feature_indices, depth, node):
        """
        Generates a Decision Tree using the ID3 algorithm.

        :param examples: numpy array of `Example`s
        :param feature_indices:
        :param depth:
        :param parent:
        :return:
        """
        if len(examples) == 0:
            raise Exception("No examples provided. ID3 failed.")

        print "DEPTH: {}\n\n".format(depth)
        root = None
        if depth == 0:
            root = DecisionTreeNode()
        else:
            root = node

        if self.max_depth > 0 and depth == self.max_depth:
            print 'reached max_depth'
            leaf = DecisionTreeNode(label=utils.most_common_value(examples))
            root.add_child(leaf)
            return root

        if utils.is_homogeneous(examples, positive=True):  # test if all examples for class label are positive
            print 'is positive homogeneous'
            root.label = True
            return root
        elif utils.is_homogeneous(examples, positive=False):
            print 'is negative homogeneous'
            root.label = False
            return root

        if not feature_indices:
            print 'no feature indices'
            root.label = utils.most_common_value(examples)
            return root

        feature_index = get_best_feature_index(examples, self.schema, feature_indices)
        print "feature_index: {}".format(feature_index)
        for feature_value in self.schema[feature_index].values:
            print feature_value
            feature_type = self.schema[feature_index].type
            feature_test = generate_feature_test(feature_type, feature_value)
            child = DecisionTreeNode(feature_test=feature_test, feature_index=feature_index)
            root.add_child(child)
            examples_matching_feature_value = utils.subset(examples, feature_index, feature_value)
            print "examples matching feature value: {}".format(len(examples_matching_feature_value))
            if not examples_matching_feature_value:
                leaf = DecisionTreeNode(feature_index=feature_index, label=utils.most_common_value(examples))
                child.add_child(leaf)
            else:
                features_without_best_classifier = [index for index in feature_indices if index != feature_index]
                print "features without best classifier: {}".format(len(features_without_best_classifier))
                self._generate_tree(
                    examples=examples_matching_feature_value,
                    feature_indices=features_without_best_classifier,
                    depth=depth + 1,
                    node=child,
                )

        return root

    def __repr__(self):
        return self.root


def print_tree(decision_tree):
    graph = pydot.Dot(graph_type='graph# the idea here is not to cover how to represent the hierarchical data')
    graph.write_png('decision_tree_{}.png'.format(str(datetime.datetime.now())))
