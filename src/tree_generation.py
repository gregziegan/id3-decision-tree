# coding: utf-8
import datetime
import pydot

from feature_selection import get_best_feature_index_and_value, test_all_feature_values, get_continuous_feature_values, get_best_threshold
import utils


class DecisionTreeNode(object):

    def __init__(self, feature_index=None, feature_values=None, label=None, parent=None, depth=None):
        if feature_index and label:
            raise Exception("Node cannot be a leaf and a decision node")

        self.feature_index = feature_index
        self.feature_values = feature_values
        self.parent = parent
        self._children = []
        self.label = label
        self.depth = depth

    def get_children(self):
        return self._children

    def add_child(self, new_child):
        self._children.append(new_child)

    @property
    def full_description(self):
        return '<Node: label: {}, feature_index: {}, feature_values: {}, parent: {}, children: {}>'.format(
            self.label, self.feature_index, self.feature_values, self.parent, self._children
        )

    def __repr__(self):
        if self.label != None:
            return "{}-{}".format(self.label, self.parent)
        else:
            parent = self.parent.feature_index if self.parent else 'null'
            return "{}-{}".format(self.feature_index, parent)


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

        print "\nDEPTH: {}\n".format(depth)
        root = None
        if depth == 0:
            root = DecisionTreeNode(depth=depth)
        else:
            root = node

        print "\n\n{}\n\n".format(root.full_description)

        if self.max_depth > 0 and depth == self.max_depth:
            print 'reached max_depth'
            root.label = utils.most_common_value(examples)
            return root

        if utils.is_homogeneous(examples, positive=True):  # test if all examples for class label are positive
            print 'all examples are class label: True'
            root.label = True
            return root
        elif utils.is_homogeneous(examples, positive=False):
            print 'all examples are class label: False'
            root.label = False
            return root

        if not feature_indices:
            print 'no feature indices'
            root.label = utils.most_common_value(examples)
            return root

        root.feature_index, root.feature_values = get_best_feature_index_and_value(examples, self.schema, feature_indices)
        print "feature_to_test: {}, index: {}".format(self.schema[root.feature_index].name, root.feature_index)
        for feature_value in root.feature_values:
            child = DecisionTreeNode(parent=root, depth=depth+1)
            root.add_child(child)
            examples_matching_feature_value = utils.subset(examples, root.feature_index, feature_value)
            print "# of examples matching '{}': {}".format(feature_value, len(examples_matching_feature_value))
            if not examples_matching_feature_value:
                print "no examples match '{}'. # of examples: {}".format(feature_value, len(examples_matching_feature_value))
                child.label=utils.most_common_value(examples)
            else:
                features_without_best_classifier = [index for index in feature_indices if index != root.feature_index]
                print "# of features without best classifier: {}".format(len(features_without_best_classifier))
                self._generate_tree(
                    examples=examples_matching_feature_value,
                    feature_indices=features_without_best_classifier,
                    depth=depth + 1,
                    node=child,
                )

        return root

    def classify(self, example):
        node = self.root
        while node.label is None:
            child_index = test_all_feature_values(example, node)
            node = node.get_children()[child_index]
        print("Assigned label: {}\tActual Class label: {}\n".format(node.label, example[-1]))
        return node.label

    def get_accuracy(self, examples):
        accuracy = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
        for example in examples:
            predicted_label = self.classify(example)
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

    def get_size_and_depth(self):
        tree_size = 0
        tree_depth = -1
        parents = [self.root]
        while parents:
            for n in parents:
                new_parents = []
                for child in n.get_children():
                    if(child.depth > tree_depth):
                        tree_depth = child.depth
                    tree_size += 1
                    new_parents.append(child)
                parents = new_parents
        return tree_size, tree_depth

    def __repr__(self):
        return self.root


def add_children(graph, node, schema):
    if not node.get_children():
        return

    for child in node.get_children():
        edge = pydot.Edge(str(child.parent), str(child))
        graph.add_edge(edge)
        add_children(graph, child, schema)


def print_tree(decision_tree):
    graph = pydot.Dot(graph_type='graph')
    add_children(graph, decision_tree.root, decision_tree.schema)
    graph.write_png('/tmp/decision_tree_{}.png'.format(str(datetime.datetime.now())))
