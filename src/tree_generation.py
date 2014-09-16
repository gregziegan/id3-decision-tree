import utils
from feature_selection import get_best_feature, generate_feature_test


class DecisionTreeNode(object):

    def __init__(self, feature_test=None, parent=None, label=None):
        if feature_test and label:
            raise Exception("Node cannot contain a feature test and a label.")

        self.feature_test = feature_test
        self.parent = parent
        self._children = []
        self.label = label

    def get_children(self):
        return self._children

    def add_child(self, new_child):
        self._children.append(new_child)


class DecisionTree(object):

    def __init__(self, examples, target_feature, features, max_depth):
        self.root = self._generate_tree(examples, target_feature, features, 0)
        self.max_depth = max_depth

    def _generate_tree(self, examples, target_feature, features, depth, parent=None):
        root = DecisionTreeNode(parent=parent)
        parent.add_child(root)

        if not examples:
            raise Exception("No examples provided. ID3 failed.")

        if depth == self.max_depth:
            leaf = DecisionTreeNode(parent=root, label=utils.most_common_value(examples))
            root.add_child(leaf)
            return root


        if utils.is_homogeneous(examples, positive=True): # test if all examples for class label are positive
            root.label = True
            return root
        elif utils.is_homogeneous(examples, positive=False):
            root.label = False
            return root

        if not features:
            root.label = utils.most_common_value(examples)
            return root

        best_feature = get_best_feature(examples, features)
        for feature_value in best_feature.values():
            feature_test = generate_feature_test(feature_value)
            node = DecisionTreeNode(feature_test=feature_test, parent=root)
            examples_matching_feature_value = utils.subset(examples, best_feature, feature_value)
            if not examples_matching_feature_value:
                leaf = DecisionTreeNode(parent=node, label=utils.most_common_value(examples))
                node.add_child(leaf)
            else:
                features_without_best_classifier = [feature for feature in features if feature != best_feature]
                self._generate_tree(
                    examples=examples_matching_feature_value,
                    target_feature=target_feature,
                    features=features_without_best_classifier,
                   depth=depth + 1,
                    parent=node,
                )

        return root