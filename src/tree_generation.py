import utils

class DecisionTreeNode(object):

    def __init__(self, feature_test=None, parent=None, label=None):
        if feature_test and label:
            raise Exception("Node cannot contain a feature test and a label.")

        self.test = feature_test
        self.parent = parent
        self.label = label


class DecisionTree(object):

    def __init__(self, dataset):
        self.root = self._generate_tree(dataset)

    def _generate_tree(self, examples, target_feature, features):
        root = DecisionTreeNode()

        if utils.is_homogenous(examples, test='+'):
            root.label = '+'
            return root
        elif utils.is_homogenous(examples, test='-'):
            root.label = '-'
            return root

        if not features:
            root.label = utils.most_common_value(target_feature, features)
            return root

        best_feature = utils.get_best_feature(examples)
        feature_test = utils.generate_feature_test(best_feature)
        for value in best_feature.values():
            node = DecisionTreeNode(feature_test=feature_test, parent=root)
            examples_with_this_value = utils.subset(best_feature, value)
            if not examples_with_this_value:
                leaf = DecisionTreeNode(parent=node, label=utils.most_common_value(best_feature, features))
            else:
                features_without_best_classifier = [f for f in features if f != best_feature]
                self._generate_tree(examples_with_this_value, target_feature, features_without_best_classifier)

        return root