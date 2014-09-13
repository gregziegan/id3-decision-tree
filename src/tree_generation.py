import utils

class DecisionTreeNode(object):

    def __init__(self, test, parent=None):
        self.test = test
        self.parent = parent


class DecisionTree(object):

    def __init__(self, dataset):
        self._generate_tree(dataset)

    def _generate_tree(self, dataset):
        pass


    @staticmethod
    def create_node(examples, parent=None):
        best_feature = utils.get_best_feature(examples)
        feature_test = utils.generate_feature_test(best_feature)
        node = DecisionTreeNode(test=feature_test, parent=parent)
        return node
    
