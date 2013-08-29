import unittest as unittest
import numpy as np

import load_data 

import rftk



class TestNew(unittest.TestCase):

    def test_vanilia_classifier(self):
        learner = rftk.learn.create_vanilia_classifier()
        x = np.array([[3,1],[3,2], [3,3], [0,1], [0,2]], dtype=np.float32)
        classes = np.array([0,0,0,1,2], dtype=np.int32)
        predictor = learner.fit(x=x, classes=classes, bootstrap=False, number_of_features=2)
        result = predictor.predict(x=x).argmax(axis=1)
        self.assertEqual(result[0], 0)
        self.assertEqual(result[1], 0)
        self.assertEqual(result[2], 0)
        self.assertEqual(result[3], 1)
        self.assertEqual(result[4], 2)

    def test_vanilia_classifier_breadth_first(self):
        learner = rftk.learn.create_vanilia_classifier()
        x = np.array([[3,1],[3,2], [3,3], [0,1], [0,2]], dtype=np.float32)
        classes = np.array([0,0,0,1,2], dtype=np.int32)
        predictor = learner.fit(x=x, classes=classes, bootstrap=False, tree_order='breadth_first', number_of_features=2)
        result = predictor.predict(x=x).argmax(axis=1)
        self.assertEqual(result[0], 0)
        self.assertEqual(result[1], 0)
        self.assertEqual(result[2], 0)
        self.assertEqual(result[3], 1)
        self.assertEqual(result[4], 2)

    def test_one_stream_classifier(self):
        learner = rftk.learn.create_one_stream_classifier()
        x = np.array([[3,1],[3,2], [3,3], [0,1], [0,2]], dtype=np.float32)
        classes = np.array([0,0,0,1,2], dtype=np.int32)
        predictor = learner.fit(x=x, classes=classes,
                                bootstrap=False,
                                number_of_features=2,
                                number_of_splitpoints=10,
                                min_child_size=1)
        result = predictor.predict(x=x).argmax(axis=1)
        self.assertEqual(result[0], 0)
        self.assertEqual(result[1], 0)
        self.assertEqual(result[2], 0)
        self.assertEqual(result[3], 1)
        self.assertEqual(result[4], 2)

    def test_two_stream_classifier_balanced(self):
        learner = rftk.learn.create_two_stream_classifier()
        x = np.array([[3,1],[3,2], [3,3], [0,1], [0,1], [0,2], [0,2]], dtype=np.float32)
        classes = np.array([0,0,0,1,1,2,2], dtype=np.int32)
        predictor = learner.fit(x=x, classes=classes,
                                bootstrap=False,
                                number_of_trees=100,
                                number_of_features=2,
                                number_of_splitpoints=10,
                                probability_of_impurity_stream=0.5,
                                min_child_size=1)
        result_prob = predictor.predict(x=x)
        result = predictor.predict(x=x).argmax(axis=1)
        self.assertEqual(result[0], 0)
        self.assertEqual(result[1], 0)
        self.assertEqual(result[2], 0)
        self.assertEqual(result[3], 1)
        self.assertEqual(result[4], 1)
        self.assertEqual(result[5], 2)
        self.assertEqual(result[6], 2)

    def test_standard_regression(self):
        learner = rftk.learn.create_standard_regression()
        x = np.array([[3,1],[3,2], [3,3], [0,1], [0.5,1.5], [1,1.8], [1.5,2]], dtype=np.float32)
        y = np.array([[0,0],[0,0],[0,0],[1,3],[1,3],[2,-1],[2,-1]], dtype=np.float32)
        predictor = learner.fit(x=x, y=y, bootstrap=False, number_of_features=2, number_of_trees=10, number_of_leaves=10000000)
        result = predictor.predict(x=x)
        self.assertEqual(result[0,0], 0)
        self.assertEqual(result[0,1], 0)
        self.assertEqual(result[1,0], 0)
        self.assertEqual(result[1,1], 0)
        self.assertEqual(result[2,0], 0)
        self.assertEqual(result[2,1], 0)
        self.assertEqual(result[3,0], 1)
        self.assertEqual(result[3,1], 3)
        self.assertEqual(result[4,0], 1)
        self.assertEqual(result[4,1], 3)
        self.assertEqual(result[5,0], 2)
        self.assertEqual(result[5,1], -1)
        self.assertEqual(result[5,0], 2)
        self.assertEqual(result[5,1], -1)

    def test_ecoli_classifiers(self):
        x_train, y_train, x_test, y_test = load_data.load_ecoli_data()
        learner = rftk.learn.create_vanilia_classifier()
        predictor = learner.fit(x=x_train, classes=y_train, bootstrap=False, number_of_trees=10, number_of_jobs=5)
        y_hat = predictor.predict(x=x_test).argmax(axis=1)
        acc = np.mean(y_test == y_hat)
        self.assertGreater(acc, 0.75)

        forest = predictor.get_forest()
        tree = forest.GetTree(0)
        extraInfo = tree.mExtraInfo
        extraInfo.Print()

    def test_wine_regression(self):
        x_train, y_train, x_test, y_test = load_data.load_wine_data()
        learner = rftk.learn.create_standard_regression()
        predictor = learner.fit(x=x_train, y=y_train, bootstrap=False, number_of_trees=10, number_of_jobs=5)
        y_hat = predictor.predict(x=x_test)
        mse = np.mean((y_test - y_hat)**2)
        print mse
        self.assertLess(mse, 0.5)

        forest = predictor.get_forest()
        tree = forest.GetTree(0)
        extraInfo = tree.mExtraInfo
        extraInfo.Print()

    # def test_extra_info(self):
    #     learner = rftk.learn.create_vanilia_classifier()
    #     x = np.array(np.random.rand(1000,10), dtype=np.float32)
    #     classes = np.array(np.random.randint(5,size=(1000)), dtype=np.int32)
    #     predictor = learner.fit(x=x, classes=classes, bootstrap=False, number_of_features=3, number_of_trees=5, number_of_jobs=5)
    #     forest = predictor.get_forest()
    #     for i in range(forest.GetNumberOfTrees()):
    #         print('\n\nTree %d' % i)
    #         tree = forest.GetTree(i)
    #         extraInfo = tree.mExtraInfo
    #         extraInfo.Print()


if __name__ == '__main__':
    unittest.main()
