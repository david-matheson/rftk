import unittest as unittest
import numpy as np

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

    def test_one_stream_classifier(self):
        learner = rftk.learn.create_one_stream_classifier()
        x = np.array([[3,1],[3,2], [3,3], [0,1], [0,2]], dtype=np.float32)
        classes = np.array([0,0,0,1,2], dtype=np.int32)
        predictor = learner.fit(x=x, classes=classes,
                                bootstrap=False,
                                number_of_features=2,
                                number_of_splitpoints=10)
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
                                probability_of_impurity_stream=0.5)
        result_prob = predictor.predict(x=x)
        result = predictor.predict(x=x).argmax(axis=1)
        self.assertEqual(result[0], 0)
        self.assertEqual(result[1], 0)
        self.assertEqual(result[2], 0)
        self.assertEqual(result[3], 1)
        self.assertEqual(result[4], 1)
        self.assertEqual(result[5], 2)
        self.assertEqual(result[6], 2)

if __name__ == '__main__':
    unittest.main()
