import unittest as unittest
import numpy as np

import rftk
import rftk.buffers as buffers
import rftk.bootstrap as bootstrap
import rftk.pipeline as pipeline
import rftk.matrix_features as matrix_features
import rftk.classification as classification
import rftk.try_split as try_split
import rftk.splitpoints as splitpoints
import rftk.should_split as should_split
import rftk.forest_data as forest_data
import rftk.learn as learn
import rftk.predict as predict

class TestNew(unittest.TestCase):

    def test_vanilia_classifier(self):
        learner = rftk.learn.create_vanilia_classifier()
        x = np.array([[3,1],[3,2], [3,3], [0,1], [0,2]], dtype=np.float32)
        classes = np.array([0,0,0,1,2], dtype=np.int32)
        predictor = learner.fit(x=x, classes=classes, number_of_features=2, bootstrap=False)
        result = predictor.predict(x=x).argmax(axis=1)
        self.assertEqual(result[0], 0)
        self.assertEqual(result[1], 0)
        self.assertEqual(result[2], 0)
        self.assertEqual(result[3], 1)
        self.assertEqual(result[4], 2)

if __name__ == '__main__':
    unittest.main()
