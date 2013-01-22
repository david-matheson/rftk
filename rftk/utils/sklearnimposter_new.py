import numpy as np

import rftk.native.assert_util
import rftk.native.bootstrap
import rftk.native.buffers as buffers
import rftk.native.forest_data
import rftk.native.feature_extractors as feature_extractors
import rftk.native.best_split as best_splits
import rftk.native.predict as predict
import rftk.native.train as train

import rftk.utils.buffer_converters as buffer_converters
import rftk.utils.predict as predict_utils

class RandomForestClassifier:
    def __init__(self, max_features, n_estimators, max_depth, min_samples_split, n_jobs=1):
        self.max_features = max_features
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.number_of_jobs = n_jobs


    def fit(self, x, y):
        (x_m,x_n) = x.shape
        y_len = len(y)
        assert(x_m == y_len)

        data = buffers.BufferCollection()
        data.AddMatrixBufferFloat(buffers.X_FLOAT_DATA, buffer_converters.as_matrix_buffer(x))
        data.AddMatrixBufferInt(buffers.CLASS_LABELS, buffer_converters.as_matrix_buffer(y))

        # feature_extractor = feature_extractors.RandomProjectionFeatureExtractor( self.max_features, x_n, x_n, True)
        # feature_extractor = feature_extractors.AxisAlignedFeatureExtractor( self.max_features, x_n)
        # node_data_collector = train.AllNodeDataCollectorFactory()
        # class_infogain_best_split = best_splits.ClassInfoGainAllThresholdsBestSplit(1.0, 1, int(np.max(y)) + 1)

        node_data_collector = train.RandomThresholdHistogramDataCollectorFactory(int(np.max(y)) + 1, 10, 20)
        class_infogain_best_split = best_splits.ClassInfoGainHistogramsBestSplit(int(np.max(y)) + 1)

        split_criteria = train.OfflineSplitCriteria( self.max_depth,
                                                    0.001,
                                                    self.min_samples_split,
                                                    1)
        list_test = [feature_extractor]
        train_config = train.TrainConfigParams(list_test,
                                                node_data_collector,
                                                class_infogain_best_split,
                                                split_criteria,
                                                self.n_estimators,
                                                10000)

        sampling_config = train.OfflineSamplingParams(x_m, True)
        indices = buffer_converters.as_matrix_buffer( np.arange(x_m) )

        depth_first_learner = train.DepthFirstParallelForestLearner(train_config)
        forest = depth_first_learner.Train(data, indices, sampling_config, self.number_of_jobs)
        self.vec_predict_forest = predict.VecForestPredictor(forest)

    def predict_class(self, x):
        yhat = self.predict(x)
        return yhat.argmax(axis=1)

    def predict(self, x):
        yhat = predict_utils.vec_predict_ys(self.vec_predict_forest, x)
        return yhat

