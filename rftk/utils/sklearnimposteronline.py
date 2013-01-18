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

class OnlineRandomForestClassifier:
    def __init__(self, max_features, n_estimators, max_depth, min_samples_split, y_dim, x_dim, min_impurity=0.001, n_jobs=1):
        self.number_of_jobs = n_jobs

        self.feature_extractor = feature_extractors.AxisAlignedFeatureExtractor( max_features, x_dim)
        self.node_data_collector = train.AllNodeDataCollectorFactory()
        self.class_infogain_best_split = best_splits.ClassInfoGainAllThresholdsBestSplit(1.0, 1, y_dim)
        self.split_criteria = train.OnlineAlphaBetaSplitCriteria(    max_depth,
                                                                min_impurity,
                                                                min_samples_split)
        self.list_test = [self.feature_extractor]
        self.train_config = train.TrainConfigParams(self.list_test,
                                                self.node_data_collector,
                                                self.class_infogain_best_split,
                                                self.split_criteria,
                                                n_estimators,
                                                10000)
        self.sampling_config = train.OnlineSamplingParams(False, 1.0)
        self.online_learner = train.OnlineForestLearner(self.train_config)


    def fit(self, x, y):
        (x_m,x_n) = x.shape
        y_len = len(y)
        assert(x_m == y_len)

        data = buffers.BufferCollection()
        data.AddMatrixBufferFloat(buffers.X_FLOAT_DATA, buffer_converters.as_matrix_buffer(x))
        data.AddMatrixBufferInt(buffers.CLASS_LABELS, buffer_converters.as_matrix_buffer(y))

        sampling_config = train.OfflineSamplingParams(x_m, True)
        indices = buffer_converters.as_matrix_buffer( np.arange(x_m) )

        self.online_learner.Train(data, indices, self.sampling_config)
        self.vec_predict_forest = predict.VecForestPredictor(self.online_learner.GetForest())

    def predict_class(self, x):
        yhat = self.predict(x)
        return yhat.argmax(axis=1)

    def predict(self, x):
        yhat = predict_utils.vec_predict_ys(self.vec_predict_forest, x)
        return yhat

