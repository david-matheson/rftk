import numpy as np

from datetime import datetime

import rftk.native.assert_util
import rftk.native.bootstrap
import rftk.native.buffers as buffers
import rftk.native.forest_data  as forest_data
import rftk.native.feature_extractors as feature_extractors
import rftk.native.best_split as best_splits
import rftk.native.predict as predict
import rftk.native.train as train

import rftk.utils.buffer_converters as buffer_converters
import rftk.utils.predict as predict_utils

import plot_utils
import dist_utils


if __name__ == "__main__":
    dist = dist_utils.mog_2d_3class_example2()
    n_per = 10000

    X_test,Y_test = dist.sample(n_per)

    print datetime.now()

    # Configure
    max_features=1
    number_of_trees=25
    max_depth=15
    min_impurity=0.001
    min_samples_split=10
    (_,x_dim) = X_test.shape
    y_dim = 3
    number_of_jobs = 1
    # feature_extractor = feature_extractors.RandomProjectionFeatureExtractor( max_features, x_dim, x_dim, True)
    feature_extractor = feature_extractors.AxisAlignedFeatureExtractor( max_features, x_dim, True)
    # node_data_collector = train.AllNodeDataCollectorFactory()
    # class_infogain_best_split = best_splits.ClassInfoGainAllThresholdsBestSplit(1.0, 1, y_dim)
    node_data_collector = train.RandomThresholdHistogramDataCollectorFactory(y_dim, 10, 0.7)
    class_infogain_best_split = best_splits.ClassInfoGainHistogramsBestSplit(y_dim)
    # self.split_criteria = train.OnlineAlphaBetaSplitCriteria(   max_depth,
    #                                                             min_impurity,
    #                                                             min_samples_split)
    split_criteria = train.OnlineConsistentSplitCriteria(  2.0,
                                                                min_impurity,
                                                                min_samples_split,
                                                                10 * min_samples_split)

    extractor_list = [feature_extractor]
    train_config = train.TrainConfigParams(extractor_list,
                                            node_data_collector,
                                            class_infogain_best_split,
                                            split_criteria,
                                            number_of_trees,
                                            10000)
    sampling_config = train.OnlineSamplingParams(False, 1.0)
    online_learner = train.OnlineForestLearner(train_config)

    for epoch_id in range(1,2000):
        print "Fitting epoch %d" % (epoch_id)
        epoch_per = epoch_id * epoch_id
        X_train,Y_train = dist.sample(epoch_per)
        (x_m,_) = X_train.shape

        data = buffers.BufferCollection()
        data.AddMatrixBufferFloat(buffers.X_FLOAT_DATA, buffer_converters.as_matrix_buffer(X_train))
        data.AddMatrixBufferInt(buffers.CLASS_LABELS, buffer_converters.as_matrix_buffer(Y_train))
        indices = buffer_converters.as_matrix_buffer( np.arange(x_m) )
        online_learner.Train(data, indices, sampling_config)
        predict_forest = predict_utils.MatrixForestPredictor(online_learner.GetForest())

        print datetime.now()

        y_probs = predict_forest.predict_proba(X_test)
        y_hat = y_probs.argmax(axis=1)

        print "Synthetic (classification):"
        print "    Accuracy:", np.mean(Y_test == y_hat)

        plot_utils.grid_plot(predict_forest, X_train, Y_train, X_test, "output-all/synthetic_online_classification-%d.png" % (epoch_per))

        online_forest_data = online_learner.GetForest()

        for tree_id in range(online_forest_data.GetNumberOfTrees()):
            print tree_id
            single_tree_forest_data = forest_data.Forest([online_forest_data.GetTree(tree_id)])
            single_tree_forest_predictor = predict_utils.MatrixForestPredictor(single_tree_forest_data)
            plot_utils.grid_plot(single_tree_forest_predictor, X_train, Y_train, X_test, "output-trees/synthetic_online_classification-%d-%d.png" % (epoch_per, tree_id))