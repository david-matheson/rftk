import numpy as np

from datetime import datetime
import sklearn.ensemble

import rftk.native.assert_util
import rftk.native.bootstrap
import rftk.native.buffers as buffers
import rftk.native.forest_data as forest_data
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

    X_train,Y_train = dist.sample(n_per)
    X_test,Y_test = dist.sample(n_per)

    print datetime.now()

    useSklearn = False
    max_features = 1
    number_to_trees = 25
    max_depth = 15
    min_samples_split = 5
    number_random_thresholds = 10
    number_of_jobs = 1

    if useSklearn:
        forest = sklearn.ensemble.RandomForestClassifier(   criterion="entropy",
                                                            max_features=max_features,
                                                            n_estimators=number_to_trees,
                                                            max_depth=max_depth,
                                                            min_samples_split=min_samples_split,
                                                            n_jobs=number_of_jobs)
        forest.fit(X_train, Y_train)
    else:
        (x_m,x_n) = X_train.shape
        assert(x_m == len(Y_train))

        # feature_extractor = feature_extractors.RandomProjectionFeatureExtractor( max_features, x_n, x_n)
        feature_extractor = feature_extractors.AxisAlignedFeatureExtractor( max_features, x_n)
        # node_data_collector = train.AllNodeDataCollectorFactory()
        # class_infogain_best_split = best_splits.ClassInfoGainAllThresholdsBestSplit(1.0, 1, int(np.max(Y_train)) + 1)
        node_data_collector = train.RandomThresholdHistogramDataCollectorFactory(int(np.max(Y_train)) + 1, number_random_thresholds, 0)
        class_infogain_best_split = best_splits.ClassInfoGainHistogramsBestSplit(int(np.max(Y_train)) + 1)
        split_criteria = train.OfflineSplitCriteria( max_depth,
                                                    0.001,
                                                    min_samples_split,
                                                    1)
        extractor_list = [feature_extractor]
        train_config = train.TrainConfigParams(extractor_list,
                                                node_data_collector,
                                                class_infogain_best_split,
                                                split_criteria,
                                                number_to_trees,
                                                10000)
        depth_first_learner = train.DepthFirstParallelForestLearner(train_config)

        data = buffers.BufferCollection()
        data.AddMatrixBufferFloat(buffers.X_FLOAT_DATA, buffer_converters.as_matrix_buffer(X_train))
        data.AddMatrixBufferInt(buffers.CLASS_LABELS, buffer_converters.as_matrix_buffer(Y_train))
        sampling_config = train.OfflineSamplingParams(x_m, True)
        indices = buffer_converters.as_matrix_buffer( np.arange(x_m) )
        full_forest_data = depth_first_learner.Train(data, indices, sampling_config, number_of_jobs)
        forest = predict_utils.MatrixForestPredictor(full_forest_data)

    y_probs = forest.predict_proba(X_test)
    y_hat = y_probs.argmax(axis=1)

    print "Synthetic (classification):"
    print "    Accuracy:", np.mean(Y_test == y_hat)

    print datetime.now()

    plot_utils.grid_plot(forest, X_train, Y_train, X_test, "synthetic_classification.png")

    if not useSklearn:
        for tree_id in range(full_forest_data.GetNumberOfTrees()):
            print tree_id
            single_tree_forest_data = forest_data.Forest([full_forest_data.GetTree(tree_id)])
            single_tree_forest_predictor = predict_utils.MatrixForestPredictor(single_tree_forest_data)
            plot_utils.grid_plot(single_tree_forest_predictor, X_train, Y_train, X_test, "synthetic_classification-%d.png" % (tree_id))



