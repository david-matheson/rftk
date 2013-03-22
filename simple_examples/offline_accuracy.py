import numpy as np
import argparse
import cPickle as pickle
from datetime import datetime
import sklearn.ensemble

import dist_utils

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Sklearn accuracy on data')
    parser.add_argument('-i', '--in_folder', help='train and test data file', required=True)
    parser.add_argument('-n', '--number_of_trees', default=25, type=int, help='number of epochs')
    parser.add_argument('-o', '--out_folder', help='train and test data file', required=True)
    args = parser.parse_args()

    accuracy_file = open("%s/accuracies.pkl" % (args.in_folder), "rb")
    (epoch_id, total_sample_list, forest_accuracy, tree_accuracy, bayes_accuracy) = pickle.load(accuracy_file)

    data_file = open("%s/data.pkl" % (args.in_folder), "rb")
    (X_train, Y_train, X_test, Y_test) = pickle.load(data_file)


    print datetime.now()

    useSklearn = True
    max_features = 1
    max_depth = 15
    min_samples_split = 5
    number_random_thresholds = 10
    number_of_jobs = 1

    index = 0
    forest_accuracy = np.zeros(len(total_sample_list))
    for sample_count in total_sample_list:
        print datetime.now()
        print "Fitting sample count %d" % sample_count
        X_train_sample = X_train[0:sample_count, :]
        Y_train_sample = Y_train[0:sample_count]


        if useSklearn:
            forest = sklearn.ensemble.RandomForestClassifier(   criterion="entropy",
                                                                max_features=max_features,
                                                                n_estimators=args.number_of_trees,
                                                                max_depth=max_depth,
                                                                min_samples_split=min_samples_split,
                                                                n_jobs=number_of_jobs)
            forest.fit(X_train_sample, Y_train_sample)
        else:
            (x_m,x_n) = X_train_sample.shape
            assert(x_m == len(Y_train_sample))

            feature_extractor = feature_extractors.RandomProjectionFeatureExtractor( max_features, x_n, x_n)
            # feature_extractor = feature_extractors.AxisAlignedFeatureExtractor( max_features, x_n)
            # node_data_collector = train.AllNodeDataCollectorFactory()
            # class_infogain_best_split = best_splits.ClassInfoGainAllThresholdsBestSplit(1.0, 1, int(np.max(Y_train_sample)) + 1)
            node_data_collector = train.RandomThresholdHistogramDataCollectorFactory(int(np.max(Y_train_sample)) + 1, number_random_thresholds, 0)
            class_infogain_best_split = best_splits.ClassInfoGainHistogramsBestSplit(int(np.max(Y_train_sample)) + 1,
                buffers.HISTOGRAM_LEFT, buffers.HISTOGRAM_RIGHT, buffers.HISTOGRAM_LEFT, buffers.HISTOGRAM_RIGHT)
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
            data.AddFloat32MatrixBuffer(buffers.X_FLOAT_DATA, buffers.as_matrix_buffer(X_train_sample))
            data.AddInt32VectorBuffer(buffers.CLASS_LABELS, buffers.Int32Vector(Y_train_sample))
            sampling_config = train.OfflineSamplingParams(x_m, True)
            indices = buffers.Int32Vector( np.array(np.arange(x_m), dtype=np.int32) )
            full_forest_data = depth_first_learner.Train(data, indices, sampling_config, number_of_jobs)
            forest = predict.MatrixForestPredictor(full_forest_data)

        y_probs = forest.predict_proba(X_test)
        y_hat = y_probs.argmax(axis=1)
        accuracy = np.mean(Y_test == y_hat)
        print "    Accuracy:", accuracy
        forest_accuracy[index] = accuracy
        index += 1
        pickle.dump(forest_accuracy, open("%s/offline-accuracies.pkl" % (args.out_folder), "wb"))





