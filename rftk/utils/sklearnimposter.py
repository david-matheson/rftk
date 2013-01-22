import numpy as np

import rftk.utils.bootstrap as bootstrap_utils
import rftk.utils.predict as predict_utils
import rftk.leafstats.histogram_leaf_stats as leaf_stats
import rftk.feature_candidates.axisaligned as feature_candidates_axisaligned
import rftk.feature_candidates.collection as feature_candidates_collection
import rftk.best_split_factory.class_entropy_best_split_factory as class_entropy_best_split_factory
import rftk.forest_builder_offline.node_splitter as node_splitter_module
import rftk.forest_builder_offline.joblib_forest_builder as default_forest_builder
import rftk.forest_builder_offline.depth_first_tree_builder as default_tree_builder
import rftk.stop_criteria.criteria as stop_criteria

import rftk.native.predict as predict


class RandomForestClassifier:
    def __init__(self, max_features, n_estimators, max_depth, min_samples_split,
                        number_of_jobs=1,
                        forest_builder=default_forest_builder,
                        tree_builder=default_tree_builder):
        self.max_features = max_features
        self.n_estimators = n_estimators


        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.number_of_jobs = number_of_jobs
        self.forest_builder = forest_builder
        self.tree_builder = tree_builder

    def fit(self, x, y):
        (x_m,x_n) = x.shape
        y_len = len(y)
        assert(x_m == y_len)

        axis_aligned_feature_candidates = feature_candidates_axisaligned.AxisAlignedFeatureCandidates(number_of_candidates=self.max_features, x_dim=x_n)
        fc_collection = feature_candidates_collection.FeatureCandidateCollection([axis_aligned_feature_candidates])
        split_factory = class_entropy_best_split_factory.ClassEntropyBestSplitFactory(ratio_of_thresholds_to_test=1.0, min_thresholds_to_test=1, max_class=int(np.max(y)))
        stop_criteria_list = [  stop_criteria.DepthCriteria(max_depth=self.max_depth),
                                stop_criteria.MinSamples(min_samples=self.min_samples_split),
                                stop_criteria.ImpurityGain(min_impurity_gain=0.001)]

        node_splitter_init_params = node_splitter_module.NodeSplitterInitParams(feature_candidate_collection=fc_collection,
                                                                                best_splitter_factory=split_factory,
                                                                                stop_criteria_list=stop_criteria_list)

        builder = self.forest_builder.ForestBuilder(tree_builder_module=self.tree_builder,
                                                weight_sampler=bootstrap_utils.BootstapSampler(x_m),
                                                leaf_stats_factory=leaf_stats.HistogramLeafStatsFactory(np.amax(y) + 1),
                                                node_splitter_init_params=node_splitter_init_params,
                                                number_of_trees=self.n_estimators,
                                                number_of_jobs=self.number_of_jobs)
        self.tree_list = builder.train_forest(data=x, indices=np.arange(x_m), ys=y)

        predict_forest = predict_utils.as_predict_forest(self.tree_list)
        self.predict_forest = predict.ForestPredictor(predict_forest)

    def predict_class(self, x):
        yhat = self.predict(x)
        return yhat.argmax(axis=1)

    def predict(self, x):
        yhat = predict_utils.vec_predict_ys(self.predict_forest, x)
        return yhat

