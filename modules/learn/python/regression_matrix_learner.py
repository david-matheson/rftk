import numpy as np

import rftk.buffers as buffers
import rftk.pipeline as pipeline
import rftk.matrix_features as matrix_features
import rftk.splitpoints as splitpoints
import rftk.regression as regression
import rftk.predict as predict
import learn
from wrappers import *
from split_criteria import *

def matrix_regression_data_prepare(**kwargs):
    bufferCollection = buffers.BufferCollection()
    bufferCollection.AddBuffer(buffers.X_FLOAT_DATA, buffers.as_matrix_buffer(kwargs['x']))
    if 'y' in kwargs:
        bufferCollection.AddBuffer(buffers.YS, buffers.as_matrix_buffer(kwargs['y']))
    return bufferCollection

def create_matrix_regression_predictor_32f(forest, **kwargs):
    dimension_of_y = forest.GetTree(0).mYs.GetN() / 2
    all_samples_step = pipeline.AllSamplesStep_f32f32i32(buffers.X_FLOAT_DATA)
    combiner = regression.MeanVarianceCombiner_f32(dimension_of_y)
    matrix_feature = matrix_features.LinearFloat32MatrixFeature_f32i32(all_samples_step.IndicesBufferId,
                                                                        buffers.X_FLOAT_DATA)
    forest_predicter = predict.LinearMatrixRegressionPredictin_f32i32(forest, matrix_feature, combiner, all_samples_step)
    return PredictorWrapper_32f(forest_predicter, matrix_regression_data_prepare)

def create_regression_axis_aligned_matrix_learner_32f(**kwargs):
    number_of_trees = int( kwargs.get('number_of_trees', 10) )
    number_of_leaves = int( kwargs.get('number_of_leaves', kwargs['y'].shape[0] / 5 + 1) )
    number_of_features = int( kwargs.get('number_of_features', (kwargs['x'].shape[1])/3 + 0.5))
    # number_of_features = int( kwargs.get('number_of_features', np.sqrt(kwargs['x'].shape[1])))
    feature_ordering = int( kwargs.get('feature_ordering', pipeline.FEATURES_BY_DATAPOINTS) )
    number_of_jobs = int( kwargs.get('number_of_jobs', 1) )
    dimension_of_y = int( kwargs['y'].shape[1] )

    try_split_criteria = create_try_split_criteria(**kwargs)

    if 'bootstrap' in kwargs and kwargs.get('bootstrap'):
        sample_data_step = pipeline.BootstrapSamplesStep_f32f32i32(buffers.X_FLOAT_DATA)
    else:
        sample_data_step = pipeline.AllSamplesStep_f32f32i32(buffers.X_FLOAT_DATA)

    number_of_features_buffer = buffers.as_vector_buffer(np.array([number_of_features], dtype=np.int32))
    set_number_features_step = pipeline.SetInt32VectorBufferStep(number_of_features_buffer, pipeline.WHEN_NEW)
    tree_steps_pipeline = pipeline.Pipeline([sample_data_step, set_number_features_step])

    feature_params_step = matrix_features.AxisAlignedParamsStep_f32i32(set_number_features_step.OutputBufferId, buffers.X_FLOAT_DATA)
    matrix_feature = matrix_features.LinearFloat32MatrixFeature_f32i32(feature_params_step.FloatParamsBufferId,
                                                                      feature_params_step.IntParamsBufferId,
                                                                      sample_data_step.IndicesBufferId,
                                                                      buffers.X_FLOAT_DATA)
    matrix_feature_extractor_step = matrix_features.LinearFloat32MatrixFeatureExtractorStep_f32i32(matrix_feature, feature_ordering)
    slice_ys_step = pipeline.SliceFloat32MatrixBufferStep_i32(buffers.YS, sample_data_step.IndicesBufferId)
    slice_weights_step = pipeline.SliceFloat32VectorBufferStep_i32(sample_data_step.WeightsBufferId, sample_data_step.IndicesBufferId)
    impurity_walker = regression.SumOfVarianceWalker_f32i32(slice_weights_step.SlicedBufferId,
                                                            slice_ys_step.SlicedBufferId,
                                                            dimension_of_y)
    best_splitpint_step = regression.SumOfVarianceBestSplitpointsWalkingSortedStep_f32i32(impurity_walker,
                                                                        matrix_feature_extractor_step.FeatureValuesBufferId,
                                                                        feature_ordering)
    node_steps_pipeline = pipeline.Pipeline([feature_params_step, matrix_feature_extractor_step,
                                            slice_ys_step, slice_weights_step, best_splitpint_step])

    split_buffers = splitpoints.SplitSelectorBuffers(best_splitpint_step.ImpurityBufferId,
                                                          best_splitpint_step.SplitpointBufferId,
                                                          best_splitpint_step.SplitpointCountsBufferId,
                                                          best_splitpint_step.ChildCountsBufferId,
                                                          best_splitpint_step.LeftYsBufferId,
                                                          best_splitpint_step.RightYsBufferId,
                                                          feature_params_step.FloatParamsBufferId,
                                                          feature_params_step.IntParamsBufferId,
                                                          matrix_feature_extractor_step.FeatureValuesBufferId,
                                                          feature_ordering,
                                                          matrix_feature_extractor_step)
    should_split_criteria = create_should_split_criteria(**kwargs)
    finalizer = regression.MeanVarianceEstimatorFinalizer_f32()
    split_indices = splitpoints.SplitIndices_f32i32(sample_data_step.IndicesBufferId)
    split_selector = splitpoints.SplitSelector_f32i32([split_buffers], should_split_criteria, finalizer, split_indices )

    tree_learner = learn.BreadthFirstTreeLearner_f32i32(try_split_criteria, tree_steps_pipeline, node_steps_pipeline, split_selector, number_of_leaves)
    forest_learner = learn.ParallelForestLearner(tree_learner, number_of_trees, 5, 5, dimension_of_y*2, number_of_jobs)
    return forest_learner


def create_biau2008_regression_axis_aligned_matrix_learner_32f(**kwargs):
    number_of_trees = int( kwargs.get('number_of_trees', 10) )
    number_of_leaves = int( kwargs.get('number_of_leaves', kwargs['y'].shape[0] / 5 + 1) )
    number_of_split_retries = int( kwargs.get('number_of_split_retries', 10) )
    number_of_features = 1
    feature_ordering = int( kwargs.get('feature_ordering', pipeline.FEATURES_BY_DATAPOINTS) )
    number_of_jobs = int( kwargs.get('number_of_jobs', 1) )
    dimension_of_y = int( kwargs['y'].shape[1] )

    try_split_criteria = try_split.MinNodeSizeCriteria(2)

    sample_data_step = pipeline.AllSamplesStep_f32f32i32(buffers.X_FLOAT_DATA)

    number_of_features_buffer = buffers.as_vector_buffer(np.array([number_of_features], dtype=np.int32))
    set_number_features_step = pipeline.SetInt32VectorBufferStep(number_of_features_buffer, pipeline.WHEN_NEW)
    tree_steps_pipeline = pipeline.Pipeline([sample_data_step, set_number_features_step])

    feature_params_step = matrix_features.AxisAlignedParamsStep_f32i32(set_number_features_step.OutputBufferId, buffers.X_FLOAT_DATA)
    matrix_feature = matrix_features.LinearFloat32MatrixFeature_f32i32(feature_params_step.FloatParamsBufferId,
                                                                      feature_params_step.IntParamsBufferId,
                                                                      sample_data_step.IndicesBufferId,
                                                                      buffers.X_FLOAT_DATA)
    matrix_feature_extractor_step = matrix_features.LinearFloat32MatrixFeatureExtractorStep_f32i32(matrix_feature, feature_ordering)
    slice_ys_step = pipeline.SliceFloat32MatrixBufferStep_i32(buffers.YS, sample_data_step.IndicesBufferId)
    slice_weights_step = pipeline.SliceFloat32VectorBufferStep_i32(sample_data_step.WeightsBufferId, sample_data_step.IndicesBufferId)

    impurity_walker = regression.SumOfVarianceWalker_f32i32(slice_weights_step.SlicedBufferId,
                                                            slice_ys_step.SlicedBufferId,
                                                            dimension_of_y)

    best_splitpint_step = regression.SumOfVarianceRandomGapSplitpointsStep_f32i32(impurity_walker,
                                                                matrix_feature_extractor_step.FeatureValuesBufferId,
                                                                feature_ordering)

    node_steps_pipeline = pipeline.Pipeline([feature_params_step, matrix_feature_extractor_step,
                                            slice_ys_step, slice_weights_step, best_splitpint_step])

    split_buffers = splitpoints.SplitSelectorBuffers(best_splitpint_step.ImpurityBufferId,
                                                          best_splitpint_step.SplitpointBufferId,
                                                          best_splitpint_step.SplitpointCountsBufferId,
                                                          best_splitpint_step.ChildCountsBufferId,
                                                          best_splitpint_step.LeftYsBufferId,
                                                          best_splitpint_step.RightYsBufferId,
                                                          feature_params_step.FloatParamsBufferId,
                                                          feature_params_step.IntParamsBufferId,
                                                          matrix_feature_extractor_step.FeatureValuesBufferId,
                                                          feature_ordering,
                                                          matrix_feature_extractor_step)
    should_split_criteria = min_child_size_criteria = should_split.MinChildSizeCriteria(1)
    finalizer = regression.MeanVarianceEstimatorFinalizer_f32()
    split_indices = splitpoints.SplitIndices_f32i32(sample_data_step.IndicesBufferId)
    split_selector = splitpoints.SplitSelector_f32i32([split_buffers], should_split_criteria, finalizer, split_indices )

    tree_learner = learn.Biau2008TreeLearner_f32i32(try_split_criteria, tree_steps_pipeline, node_steps_pipeline, split_selector, number_of_leaves, number_of_split_retries)
    forest_learner = learn.ParallelForestLearner(tree_learner, number_of_trees, 5, 5, dimension_of_y*2, number_of_jobs)
    return forest_learner


def create_biau2012_regression_axis_aligned_matrix_learner_32f(**kwargs):
    number_of_trees = int( kwargs.get('number_of_trees', 10) )
    number_of_leaves = int( kwargs.get('number_of_leaves', kwargs['y'].shape[0] / 5 + 1) )
    number_of_features = int( kwargs.get('number_of_features', (kwargs['x'].shape[1])/3 + 0.5))
    # number_of_features = int( kwargs.get('number_of_features', np.sqrt(kwargs['x'].shape[1])))
    feature_ordering = int( kwargs.get('feature_ordering', pipeline.FEATURES_BY_DATAPOINTS) )
    number_of_jobs = int( kwargs.get('number_of_jobs', 1) )
    dimension_of_y = int(  kwargs['y'].shape[1] )
    probability_of_impurity_stream = float(kwargs.get('probability_of_impurity_stream', 0.5) )

    try_split_criteria = create_try_split_criteria(**kwargs)

    sample_data_step = pipeline.AllSamplesStep_f32f32i32(buffers.X_FLOAT_DATA)

    number_of_features_buffer = buffers.as_vector_buffer(np.array([number_of_features], dtype=np.int32))
    set_number_features_step = pipeline.SetInt32VectorBufferStep(number_of_features_buffer, pipeline.WHEN_NEW)
    assert(np.max(np.abs(kwargs['x'])) <= 1.00) # double check that the data has been scaled into a -1,1 hypercube
    feature_range_buffer = buffers.as_vector_buffer(np.array([-1, 1], dtype=np.float32))
    set_feature_range_buffer_step = pipeline.SetFloat32VectorBufferStep(feature_range_buffer, pipeline.WHEN_NEW)

    assign_stream_step = splitpoints.AssignStreamStep_f32i32(sample_data_step.WeightsBufferId, probability_of_impurity_stream, False)
    forest_steps_pipeline = pipeline.Pipeline([sample_data_step, set_number_features_step, set_feature_range_buffer_step, assign_stream_step])
    tree_steps_pipeline = pipeline.Pipeline([])

    feature_params_step = matrix_features.AxisAlignedParamsStep_f32i32(set_number_features_step.OutputBufferId, buffers.X_FLOAT_DATA)
    matrix_feature = matrix_features.LinearFloat32MatrixFeature_f32i32(feature_params_step.FloatParamsBufferId,
                                                                      feature_params_step.IntParamsBufferId,
                                                                      sample_data_step.IndicesBufferId,
                                                                      buffers.X_FLOAT_DATA)
    matrix_feature_extractor_step = matrix_features.LinearFloat32MatrixFeatureExtractorStep_f32i32(matrix_feature, feature_ordering)
    slice_ys_step = pipeline.SliceFloat32MatrixBufferStep_i32(buffers.YS, sample_data_step.IndicesBufferId)
    slice_weights_step = pipeline.SliceFloat32VectorBufferStep_i32(sample_data_step.WeightsBufferId, sample_data_step.IndicesBufferId)
    slice_assign_stream_step = pipeline.SliceInt32VectorBufferStep_i32(assign_stream_step.StreamTypeBufferId, sample_data_step.IndicesBufferId)


    quantized_feature_equal = pipeline.FeatureEqualQuantized_f32i32(1.0)

    midpoint_step = splitpoints.RangeMidpointStep_f32i32(feature_params_step.FloatParamsBufferId,
                                                        feature_params_step.IntParamsBufferId,
                                                        set_feature_range_buffer_step.OutputBufferId,
                                                        quantized_feature_equal)

    mean_variance_stats_updater = regression.MeanVarianceStatsUpdater_f32i32(slice_weights_step.SlicedBufferId,
                                                                              slice_ys_step.SlicedBufferId,
                                                                              dimension_of_y)

    two_stream_split_stats_step = regression.SumOfVarianceTwoStreamStep_f32i32(midpoint_step.SplitpointsBufferId,
                                                                          midpoint_step.SplitpointsCountsBufferId,
                                                                          slice_assign_stream_step.SlicedBufferId,
                                                                          matrix_feature_extractor_step.FeatureValuesBufferId,
                                                                          feature_ordering,
                                                                          mean_variance_stats_updater)


    impurity_step = regression.SumOfVarianceSplitpointsImpurity_f32i32(midpoint_step.SplitpointsCountsBufferId,
                                                                          two_stream_split_stats_step.ChildCountsImpurityBufferId,
                                                                          two_stream_split_stats_step.LeftImpurityStatsBufferId,
                                                                          two_stream_split_stats_step.RightImpurityStatsBufferId)

    node_steps_pipeline = pipeline.Pipeline([feature_params_step, matrix_feature_extractor_step,
                                            slice_ys_step, slice_weights_step, slice_assign_stream_step,
                                            midpoint_step, two_stream_split_stats_step, impurity_step])

    split_buffers = splitpoints.SplitSelectorBuffers(impurity_step.ImpurityBufferId,
                                                          midpoint_step.SplitpointsBufferId,
                                                          midpoint_step.SplitpointsCountsBufferId,
                                                          two_stream_split_stats_step.ChildCountsEstimatorBufferId,
                                                          two_stream_split_stats_step.LeftEstimatorStatsBufferId,
                                                          two_stream_split_stats_step.RightEstimatorStatsBufferId,
                                                          feature_params_step.FloatParamsBufferId,
                                                          feature_params_step.IntParamsBufferId,
                                                          matrix_feature_extractor_step.FeatureValuesBufferId,
                                                          feature_ordering,
                                                          matrix_feature_extractor_step)
    should_split_criteria = no_split_criteria(**kwargs)
    finalizer = regression.MeanVarianceEstimatorFinalizer_f32()
    split_indices = splitpoints.SplitIndices_f32i32(sample_data_step.IndicesBufferId)
    split_midpoint_ranges = splitpoints.SplitBuffersFeatureRange_f32i32(midpoint_step.PastFloatParamsBufferId,
                                                                        midpoint_step.PastIntParamsBufferId,
                                                                        midpoint_step.PastRangesBufferId,
                                                                        set_feature_range_buffer_step.OutputBufferId,
                                                                        quantized_feature_equal)
    split_steps = splitpoints.SplitBuffersList([split_indices, split_midpoint_ranges])

    split_selector = splitpoints.SplitSelector_f32i32([split_buffers],
                                                        should_split_criteria,
                                                        finalizer,
                                                        split_steps)

    tree_learner = learn.BreadthFirstTreeLearner_f32i32(try_split_criteria, tree_steps_pipeline, node_steps_pipeline, split_selector, number_of_leaves)
    forest_learner = learn.ParallelForestLearner(tree_learner, forest_steps_pipeline, number_of_trees, 5, 5, dimension_of_y*2, number_of_jobs)
    return forest_learner

def create_consistent_two_stream_regression_axis_aligned_matrix_learner_32f(**kwargs):
    number_of_trees = int( kwargs.get('number_of_trees', 10) )
    number_of_leaves = int( kwargs.get('number_of_leaves', kwargs['y'].shape[0] / 5 + 1) )
    number_of_features = int( kwargs.get('number_of_features', (kwargs['x'].shape[1])/3 + 0.5))
    # number_of_features = int( kwargs.get('number_of_features', np.sqrt(kwargs['x'].shape[1])))
    feature_ordering = int( kwargs.get('feature_ordering', pipeline.FEATURES_BY_DATAPOINTS) )
    number_of_jobs = int( kwargs.get('number_of_jobs', 1) )
    dimension_of_y = int(  kwargs['y'].shape[1] )

    probability_of_impurity_stream = float(kwargs.get('probability_of_impurity_stream', 0.5) )
    in_bounds_number_of_points = int(kwargs.get('in_bounds_number_of_points', kwargs['y'].shape[0]/2) )

    try_split_criteria = create_try_split_criteria(**kwargs)

    if 'bootstrap' in kwargs and kwargs.get('bootstrap'):
        sample_data_step = pipeline.BootstrapSamplesStep_f32f32i32(buffers.X_FLOAT_DATA)
    else:
        sample_data_step = pipeline.AllSamplesStep_f32f32i32(buffers.X_FLOAT_DATA)

    assign_stream_step = splitpoints.AssignStreamStep_f32i32(sample_data_step.WeightsBufferId, probability_of_impurity_stream)
    tree_steps_pipeline = pipeline.Pipeline([sample_data_step, assign_stream_step])

    if 'poisson_number_of_features' in kwargs and kwargs.get('poisson_number_of_features'):
        set_number_features_step = pipeline.PoissonStep_f32i32(number_of_features, 1)
    else:
        number_of_features_buffer = buffers.as_vector_buffer(np.array([number_of_features], dtype=np.int32))
        set_number_features_step = pipeline.SetInt32VectorBufferStep(number_of_features_buffer, pipeline.WHEN_NEW)

    feature_params_step = matrix_features.AxisAlignedParamsStep_f32i32(set_number_features_step.OutputBufferId, buffers.X_FLOAT_DATA)
    matrix_feature = matrix_features.LinearFloat32MatrixFeature_f32i32(feature_params_step.FloatParamsBufferId,
                                                                      feature_params_step.IntParamsBufferId,
                                                                      sample_data_step.IndicesBufferId,
                                                                      buffers.X_FLOAT_DATA)
    matrix_feature_extractor_step = matrix_features.LinearFloat32MatrixFeatureExtractorStep_f32i32(matrix_feature, feature_ordering)

    slice_ys_step = pipeline.SliceFloat32MatrixBufferStep_i32(buffers.YS, sample_data_step.IndicesBufferId)
    slice_weights_step = pipeline.SliceFloat32VectorBufferStep_i32(sample_data_step.WeightsBufferId, sample_data_step.IndicesBufferId)
    slice_stream_step = pipeline.SliceInt32VectorBufferStep_i32(assign_stream_step.StreamTypeBufferId, sample_data_step.IndicesBufferId)

    impurity_walker = regression.SumOfVarianceTwoStreamWalker_f32i32(slice_weights_step.SlicedBufferId,
                                                            slice_stream_step.SlicedBufferId,
                                                            slice_ys_step.SlicedBufferId,
                                                            dimension_of_y)

    best_splitpint_step = regression.SumOfVarianceTwoStreamBestSplitpointsWalkingSortedStep_f32i32(impurity_walker,
                                                                        slice_stream_step.SlicedBufferId,
                                                                        matrix_feature_extractor_step.FeatureValuesBufferId,
                                                                        feature_ordering,
                                                                        in_bounds_number_of_points)

    node_steps_pipeline = pipeline.Pipeline([set_number_features_step, feature_params_step, matrix_feature_extractor_step,
                                            slice_ys_step, slice_weights_step, slice_stream_step, best_splitpint_step])

    split_buffers = splitpoints.SplitSelectorBuffers(best_splitpint_step.ImpurityBufferId,
                                                          best_splitpint_step.SplitpointBufferId,
                                                          best_splitpint_step.SplitpointCountsBufferId,
                                                          best_splitpint_step.ChildCountsEstimationBufferId,
                                                          best_splitpint_step.LeftEstimationYsBufferId,
                                                          best_splitpint_step.RightEstimationYsBufferId,
                                                          feature_params_step.FloatParamsBufferId,
                                                          feature_params_step.IntParamsBufferId,
                                                          matrix_feature_extractor_step.FeatureValuesBufferId,
                                                          feature_ordering,
                                                          matrix_feature_extractor_step)
    should_split_criteria = create_should_split_criteria(**kwargs)
    finalizer = regression.MeanVarianceEstimatorFinalizer_f32()
    split_indices = splitpoints.SplitIndices_f32i32(sample_data_step.IndicesBufferId)
    split_selector = splitpoints.SplitSelector_f32i32([split_buffers], should_split_criteria, finalizer, split_indices )

    tree_learner = learn.BreadthFirstTreeLearner_f32i32(try_split_criteria, tree_steps_pipeline, node_steps_pipeline, split_selector, number_of_leaves)
    forest_learner = learn.ParallelForestLearner(tree_learner, number_of_trees, 5, 5, dimension_of_y*2, number_of_jobs)
    return forest_learner


def create_standard_regression(**kwargs):
    return LearnerWrapper(  matrix_regression_data_prepare,
                            create_regression_axis_aligned_matrix_learner_32f,
                            create_matrix_regression_predictor_32f,
                            kwargs)

def create_biau2008_regression(**kwargs):
    return LearnerWrapper(  matrix_regression_data_prepare,
                            create_biau2008_regression_axis_aligned_matrix_learner_32f,
                            create_matrix_regression_predictor_32f,
                            kwargs)

def create_biau2012_regression(**kwargs):
    return LearnerWrapper(  matrix_regression_data_prepare,
                            create_biau2012_regression_axis_aligned_matrix_learner_32f,
                            create_matrix_regression_predictor_32f,
                            kwargs)

def create_consistent_regression(**kwargs):
    return LearnerWrapper(  matrix_regression_data_prepare,
                            create_consistent_two_stream_regression_axis_aligned_matrix_learner_32f,
                            create_matrix_regression_predictor_32f,
                            kwargs)