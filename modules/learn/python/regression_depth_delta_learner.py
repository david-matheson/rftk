import numpy as np

import rftk.buffers as buffers
import rftk.pipeline as pipeline
import rftk.image_features as image_features
import rftk.splitpoints as splitpoints
import rftk.should_split as should_split
import rftk.regression as regression
import rftk.predict as predict
import learn
from wrappers import *
from split_criteria import *


def depth_delta_regression_data_prepare(**kwargs):
    bufferCollection = buffers.BufferCollection()
    bufferCollection.AddBuffer(buffers.DEPTH_IMAGES, kwargs['depth_images'])
    bufferCollection.AddBuffer(buffers.PIXEL_INDICES, kwargs['pixel_indices'])
    if 'offset_scales' in kwargs:
        bufferCollection.AddBuffer(buffers.OFFSET_SCALES, kwargs['offset_scales'])
    if 'y' in kwargs:
        bufferCollection.AddBuffer(buffers.YS, kwargs['y'])
    return bufferCollection

def create_regression_depth_delta_predictor_32f(forest, **kwargs):
    dimension_of_y = forest.GetTree(0).mYs.GetN() / 2
    all_samples_step = pipeline.AllSamplesStep_i32f32i32(buffers.PIXEL_INDICES)
    combiner = regression.MeanVarianceCombiner_f32(dimension_of_y)
    depth_delta_feature = image_features.ScaledDepthDeltaFeature_f32i32(all_samples_step.IndicesBufferId,
                                                                        buffers.PIXEL_INDICES,
                                                                        buffers.DEPTH_IMAGES)
    forest_predicter = predict.ScaledDepthDeltaRegressionPrediction_f32i32(forest, depth_delta_feature, combiner, all_samples_step)
    return PredictorWrapper_32f(forest_predicter, depth_delta_regression_data_prepare)


def create_regression_scaled_depth_delta_learner_32f(**kwargs):
    ux = float( kwargs.get('ux') )
    uy = float( kwargs.get('uy') )
    vx = float( kwargs.get('vx') )
    vy = float( kwargs.get('vy') )

    number_of_trees = int( kwargs.get('number_of_trees', 10) )
    number_of_features = int( kwargs.get('number_of_features', 1) )
    feature_ordering = int( kwargs.get('feature_ordering', pipeline.FEATURES_BY_DATAPOINTS) )
    number_of_jobs = int( kwargs.get('number_of_jobs', 1) )
    dimension_of_y = int( kwargs['y'].GetN() )

    try_split_criteria = create_try_split_criteria(**kwargs)

    if 'bootstrap' in kwargs and kwargs.get('bootstrap'):
        sample_data_step = pipeline.BootstrapSamplesStep_i32f32i32(buffers.PIXEL_INDICES)
    else:
        sample_data_step = pipeline.AllSamplesStep_i32f32i32(buffers.PIXEL_INDICES)

    number_of_features_buffer = buffers.as_vector_buffer(np.array([number_of_features], dtype=np.int32))
    set_number_features_step = pipeline.SetInt32VectorBufferStep(number_of_features_buffer, pipeline.WHEN_NEW)
    tree_steps_pipeline = pipeline.Pipeline([sample_data_step, set_number_features_step])

    feature_params_step = image_features.PixelPairGaussianOffsetsStep_f32i32(set_number_features_step.OutputBufferId, ux, uy, vx, vy )
    depth_delta_feature = image_features.ScaledDepthDeltaFeature_f32i32(feature_params_step.FloatParamsBufferId,
                                                                      feature_params_step.IntParamsBufferId,
                                                                      sample_data_step.IndicesBufferId,
                                                                      buffers.PIXEL_INDICES,
                                                                      buffers.DEPTH_IMAGES,
                                                                      buffers.OFFSET_SCALES)
    depth_delta_feature_extractor_step = image_features.ScaledDepthDeltaFeatureExtractorStep_f32i32(depth_delta_feature, feature_ordering)
    slice_ys_step = pipeline.SliceFloat32MatrixBufferStep_i32(buffers.YS, sample_data_step.IndicesBufferId)
    slice_weights_step = pipeline.SliceFloat32VectorBufferStep_i32(sample_data_step.WeightsBufferId, sample_data_step.IndicesBufferId)

    impurity_walker = regression.SumOfVarianceWalker_f32i32(slice_weights_step.SlicedBufferId,
                                                            slice_ys_step.SlicedBufferId,
                                                            dimension_of_y)

    best_splitpint_step = regression.SumOfVarianceBestSplitpointsWalkingSortedStep_f32i32(impurity_walker,
                                                                        depth_delta_feature_extractor_step.FeatureValuesBufferId,
                                                                        feature_ordering)

    node_steps_pipeline = pipeline.Pipeline([feature_params_step, depth_delta_feature_extractor_step,
                                            slice_ys_step, slice_weights_step, best_splitpint_step])

    split_buffers = splitpoints.SplitSelectorBuffers(best_splitpint_step.ImpurityBufferId,
                                                          best_splitpint_step.SplitpointBufferId,
                                                          best_splitpint_step.SplitpointCountsBufferId,
                                                          best_splitpint_step.ChildCountsBufferId,
                                                          best_splitpint_step.LeftYsBufferId,
                                                          best_splitpint_step.RightYsBufferId,
                                                          feature_params_step.FloatParamsBufferId,
                                                          feature_params_step.IntParamsBufferId,
                                                          depth_delta_feature_extractor_step.FeatureValuesBufferId,
                                                          feature_ordering,
                                                          sample_data_step.IndicesBufferId)
    should_split_criteria = create_should_split_criteria(**kwargs)
    finalizer = regression.MeanVarianceEstimatorFinalizer_f32()
    split_selector = splitpoints.SplitSelector_f32i32([split_buffers], should_split_criteria, finalizer )

    tree_learner = learn.DepthFirstTreeLearner_f32i32(try_split_criteria, tree_steps_pipeline, node_steps_pipeline, split_selector)
    forest_learner = learn.ParallelForestLearner(tree_learner, number_of_trees, 5, 5, dimension_of_y*2, number_of_jobs)
    return forest_learner

def create_consistent_two_stream_regression_scaled_depth_delta_learner_32f(**kwargs):
    ux = float( kwargs.get('ux') )
    uy = float( kwargs.get('uy') )
    vx = float( kwargs.get('vx') )
    vy = float( kwargs.get('vy') )

    number_of_trees = int( kwargs.get('number_of_trees', 10) )
    number_of_features = int( kwargs.get('number_of_features', 1) )
    feature_ordering = int( kwargs.get('feature_ordering', pipeline.FEATURES_BY_DATAPOINTS) )
    number_of_jobs = int( kwargs.get('number_of_jobs', 1) )
    dimension_of_y = int( kwargs['y'].GetN() )
    probability_of_impurity_stream = float(kwargs.get('probability_of_impurity_stream', 0.5) )
    in_bounds_sampling_rate = float(kwargs.get('in_bounds_sampling_rate', 0.9) )

    try_split_criteria = create_try_split_criteria(**kwargs)

    if 'bootstrap' in kwargs and kwargs.get('bootstrap'):
        sample_data_step = pipeline.BootstrapSamplesStep_i32f32i32(buffers.PIXEL_INDICES)
    else:
        sample_data_step = pipeline.AllSamplesStep_i32f32i32(buffers.PIXEL_INDICES)

    set_number_features_step = pipeline.PoissonStep_f32i32(number_of_features, 1)
    assign_stream_step = splitpoints.AssignStreamStep_f32i32(sample_data_step.IndicesBufferId, probability_of_impurity_stream)
    tree_steps_pipeline = pipeline.Pipeline([sample_data_step, set_number_features_step, assign_stream_step])

    feature_params_step = image_features.PixelPairGaussianOffsetsStep_f32i32(set_number_features_step.OutputBufferId, ux, uy, vx, vy )
    depth_delta_feature = image_features.ScaledDepthDeltaFeature_f32i32(feature_params_step.FloatParamsBufferId,
                                                                      feature_params_step.IntParamsBufferId,
                                                                      sample_data_step.IndicesBufferId,
                                                                      buffers.PIXEL_INDICES,
                                                                      buffers.DEPTH_IMAGES,
                                                                      buffers.OFFSET_SCALES)
    depth_delta_feature_extractor_step = image_features.ScaledDepthDeltaFeatureExtractorStep_f32i32(depth_delta_feature, feature_ordering)
    slice_ys_step = pipeline.SliceFloat32MatrixBufferStep_i32(buffers.YS, sample_data_step.IndicesBufferId)
    slice_weights_step = pipeline.SliceFloat32VectorBufferStep_i32(sample_data_step.WeightsBufferId, sample_data_step.IndicesBufferId)
    slice_stream_step = pipeline.SliceInt32VectorBufferStep_i32(assign_stream_step.StreamTypeBufferId, sample_data_step.IndicesBufferId)

    impurity_walker = regression.SumOfVarianceTwoStreamWalker_f32i32(slice_weights_step.SlicedBufferId,
                                                            slice_stream_step.SlicedBufferId,
                                                            slice_ys_step.SlicedBufferId,
                                                            dimension_of_y)

    best_splitpint_step = regression.SumOfVarianceTwoStreamBestSplitpointsWalkingSortedStep_f32i32(impurity_walker,
                                                                        slice_stream_step.SlicedBufferId,
                                                                        depth_delta_feature_extractor_step.FeatureValuesBufferId,
                                                                        feature_ordering,
                                                                        in_bounds_sampling_rate)

    node_steps_pipeline = pipeline.Pipeline([feature_params_step, depth_delta_feature_extractor_step,
                                            slice_ys_step, slice_weights_step, slice_stream_step, best_splitpint_step])

    split_buffers = splitpoints.SplitSelectorBuffers(best_splitpint_step.ImpurityBufferId,
                                                          best_splitpint_step.SplitpointBufferId,
                                                          best_splitpint_step.SplitpointCountsBufferId,
                                                          best_splitpint_step.ChildCountsEstimationBufferId,
                                                          best_splitpint_step.LeftEstimationYsBufferId,
                                                          best_splitpint_step.RightEstimationYsBufferId,
                                                          feature_params_step.FloatParamsBufferId,
                                                          feature_params_step.IntParamsBufferId,
                                                          depth_delta_feature_extractor_step.FeatureValuesBufferId,
                                                          feature_ordering,
                                                          sample_data_step.IndicesBufferId)
    should_split_criteria = create_should_split_criteria(**kwargs)
    finalizer = regression.MeanVarianceEstimatorFinalizer_f32()
    split_selector = splitpoints.SplitSelector_f32i32([split_buffers], should_split_criteria, finalizer )

    tree_learner = learn.DepthFirstTreeLearner_f32i32(try_split_criteria, tree_steps_pipeline, node_steps_pipeline, split_selector)
    forest_learner = learn.ParallelForestLearner(tree_learner, number_of_trees, 5, 5, dimension_of_y*2, number_of_jobs)
    return forest_learner

def create_vanilia_scaled_depth_delta_regression(**kwargs):
    return LearnerWrapper(  depth_delta_regression_data_prepare,
                            create_regression_scaled_depth_delta_learner_32f,
                            create_regression_depth_delta_predictor_32f,
                            kwargs)

def create_consistent_scaled_depth_delta_regression(**kwargs):
    return LearnerWrapper(  depth_delta_regression_data_prepare,
                            create_consistent_two_stream_regression_scaled_depth_delta_learner_32f,
                            create_regression_depth_delta_predictor_32f,
                            kwargs)
