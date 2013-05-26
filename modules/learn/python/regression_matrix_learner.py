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

def create_regression_axis_aligned_matrix_walking_learner_32f(**kwargs):
    number_of_trees = int( kwargs.get('number_of_trees', 10) )
    number_of_features = int( kwargs.get('number_of_features', np.sqrt(kwargs['x'].shape[1])) )
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
                                                          feature_ordering)
    should_split_criteria = create_should_split_criteria(**kwargs)
    finalizer = regression.MeanVarianceEstimatorFinalizer_f32()
    split_indices = splitpoints.SplitIndices_f32i32(sample_data_step.IndicesBufferId)
    split_selector = splitpoints.SplitSelector_f32i32([split_buffers], should_split_criteria, finalizer, split_indices )

    tree_learner = learn.DepthFirstTreeLearner_f32i32(try_split_criteria, tree_steps_pipeline, node_steps_pipeline, split_selector)
    forest_learner = learn.ParallelForestLearner(tree_learner, number_of_trees, 5, 5, dimension_of_y*2, number_of_jobs)
    return forest_learner

def create_vanilia_regression(**kwargs):
    return LearnerWrapper(  matrix_regression_data_prepare,
                            create_regression_axis_aligned_matrix_walking_learner_32f,
                            create_matrix_regression_predictor_32f,
                            kwargs)
