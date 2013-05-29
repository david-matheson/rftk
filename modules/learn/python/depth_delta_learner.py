import numpy as np

import rftk.buffers as buffers
import rftk.pipeline as pipeline
import rftk.image_features as image_features
import rftk.splitpoints as splitpoints
import rftk.should_split as should_split
import rftk.classification as classification
import rftk.predict as predict
import learn
from wrappers import *
from split_criteria import *


def depth_delta_classification_data_prepare(**kwargs):
    bufferCollection = buffers.BufferCollection()
    bufferCollection.AddBuffer(buffers.DEPTH_IMAGES, kwargs['depth_images'])
    bufferCollection.AddBuffer(buffers.PIXEL_INDICES, kwargs['pixel_indices'])
    if 'offset_scales' in kwargs:
        bufferCollection.AddBuffer(buffers.OFFSET_SCALES, kwargs['offset_scales'])
    if 'classes' in kwargs:
        bufferCollection.AddBuffer(buffers.CLASS_LABELS, kwargs['classes'])
    return bufferCollection

def create_depth_delta_predictor_32f(forest, **kwargs):
    number_of_classes = forest.GetTree(0).mYs.GetN()
    all_samples_step = pipeline.AllSamplesStep_i32f32i32(buffers.PIXEL_INDICES)
    combiner = classification.ClassProbabilityCombiner_f32(number_of_classes)
    depth_delta_feature = image_features.ScaledDepthDeltaFeature_f32i32(all_samples_step.IndicesBufferId,
                                                                        buffers.PIXEL_INDICES,
                                                                        buffers.DEPTH_IMAGES)
    forest_predicter = predict.ScaledDepthDeltaClassificationPredictin_f32i32(forest, depth_delta_feature, combiner, all_samples_step)
    return PredictorWrapper_32f(forest_predicter, depth_delta_classification_data_prepare)

def create_scaled_depth_delta_learner_32f(**kwargs):
    ux = float( kwargs.get('ux') )
    uy = float( kwargs.get('uy') )
    vx = float( kwargs.get('vx') )
    vy = float( kwargs.get('vy') )

    number_of_trees = int( kwargs.get('number_of_trees', 10) )
    number_of_features = int( kwargs.get('number_of_features', 1) )
    feature_ordering = int( kwargs.get('feature_ordering', pipeline.FEATURES_BY_DATAPOINTS) )
    number_of_jobs = int( kwargs.get('number_of_jobs', 1) )
    number_of_classes = int( kwargs['classes'].GetMax() + 1 )

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
    slice_classes_step = pipeline.SliceInt32VectorBufferStep_i32(buffers.CLASS_LABELS, sample_data_step.IndicesBufferId)
    slice_weights_step = pipeline.SliceFloat32VectorBufferStep_i32(sample_data_step.WeightsBufferId, sample_data_step.IndicesBufferId)
    class_infogain_walker = classification.ClassInfoGainWalker_f32i32(slice_weights_step.SlicedBufferId,
                                                                      slice_classes_step.SlicedBufferId,
                                                                      number_of_classes)
    best_splitpint_step = classification.ClassInfoGainBestSplitpointsWalkingSortedStep_f32i32(class_infogain_walker,
                                                                        depth_delta_feature_extractor_step.FeatureValuesBufferId,
                                                                        feature_ordering)
    node_steps_pipeline = pipeline.Pipeline([feature_params_step, depth_delta_feature_extractor_step,
                                            slice_classes_step, slice_weights_step, best_splitpint_step])

    split_buffers = splitpoints.SplitSelectorBuffers(best_splitpint_step.ImpurityBufferId,
                                                          best_splitpint_step.SplitpointBufferId,
                                                          best_splitpint_step.SplitpointCountsBufferId,
                                                          best_splitpint_step.ChildCountsBufferId,
                                                          best_splitpint_step.LeftYsBufferId,
                                                          best_splitpint_step.RightYsBufferId,
                                                          feature_params_step.FloatParamsBufferId,
                                                          feature_params_step.IntParamsBufferId,
                                                          depth_delta_feature_extractor_step.FeatureValuesBufferId,
                                                          feature_ordering)
    should_split_criteria = create_should_split_criteria(**kwargs)
    finalizer = classification.ClassEstimatorFinalizer_f32()
    split_indices = splitpoints.SplitIndices_f32i32(sample_data_step.IndicesBufferId)
    split_selector = splitpoints.SplitSelector_f32i32([split_buffers], should_split_criteria, finalizer, split_indices )

    tree_learner = learn.DepthFirstTreeLearner_f32i32(try_split_criteria, tree_steps_pipeline, node_steps_pipeline, split_selector)
    forest_learner = learn.ParallelForestLearner(tree_learner, number_of_trees, 5, 5, number_of_classes, number_of_jobs)
    return forest_learner


def create_online_scaled_depth_delta_one_stream_learner_32f(**kwargs):
    number_of_trees = int( kwargs.get('number_of_trees', 10) )
    number_of_features = int( kwargs.get('number_of_features', 1))
    feature_ordering = int( kwargs.get('feature_ordering', pipeline.FEATURES_BY_DATAPOINTS) )
    number_of_splitpoints = int( kwargs.get('number_of_splitpoints', 1 ))
    number_of_classes = int( kwargs['classes'].GetMax() + 1 )
    max_frontier_size = int( kwargs.get('max_frontier_size', 10000000) )
    impurity_update_period = int( kwargs.get('impurity_update_period', 1) )

    ux = float( kwargs.get('ux') )
    uy = float( kwargs.get('uy') )
    vx = float( kwargs.get('vx') )
    vy = float( kwargs.get('vy') )

    try_split_criteria = create_try_split_criteria(**kwargs)

    if 'bootstrap' in kwargs and kwargs.get('bootstrap'):
        sample_data_step = pipeline.BootstrapSamplesStep_i32f32i32(buffers.PIXEL_INDICES)
    elif 'poisson_sample' in kwargs:
        poisson_sample_mean = float(kwargs.get('poisson_sample'))
        sample_data_step = pipeline.PoissonSamplesStep_i32i32(buffers.PIXEL_INDICES, poisson_sample_mean)
    else:
        sample_data_step = pipeline.AllSamplesStep_i32f32i32(buffers.PIXEL_INDICES)

    number_of_features_buffer = buffers.as_vector_buffer(np.array([number_of_features], dtype=np.int32))
    set_number_features_step = pipeline.SetInt32VectorBufferStep(number_of_features_buffer, pipeline.WHEN_NEW)
    tree_steps_pipeline = pipeline.Pipeline([sample_data_step, set_number_features_step])

    feature_params_step = image_features.PixelPairGaussianOffsetsStep_f32i32(set_number_features_step.OutputBufferId, ux, uy, vx, vy )
    init_node_steps_pipeline = pipeline.Pipeline([feature_params_step])

    depth_delta_feature = image_features.ScaledDepthDeltaFeature_f32i32(feature_params_step.FloatParamsBufferId,
                                                                      feature_params_step.IntParamsBufferId,
                                                                      sample_data_step.IndicesBufferId,
                                                                      buffers.PIXEL_INDICES,
                                                                      buffers.DEPTH_IMAGES,
                                                                      buffers.OFFSET_SCALES)
    depth_delta_feature_extractor_step = image_features.ScaledDepthDeltaFeatureExtractorStep_f32i32(depth_delta_feature, feature_ordering)
    slice_classes_step = pipeline.SliceInt32VectorBufferStep_i32(buffers.CLASS_LABELS, sample_data_step.IndicesBufferId)
    slice_weights_step = pipeline.SliceFloat32VectorBufferStep_i32(sample_data_step.WeightsBufferId, sample_data_step.IndicesBufferId)
    random_splitpoint_selection_step = splitpoints.RandomSplitpointsStep_f32i32(depth_delta_feature_extractor_step.FeatureValuesBufferId,
                                                                                number_of_splitpoints,
                                                                                feature_ordering)
    class_stats_updater = classification.ClassStatsUpdater_f32i32(slice_weights_step.SlicedBufferId,
                                                                      slice_classes_step.SlicedBufferId,
                                                                      number_of_classes)
    one_stream_split_stats_step = classification.ClassStatsUpdaterOneStreamStep_f32i32(random_splitpoint_selection_step.SplitpointsBufferId,
                                                                          random_splitpoint_selection_step.SplitpointsCountsBufferId,
                                                                          depth_delta_feature_extractor_step.FeatureValuesBufferId,
                                                                          feature_ordering,
                                                                          class_stats_updater)
    update_stats_node_steps_pipeline = pipeline.Pipeline([depth_delta_feature_extractor_step,
                                                          slice_classes_step,
                                                          slice_weights_step,
                                                          random_splitpoint_selection_step,
                                                          one_stream_split_stats_step])

    impurity_step = classification.ClassInfoGainSplitpointsImpurity_f32i32(random_splitpoint_selection_step.SplitpointsCountsBufferId,
                                                                          one_stream_split_stats_step.ChildCountsBufferId,
                                                                          one_stream_split_stats_step.LeftStatsBufferId,
                                                                          one_stream_split_stats_step.RightStatsBufferId)
    update_impurity_node_steps_pipeline = pipeline.Pipeline([impurity_step])

    split_buffers = splitpoints.SplitSelectorBuffers(impurity_step.ImpurityBufferId,
                                                          random_splitpoint_selection_step.SplitpointsBufferId,
                                                          random_splitpoint_selection_step.SplitpointsCountsBufferId,
                                                          one_stream_split_stats_step.ChildCountsBufferId,
                                                          one_stream_split_stats_step.LeftStatsBufferId,
                                                          one_stream_split_stats_step.RightStatsBufferId,
                                                          feature_params_step.FloatParamsBufferId,
                                                          feature_params_step.IntParamsBufferId,
                                                          depth_delta_feature_extractor_step.FeatureValuesBufferId,
                                                          feature_ordering)
    should_split_criteria = create_should_split_criteria(**kwargs)
    finalizer = classification.ClassEstimatorFinalizer_f32()
    split_selector = splitpoints.WaitForBestSplitSelector_f32i32([split_buffers], should_split_criteria, finalizer )

    feature_prediction = image_features.ScaledDepthDeltaFeature_f32i32(sample_data_step.IndicesBufferId,
                                                                              buffers.PIXEL_INDICES,
                                                                              buffers.DEPTH_IMAGES)
    estimator_params_updater = classification.ClassEstimatorUpdater_f32i32(sample_data_step.WeightsBufferId, buffers.CLASS_LABELS, number_of_classes)
    forest_learner = learn.OnlineForestScaledDepthDeltaClassLearner_f32i32(
                                                              try_split_criteria,
                                                              tree_steps_pipeline,
                                                              init_node_steps_pipeline,
                                                              update_stats_node_steps_pipeline,
                                                              update_impurity_node_steps_pipeline,
                                                              impurity_update_period, split_selector,
                                                              max_frontier_size, number_of_trees, 5, 5, number_of_classes,
                                                              sample_data_step.IndicesBufferId, sample_data_step.WeightsBufferId,
                                                              feature_prediction, estimator_params_updater)
    return forest_learner

def create_online_scaled_depth_delta_two_stream_consistent_learner_32f(**kwargs):
    number_of_trees = int( kwargs.get('number_of_trees', 10) )
    number_of_features = int( kwargs.get('number_of_features', 1))
    feature_ordering = int( kwargs.get('feature_ordering', pipeline.FEATURES_BY_DATAPOINTS) )
    number_of_splitpoints = int( kwargs.get('number_of_splitpoints', 1 ))
    number_of_classes = int( kwargs['classes'].GetMax() + 1 )
    max_frontier_size = int( kwargs.get('max_frontier_size', 10000000) )
    impurity_update_period = int( kwargs.get('impurity_update_period', 1) )
    probability_of_impurity_stream = float(kwargs.get('probability_of_impurity_stream', 0.5) )

    ux = float( kwargs.get('ux') )
    uy = float( kwargs.get('uy') )
    vx = float( kwargs.get('vx') )
    vy = float( kwargs.get('vy') )

    try_split_criteria = create_try_split_criteria(**kwargs)

    if 'bootstrap' in kwargs and kwargs.get('bootstrap'):
        sample_data_step = pipeline.BootstrapSamplesStep_i32f32i32(buffers.PIXEL_INDICES)
    elif 'poisson_sample' in kwargs:
        poisson_sample_mean = float(kwargs.get('poisson_sample'))
        sample_data_step = pipeline.PoissonSamplesStep_i32i32(buffers.PIXEL_INDICES, poisson_sample_mean)
    else:
        sample_data_step = pipeline.AllSamplesStep_i32f32i32(buffers.PIXEL_INDICES)

    assign_stream_step = splitpoints.AssignStreamStep_f32i32(sample_data_step.WeightsBufferId, probability_of_impurity_stream)
    tree_steps_pipeline = pipeline.Pipeline([sample_data_step, assign_stream_step])

    # On init
    set_number_features_step = pipeline.PoissonStep_f32i32(number_of_features, 1)
    feature_params_step = image_features.PixelPairGaussianOffsetsStep_f32i32(set_number_features_step.OutputBufferId, ux, uy, vx, vy )
    init_node_steps_pipeline = pipeline.Pipeline([set_number_features_step, feature_params_step])

    # On update
    depth_delta_feature = image_features.ScaledDepthDeltaFeature_f32i32(feature_params_step.FloatParamsBufferId,
                                                                      feature_params_step.IntParamsBufferId,
                                                                      sample_data_step.IndicesBufferId,
                                                                      buffers.PIXEL_INDICES,
                                                                      buffers.DEPTH_IMAGES,
                                                                      buffers.OFFSET_SCALES)
    depth_delta_feature_extractor_step = image_features.ScaledDepthDeltaFeatureExtractorStep_f32i32(depth_delta_feature, feature_ordering)
    slice_classes_step = pipeline.SliceInt32VectorBufferStep_i32(buffers.CLASS_LABELS, sample_data_step.IndicesBufferId)
    slice_weights_step = pipeline.SliceFloat32VectorBufferStep_i32(sample_data_step.WeightsBufferId, sample_data_step.IndicesBufferId)
    slice_assign_stream_step = pipeline.SliceInt32VectorBufferStep_i32(assign_stream_step.StreamTypeBufferId, sample_data_step.IndicesBufferId)

    random_splitpoint_selection_step = splitpoints.RandomSplitpointsStep_f32i32(depth_delta_feature_extractor_step.FeatureValuesBufferId,
                                                                                number_of_splitpoints,
                                                                                feature_ordering,
                                                                                slice_assign_stream_step.SlicedBufferId)

    class_stats_updater = classification.ClassStatsUpdater_f32i32(slice_weights_step.SlicedBufferId,
                                                                      slice_classes_step.SlicedBufferId,
                                                                      number_of_classes)

    two_stream_split_stats_step = classification.ClassStatsUpdaterTwoStreamStep_f32i32(random_splitpoint_selection_step.SplitpointsBufferId,
                                                                          random_splitpoint_selection_step.SplitpointsCountsBufferId,
                                                                          slice_assign_stream_step.SlicedBufferId,
                                                                          depth_delta_feature_extractor_step.FeatureValuesBufferId,
                                                                          feature_ordering,
                                                                          class_stats_updater)

    update_stats_node_steps_pipeline = pipeline.Pipeline([depth_delta_feature_extractor_step,
                                                          slice_classes_step,
                                                          slice_weights_step,
                                                          slice_assign_stream_step,
                                                          random_splitpoint_selection_step,
                                                          two_stream_split_stats_step])

    # On impurity
    impurity_step = classification.ClassInfoGainSplitpointsImpurity_f32i32(random_splitpoint_selection_step.SplitpointsCountsBufferId,
                                                                          two_stream_split_stats_step.ChildCountsImpurityBufferId,
                                                                          two_stream_split_stats_step.LeftImpurityStatsBufferId,
                                                                          two_stream_split_stats_step.RightImpurityStatsBufferId)

    update_impurity_node_steps_pipeline = pipeline.Pipeline([impurity_step])

    split_buffers = splitpoints.SplitSelectorBuffers(impurity_step.ImpurityBufferId,
                                                          random_splitpoint_selection_step.SplitpointsBufferId,
                                                          random_splitpoint_selection_step.SplitpointsCountsBufferId,
                                                          two_stream_split_stats_step.ChildCountsEstimatorBufferId,
                                                          two_stream_split_stats_step.LeftEstimatorStatsBufferId,
                                                          two_stream_split_stats_step.RightEstimatorStatsBufferId,
                                                          feature_params_step.FloatParamsBufferId,
                                                          feature_params_step.IntParamsBufferId,
                                                          depth_delta_feature_extractor_step.FeatureValuesBufferId,
                                                          feature_ordering)
    should_split_criteria = create_should_split_consistent_criteria(**kwargs)
    finalizer = classification.ClassEstimatorFinalizer_f32()
    split_selector = splitpoints.WaitForBestSplitSelector_f32i32([split_buffers], should_split_criteria, finalizer )

    feature_prediction = image_features.ScaledDepthDeltaFeature_f32i32(sample_data_step.IndicesBufferId,
                                                                              buffers.PIXEL_INDICES,
                                                                              buffers.DEPTH_IMAGES)
    estimator_params_updater = classification.ClassEstimatorUpdater_f32i32(sample_data_step.WeightsBufferId, buffers.CLASS_LABELS, number_of_classes)
    forest_learner = learn.OnlineForestScaledDepthDeltaClassLearner_f32i32(
                                                try_split_criteria,
                                                tree_steps_pipeline,
                                                init_node_steps_pipeline,
                                                update_stats_node_steps_pipeline,
                                                update_impurity_node_steps_pipeline,
                                                impurity_update_period, split_selector,
                                                max_frontier_size, number_of_trees, 5, 5, number_of_classes,
                                                sample_data_step.IndicesBufferId, sample_data_step.WeightsBufferId,
                                                feature_prediction, estimator_params_updater)
    return forest_learner

def create_vanilia_scaled_depth_delta_classifier(**kwargs):
    return LearnerWrapper(  depth_delta_classification_data_prepare,
                            create_scaled_depth_delta_learner_32f,
                            create_depth_delta_predictor_32f,
                            kwargs)

def create_online_one_stream_depth_delta_classifier(**kwargs):
    return LearnerWrapper(  depth_delta_classification_data_prepare,
                            create_online_scaled_depth_delta_one_stream_learner_32f,
                            create_depth_delta_predictor_32f,
                            kwargs)

def create_online_two_stream_consistent_depth_delta_classifier(**kwargs):
    return LearnerWrapper(  depth_delta_classification_data_prepare,
                            create_online_scaled_depth_delta_two_stream_consistent_learner_32f,
                            create_depth_delta_predictor_32f,
                            kwargs)