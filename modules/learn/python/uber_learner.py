import numpy as np

import rftk.buffers as buffers
import rftk.pipeline as pipeline
import rftk.matrix_features as matrix_features
import rftk.image_features as image_features
import rftk.splitpoints as splitpoints
import rftk.classification as classification
import rftk.predict as predict
import learn
from wrappers import *
from greedy_add_swap_learner import *
from split_criteria import *

def make_uber_data_prepare(learner_kwargs):

    def uber_data_prepare(**kwargs):
        bufferCollection = buffers.BufferCollection()

        # add input data buffers
        data_type = learner_kwargs.get('data_type')
        if data_type == 'matrix':
            bufferCollection.AddBuffer(buffers.X_FLOAT_DATA, kwargs['x'])
        elif data_type == 'depth_image':
            bufferCollection.AddBuffer(buffers.DEPTH_IMAGES, kwargs['depth_images'])
            bufferCollection.AddBuffer(buffers.PIXEL_INDICES, kwargs['pixel_indices'])
            if 'offset_scales' in kwargs:
                bufferCollection.AddBuffer(buffers.OFFSET_SCALES, kwargs['offset_scales'])
        else:
            raise Exception("unknown data_type")

        # add y data buffers
        prediction_type = learner_kwargs.get('prediction_type')
        if prediction_type == 'classification':
            if 'classes' in kwargs:
                bufferCollection.AddBuffer(buffers.CLASS_LABELS, kwargs['classes'])
        elif prediction_type == 'regression':
            if 'y' in kwargs:
                bufferCollection.AddBuffer(buffers.YS, kwargs['y'])
        else:
            raise Exception("unknown prediction_type")

        return bufferCollection

    return uber_data_prepare 

def make_uber_create_predictor(learner_kwargs):

    def uber_create_predictor(forest, **kwargs):

        data_type = learner_kwargs.get('data_type')
        if data_type == 'matrix':
            all_samples_step = pipeline.AllSamplesStep_f32f32i32(buffers.X_FLOAT_DATA)
            feature = matrix_features.LinearFloat32MatrixFeature_f32i32(all_samples_step.IndicesBufferId, buffers.X_FLOAT_DATA)
        elif data_type == 'depth_image':
            all_samples_step = pipeline.AllSamplesStep_i32f32i32(buffers.PIXEL_INDICES)
            feature = image_features.ScaledDepthDeltaFeature_f32i32(all_samples_step.IndicesBufferId,
                                                                    buffers.PIXEL_INDICES,
                                                                    buffers.DEPTH_IMAGES)
        else:
            raise Exception("unknown data_type")

        prediction_type = learner_kwargs.get('prediction_type')
        if prediction_type == 'classification':
            number_of_classes = forest.GetTree(0).mYs.GetN()    
            combiner = classification.ClassProbabilityCombiner_f32(number_of_classes)   
        elif prediction_type == 'regression':
            dimension_of_y = forest.GetTree(0).mYs.GetN() / 2
            combiner = regression.MeanVarianceCombiner_f32(dimension_of_y)
        else:
            raise Exception("unknown prediction_type")

        if data_type == 'matrix' and prediction_type == 'classification':
            forest_predicter = predict.LinearMatrixClassificationPredictin_f32i32(forest, feature, combiner, all_samples_step)
        elif data_type == 'matrix' and prediction_type == 'regression':
            forest_predicter = predict.LinearMatrixRegressionPredictin_f32i32(forest, feature, combiner, all_samples_step)
        elif data_type == 'depth_image' and prediction_type == 'classification':
            forest_predicter = predict.ScaledDepthDeltaClassificationPredictin_f32i32(forest, feature, combiner, all_samples_step)
        elif data_type == 'depth_image' and prediction_type == 'regression':
            forest_predicter = predict.ScaledDepthDeltaRegressionPrediction_f32i32(forest, feature, combiner, all_samples_step)

        return PredictorWrapper_32f(forest_predicter, make_uber_data_prepare(learner_kwargs))

    return uber_create_predictor


def uber_create_learner(**kwargs):
    forest_steps = []
    tree_steps = []
    node_steps_init = []
    node_steps_update = []
    node_steps_impurity = []

    data_type = kwargs.get('data_type')
    extractor_type = kwargs.get('extractor_type')
    prediction_type = kwargs.get('prediction_type')
    split_type = kwargs.get('split_type')
    tree_type = kwargs.get('tree_type')
    feature_ordering = int( kwargs.get('feature_ordering', pipeline.FEATURES_BY_DATAPOINTS) )
    streams_type = kwargs.get('streams_type', 'one_stream')

    
    # Setup sampling of weights step
    if 'bootstrap' in kwargs and kwargs.get('bootstrap'):
        if data_type == 'matrix':
            sample_data_step = pipeline.BootstrapSamplesStep_f32f32i32(buffers.X_FLOAT_DATA)
        elif data_type == 'depth_image':
            sample_data_step = pipeline.BootstrapSamplesStep_i32f32i32(buffers.PIXEL_INDICES)
        else:
            raise Exception("unknown data_type %s" % (data_type))
    elif 'poisson_sample' in kwargs:
        poisson_sample_mean = float(kwargs.get('poisson_sample'))
        if data_type == 'matrix':
            sample_data_step = pipeline.PoissonSamplesStep_f32i32(buffers.X_FLOAT_DATA, poisson_sample_mean)
        elif data_type == 'depth_image':
            sample_data_step = pipeline.PoissonSamplesStep_i32i32(buffers.PIXEL_INDICES, poisson_sample_mean)
        else:
            raise Exception("unknown data_type %s" % (data_type))
    else:
        if data_type == 'matrix':
            sample_data_step = pipeline.AllSamplesStep_f32f32i32(buffers.X_FLOAT_DATA)
        elif data_type == 'depth_image':
            sample_data_step = pipeline.AllSamplesStep_i32f32i32(buffers.PIXEL_INDICES)
        else:
            raise Exception("unknown data_type %s" % (data_type))
    tree_steps.append(sample_data_step)

    # Default number of features to extract
    if data_type == 'matrix' and prediction_type == 'classification':
        default_number_of_features = int(np.sqrt(kwargs['x'].shape[1]))
    elif data_type == 'matrix' and prediction_type == 'regression':
        default_number_of_features = int(kwargs['x'].shape[1]/3 + 0.5)
    elif data_type == 'depth_image':
        default_number_of_features = 1
    else:
        raise Exception("unknown data_type")

    # Set number of features to extract
    # need to add possion and uniform
    number_of_features = int( kwargs.get('number_of_features', default_number_of_features) )
    possion_number_of_features = bool( kwargs.get('possion_number_of_features', False))
    if possion_number_of_features:
        set_number_features_step = pipeline.PoissonStep_f32i32(number_of_features, 1)
        node_steps_init.append(set_number_features_step)
    else:
        number_of_features_buffer = buffers.as_vector_buffer(np.array([number_of_features], dtype=np.int32))
        set_number_features_step = pipeline.SetInt32VectorBufferStep(number_of_features_buffer, pipeline.WHEN_NEW)
        tree_steps.append(set_number_features_step)


    # Setup extraction step
    if data_type == 'matrix':
        if extractor_type == 'axis_aligned':
            feature_params_step = matrix_features.AxisAlignedParamsStep_f32i32(set_number_features_step.OutputBufferId, buffers.X_FLOAT_DATA)
        elif extractor_type == 'dimension_pair_diff':
            feature_params_step = matrix_features.DimensionPairDifferenceParamsStep_f32i32(set_number_features_step.OutputBufferId, buffers.X_FLOAT_DATA ) 
        elif extractor_type == 'class_pair_diff':
            feature_params_step = matrix_features.ClassPairDifferenceParamsStep_f32i32(set_number_features_step.OutputBufferId,
                                                                                      buffers.X_FLOAT_DATA,
                                                                                      buffers.CLASS_LABELS,
                                                                                      sample_data_step.IndicesBufferId )
        else:
            raise Exception("unknown extractor_type %s" % extractor_type)

        feature = matrix_features.LinearFloat32MatrixFeature_f32i32(feature_params_step.FloatParamsBufferId,
                                                                  feature_params_step.IntParamsBufferId,
                                                                  sample_data_step.IndicesBufferId,
                                                                  buffers.X_FLOAT_DATA)
        feature_extractor_step = matrix_features.LinearFloat32MatrixFeatureExtractorStep_f32i32(feature, feature_ordering)
    elif data_type == 'depth_image' and extractor_type == 'pixel_pair_diff':
        ux = float( kwargs.get('ux') )
        uy = float( kwargs.get('uy') )
        vx = float( kwargs.get('vx') )
        vy = float( kwargs.get('vy') )

        feature_params_step = image_features.PixelPairGaussianOffsetsStep_f32i32(set_number_features_step.OutputBufferId, ux, uy, vx, vy )
        feature = image_features.ScaledDepthDeltaFeature_f32i32(feature_params_step.FloatParamsBufferId,
                                                                      feature_params_step.IntParamsBufferId,
                                                                      sample_data_step.IndicesBufferId,
                                                                      buffers.PIXEL_INDICES,
                                                                      buffers.DEPTH_IMAGES,
                                                                      buffers.OFFSET_SCALES)
        feature_extractor_step = image_features.ScaledDepthDeltaFeatureExtractorStep_f32i32(feature, feature_ordering)

    else:
        raise Exception("unknown data_type %s extractor_type %s" % (data_type, extractor_type))

    node_steps_init.append(feature_params_step)
    node_steps_update.append(feature_extractor_step)

    # Slice weights for datapoints in the node
    slice_weights_step = pipeline.SliceFloat32VectorBufferStep_i32(sample_data_step.WeightsBufferId, sample_data_step.IndicesBufferId)
    node_steps_update.append(slice_weights_step)

    # Setup stream ids
    if streams_type == 'two_stream_per_tree' or streams_type == 'two_stream_per_forest':
        probability_of_impurity_stream = float(kwargs.get('probability_of_impurity_stream', 0.5) )
        assign_stream_step = splitpoints.AssignStreamStep_f32i32(sample_data_step.WeightsBufferId, probability_of_impurity_stream)
        slice_assign_stream_step = pipeline.SliceInt32VectorBufferStep_i32(assign_stream_step.StreamTypeBufferId, sample_data_step.IndicesBufferId)
        if streams_type == 'two_stream_per_tree':
            tree_steps.append(assign_stream_step)
        elif streams_type == 'two_stream_per_forest':
            forest_steps.append(assign_stream_step)
        node_steps_update.append(slice_assign_stream_step)
    elif streams_type != 'one_stream':
        raise Exception("unknown streams_type %s" % streams_type)

    # Slice ys for datapoints in the node
    if prediction_type == 'classification':
        slice_ys_step = pipeline.SliceInt32VectorBufferStep_i32(buffers.CLASS_LABELS, sample_data_step.IndicesBufferId)
        finalizer = classification.ClassEstimatorFinalizer_f32()
        # fix this... shouldn't be passing numpy arrays for one and rftk.buffers for the other
        if data_type == 'matrix':
            number_of_classes = int( np.max(kwargs['classes']) + 1 )
        else:
            number_of_classes = int( kwargs['classes'].GetMax() + 1 )
        y_estimator_dimension = number_of_classes
    elif prediction_type == 'regression':
        slice_ys_step = pipeline.SliceFloat32MatrixBufferStep_i32(buffers.YS, sample_data_step.IndicesBufferId)
        raise Exception("not implemented")
    else:
        raise Exception("unknown prediction_type %s" % prediction_type)
    node_steps_update.append(slice_ys_step)


    # Setup finding splitpoint steps
    if split_type == 'all_midpoints':

        if prediction_type == 'classification':
            class_infogain_walker = classification.ClassInfoGainWalker_f32i32(slice_weights_step.SlicedBufferId,
                                                                              slice_ys_step.SlicedBufferId,
                                                                              number_of_classes)
            best_splitpint_step = classification.ClassInfoGainBestSplitpointsWalkingSortedStep_f32i32(class_infogain_walker,
                                                                                feature_extractor_step.FeatureValuesBufferId,
                                                                                feature_ordering)
            impurity_buffer = best_splitpint_step.ImpurityBufferId
            splitpoint_buffer = best_splitpint_step.SplitpointBufferId
            splitpoint_counts_buffer = best_splitpint_step.SplitpointCountsBufferId
            child_count_buffer = best_splitpint_step.ChildCountsBufferId
            left_estimator_buffer = best_splitpint_step.LeftYsBufferId
            right_estimator_buffer = best_splitpint_step.RightYsBufferId

        # elif prediction_type == 'regression':
        else:
            raise Exception("unknown prediction_type %s" % prediction_type)
        node_steps_impurity.append(best_splitpint_step)

    elif split_type == 'constant_splitpoints':

        constant_splitpoints_type = kwargs.get('constant_splitpoints_type')
        number_of_splitpoints = int(kwargs.get('number_of_splitpoints'))

        # Select splitpoints
        if constant_splitpoints_type == 'at_random_datapoints':
            if streams_type == 'one_stream':
                splitpoint_selection_step = splitpoints.RandomSplitpointsStep_f32i32(feature_extractor_step.FeatureValuesBufferId,
                                                                                        number_of_splitpoints,
                                                                                        feature_ordering)
            elif streams_type == 'two_stream_per_tree' or streams_type == 'two_stream_per_forest':
                splitpoint_selection_step = splitpoints.RandomSplitpointsStep_f32i32(feature_extractor_step.FeatureValuesBufferId,
                                                                                        number_of_splitpoints,
                                                                                        feature_ordering,
                                                                                        slice_assign_stream_step.SlicedBufferId)
            else:
                raise Exception("unknown constant_splitpoints_type %s" % streams_type)
        else:
            raise Exception("unknown constant_splitpoints_type %s" % constant_splitpoints_type)
        node_steps_update.append(splitpoint_selection_step)

        # Update stats and impurity for splitpoints
        if prediction_type == 'classification':
            class_stats_updater = classification.ClassStatsUpdater_f32i32(slice_weights_step.SlicedBufferId,
                                                                          slice_ys_step.SlicedBufferId,
                                                                          number_of_classes)
            if streams_type == 'one_stream':
                one_stream_split_stats_step = classification.ClassStatsUpdaterOneStreamStep_f32i32(splitpoint_selection_step.SplitpointsBufferId,
                                                                                      splitpoint_selection_step.SplitpointsCountsBufferId,
                                                                                      feature_extractor_step.FeatureValuesBufferId,
                                                                                      feature_ordering,
                                                                                      class_stats_updater)

                impurity_step = classification.ClassInfoGainSplitpointsImpurity_f32i32(splitpoint_selection_step.SplitpointsCountsBufferId,
                                                                                      one_stream_split_stats_step.ChildCountsBufferId,
                                                                                      one_stream_split_stats_step.LeftStatsBufferId,
                                                                                      one_stream_split_stats_step.RightStatsBufferId)
                impurity_buffer = impurity_step.ImpurityBufferId
                splitpoint_buffer = splitpoint_selection_step.SplitpointsBufferId
                splitpoint_counts_buffer = splitpoint_selection_step.SplitpointsCountsBufferId
                child_count_buffer = one_stream_split_stats_step.ChildCountsBufferId
                left_estimator_buffer = one_stream_split_stats_step.LeftStatsBufferId
                right_estimator_buffer = one_stream_split_stats_step.RightStatsBufferId
                node_steps_update.append(one_stream_split_stats_step)
                node_steps_impurity.append(impurity_step)

            elif streams_type == 'two_stream_per_tree' or streams_type == 'two_stream_per_forest':
                two_stream_split_stats_step = classification.ClassStatsUpdaterTwoStreamStep_f32i32(splitpoint_selection_step.SplitpointsBufferId,
                                                                                      splitpoint_selection_step.SplitpointsCountsBufferId,
                                                                                      slice_assign_stream_step.SlicedBufferId,
                                                                                      feature_extractor_step.FeatureValuesBufferId,
                                                                                      feature_ordering,
                                                                                      class_stats_updater)

                impurity_step = classification.ClassInfoGainSplitpointsImpurity_f32i32(splitpoint_selection_step.SplitpointsCountsBufferId,
                                                                                      two_stream_split_stats_step.ChildCountsImpurityBufferId,
                                                                                      two_stream_split_stats_step.LeftImpurityStatsBufferId,
                                                                                      two_stream_split_stats_step.RightImpurityStatsBufferId)
                impurity_buffer = impurity_step.ImpurityBufferId
                splitpoint_buffer = splitpoint_selection_step.SplitpointsBufferId
                splitpoint_counts_buffer = splitpoint_selection_step.SplitpointsCountsBufferId
                child_count_buffer = two_stream_split_stats_step.ChildCountsEstimatorBufferId
                left_estimator_buffer = two_stream_split_stats_step.LeftEstimatorStatsBufferId
                right_estimator_buffer = two_stream_split_stats_step.RightEstimatorStatsBufferId
                node_steps_update.append(two_stream_split_stats_step)
                node_steps_impurity.append(impurity_step)
            else:
                raise Exception("unknown constant_splitpoints_type %s" % streams_type)


        # elif prediction_type == 'regression':
        else:
            raise Exception("unknown prediction_type %s" % prediction_type)


    else:
        raise Exception("unknown split_type %s" % split_type)


    split_buffers = splitpoints.SplitSelectorBuffers(impurity_buffer,
                                                    splitpoint_buffer,
                                                    splitpoint_counts_buffer,
                                                    child_count_buffer,
                                                    left_estimator_buffer,
                                                    right_estimator_buffer,
                                                    feature_params_step.FloatParamsBufferId,
                                                    feature_params_step.IntParamsBufferId,
                                                    feature_extractor_step.FeatureValuesBufferId,
                                                    feature_ordering,
                                                    feature_extractor_step)

    forest_steps_pipeline = pipeline.Pipeline(forest_steps)
    tree_steps_pipeline = pipeline.Pipeline(tree_steps)
    node_steps_pipeline = pipeline.Pipeline(node_steps_init + node_steps_update + node_steps_impurity)

    try_split_criteria = create_try_split_criteria(**kwargs)
    should_split_criteria = create_should_split_criteria(**kwargs)

    split_indices = splitpoints.SplitIndices_f32i32(sample_data_step.IndicesBufferId)
    if tree_type == 'online':
        split_selector = splitpoints.WaitForBestSplitSelector_f32i32([split_buffers], should_split_criteria, finalizer, split_indices )
    else:
        split_selector = splitpoints.SplitSelector_f32i32([split_buffers], should_split_criteria, finalizer, split_indices)

    number_of_trees = int( kwargs.get('number_of_trees', 1) )
    number_of_jobs = int( kwargs.get('number_of_jobs', 1) )
    if tree_type == 'depth_first':
        tree_learner = learn.DepthFirstTreeLearner_f32i32(try_split_criteria, tree_steps_pipeline, node_steps_pipeline, split_selector)
        forest_learner = learn.ParallelForestLearner(tree_learner, forest_steps_pipeline, number_of_trees, y_estimator_dimension, number_of_jobs)
    elif tree_type == 'breadth_first':
        tree_learner = learn.BreadthFirstTreeLearner_f32i32(try_split_criteria, tree_steps_pipeline, node_steps_pipeline, split_selector)
        forest_learner = learn.ParallelForestLearner(tree_learner, forest_steps_pipeline, number_of_trees, y_estimator_dimension, number_of_jobs)
    elif tree_type == 'online':
        max_frontier_size = int( kwargs.get('max_frontier_size', 10000000) )
        impurity_update_period = int( kwargs.get('impurity_update_period', 1) )

        node_steps_init_pipeline = pipeline.Pipeline(node_steps_init)
        node_steps_update_pipeline = pipeline.Pipeline(node_steps_update)
        node_steps_impurity_pipeline = pipeline.Pipeline(node_steps_impurity)

        if data_type == 'matrix' and prediction_type == 'classification':
            feature_prediction = matrix_features.LinearFloat32MatrixFeature_f32i32(sample_data_step.IndicesBufferId, buffers.X_FLOAT_DATA)
            estimator_params_updater = classification.ClassEstimatorUpdater_f32i32(sample_data_step.WeightsBufferId, buffers.CLASS_LABELS, number_of_classes)
            forest_learner = learn.OnlineForestMatrixClassLearner_f32i32(
                                                        try_split_criteria,
                                                        tree_steps_pipeline,
                                                        node_steps_init_pipeline,
                                                        node_steps_update_pipeline,
                                                        node_steps_impurity_pipeline,
                                                        impurity_update_period, split_selector,
                                                        max_frontier_size, number_of_trees, 2, 2, number_of_classes,
                                                        sample_data_step.IndicesBufferId, sample_data_step.WeightsBufferId,
                                                        feature_prediction, estimator_params_updater)
        elif data_type == 'depth_image' and prediction_type == 'classification':
            feature_prediction = image_features.ScaledDepthDeltaFeature_f32i32(sample_data_step.IndicesBufferId,
                                                                              buffers.PIXEL_INDICES,
                                                                              buffers.DEPTH_IMAGES)
            estimator_params_updater = classification.ClassEstimatorUpdater_f32i32(sample_data_step.WeightsBufferId, buffers.CLASS_LABELS, number_of_classes)
            forest_learner = learn.OnlineForestScaledDepthDeltaClassLearner_f32i32(
                                                        try_split_criteria,
                                                        tree_steps_pipeline,
                                                        node_steps_init_pipeline,
                                                        node_steps_update_pipeline,
                                                        node_steps_impurity_pipeline,
                                                        impurity_update_period, split_selector,
                                                        max_frontier_size, number_of_trees, 5, 5, number_of_classes,
                                                        sample_data_step.IndicesBufferId, sample_data_step.WeightsBufferId,
                                                        feature_prediction, estimator_params_updater)
        else:            # todo sort this out for regression
            raise Exception("online: unknown data_type %s prediction_type %s" % (data_type,prediction_type))
    else:
        raise Exception("unknown tree_type")

    return forest_learner


def create_uber_learner(**kwargs):
    return LearnerWrapper(  make_uber_data_prepare(kwargs),
                            uber_create_learner,
                            make_uber_create_predictor(kwargs),
                            kwargs)
