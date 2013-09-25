import numpy as np

import rftk.buffers as buffers
import rftk.pipeline as pipeline
import rftk.matrix_features as matrix_features
import rftk.image_features as image_features
import rftk.splitpoints as splitpoints
import rftk.classification as classification
import rftk.regression as regression
import rftk.predict as predict
import learn
from utils import *
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
            number_of_classes = forest.GetTree(0).GetYs().GetN()    
            combiner = classification.ClassProbabilityCombiner_f32(number_of_classes)   
        elif prediction_type == 'regression':
            dimension_of_y = forest.GetTree(0).GetYs().GetN() / 2
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

def remove_prepare_data(unused_kwargs_keys):
    prepare_data_ignore_list = ['x', 'y', 'classes', 'depth_images', 'pixel_indices', 'offset_scales']
    unused_kwargs_keys = [x for x in unused_kwargs_keys if x not in prepare_data_ignore_list]
    return unused_kwargs_keys

def get_number_of_leaves(kwargs, unused_kwargs_keys, number_of_datapoints):
    number_of_leaves = number_of_datapoints
    is_default = True
    if 'number_of_leaves' in kwargs:
        number_of_leaves = int(pop_kwargs(kwargs, 'number_of_leaves', unused_kwargs_keys))
        is_default = False
    elif 'number_of_leaves_ratio' in kwargs:
        number_of_leaves_ratio = float(pop_kwargs(kwargs, 'number_of_leaves_ratio', unused_kwargs_keys))
        number_of_leaves = int(number_of_datapoints * number_of_leaves_ratio)
        is_default = False
    return number_of_leaves, is_default

def uber_create_learner(**kwargs):
    unused_kwargs_keys = kwargs.keys()
    unused_kwargs_keys = remove_prepare_data(unused_kwargs_keys)

    forest_steps = []
    tree_steps = []
    node_steps_init = []
    node_steps_update = []
    node_steps_impurity = []

    split_steps_list = []

    data_type = pop_kwargs(kwargs, 'data_type', unused_kwargs_keys)
    extractor_type = pop_kwargs(kwargs, 'extractor_type', unused_kwargs_keys)
    prediction_type = pop_kwargs(kwargs, 'prediction_type', unused_kwargs_keys)
    split_type = pop_kwargs(kwargs, 'split_type', unused_kwargs_keys)
    tree_type = pop_kwargs(kwargs, 'tree_type', unused_kwargs_keys)
    feature_ordering = int( pop_kwargs(kwargs, 'feature_ordering', unused_kwargs_keys, pipeline.FEATURES_BY_DATAPOINTS) )
    streams_type = pop_kwargs(kwargs, 'streams_type', unused_kwargs_keys, 'one_stream')

    selector_type_default = 'best_valid'
    if tree_type == 'online':
        selector_type_default = 'only_best'
    selector_type = pop_kwargs(kwargs, 'selector_type', unused_kwargs_keys, selector_type_default) #or only_best
    
    # Setup sampling of weights step
    if pop_kwargs(kwargs, 'bootstrap', unused_kwargs_keys, False):
        if data_type == 'matrix':
            sample_data_step = pipeline.BootstrapSamplesStep_f32f32i32(buffers.X_FLOAT_DATA)
        elif data_type == 'depth_image':
            sample_data_step = pipeline.BootstrapSamplesStep_i32f32i32(buffers.PIXEL_INDICES)
        else:
            raise Exception("unknown data_type %s" % (data_type))
        tree_steps.append(sample_data_step)
    elif 'poisson_sample' in kwargs:
        poisson_sample_mean = float(pop_kwargs(kwargs, 'poisson_sample', unused_kwargs_keys))
        if data_type == 'matrix':
            sample_data_step = pipeline.PoissonSamplesStep_f32i32(buffers.X_FLOAT_DATA, poisson_sample_mean)
        elif data_type == 'depth_image':
            sample_data_step = pipeline.PoissonSamplesStep_i32i32(buffers.PIXEL_INDICES, poisson_sample_mean)
        else:
            raise Exception("unknown data_type %s" % (data_type))
        tree_steps.append(sample_data_step)
    else:
        if data_type == 'matrix':
            sample_data_step = pipeline.AllSamplesStep_f32f32i32(buffers.X_FLOAT_DATA)
        elif data_type == 'depth_image':
            sample_data_step = pipeline.AllSamplesStep_i32f32i32(buffers.PIXEL_INDICES)
        else:
            raise Exception("unknown data_type %s" % (data_type))
        forest_steps.append(sample_data_step)

    # Default number of features to extract
    if data_type == 'matrix' and prediction_type == 'classification':
        default_number_of_features = int(np.sqrt(pop_kwargs(kwargs, 'x', unused_kwargs_keys).shape[1]))
        number_of_datapoints = pop_kwargs(kwargs, 'classes', unused_kwargs_keys).shape[0]/2
    elif data_type == 'matrix' and prediction_type == 'regression':
        default_number_of_features = int(pop_kwargs(kwargs, 'x', unused_kwargs_keys).shape[1]/3 + 0.5)
        number_of_datapoints = pop_kwargs(kwargs, 'y', unused_kwargs_keys).shape[0]/2
    elif data_type == 'depth_image' and prediction_type == 'classification':
        default_number_of_features = 1
        number_of_datapoints = pop_kwargs(kwargs, 'classes', unused_kwargs_keys).GetN()/2
    elif data_type == 'depth_image' and prediction_type == 'regression':
        default_number_of_features = 1
        number_of_datapoints = pop_kwargs(kwargs, 'y', unused_kwargs_keys).GetM()/2
    else:
        raise Exception("unknown data_type")

    # Set number of features to extract
    # need to add possion and uniform
    number_of_features = int( pop_kwargs(kwargs, 'number_of_features', unused_kwargs_keys, default_number_of_features) )
    possion_number_of_features = bool( pop_kwargs(kwargs, 'possion_number_of_features', unused_kwargs_keys, False) )
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
        ux = float( pop_kwargs(kwargs, 'ux', unused_kwargs_keys) )
        uy = float( pop_kwargs(kwargs, 'uy', unused_kwargs_keys) )
        vx = float( pop_kwargs(kwargs, 'vx', unused_kwargs_keys) )
        vy = float( pop_kwargs(kwargs, 'vy', unused_kwargs_keys) )

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
        probability_of_impurity_stream = float(pop_kwargs(kwargs, 'probability_of_impurity_stream', unused_kwargs_keys, 0.5))
        assign_stream_step = splitpoints.AssignStreamStep_f32i32(sample_data_step.WeightsBufferId, probability_of_impurity_stream)
        slice_stream_step = pipeline.SliceInt32VectorBufferStep_i32(assign_stream_step.StreamTypeBufferId, sample_data_step.IndicesBufferId)
        if streams_type == 'two_stream_per_tree':
            tree_steps.append(assign_stream_step)
        elif streams_type == 'two_stream_per_forest':
            forest_steps.append(assign_stream_step)
        node_steps_update.append(slice_stream_step)
    elif streams_type != 'one_stream':
        raise Exception("unknown streams_type %s" % streams_type)

    # Slice ys for datapoints in the node
    if prediction_type == 'classification':
        slice_ys_step = pipeline.SliceInt32VectorBufferStep_i32(buffers.CLASS_LABELS, sample_data_step.IndicesBufferId)
        finalizer = classification.ClassEstimatorFinalizer_f32()
        # fix this... shouldn't be passing numpy arrays for one and rftk.buffers for the other
        if data_type == 'matrix':
            number_of_classes = int( np.max(pop_kwargs(kwargs, 'classes', unused_kwargs_keys)) + 1 )
        else:
            number_of_classes = int( pop_kwargs(kwargs, 'classes', unused_kwargs_keys).GetMax() + 1 )
        y_estimator_dimension = number_of_classes
    elif prediction_type == 'regression':
        slice_ys_step = pipeline.SliceFloat32MatrixBufferStep_i32(buffers.YS, sample_data_step.IndicesBufferId)
        finalizer = regression.MeanVarianceEstimatorFinalizer_f32()
        if data_type == 'matrix':
            dimension_of_y = int( pop_kwargs(kwargs, 'y', unused_kwargs_keys).shape[1] )
        else:
            dimension_of_y = int( pop_kwargs(kwargs, 'y', unused_kwargs_keys).GetN() )
        y_estimator_dimension = dimension_of_y*2
    else:
        raise Exception("unknown prediction_type %s" % prediction_type)
    node_steps_update.append(slice_ys_step)


    # Setup finding splitpoint steps
    if split_type == 'all_midpoints':

        if prediction_type == 'classification':
            if streams_type == 'one_stream':
                impurity_walker = classification.ClassInfoGainWalker_f32i32(slice_weights_step.SlicedBufferId,
                                                                                  slice_ys_step.SlicedBufferId,
                                                                                  number_of_classes)
                best_splitpoint_step = classification.ClassInfoGainBestSplitpointsWalkingSortedStep_f32i32(impurity_walker,
                                                                                    feature_extractor_step.FeatureValuesBufferId,
                                                                                    feature_ordering)
            else:
                raise Exception("unknown streams_type %s" % streams_type)

        elif prediction_type == 'regression':
            if streams_type == 'one_stream':
                impurity_walker = regression.SumOfVarianceWalker_f32i32(slice_weights_step.SlicedBufferId,
                                                                        slice_ys_step.SlicedBufferId,
                                                                        dimension_of_y)
                best_splitpoint_step = regression.SumOfVarianceBestSplitpointsWalkingSortedStep_f32i32(impurity_walker,
                                                                                    feature_extractor_step.FeatureValuesBufferId,
                                                                                    feature_ordering)
            elif streams_type == 'two_stream_per_tree' or streams_type == 'two_stream_per_forest':

                in_bounds_number_of_points = int(pop_kwargs(kwargs, 
                                                                'in_bounds_number_of_points', 
                                                                unused_kwargs_keys, 
                                                                number_of_datapoints / 2))
                impurity_walker = regression.SumOfVarianceTwoStreamWalker_f32i32(slice_weights_step.SlicedBufferId,
                                                                                    slice_stream_step.SlicedBufferId,
                                                                                    slice_ys_step.SlicedBufferId,
                                                                                    dimension_of_y)
                best_splitpoint_step = regression.SumOfVarianceTwoStreamBestSplitpointsWalkingSortedStep_f32i32(impurity_walker,
                                                                    slice_stream_step.SlicedBufferId,
                                                                    feature_extractor_step.FeatureValuesBufferId,
                                                                    feature_ordering,
                                                                    in_bounds_number_of_points)
            else:
                raise Exception("unknown streams_type %s" % streams_type)
        else:
            raise Exception("unknown prediction_type %s" % prediction_type)

        if streams_type == 'one_stream':
            impurity_buffer = best_splitpoint_step.ImpurityBufferId
            splitpoint_buffer = best_splitpoint_step.SplitpointBufferId
            splitpoint_counts_buffer = best_splitpoint_step.SplitpointCountsBufferId
            child_count_buffer = best_splitpoint_step.ChildCountsBufferId
            left_estimator_buffer = best_splitpoint_step.LeftYsBufferId
            right_estimator_buffer = best_splitpoint_step.RightYsBufferId
        elif streams_type == 'two_stream_per_tree' or streams_type == 'two_stream_per_forest':
            impurity_buffer = best_splitpoint_step.ImpurityBufferId
            splitpoint_buffer = best_splitpoint_step.SplitpointBufferId
            splitpoint_counts_buffer = best_splitpoint_step.SplitpointCountsBufferId
            child_count_buffer = best_splitpoint_step.ChildCountsEstimationBufferId
            left_estimator_buffer = best_splitpoint_step.LeftEstimationYsBufferId
            right_estimator_buffer = best_splitpoint_step.RightEstimationYsBufferId
        else:
            raise Exception("unknown streams_type %s" % streams_type)
        node_steps_impurity.append(best_splitpoint_step)

    elif split_type == 'random_gap':
        if prediction_type == 'regression':
            impurity_walker = regression.SumOfVarianceWalker_f32i32(slice_weights_step.SlicedBufferId,
                                                                    slice_ys_step.SlicedBufferId,
                                                                    dimension_of_y)

            best_splitpoint_step = regression.SumOfVarianceRandomGapSplitpointsStep_f32i32(impurity_walker,
                                                                        feature_extractor_step.FeatureValuesBufferId,
                                                                        feature_ordering)
        else:
            raise Exception("unknown prediction_type %s" % prediction_type)

        impurity_buffer = best_splitpoint_step.ImpurityBufferId
        splitpoint_buffer = best_splitpoint_step.SplitpointBufferId
        splitpoint_counts_buffer = best_splitpoint_step.SplitpointCountsBufferId
        child_count_buffer = best_splitpoint_step.ChildCountsBufferId
        left_estimator_buffer = best_splitpoint_step.LeftYsBufferId
        right_estimator_buffer = best_splitpoint_step.RightYsBufferId
        node_steps_impurity.append(best_splitpoint_step)

    elif split_type == 'constant_splitpoints':

        constant_splitpoints_type = pop_kwargs(kwargs, 'constant_splitpoints_type', unused_kwargs_keys)

        # Select splitpoints at random datapoints
        if constant_splitpoints_type == 'at_random_datapoints':
            number_of_splitpoints = int(pop_kwargs(kwargs, 'number_of_splitpoints', unused_kwargs_keys))
            if streams_type == 'one_stream':
                splitpoint_selection_step = splitpoints.RandomSplitpointsStep_f32i32(feature_extractor_step.FeatureValuesBufferId,
                                                                                        number_of_splitpoints,
                                                                                        feature_ordering)
            elif streams_type == 'two_stream_per_tree' or streams_type == 'two_stream_per_forest':
                splitpoint_selection_step = splitpoints.RandomSplitpointsStep_f32i32(feature_extractor_step.FeatureValuesBufferId,
                                                                                        number_of_splitpoints,
                                                                                        feature_ordering,
                                                                                        slice_stream_step.SlicedBufferId)
            else:
                raise Exception("unknown constant_splitpoints_type %s" % streams_type)

        # Select splitpoints at midpoint of the range
        elif constant_splitpoints_type == 'at_range_midpoints':
            feature_range = int(pop_kwargs(kwargs, 'feature_range', unused_kwargs_keys, 1))
            feature_range_buffer = buffers.as_vector_buffer(np.array([-feature_range, feature_range], dtype=np.float32))
            set_feature_range_buffer_step = pipeline.SetFloat32VectorBufferStep(feature_range_buffer, pipeline.WHEN_NEW)
            quantized_feature_equal = pipeline.FeatureEqualQuantized_f32i32(1.0)
            splitpoint_selection_step = splitpoints.RangeMidpointStep_f32i32(feature_params_step.FloatParamsBufferId,
                                                                            feature_params_step.IntParamsBufferId,
                                                                            set_feature_range_buffer_step.OutputBufferId,
                                                                            quantized_feature_equal)
            split_midpoint_ranges = splitpoints.SplitBuffersFeatureRange_f32i32(splitpoint_selection_step.PastFloatParamsBufferId,
                                                                                splitpoint_selection_step.PastIntParamsBufferId,
                                                                                splitpoint_selection_step.PastRangesBufferId,
                                                                                set_feature_range_buffer_step.OutputBufferId,
                                                                                quantized_feature_equal)
            forest_steps.append(set_feature_range_buffer_step)
            split_steps_list.append(split_midpoint_ranges)
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

            elif streams_type == 'two_stream_per_tree' or streams_type == 'two_stream_per_forest':
                two_stream_split_stats_step = classification.ClassStatsUpdaterTwoStreamStep_f32i32(splitpoint_selection_step.SplitpointsBufferId,
                                                                                      splitpoint_selection_step.SplitpointsCountsBufferId,
                                                                                      slice_stream_step.SlicedBufferId,
                                                                                      feature_extractor_step.FeatureValuesBufferId,
                                                                                      feature_ordering,
                                                                                      class_stats_updater)

                impurity_step = classification.ClassInfoGainSplitpointsImpurity_f32i32(splitpoint_selection_step.SplitpointsCountsBufferId,
                                                                                      two_stream_split_stats_step.ChildCountsImpurityBufferId,
                                                                                      two_stream_split_stats_step.LeftImpurityStatsBufferId,
                                                                                      two_stream_split_stats_step.RightImpurityStatsBufferId)
            else:
                raise Exception("unknown constant_splitpoints_type %s" % streams_type)


        elif prediction_type == 'regression':
            mean_variance_stats_updater = regression.MeanVarianceStatsUpdater_f32i32(slice_weights_step.SlicedBufferId,
                                                                                      slice_ys_step.SlicedBufferId,
                                                                                      dimension_of_y)
            if streams_type == 'one_stream':
                one_stream_split_stats_step = regression.SumOfVarianceOneStreamStep_f32i32(splitpoint_selection_step.SplitpointsBufferId,
                                                                                      splitpoint_selection_step.SplitpointsCountsBufferId,
                                                                                      feature_extractor_step.FeatureValuesBufferId,
                                                                                      feature_ordering,
                                                                                      class_stats_updater)

                impurity_step = regression.SumOfVarianceSplitpointsImpurity_f32i32(splitpoint_selection_step.SplitpointsCountsBufferId,
                                                                                      one_stream_split_stats_step.ChildCountsBufferId,
                                                                                      one_stream_split_stats_step.LeftStatsBufferId,
                                                                                      one_stream_split_stats_step.RightStatsBufferId)

            elif streams_type == 'two_stream_per_tree' or streams_type == 'two_stream_per_forest':
                two_stream_split_stats_step = regression.SumOfVarianceTwoStreamStep_f32i32(splitpoint_selection_step.SplitpointsBufferId,
                                                                                      splitpoint_selection_step.SplitpointsCountsBufferId,
                                                                                      slice_stream_step.SlicedBufferId,
                                                                                      feature_extractor_step.FeatureValuesBufferId,
                                                                                      feature_ordering,
                                                                                      mean_variance_stats_updater)
                impurity_step = regression.SumOfVarianceSplitpointsImpurity_f32i32(splitpoint_selection_step.SplitpointsCountsBufferId,
                                                                                      two_stream_split_stats_step.ChildCountsImpurityBufferId,
                                                                                      two_stream_split_stats_step.LeftImpurityStatsBufferId,
                                                                                      two_stream_split_stats_step.RightImpurityStatsBufferId)

            else:
                raise Exception("unknown constant_splitpoints_type %s" % streams_type)

        else:
            raise Exception("unknown prediction_type %s" % prediction_type)

        if streams_type == 'one_stream':
            impurity_buffer = impurity_step.ImpurityBufferId
            splitpoint_buffer = splitpoint_selection_step.SplitpointsBufferId
            splitpoint_counts_buffer = splitpoint_selection_step.SplitpointsCountsBufferId
            child_count_buffer = one_stream_split_stats_step.ChildCountsBufferId
            left_estimator_buffer = one_stream_split_stats_step.LeftStatsBufferId
            right_estimator_buffer = one_stream_split_stats_step.RightStatsBufferId
            node_steps_update.append(one_stream_split_stats_step)
            node_steps_impurity.append(impurity_step)

        elif streams_type == 'two_stream_per_tree' or streams_type == 'two_stream_per_forest':
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
    forest_and_tree_steps_pipeline = pipeline.Pipeline(forest_steps+tree_steps)
    node_steps_pipeline = pipeline.Pipeline(node_steps_init + node_steps_update + node_steps_impurity)

    try_split_criteria = create_try_split_criteria(unused_kwargs_keys=unused_kwargs_keys, **kwargs)
    should_split_criteria = create_should_split_criteria(unused_kwargs_keys=unused_kwargs_keys, **kwargs)

    split_indices = splitpoints.SplitIndices_f32i32(sample_data_step.IndicesBufferId)
    split_steps_list.append(split_indices)
    split_steps = splitpoints.SplitBuffersList(split_steps_list)

    if selector_type == 'best_valid':
        split_selector = splitpoints.SplitSelector_f32i32([split_buffers], should_split_criteria, finalizer, split_steps)
    elif selector_type == 'only_best':
        split_selector = splitpoints.WaitForBestSplitSelector_f32i32([split_buffers], should_split_criteria, finalizer, split_steps )

    number_of_trees = int( pop_kwargs(kwargs, 'number_of_trees', unused_kwargs_keys) )
    number_of_jobs = int( pop_kwargs(kwargs, 'number_of_jobs', unused_kwargs_keys, 1) )
    if tree_type == 'depth_first':
        tree_learner = learn.DepthFirstTreeLearner_f32i32(try_split_criteria, tree_steps_pipeline, node_steps_pipeline, split_selector)
        forest_learner = learn.ParallelForestLearner(tree_learner, forest_steps_pipeline, number_of_trees, y_estimator_dimension, number_of_jobs)
    elif tree_type == 'breadth_first':
        number_of_leaves, is_default = get_number_of_leaves(kwargs, unused_kwargs_keys, number_of_datapoints)
        if is_default:
            tree_learner = learn.BreadthFirstTreeLearner_f32i32(try_split_criteria, tree_steps_pipeline, node_steps_pipeline, split_selector)
        else:
            tree_learner = learn.BreadthFirstTreeLearner_f32i32(try_split_criteria, tree_steps_pipeline, node_steps_pipeline, split_selector, number_of_leaves)
        forest_learner = learn.ParallelForestLearner(tree_learner, forest_steps_pipeline, number_of_trees, y_estimator_dimension, number_of_jobs)
    elif tree_type == 'biau2008':
        number_of_leaves, is_default = get_number_of_leaves(kwargs, unused_kwargs_keys, number_of_datapoints)
        number_of_split_retries = int(pop_kwargs(kwargs, 'number_of_split_retries', unused_kwargs_keys))
        tree_learner = learn.Biau2008TreeLearner_f32i32(try_split_criteria, forest_and_tree_steps_pipeline, node_steps_pipeline, split_selector, number_of_leaves, number_of_split_retries)
        forest_learner = learn.ParallelForestLearner(tree_learner, number_of_trees, dimension_of_y, number_of_jobs)
    elif tree_type == 'online':
        max_frontier_size = int(pop_kwargs(kwargs, 'max_frontier_size', unused_kwargs_keys, 10000000))
        impurity_update_period = int(pop_kwargs(kwargs, 'impurity_update_period', unused_kwargs_keys, 1))

        node_steps_init_pipeline = pipeline.Pipeline(node_steps_init)
        node_steps_update_pipeline = pipeline.Pipeline(node_steps_update)
        node_steps_impurity_pipeline = pipeline.Pipeline(node_steps_impurity)

        if data_type == 'matrix' and prediction_type == 'classification':
            feature_prediction = matrix_features.LinearFloat32MatrixFeature_f32i32(sample_data_step.IndicesBufferId, buffers.X_FLOAT_DATA)
            estimator_params_updater = classification.ClassEstimatorUpdater_f32i32(sample_data_step.WeightsBufferId, buffers.CLASS_LABELS, number_of_classes)
            forest_learner = learn.OnlineForestMatrixClassLearner_f32i32(
                                                        try_split_criteria,
                                                        forest_and_tree_steps_pipeline,
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
                                                        forest_and_tree_steps_pipeline,
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

    if unused_kwargs_keys:
        raise Exception("The following arguments were not used. You have a typo or an invalid config %s" % (str(unused_kwargs_keys)))

    return forest_learner


def create_uber_learner(**kwargs):
    return LearnerWrapper(  make_uber_data_prepare(kwargs),
                            uber_create_learner,
                            make_uber_create_predictor(kwargs),
                            kwargs)
