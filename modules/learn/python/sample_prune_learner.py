
import numpy as np
from scipy import stats
import itertools
import collections
import rftk.forest_data as forest_data
import rftk.buffers as buffers
from learn import *
from uber_learner import *
from greedy_prune import *


class SamplePruneWrapper:

    def __init__(self, prepare_data, create_learner, create_predictor, error_calculator, kwargs ):
        self.prepare_data = prepare_data
        self.create_learner = create_learner
        self.create_predictor = create_predictor
        self.full_forest = forest_data.Forest()
        self.pruned_forest = None
        self.error_calculator = error_calculator
        self.init_kwargs = kwargs

    def fit(self, **kwargs):
        all_kwargs = dict(self.init_kwargs.items() + kwargs.items())

        parameter_ranges = []

        if 'number_of_features_range' in all_kwargs:
            r = all_kwargs['number_of_features_range']
            parameter_ranges.append(('number_of_features', r[0], r[1], r[2]))
            del all_kwargs['number_of_features_range']

        if 'min_child_size_range' in all_kwargs:
            r = all_kwargs['min_child_size_range']
            parameter_ranges.append(('min_child_size', r[0], r[1], r[2]))
            del all_kwargs['min_child_size_range']

        max_number_of_trees = 100
        if 'max_number_of_trees' in all_kwargs:
            max_number_of_trees = all_kwargs['max_number_of_trees']
            del all_kwargs['max_number_of_trees']

        batch_size = 1
        if 'batch_size' in all_kwargs:
            batch_size = all_kwargs.get(batch_size)
            del all_kwargs['batch_size']

        use_all_trees = False
        if 'use_all_trees' in all_kwargs:
            use_all_trees = all_kwargs['use_all_trees']
            del all_kwargs['use_all_trees']

        # First learn a forest with default parameters (ie with Breiman's recommendations)
        bufferCollection = self.prepare_data(**kwargs)

        parameter_ranges_values = {}
        for (name, range_min, range_max, use_log_space) in parameter_ranges:
            parameter_ranges_values[name] = []

        explore = True
        burnined_in_period = max_number_of_trees
        prune_period = max_number_of_trees
        included_params = None

        while self.full_forest.GetNumberOfTrees() < max_number_of_trees:

            for i in range(prune_period):
                # if self.full_forest.GetNumberOfTrees() > burnined_in_period:
                #     explore = not explore or included_params is None

                # print self.full_forest.GetNumberOfTrees()
                if explore:
                    for (name, range_min, range_max, use_log_space) in parameter_ranges:
                        all_kwargs[name] = int(np.random.uniform(range_min, range_max))
                        parameter_ranges_values[name].append(all_kwargs[name])
                        # print('explore %s - %d' % (name, all_kwargs[name]))
                else: #exploit
                    #sample from
                    included_density = stats.gaussian_kde(included_params)
                    if np.sum(excluded_trees_mask) > burnined_in_period:
                        excluded_density = stats.gaussian_kde(excluded_params)
                    reject = True
                    while reject:
                        sample = included_density.resample()[:,0].T
                        if np.sum(excluded_trees_mask) > burnined_in_period:
                            include_prob = included_density.evaluate(sample)
                            exclude_prob = excluded_density.evaluate(sample)
                            reject_rate = exclude_prob / (include_prob + exclude_prob)
                        else:
                            reject_rate = 0
                        reject = np.random.uniform(0.0,1.0) < reject_rate
                    for i in range(len(sample)):
                        name = parameter_ranges[i][0]
                        all_kwargs[name] = int(np.clip(sample[i], parameter_ranges[i][1], parameter_ranges[i][2]))
                        parameter_ranges_values[name].append(all_kwargs[name])
                        # print('exploit %s - %d' % (name, all_kwargs[name]))

                # todo: support learning batches of trees
                all_kwargs['number_of_trees'] = 1

                learner = self.create_learner(**all_kwargs)
                new_forest = learner.Learn(bufferCollection)
                self.full_forest.AddForest(new_forest)

            # prune forest
            predictor_wrapper = self.create_predictor(self.full_forest, **kwargs)
            self.pruned_forest, tree_weights, error = greedy_prune(predictor_wrapper, self.error_calculator, 1.0, **kwargs)

            included_trees_mask = tree_weights == 1
            excluded_trees_mask = tree_weights == 0
            print np.sum(included_trees_mask)
            print len(parameter_ranges)
            included_params = np.zeros((len(parameter_ranges), np.sum(included_trees_mask)))
            excluded_params = np.zeros((len(parameter_ranges), np.sum(excluded_trees_mask)))
            # print included_trees_mask
            for (i, (name, range_min, range_max, use_log_space)) in enumerate(parameter_ranges):
                included_params[i,:] = np.array([x for x in itertools.compress(parameter_ranges_values[name], included_trees_mask)])
                excluded_params[i,:] = np.array([x for x in itertools.compress(parameter_ranges_values[name], excluded_trees_mask)])

                print('%s included ' % name)
                print np.histogram(included_params[i,:], bins=np.arange(range_min, range_max+1))
                print('%s excluded ' % name)
                print np.histogram(excluded_params[i,:], bins=np.arange(range_min, range_max+1))

        # prune forest using densities
        # self.pruned_forest = forest_data.Forest()
        # tree_params = np.zeros((len(tree_weights), len(parameter_ranges)))
        # for (i, (name, range_min, range_max, use_log_space)) in enumerate(parameter_ranges):
        #     tree_params[:,i] = np.array(parameter_ranges_values[name])
        # for i in range(len(tree_weights)):
        #     include_prob = included_density.evaluate(tree_params[i,:])
        #     exclude_prob = excluded_density.evaluate(tree_params[i,:])
        #     reject_rate = exclude_prob / (include_prob + exclude_prob)
        #     accept = np.random.uniform(0.0,1.0) > reject_rate
        #     if accept:
        #         print('add %d %s %f' % (i, str(tree_params[i]), reject_rate))
        #         self.pruned_forest.AddTree(self.full_forest.GetTree(i))

     
        if use_all_trees:
            final_predictor_wrapper = self.create_predictor(self.full_forest, **kwargs)
        else:
            predictor_wrapper = self.create_predictor(self.full_forest, **kwargs)
            self.pruned_forest, tree_weights, error = greedy_add(predictor_wrapper, self.error_calculator, **kwargs)
            final_predictor_wrapper = self.create_predictor(self.pruned_forest, **kwargs)

        return final_predictor_wrapper





def create_sample_prune_learner(**kwargs):
    data_type = kwargs.get('data_type')
    prediction_type = kwargs.get('prediction_type')

    if data_type == 'matrix' and prediction_type == 'classification':
        return SamplePruneWrapper(  make_uber_data_prepare(kwargs),
                              uber_create_learner,
                              make_uber_create_predictor(kwargs),
                              CrossEntropyError(),
                              kwargs)
    elif data_type == 'matrix' and prediction_type == 'regression':
        return SamplePruneWrapper(  make_uber_data_prepare(kwargs),
                              uber_create_learner,
                              make_uber_create_predictor(kwargs),
                              MseRegressionError(),
                              kwargs)
    else:
        raise Exception("create_sample_prune_learner data_type=%s prediction_type=%s is not supported" % 
                    (data_type, prediction_type))

