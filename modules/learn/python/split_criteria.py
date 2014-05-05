import rftk.try_split as try_split
import rftk.should_split as should_split
from utils import *

def create_try_split_criteria(unused_kwargs_keys=[], **kwargs):
    try_split_criteria_list = []

    min_node_size = int( pop_kwargs(kwargs, 'min_node_size', unused_kwargs_keys, 2) )
    min_node_size_criteria = try_split.MinNodeSizeCriteria(min_node_size)
    try_split_criteria_list.append( min_node_size_criteria )

    if 'max_depth' in kwargs:
        max_depth = int( pop_kwargs(kwargs, 'max_depth', unused_kwargs_keys) )
        max_depth_criteria = try_split.MaxDepthCriteria(max_depth)
        try_split_criteria_list.append(max_depth_criteria)
    if 'max_seconds_to_learn' in kwargs:
        max_seconds_to_learn = int( pop_kwargs(kwargs, 'max_seconds_to_learn', unused_kwargs_keys) )
        time_limit_criteria = try_split.TimeLimitCriteria(max_seconds_to_learn)
        try_split_criteria_list.append(time_limit_criteria)
    if not try_split_criteria_list:
        try_split_no_criteria = try_split.TrySplitNoCriteria()
        try_split_criteria_list.append(try_split_no_criteria)
    return try_split.TrySplitCombinedCriteria(try_split_criteria_list)


def create_should_split_criteria(unused_kwargs_keys=[], **kwargs):
    split_criteria_type =  pop_kwargs(kwargs, 'split_criteria_type', unused_kwargs_keys, 'standard')
    should_split_criteria_list = []

    if split_criteria_type == 'standard':
        min_impurity = float( pop_kwargs(kwargs, 'min_impurity', unused_kwargs_keys, 0.0) )
        min_impurity_criteria = should_split.MinImpurityCriteria(min_impurity)
        should_split_criteria_list.append(min_impurity_criteria)

        if 'min_child_size' in kwargs:
            min_child_size = int(pop_kwargs(kwargs, 'min_child_size', unused_kwargs_keys))
            min_child_size_criteria = should_split.MinChildSizeCriteria(min_child_size)
            should_split_criteria_list.append(min_child_size_criteria)
        if 'min_child_size_sum' in kwargs:
            min_child_size_sum = int(pop_kwargs(kwargs, 'min_child_size_sum', unused_kwargs_keys))
            min_child_size_sum_criteria = should_split.MinChildSizeSumCriteria(min_child_size_sum)
            should_split_criteria_list.append(min_child_size_sum_criteria)      

    elif split_criteria_type == 'biau2008':
            min_child_size_criteria = should_split.MinChildSizeCriteria(1)
            should_split_criteria_list.append(min_child_size_criteria)
    elif split_criteria_type == 'biau2012':
        pass
    elif split_criteria_type == 'online_consistent':
        min_impurity = float( pop_kwargs(kwargs, 'min_impurity', unused_kwargs_keys, 0.0))
        number_of_data_to_split_root = float( pop_kwargs(kwargs, 'number_of_data_to_split_root', unused_kwargs_keys) )
        number_of_data_to_force_split_root_ratio = float( pop_kwargs(kwargs, 'number_of_data_to_force_split_root_ratio', unused_kwargs_keys))
        split_rate_growth = float( pop_kwargs(kwargs, 'split_rate_growth', unused_kwargs_keys)  )

        online_consistent_criteria = should_split.OnlineConsistentCriteria( min_impurity,
                                                                            number_of_data_to_split_root,
                                                                            number_of_data_to_split_root*number_of_data_to_force_split_root_ratio,
                                                                            split_rate_growth)
        should_split_criteria_list.append(online_consistent_criteria)
    else:
        raise Exception("unknown split_criteria_type %s" % split_criteria_type)

    return should_split.ShouldSplitCombinedCriteria(should_split_criteria_list)

def no_split_criteria(**kwargs):
    should_split_criteria_list = []
    should_split_no_criteria = should_split.ShouldSplitNoCriteria()
    should_split_criteria_list.append(should_split_no_criteria)
    return should_split.ShouldSplitCombinedCriteria(should_split_criteria_list)

def create_should_split_consistent_criteria(**kwargs):

    min_impurity = float( kwargs.get('min_impurity', 0.0) )
    number_of_data_to_split_root = float( kwargs.get('number_of_data_to_split_root') )
    number_of_data_to_force_split_root = float( kwargs.get('number_of_data_to_force_split_root') )
    split_rate_growth = float( kwargs.get('split_rate_growth') )

    return should_split.OnlineConsistentCriteria(  min_impurity,
                                                    number_of_data_to_split_root,
                                                    number_of_data_to_force_split_root,
                                                    split_rate_growth)
