import rftk.try_split as try_split
import rftk.should_split as should_split

def create_try_split_criteria(**kwargs):
    try_split_criteria_list = []
    if 'min_node_size' in kwargs:
        min_node_size = int( kwargs.get('min_node_size', 1) )
        print 'min_node_size %d' % min_node_size
        min_node_size_criteria = try_split.MinNodeSizeCriteria(min_node_size)
        try_split_criteria_list.append( min_node_size_criteria )
    if 'max_depth' in kwargs:
        max_depth = int( kwargs.get('max_depth') )
        print 'max_depth %d' % max_depth
        max_depth_criteria = try_split.MaxDepthCriteria(max_depth)
        try_split_criteria_list.append(max_depth_criteria)
    if 'max_seconds_to_learn' in kwargs:
        max_seconds_to_learn = int( kwargs.get('max_seconds_to_learn') )
        print 'max_seconds_to_learn %d' % max_seconds_to_learn
        time_limit_criteria = try_split.TimeLimitCriteria(max_seconds_to_learn)
        try_split_criteria_list.append(time_limit_criteria)
    if not try_split_criteria_list:
        try_split_no_criteria = try_split.TrySplitNoCriteria()
        try_split_criteria_list.append(try_split_no_criteria)
    return try_split.TrySplitCombinedCriteria(try_split_criteria_list)


def create_should_split_criteria(**kwargs):
    should_split_criteria_list = []
    if 'min_child_size' in kwargs:
        min_child_size = int( kwargs.get('min_child_size') )
        print 'min_child_size %d' % min_child_size
        min_child_size_criteria = should_split.MinChildSizeCriteria(min_child_size)
        should_split_criteria_list.append(min_child_size_criteria)
    if 'min_child_size_sum' in kwargs:
        min_child_size_sum = int( kwargs.get('min_child_size_sum') )
        print 'min_child_size_sum %d' % min_child_size_sum
        min_child_size_sum_criteria = should_split.MinChildSizeSumCriteria(min_child_size_sum)
        should_split_criteria_list.append(min_child_size_sum_criteria)      
    if 'min_impurity' in kwargs:
        min_impurity = float( kwargs.get('min_impurity') )
        print 'min_impurity %f' % min_impurity
        min_impurity_criteria = should_split.MinImpurityCriteria(min_impurity)
        should_split_criteria_list.append(min_impurity_criteria)
    if not should_split_criteria_list:
        should_split_no_criteria = should_split.ShouldSplitNoCriteria()
        should_split_criteria_list.append(should_split_no_criteria)
    return should_split.ShouldSplitCombinedCriteria(should_split_criteria_list)