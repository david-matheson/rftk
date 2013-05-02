import rftk.try_split as try_split

def create_try_split_criteria(**kwargs):
    try_split_criteria_list = []
    min_node_size = int( kwargs.get('min_node_size', 1) )
    min_node_size_criteria = try_split.MinNodeSizeCriteria(min_node_size)
    try_split_criteria_list.append( min_node_size_criteria )
    if 'max_depth' in kwargs:
        max_depth = int( kwargs.get('max_depth') )
        max_depth_criteria = try_split.MaxDepthCriteria(max_depth)
        try_split_criteria_list.append(max_depth_criteria)
    if 'max_seconds_to_learn' in kwargs:
        max_seconds_to_learn = int( kwargs.get('max_seconds_to_learn') )
        time_limit_criteria = try_split.TimeLimitCriteria(max_seconds_to_learn)
        try_split_criteria_list.append(time_limit_criteria)
    return try_split.TrySplitCombinedCriteria(try_split_criteria_list)