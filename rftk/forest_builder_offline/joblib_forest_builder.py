from joblib import Parallel, delayed
from datetime import datetime

# import rftk.forest_builder_offline.
import rftk.forest_builder_offline.node_splitter as node_splitter
import rftk.utils.buffer_converters as buffer_converters

# Use global variable to get around joblib pickle
# Globals use the the os fork memory standard 
gData = None
gIndices = None
gYs = None
gTreeBuilder_Module = None

def _build_tree(weight_sampler, leaf_stats_factory, node_splitter_init_params, tree_id, random_seed):
    print "tree %d" % tree_id
    node_splitter_init_params.set_seed(random_seed)
    tree_builder = gTreeBuilder_Module.TreeBuilder(weight_sampler=weight_sampler,
                                                        leaf_stats_factory=leaf_stats_factory,
                                                        node_splitter_init_params=node_splitter_init_params)
    tree_builder.train(gData, gIndices, gYs)
    return tree_builder.get_data()

class ForestBuilder:

    def __init__(self, tree_builder_module, 
                        weight_sampler, 
                        leaf_stats_factory, 
                        node_splitter_init_params, 
                        number_of_trees, 
                        number_of_jobs ):
        self.tree_builder_module = tree_builder_module
        self.weight_sampler = weight_sampler
        self.leaf_stats_factory = leaf_stats_factory
        self.node_splitter_init_params = node_splitter_init_params

        self.number_of_trees = number_of_trees
        self.number_of_jobs = number_of_jobs

    def train_forest(self, data, indices, ys):
        global gData
        global gIndices
        global gYs
        global gTreeBuilder_Module

        gData=buffer_converters.as_matrix_buffer(data)
        gIndices=buffer_converters.as_matrix_buffer(indices)
        gYs=buffer_converters.as_matrix_buffer(ys)
        gTreeBuilder_Module=self.tree_builder_module



        self.tree_data_list = Parallel(n_jobs=self.number_of_jobs)(delayed(_build_tree)
                                                                (weight_sampler=self.weight_sampler,
                                                                leaf_stats_factory=self.leaf_stats_factory,
                                                                node_splitter_init_params=self.node_splitter_init_params,
                                                                tree_id=i,
                                                                random_seed=datetime.now().microsecond + i)
                                                                for i in range(self.number_of_trees))
        return self.tree_data_list


