from joblib import Parallel, delayed
from datetime import datetime

# import rftk.forest_builder_offline.
import rftk.forest_builder_offline.node_splitter as node_splitter

# Use global variable to get around joblib pickle
# Globals use the the os fork memory standard 
gData = None
gIndices = None
gYs = None

def _build_tree(tree_builder, tree_id, random_seed):
    treebuilder.train(gData, gIndices, gYs)
    return treebuilder.get_data()

class ForestBuilder:

    def __init__(self, tree_builder_module, 
                        weight_sampler, 
                        leaf_model_factory, 
                        node_splitter_init_params, 
                        number_of_trees, 
                        number_of_jobs ):
        self.tree_builder_module = tree_builder_module
        self.weight_sampler = weight_sampler
        self.leaf_model_factory = leaf_model_factory
        self.node_splitter_init_params = node_splitter_init_params

        self.number_of_trees = number_of_trees
        self.number_of_jobs = number_of_jobs

    def train_forest(self, data, indices, ys):
        global gData
        global gIndices
        global gYs

        gData=data
        gIndices=indices
        gYs=ys

        tree_builder = tree_builder_module.TreeBuilder(weight_sampler=self.weight_sampler, 
                                                        leaf_model_factory=self.leaf_model_factory,
                                                        node_splitter_init_params=self.node_splitter_init_params)

        self.tree_data_list = Parallel(n_jobs=self.number_of_jobs)(delayed(_build_tree)
                                                                (tree_builder=tree_builder,
                                                                tree_id=i,
                                                                random_seed=datetime.now().microsecond + i),
                                                                for i in range(self.number_of_trees))
        return self.tree_data_list


