import numpy as np

import rftk.forest_builder_offline.node_splitter as node_splitter
import rftk.forest_builder_offline.tree_data as tree_data
import rftk.stop_criteria.criteria as criteria

class TreeBuilder:
    def __init__(self, 
                weight_sampler,
                leaf_stats_factory,
                node_splitter_init_params ):
        self.weight_sampler = weight_sampler
        self.leaf_stats_factory = leaf_stats_factory
        self.node_splitter_init_params = node_splitter_init_params

        self.current_node_index = 0       
        int_params_dim = node_splitter_init_params.feature_candidate_collection.max_int_params_dim()
        float_params_dim = node_splitter_init_params.feature_candidate_collection.max_float_params_dim()
        y_dim = leaf_stats_factory.get_ydim()
        self.tree_data = tree_data.TreeData(max_int_params_dim=int_params_dim+1, 
                                            max_float_params_dim=float_params_dim+1,
                                            y_dim=y_dim)
 

    def train(self, data, indices, ys):
        # draw weights for sample indices
        number_of_samples = indices.GetM()
        sample_weights = np.array(self.weight_sampler.sample(number_of_samples), dtype=np.float32)

        self.leaf_model = self.leaf_stats_factory.construct(sample_weights=sample_weights, ys=ys)

        self.node_splitter = node_splitter.NodeSplitter(init_params=self.node_splitter_init_params,
                                                data=data,
                                                indices=indices,
                                                sample_weights=sample_weights,
                                                ys=ys)
        # start the recursion
        sample_indices = np.array( range(number_of_samples), dtype=np.int32)
        sample_indices_active = sample_indices[ (sample_weights > 0.0) ]
        self.add_node(sample_indices=sample_indices_active, tree_depth=0)
        self.tree_data.truncate(self.current_node_index)

    def add_node(self, sample_indices, tree_depth):
        node_id = self.current_node_index
        # print tree_depth, node_id, len(sample_indices)
        self.tree_data.ys[node_id, :] = self.leaf_model.get_y(sample_indices)
        try:
            int_params, float_params, left_indices, right_indices = self.node_splitter.split(sample_indices=sample_indices, tree_depth=tree_depth)
            self.current_node_index = self.current_node_index + 1
            left_node_id = self.add_node(sample_indices=left_indices, tree_depth=tree_depth+1)
            self.current_node_index = self.current_node_index + 1
            right_node_id = self.add_node(sample_indices=right_indices, tree_depth=tree_depth+1)

            self.tree_data.int_params[node_id, :] = int_params
            self.tree_data.float_params[node_id, :] = float_params            
            self.tree_data.paths[node_id, :] = np.array([left_node_id, right_node_id])
            return node_id
        except criteria.CriteriaError:
            return node_id
        except:
            raise

    def get_data(self):
        return self.tree_data











