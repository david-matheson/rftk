
import numpy as np
import rftk.forest_data as forest_data
import rftk.buffers as buffers
import learn

class ZeroOneClassificationError:
    def error(self, predictor_wrapper, tree_weights, leafs, **kwargs):
        if leafs is not None:
            y_hat = predictor_wrapper.predict_oob(x=kwargs.get('x'), tree_weights=tree_weights, leafs=leafs).argmax(axis=1)
        else:
            y_hat = predictor_wrapper.predict_oob(x=kwargs.get('x'), tree_weights=tree_weights).argmax(axis=1)
        error = 1.0 - np.mean(kwargs.get('classes') == y_hat)
        return error

def tree_weights_for_all_trees(forest_size):
    tree_weights=buffers.as_vector_buffer( np.ones(forest_size, dtype=np.float64) )
    return tree_weights
    
def tree_weights_for_swap(forest_size, tree_to_swap_out):
    tree_weights=tree_weights_for_all_trees(forest_size)
    tree_weights.Set(tree_to_swap_out, 0)
    return tree_weights

class GreedyAddSwapWrapper:

    def __init__(self, prepare_data, create_learner, create_predictor, error_calculator, kwargs ):
        self.prepare_data = prepare_data
        self.create_learner = create_learner
        self.create_predictor = create_predictor
        self.forest = None
        self.error_calculator = error_calculator
        self.init_kwargs = kwargs

    def fit(self, **kwargs):
        all_kwargs = dict(self.init_kwargs.items() + kwargs.items())

        learner = self.create_learner(**all_kwargs)
        bufferCollection = self.prepare_data(**kwargs)
        new_forest = learner.Learn(bufferCollection)

        if self.forest is None:
            self.forest = new_forest
        else:
            for new_tree_index in range(new_forest.GetNumberOfTrees()):
                operation = "None"
                tree_index_to_remove = -1
                forest_size = self.forest.GetNumberOfTrees()

                predictor_wrapper = self.create_predictor(self.forest, **kwargs)
                best_forest_error = self.error_calculator.error(predictor_wrapper, 
                                                                tree_weights=tree_weights_for_all_trees(forest_size), 
                                                                leafs=None,
                                                                **kwargs)

                # Try just adding the tree
                new_tree = new_forest.GetTree(new_tree_index)
                predictor_wrapper.add_tree(new_tree)
                # leafs = predictor_wrapper.predict_leafs(x=kwargs.get('x'))

                new_error = self.error_calculator.error(predictor_wrapper, 
                                                        tree_weights=tree_weights_for_all_trees(forest_size+1), 
                                                        leafs=None,
                                                        **kwargs)

                if new_error <= best_forest_error:
                    best_forest_error = new_error
                    operation = "Add"

                #Try swapping with each tree
                for tree_to_swap_out in range(forest_size):
                    new_error = self.error_calculator.error(predictor_wrapper, 
                                                            tree_weights=tree_weights_for_swap(forest_size+1, tree_to_swap_out),
                                                            leafs=None,
                                                             **kwargs)

                    if new_error < best_forest_error:
                        best_forest_error = new_error
                        operation = "Swap"
                        tree_index_to_remove = tree_to_swap_out


                if operation == "Add":
                    # print("Adding %d" % new_tree_index)
                    # print("error = %f" % best_forest_error)
                    self.forest.AddTree(new_forest.GetTree(new_tree_index))

                if operation == "Swap":
                    # print("Swapping %d for %d" % (tree_index_to_remove, new_tree_index))
                    # print("error = %f" % best_forest_error)
                    self.forest.RemoveTree(tree_index_to_remove)
                    self.forest.AddTree(new_forest.GetTree(new_tree_index))

        forest_predictor_wrapper = self.create_predictor(self.forest, **kwargs)
        return forest_predictor_wrapper

