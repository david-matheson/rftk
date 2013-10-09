import numpy as np
import rftk.forest_data as forest_data
import rftk.buffers as buffers
import learn

class ZeroOneClassificationError:
    def y_hat_sum(self, tree_weights, oob_weights, leaf_ys, **kwargs):
        weights = oob_weights * tree_weights
        weighted_ys = weights * leaf_ys
        y_hat_sum = weighted_ys.sum(axis=2) 
        return y_hat_sum, weights, weights.sum(axis=1)

    def error(self, y_hat_sum, weights, weights_sum, leaf_ys, **kwargs):
        y_hat_avg = y_hat_sum / weights_sum
        y_hat = y_hat_avg.argmax(axis=0)
        error = np.mean(kwargs.get('classes') != y_hat)
        return error

    def error_without_tree(self, y_hat_sum, weights, weights_sum, leaf_ys, tree_id_to_revove, **kwargs):
        y_hat_sum_without_tree = y_hat_sum - (leaf_ys[:,:,tree_id_to_revove] * weights[:,tree_id_to_revove])
        weights_without_tree = weights_sum - weights[:,tree_id_to_revove]
        y_hat_avg = y_hat_sum_without_tree / weights_without_tree
        y_hat = y_hat_avg.argmax(axis=0)
        error = np.mean(kwargs.get('classes') != y_hat)
        return error

class MseRegressionError:
    def y_hat_sum(self, tree_weights, oob_weights, leaf_ys, **kwargs):
        weights = oob_weights * tree_weights
        weighted_ys = weights * leaf_ys
        y_hat_sum = weighted_ys.sum(axis=2) 
        return y_hat_sum, weights, weights.sum(axis=1)

    def error(self, y_hat_sum, weights, weights_sum, leaf_ys, **kwargs):
        y_hat = y_hat_sum / weights_sum
        error = np.mean((kwargs.get('y') - y_hat.T)**2)
        return error

    def error_without_tree(self, y_hat_sum, weights, weights_sum, leaf_ys, tree_id_to_revove, **kwargs):
        y_hat_sum_without_tree = y_hat_sum - (leaf_ys[:,:,tree_id_to_revove] * weights[:,tree_id_to_revove])
        weights_without_tree = weights_sum - weights[:,tree_id_to_revove]
        y_hat = y_hat_sum_without_tree / weights_without_tree
        error = np.mean((kwargs.get('y') - y_hat.T)**2)
        return error


def greedy_prune(forest_predictor, error_calculator, **kwargs):
    # make predictions of each tree
    leaf_ys, oob_weights = forest_predictor.predict_leafs_ys(**kwargs)

    forest = forest_predictor.get_forest()
    forest_size = forest.GetNumberOfTrees()
    tree_weights = np.ones(forest_size, dtype=np.float32)

    # prune until removing a tree makes the oob error worse
    continue_purning = True
    while continue_purning:
        y_hat_sum, weights, weights_sum = error_calculator.y_hat_sum(tree_weights=tree_weights, 
                                                            oob_weights=oob_weights,
                                                            leaf_ys=leaf_ys,
                                                            **kwargs)

        # Measure without new tree
        best_forest_error = error_calculator.error(y_hat_sum=y_hat_sum, 
                                                        weights=weights,
                                                        weights_sum=weights_sum,
                                                        leaf_ys=leaf_ys,
                                                        **kwargs)

        # find the tree which being removed produces the least error
        tree_index_to_remove = -1
        for tree_index in range(forest_size):
            if tree_weights[tree_index] > 0:
                error_without_tree_index = error_calculator.error_without_tree(y_hat_sum=y_hat_sum, 
                                                                            weights=weights,
                                                                            weights_sum=weights_sum,
                                                                            leaf_ys=leaf_ys,
                                                                            tree_id_to_revove=tree_index,
                                                                            **kwargs)
                if(error_without_tree_index <= best_forest_error):
                    tree_index_to_remove = tree_index
                    best_forest_error = error_without_tree_index
        if tree_index_to_remove != -1:
            # print('pruning %d - %f' % (tree_index_to_remove, best_forest_error))
            tree_weights[tree_index_to_remove] = 0
        else:
            continue_purning = False

    new_forest = forest_data.Forest()
    for tree_index in range(forest_size):
        if tree_weights[tree_index] > 0:
            new_forest.AddTree( forest.GetTree(tree_index) )

    # print('pruned sizes %d => %d (%f)' % (forest_size, new_forest.GetNumberOfTrees(), best_forest_error))
    forest_predictor.set_forest(new_forest)
    return forest_predictor

