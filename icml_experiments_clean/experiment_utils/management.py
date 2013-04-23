import numpy as np
import itertools
from copy import copy

class ConfigurationDomain(object):
    def __init__(self, base_config):
        self.base_config = copy(base_config)

        self.extent = {}
        self.independent_variables = []
        for var, extent in base_config.__dict__.iteritems():
            if isinstance(extent, list):
                self.independent_variables.append(var)
                self.extent[var] = np.asarray(extent)

        # ensures two domains with the same variables and extents will have the
        # same shape
        self.independent_variables.sort()

    def variable_index_of(self, variable, variable_position):
        return np.argmax(self.extent[variable] == variable_position)

    def index_of(self, position):
        index = []
        for var, pos in zip(self.independent_variables, position):
            index.append(self.variable_index_of(var, pos))
        return tuple(index)

    def variable_position_of(self, variable, variable_index):
        return self.extent[variable][variable_index]

    def position_of(self, index):
        position = []
        for var, value in zip(self.independent_variables, index):
            position.append(self.variable_position_of(var, value))
        return tuple(position)

    def rank_of(self, variable):
        return self.independent_variables.index(variable)

    def slice_at(self, variable, position):
        new_config = copy(self.base_config)
        setattr(new_config, variable, position)
        return ConfigurationDomain(new_config)

    def extend(self, variable, extent):
        assert variable not in self.independent_variables
        assert isinstance(extent, list)
        
        new_config = copy(self.base_config)
        setattr(new_config, variable, extent)
        return ConfigurationDomain(new_config)

    @property
    def rank(self):
        return len(self.independent_variables)

    @property
    def shape(self):
        return tuple(len(self.extent[var]) for var in self.independent_variables)

    @property
    def extents(self):
        return [self.extent[var] for var in self.independent_variables]

    def __iter__(self):
        for position in itertools.product(*self.extents):
            yield position

    def configuration_at(self, position):
        new_config = copy(self.base_config)
        index = self.index_of(position)
        for var, extent, idx in zip(self.independent_variables, self.extents, index):
            setattr(new_config, var, extent[idx])
        return new_config


class MeasurementGrid(object):
    def __init__(self, domain, measurement_type):
        self.domain = domain
        self.measurement_type = measurement_type

        # allocate space for measurements
        for var in self.measurement_type.MEASURED_VALUES:
            setattr(self, var, np.zeros(self.domain.shape))

    def record_at(self, position, measurement):
        index = self.domain.index_of(position)

        for var in self.measurement_type.MEASURED_VALUES:
            tensor = getattr(self, var)
            tensor[index] = getattr(measurement, var)
            setattr(self, var, tensor)

    def value_at(self, position):
        index = self.domain.index_of(position)

        values = []
        for var in self.measurement_type.MEASURED_VALUES:
            tensor = getattr(self, var)
            values.append(tensor[index])

        return Measurement(*values)

    def slice_at(self, variable, position):
        new_domain = self.domain.slice_at(variable, position)
        new_grid = MeasurementGrid(new_domain, self.measurement_type)

        tensor_slice = [slice(None)] * self.domain.rank
        var_idx = self.domain.rank_of(variable)
        tensor_slice[var_idx] = self.domain.variable_index_of(variable, position)

        for var in self.measurement_type.MEASURED_VALUES:
            tensor = getattr(self, var)
            setattr(new_grid, var, tensor[tensor_slice])

        return new_grid

    def __str__(self):
        strings = \
            ["---- MeasurementGrid ----",
             "Measured values: {}".format(self.measurement_type.MEASURED_VALUES) ] + \
            ["Domain:"] + \
            ["  shape: {}".format(self.domain.shape)] + \
            ["  {}: {}".format(var, self.domain.extent[var])
              for var in self.domain.independent_variables ] + \
            ["-------------------------"]

        return "\n".join(strings)

            
#############

def join_grids_along_new_axis(join_axis, grids):
    # This function makes just enough sanity checks to be sure that
    # the join is possible.  It assumes you know what you're doing
    # otherwise.

    master_grid = grids[0]

    for grid in grids:
        assert join_axis not in grid.domain.independent_variables

    for grid in grids:
        assert master_grid.domain.independent_variables == grid.domain.independent_variables
        for var, extent in master_grid.domain.extent.iteritems():
            assert np.all(master_grid.domain.extent[var] == grid.domain.extent[var])

    joined_domain = master_grid.domain.extend(join_axis, range(len(grids)))
    joined_grid = MeasurementGrid(joined_domain, measurement_type=master_grid.measurement_type)

    # the joined domain doesn't put our new dimension at the end, it
    # puts it in alphabetical order so we need to permute the
    # dimensions of the new grids to match

    last_dim = len(master_grid.domain.independent_variables)
    new_dim = joined_domain.independent_variables.index(join_axis)
    perm = range(last_dim)
    perm = tuple(perm[:new_dim] + [last_dim] + perm[new_dim:])

    for var in joined_grid.measurement_type.MEASURED_VALUES:
        tensors = []
        for grid in grids:
            tensor = getattr(grid, var)
            expanded_tensor = np.transpose(tensor[...,np.newaxis], perm)
            tensors.append(expanded_tensor)

        setattr(joined_grid, var, np.concatenate(tensors, axis=new_dim))
        assert getattr(joined_grid, var).shape == joined_grid.domain.shape

    return joined_grid
