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

