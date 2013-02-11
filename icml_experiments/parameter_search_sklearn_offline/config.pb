language: PYTHON
name: "parameter_search_sklearn_offline"

variable {
  name: "number_of_features"
  type: INT
  size: 1
  min: 4
  max: 16
}

variable {
  name: "max_depth"
  type: INT
  size: 1
  min: 2
  max: 1000
}

variable {
  name: "min_samples_split"
  type: INT
  size: 1
  min: 2
  max: 100
}

variable {
  name: "min_samples_leaf"
  type: INT
  size: 1
  min: 1
  max: 100
}
