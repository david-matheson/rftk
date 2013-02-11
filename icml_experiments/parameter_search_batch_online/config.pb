language: PYTHON
name: "parameter_search_batch_online"

variable {
  name: "null_probability"
  type: FLOAT
  size: 1
  min: 0
  max: 0.9
}

variable {
  name: "min_impurity_gain"
  type: FLOAT
  size: 1
  min: 0
  max: 0.3
}

variable {
  name: "impurity_probability"
  type: FLOAT
  size: 1
  min: 0.1
  max: 0.9
}

variable {
  name: "split_rate"
  type: FLOAT
  size: 1
  min: 1
  max: 3
}

variable {
  name: "number_of_features"
  type: INT
  size: 1
  min: 4
  max: 16
}

variable {
  name: "number_of_thresholds"
  type: INT
  size: 1
  min: 1
  max: 100
}

variable {
  name: "number_of_data_to_split_root"
  type: INT
  size: 1
  min: 1
  max: 20
}

variable {
  name: "number_of_data_to_force_split_root"
  type: INT
  size: 1
  min: 20
  max: 1000
}

