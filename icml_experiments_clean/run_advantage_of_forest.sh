#!/bin/bash

RESULTS_FOLDER="results"
CONFIG_MODULE="experiment_config.advantage_of_forest"
FIGURE_FILE="figures/advantage_of_forest.png"

echo "Running experiment..."
python train_online.py -c "$CONFIG_MODULE" -o "$RESULTS_FOLDER/mog_online-{}.pkl"

python plot_advantage_of_forest.py \
    --forest "$RESULTS_FOLDER/mog_online-forest.pkl" \
    --trees $RESULTS_FOLDER/mog_online-tree*.pkl \
    --out "$FIGURE_FILE"