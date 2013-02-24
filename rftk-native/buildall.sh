#!/bin/bash
set -e
cd $PYTHONPATH/rftk-native/assert_util
python setup.py build_ext --swig=swig2.0 --inplace -f rftkinstall
cd $PYTHONPATH/rftk-native/bootstrap
python setup.py build_ext --swig=swig2.0 --inplace -f rftkinstall
cd $PYTHONPATH/rftk-native/buffers
python setup.py build_ext --swig=swig2.0 --inplace -f rftkinstall
cd $PYTHONPATH/rftk-native/forest_data
python setup.py build_ext --swig=swig2.0 --inplace -f rftkinstall
cd $PYTHONPATH/rftk-native/features
python setup.py build_ext --swig=swig2.0 --inplace -f rftkinstall
cd $PYTHONPATH/rftk-native/feature_extractors
python setup.py build_ext --swig=swig2.0 --inplace -f rftkinstall
cd $PYTHONPATH/rftk-native/best_split
python setup.py build_ext --swig=swig2.0 --inplace -f rftkinstall
cd $PYTHONPATH/rftk-native/predict
python setup.py build_ext --swig=swig2.0 --inplace -f rftkinstall
cd $PYTHONPATH/rftk-native/train
python setup.py build_ext --swig=swig2.0 --inplace -f rftkinstall
