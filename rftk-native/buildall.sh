#!/bin/bash
cd $PYTHONPATH/rftk-native/assert_util
python setup.py build_ext --inplace -f rftkinstall
cd $PYTHONPATH/rftk-native/buffers
python setup.py build_ext --inplace -f rftkinstall
cd $PYTHONPATH/rftk-native/features
python setup.py build_ext --inplace -f rftkinstall

