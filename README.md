rftk (Random Forest Toolkit)
===================

A modular and efficient C++ random forest toolkit with a python wrapper.  

Key Features
--------------
+ Customizable pipeline for determining best split
+ Pipeline steps can be run per forest, per tree or per node
+ Different forest building strategies including offline (depth first and breadth first) and online (fixed fringe)
+ Factories for common forest configurations (Breiman, Shotton, etc) 
+ Lazy evaluation of features allow features to be function of an index and the data 
+ Generic indexing allows a datapoint to be a row of a matrix or a pixel in an image
+ Support for different buffer precision (ie 32 bit vs 64 bit) and sparse buffers

Setup
--------------
Install dependencies

    > sudo apt-get install scons
    > sudo apt-get install libboost-all-dev

Clone the project

    > git clone https://github.com/david-matheson/rftk.git

Build rftk

    > cd /path/to/rftk
    > scons

Add rftk to your PYTHONPATH

    > PYTHONPATH=/path/to/rftk/
    > export PYTHONPATH  

Run unit tests.  For now, the debug version of library must be installed for all python unit tests 

    > scons test-native
    Running 113 test cases...
    Warning there is another bufferkey conflict of a different type

    *** No errors detected
    /media/data/projects/rftk-github/build/release/test-cpp
    Running 113 test cases...
    Warning there is another bufferkey conflict of a different type

    *** No errors detected

    > scons install-debug 
    > python -m unittest discover tests '*.py'
    ............................................
    ----------------------------------------------------------------------
    Ran 44 tests in 0.105s

    OK
    > scons install-release


Ideology
--------------
Modules that are called within tight loops are combined with templates.  Modules that are called at a higher level are combined with inheritance  The configuration of the forest is done in python.  Templated components are combined with swig (.i files)

How to use the library
--------------

    import rftk
    learner = rftk.learn.create_vanilia_classifier(                         
                            number_of_trees=100,
                            number_of_jobs=5)
    forest = learner.fit(x=X_train, classes=Y_train)


Under the hood
--------------
The core unit of work is a pipeline step.  A pipeline step reads from inputt buffers and writes to output buffers. Pipeline steps are chained into a pipeline. Below is simplified pipeline where "->"" lines are the steps, "-" lines are the buffers that are read at each step and "+" lines are the buffers that are written to by each step. 

  -> sample feature parameters and random split points
      + feature_params_buffer
      + split_points_buffer
  -> extract features
      - feature_params_buffer
      + feature_values_buffer
  -> estimate sufficient statistics for 
      - split_points_buffer
      - feature_values_buffer
      + statistics_buffer
  -> measure information gain  
      - statistics_buffer
      + information_gain_of_splitpoints


