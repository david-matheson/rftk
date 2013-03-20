#!/bin/bash

ls test_* | grep -v '.cpp$' | while read test; do
    echo "Beginning test suite $test"
    ./$test || echo "FAILURE in test suite $test"
done
