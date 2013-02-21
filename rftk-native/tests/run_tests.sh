#!/bin/bash

function die {
    echo $1
    exit 1
}

ls test_* | grep -v '.cpp$' | while read test; do
    ./$test || die "FAILURE in $test"
done
