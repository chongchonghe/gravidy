#!/bin/bash

pushd ../tests/jobs/
make
echo "makefile was here" $(pwd)
popd
