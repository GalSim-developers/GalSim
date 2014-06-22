#!/bin/bash

# This is just a short convenience script for pulling results from the Caltech cluster in which
# I ran the bulk of tests on #325 - anyone else may ignore this! (Barney)
rsync -acuvt --progress browe@lucius.caltech.edu:/home/browe/great3/64/GalSim/devel/external/test_sersic_highn/outputs/*.asc .
