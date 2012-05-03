#!/usr/bin/env python
"""
A script that can be run from the command-line to check the properties of images, parallel to
MeasMoments as compiled from the .cpp
"""

import sys
import os
import subprocess
import math

# This machinery lets us run Python examples even though they aren't positioned properly to find
# galsim as a package in the current directory. 
try:
    import galsim
except ImportError:
    path, filename = os.path.split(__file__)
    sys.path.append(os.path.abspath(os.path.join(path, "..")))
    import galsim

# We want to take arguments from the command-line, i.e.
# MeasMoments.py image_file [guesssig sky_level x_centroid y_centroid]
# where the ones in brackets are optional.  This will make the behavior completely parallel to the
# executable from the C++, so those who are used to its behavior can simply replace it with a call
# to this python script.

numArg = len(sys.argv)-1 # first entry in sys.argv is the name of the script
if (numArg < 1 or numArg > 5):
    raise RuntimeError("Wrong number of command-line arguments: should be in the range 1...5!")

image_file = argv[1]
