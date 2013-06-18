"""@file test_interpolants_parametric.py  Tests of interpolants using parametric galaxy models.

A companion script to `test_interpolants.py`, but instead of using `RealGalaxy` objects we instead
use Sersic models drawn into `InterpolatedImage` instances to try and get to the nitty-gritty of
the issues with interpolators.
"""

import numpy as np
import galsim
