# Copyright (c) 2012-2018 by the GalSim developers team on GitHub
# https://github.com/GalSim-developers
#
# This file is part of GalSim: The modular galaxy image simulation toolkit.
# https://github.com/GalSim-developers/GalSim
#
# GalSim is free software: redistribution and use in source and binary forms,
# with or without modification, are permitted provided that the following
# conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions, and the disclaimer given in the accompanying LICENSE
#    file.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions, and the disclaimer given in the documentation
#    and/or other materials provided with the distribution.
#

from __future__ import print_function
import os
import sys
import numpy as np

import galsim
from galsim_test_helpers import *


def check_dep(f, *args, **kwargs):
    """Check that some function raises a GalSimDeprecationWarning as a warning, but not an error.
    """
    import warnings
    # Cause all warnings to always be triggered.
    # Important in case we want to trigger the same one twice in the test suite.
    warnings.simplefilter("always")

    # Check that f() raises a warning, but not an error.
    with warnings.catch_warnings(record=True) as w:
        res = f(*args, **kwargs)
    assert len(w) >= 1, "Calling %s did not raise a warning"%str(f)
    #print([ str(wk.message) for wk in w ])
    assert issubclass(w[0].category, galsim.GalSimDeprecationWarning)
    return res

if __name__ == "__main__":
    pass
