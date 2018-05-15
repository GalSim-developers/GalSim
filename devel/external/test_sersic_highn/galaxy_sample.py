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

"""@file galaxy_sample.py Catalog handling for the COSMOS galaxy sample in sersic tests.
"""
import numpy as np

def get(filename="cosmos_sersics_sample_N300.asc"):
   """Returns (n_sersic, half_light_radius [arcsec], |g|), a tuple of NumPy arrays.
   """
   try:
       data = np.loadtxt(filename)
   except IOError:
       import os  # In case this module is being called from outside devel/external/test_sersic_hign
       modulepath, modulefile = os.path.split(__file__)
       data = np.loadtxt(os.path.join(modulepath, filename))
   n = data[:, 1]
   hlr = data[:, 2]
   gabs = data[:, 3]
   return n, hlr, gabs
