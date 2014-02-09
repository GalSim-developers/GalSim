# Copyright 2012-2014 The GalSim developers:
# https://github.com/GalSim-developers
#
# This file is part of GalSim: The modular galaxy image simulation toolkit.
#
# GalSim is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# GalSim is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with GalSim.  If not, see <http://www.gnu.org/licenses/>
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
