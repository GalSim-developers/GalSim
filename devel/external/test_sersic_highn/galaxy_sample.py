"""@file galaxy_sample.py Basic routines for handling the galaxy sample in sersic n range tests.
"""

import numpy as np


def get_galaxy_sample(filename="cosmos_sersics_sample_N300.asc"):
   """Returns (n_sersic, half_light_radius [arcsec], |g|), a tuple of NumPy arrays.
   """
   data = np.loadtxt(filename)
   n = data[:, 1]
   hlr = data[:, 2]
   gabs = data[:, 3]
   return n, hlr, gabs
