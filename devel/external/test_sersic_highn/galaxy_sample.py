"""@file galaxy_sample.py Catalog handling for the COSMOS galaxy sample in sersic tests.
"""
import numpy as np

def get(filename="cosmos_sersics_sample_N300.asc"):
   """Returns (n_sersic, half_light_radius [arcsec], |g|), a tuple of NumPy arrays.
   """
   data = np.loadtxt(filename)
   n = data[:, 1]
   hlr = data[:, 2]
   gabs = data[:, 3]
   return n, hlr, gabs
