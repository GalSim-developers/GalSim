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
"""@file getACSBandpass.py
Grab HST ACS bandpasses from the web, and then thin with rel_err = 1.e-3.  Note that the outputs of
this script, which are the files GALSIM_DIR/share/bandpasses/ACS*.dat, are already included in the
repository.  This script just lets users know where these files came from and how they were altered.
"""
from __future__ import print_function
try:
    from urllib2 import urlopen
except:
    from urllib.request import urlopen
import galsim
import numpy as np
import os

urldir = 'http://www.stsci.edu/hst/acs/analysis/throughputs/tables/'
for band in ['wfc_F435W', 'wfc_F606W', 'wfc_F775W', 'wfc_F814W', 'wfc_F850LP']:
    urlfile = urldir + band + '.dat'
    base = os.path.basename(urlfile).replace('wfc_', 'ACS_wfc_')
    file_ = urlopen(urlfile)
    x,f = np.loadtxt(file_, unpack=True)
    # For some reason, the F814W filter has repeated wavelengths in the file from STSci.  We
    # clip these out manually here.
    keep = np.concatenate([x[1:] - x[:-1] != 0.0, [True]])
    x = x[keep]
    f = f[keep]
    x /= 10.0 #Ang -> nm
    x1,f1 = galsim.utilities.thin_tabulated_values(x,f,rel_err=1.e-5, fast_search=False)
    x2,f2 = galsim.utilities.thin_tabulated_values(x,f,rel_err=1.e-4, fast_search=False)
    x3,f3 = galsim.utilities.thin_tabulated_values(x,f,rel_err=1.e-3, fast_search=False)
    print("{0} raw size = {1}".format(base,len(x)))
    print("    thinned sizes = {0}, {1}, {2}".format(len(x1),len(x2),len(x3)))

    with open(base, 'w') as out:
        out.write(
"""# ACS {0} total throughput
# File taken from {1}
#
#  Thinned by galsim.utilities.thin_tabulated_values to a relative error of 1.e-3
#  with fast_search=False.
#
# Wavelength(nm)  Throughput(0-1)
""".format(band, urlfile))
        for i in range(len(x3)):
            out.write(" {0:>10.2f}    {1:>10.5f}\n".format(x3[i], f3[i]))
