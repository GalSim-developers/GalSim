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
"""@file getWFC3Bandpass.py
Grab HST WFC3 bandpasses from the web, and then thin with rel_err = 1.e-3.  Note that the outputs of
this script, which are the files GALSIM_DIR/share/bandpasses/WFC3_?.dat, are already included in the
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

urldir = 'http://www.stsci.edu/hst/wfc3/ins_performance/throughputs/Throughput_Tables/'
for band in ['f105w', 'f125w', 'f160w']:
    urlfile = urldir + band + '.IR.tab'
    base = 'WFC3_ir_'+os.path.basename(urlfile).upper().replace('.IR.TAB', '.dat')
    file_ = urlopen(urlfile)
    i,x,f = np.loadtxt(file_, unpack=True)
    x /= 10.0 #Ang -> nm
    x1,f1 = galsim.utilities.thin_tabulated_values(x,f,rel_err=1.e-5, fast_search=False)
    x2,f2 = galsim.utilities.thin_tabulated_values(x,f,rel_err=1.e-4, fast_search=False)
    x3,f3 = galsim.utilities.thin_tabulated_values(x,f,rel_err=1.e-3, fast_search=False)
    print("{0} raw size = {1}".format(base,len(x)))
    print("    thinned sizes = {0}, {1}, {2}".format(len(x1),len(x2),len(x3)))

    with open(base, 'w') as out:
        out.write(
"""# WFC3 IR {0} total throughput
# File taken from {1}
#
#  Thinned by galsim.utilities.thin_tabulated_values to a relative error of 1.e-3
#  with fast_search=False.
#
# Wavelength(nm)  Throughput(0-1)
""".format(band, urlfile))
        for i in range(len(x3)):
            out.write(" {0:>10.2f}    {1:>10.5f}\n".format(x3[i], f3[i]))

for band in ['f275w', 'f336w']:
    urlfile = urldir + band + '.UVIS1.tab'
    urlfile2 = urldir + band + '.UVIS2.tab'
    base = 'WFC3_uvis_'+os.path.basename(urlfile).upper().replace('.UVIS1.TAB', '.dat')
    file_ = urlopen(urlfile)
    file2 = urlopen(urlfile2)
    i,x,f = np.loadtxt(file_, unpack=True)
    i2,x2,f2 = np.loadtxt(file2, unpack=True)
    x /= 10.0 #Ang -> nm
    x2 /= 10.0 #Ang -> nm

    # Average together the UVIS1 and UVIS2 throughput curves.
    assert all(x == x2)
    f = 0.5*(f+f2)

    x1,f1 = galsim.utilities.thin_tabulated_values(x,f,rel_err=1.e-5, fast_search=False)
    x2,f2 = galsim.utilities.thin_tabulated_values(x,f,rel_err=1.e-4, fast_search=False)
    x3,f3 = galsim.utilities.thin_tabulated_values(x,f,rel_err=1.e-3, fast_search=False)
    print("{0} raw size = {1}".format(base,len(x)))
    print("    thinned sizes = {0}, {1}, {2}".format(len(x1),len(x2),len(x3)))

    with open(base, 'w') as out:
        out.write(
"""# WFC3 UVIS {0} total throughput
# Average of UVIS1 and UVIS2 throughputs, from files
# {1}
# {2}
#
#  Thinned by galsim.utilities.thin_tabulated_values to a relative error of 1.e-3
#  with fast_search=False.
#
# Wavelength(nm)  Throughput(0-1)
""".format(band, urlfile, urlfile2))
        for i in range(len(x3)):
            out.write(" {0:>10.2f}    {1:>10.5f}\n".format(x3[i], f3[i]))
