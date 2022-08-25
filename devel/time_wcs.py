# Copyright (c) 2012-2022 by the GalSim developers team on GitHub
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

import time
import timeit
import numpy as np
import galsim
import astropy.io.fits as fits

fns = ["sipsample.fits", "tpv.fits", "tanpv.fits"]

rng = np.random.default_rng()

size = 1_000_000

for fn in fns:
    header = fits.getheader(f"../tests/fits_files/{fn}")
    x = rng.uniform(0, header['NAXIS1'], size=size)
    y = rng.uniform(0, header['NAXIS2'], size=size)

    wcs = galsim.GSFitsWCS(header=header)
    ra, dec = wcs.xyToradec(x, y, units='rad')

    print()
    print(fn)
    print(f"PV?  {'PV1_1' in header}")
    print(f"SIP?  {'A_ORDER' in header}")

    t = min(timeit.repeat(lambda: wcs.xyToradec(x,y,units='rad'), number=3))
    print(f"xyToradec {t:.3f}")
    #t0 = time.time()
    #ra, dec = wcs.xyToradec(x, y, units='rad')
    #t1 = time.time()
    #print(f"xyToradec {t1-t0:.3f}")

    t = min(timeit.repeat(lambda: wcs.radecToxy(ra,dec,units='rad'), number=3))
    print(f"radecToxy {t:.3f}")
    #t0 = time.time()
    #x1, y1 = wcs.radecToxy(ra, dec, units='rad')
    #t1 = time.time()
    #print(f"radecToxy {t1-t0:.3f}")
