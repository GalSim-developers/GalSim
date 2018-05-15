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

import galsim
import time

nlist = [0.3 + 0.1*k for k in range(60)]

im = galsim.Image(32,32, scale=0.28)
psf = galsim.Moffat(fwhm = 0.9, beta = 3)

for iter in range(2):
    tstart = time.time()
    for n in nlist:
        t0 = time.time()
        gal = galsim.Sersic(half_light_radius = 1.4, n=n)
        final = galsim.Convolve(psf,gal)
        im = final.drawImage(image=im)
        t1 = time.time()
        print 'n = %f, time = %f'%(n,t1-t0)
    tend = time.time()
    print 'Total time = %f'%(tend-tstart)

gsparams = galsim.GSParams(xvalue_accuracy=1.e-2, kvalue_accuracy=1.e-2,
                           maxk_threshold=1.e-2, folding_threshold=1.e-2)

for iter in range(2):
    tstart = time.time()
    for n in nlist:
        t0 = time.time()
        gal = galsim.Sersic(half_light_radius = 1.4, n=n, gsparams=gsparams)
        final = galsim.Convolve(psf,gal)
        im = final.drawImage(image=im)
        t1 = time.time()
        print 'n = %f, time = %f'%(n,t1-t0)
    tend = time.time()
    print 'Total time with loose params = %f'%(tend-tstart)
