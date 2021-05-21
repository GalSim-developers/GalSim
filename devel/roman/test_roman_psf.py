# Copyright (c) 2012-2021 by the GalSim developers team on GitHub
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
import galsim.roman
import os
import numpy as np
import time

mag = 10.

sedpath_Star   = os.path.join(galsim.meta_data.share_dir, 'SEDs', 'vega.txt')
star_sed   = galsim.SED(sedpath_Star, wave_type='nm', flux_type='flambda')
bandpass = galsim.roman.getBandpasses(AB_zeropoint=True)['H158']
rng = galsim.BaseDeviate(1234)

WCS = galsim.roman.getWCS(world_pos  = galsim.CelestialCoord(ra=-90*galsim.degrees, \
                                                    dec=-50*galsim.degrees), 
                          PA          = 0.*galsim.degrees, 
                          SCAs        = [1],
                          PA_is_FPA   = True
                         )[1]

sed_ = star_sed.withMagnitude(mag, bandpass)
flux = sed_.calculateFlux(bandpass) * galsim.roman.exptime * galsim.roman.collecting_area
print('flux = ',flux)
image = galsim.Image(galsim.roman.n_pix, galsim.roman.n_pix, wcs=WCS)
xyI = image.center
sky_level = galsim.roman.getSkyLevel(bandpass, world_pos=WCS.toWorld(xyI))

for high, app in [ (False, True), (False, False), (True, True), (True, False) ]:
    PSF = galsim.roman.getPSF(1,
                              'H158',
                              SCA_pos             = None, 
                              approximate_struts  = app,
                              wavelength          = bandpass.effective_wavelength,
                              high_accuracy       = high,
                             )

    star = galsim.DeltaFunction() * flux
    star = galsim.Convolve(star, PSF)

    t0 = time.time()
    star_stamp = star.drawImage(wcs=WCS.local(xyI), center=xyI)
    t1 = time.time()

    image.fill(sky_level)
    #Doing the sky right means doing the next line. But that's slow, and not relevant to this test.
    #WCS.makeSkyImage(image, sky_level)

    b = image.bounds & star_stamp.bounds
    image[b] += star_stamp[b]
    image += galsim.roman.thermal_backgrounds['H158']*galsim.roman.exptime
    image.addNoise(galsim.PoissonNoise(rng))
    image -= np.median(image.array)
    t2 = time.time()

    image.write('test_%d%d.fits'%(high,app))

    print('high, app = ',high,app)
    print('drawImage time = ',t1-t0)
    print('addnoise time = ',t2-t1)
