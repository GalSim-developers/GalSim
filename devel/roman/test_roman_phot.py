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

import numpy as np
import datetime
import galsim.roman

# This is written as a unit test, since I originally conceived of this as a better test of
# ChromaticOpticalPSF photon shooting than the artificial one in test_chromatic.py.
# However, it's super slow.  The FFT takes 17 seconds and the photon shooting takes 7 seconds.
# So this is not going to be sufficient for an efficient implementation

import cProfile, pstats
pr = cProfile.Profile()

def test_roman_phot():
    """Test photon shooting with a Roman PSF and a realistic galaxy SED.
    """
    import time

    image_pos = galsim.PositionD(153, 921)
    bp_dict = galsim.roman.getBandpasses()
    bandpass = bp_dict['J129']
    sed = galsim.SED('CWW_Scd_ext.sed', wave_type='nm', flux_type='flambda')

    flux = 1.e6
    gal_achrom = galsim.Sersic(n=2.8, half_light_radius=0.03, flux=flux)
    gal = (gal_achrom * sed).withFlux(flux, bandpass=bandpass)

    world_pos = galsim.CelestialCoord(
        ra = galsim.Angle.from_hms('16:01:41.01257'),  # AG Draconis
        dec = galsim.Angle.from_dms('66:48:10.1312')
    )
    PA = 112*galsim.degrees  # Random.
    date = datetime.datetime(2025, 5, 16)  # NGR's 100th birthday.
    wcs_dict = galsim.roman.getWCS(PA=PA, world_pos=world_pos, date=date)
    wcs = wcs_dict[5]

    t0 = time.time()
    psf = galsim.roman.getPSF(SCA=5, bandpass='J129', pupil_bin=8, SCA_pos=image_pos, wcs=wcs)
    t1 = time.time()
    print('create psf time = ',t1-t0)

    # First draw with FFT
    obj = galsim.Convolve(gal, psf)
    t0 = time.time()
    im1 = obj.drawImage(bandpass, wcs=wcs, nx=50, ny=50, center=image_pos)
    t1 = time.time()
    print('fft time = ',t1-t0)
    print('im1.max,sum = ', im1.array.max(), im1.array.sum())

    # Compare to photon shooting
    rng = galsim.BaseDeviate(1234)
    t0 = time.time()
    im2 = obj.drawImage(bandpass, wcs=wcs, method='phot', rng=rng, nx=50, ny=50, center=image_pos)
    t1 = time.time()
    im1.write('im1.fits')
    im2.write('im2.fits')
    print('phot time = ',t1-t0)
    print('max diff/flux = ',np.max(np.abs(im1.array-im2.array)/flux))
    print('im2.max,sum = ', im2.array.max(), im2.array.sum())
    np.testing.assert_allclose(im2.array/flux, im1.array/flux, atol=1e-3)

    # And now with interpolation
    t0 = time.time()
    psf2 = galsim.roman.getPSF(SCA=5, bandpass='J129', pupil_bin=8, SCA_pos=image_pos, wcs=wcs,
                               n_waves=10)
    t1 = time.time()
    print('create interpolated psf with 10 waves time = ',t1-t0)
    obj2 = galsim.Convolve(gal, psf2)
    t0 = time.time()
    im3 = obj2.drawImage(bandpass, wcs=wcs, method='phot', rng=rng, nx=50, ny=50, center=image_pos)
    t1 = time.time()
    im3.write('im3.fits')
    print('n_waves=10 phot time = ',t1-t0)
    print('max diff/flux = ',np.max(np.abs(im1.array-im3.array)/flux))
    print('im3.max,sum = ', im3.array.max(), im3.array.sum())
    np.testing.assert_allclose(im3.array/flux, im1.array/flux, atol=1e-3)

    # Doing a second pass of obj isn't much faster
    t0 = time.time()
    im2 = obj.drawImage(bandpass, wcs=wcs, method='phot', rng=rng, nx=50, ny=50, center=image_pos)
    t1 = time.time()
    print('phot time = ',t1-t0)
    print('max diff/flux = ',np.max(np.abs(im1.array-im2.array)/flux))
    print('im2.max,sum = ', im2.array.max(), im2.array.sum())
    np.testing.assert_allclose(im2.array/flux, im1.array/flux, atol=1e-3)

    # But the interpolated one is fast for repeated use.
    t0 = time.time()
    pr.enable()
    im3 = obj2.drawImage(bandpass, wcs=wcs, method='phot', rng=rng, nx=50, ny=50, center=image_pos)
    pr.disable()
    t1 = time.time()
    print('n_waves=10 phot time = ',t1-t0)
    print('max diff/flux = ',np.max(np.abs(im1.array-im3.array)/flux))
    print('im3.max,sum = ', im3.array.max(), im3.array.sum())
    np.testing.assert_allclose(im3.array/flux, im1.array/flux, atol=1e-3)

test_roman_phot()
ps = pstats.Stats(pr).sort_stats('tottime')
ps.print_stats(30)
