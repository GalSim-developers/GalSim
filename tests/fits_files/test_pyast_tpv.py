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

# This script is featured on pyast issue page:
# https://github.com/timj/starlink-pyast/issues/8
# It also constructs the file tanflip.fits, which we use in the test suite.
# PyAst natively flips the order of RA and Dec when writing this file as a TAN WCS.
# This was a kind of input that wasn't otherwise featured in our test suite, but is
# apparently allowed by the fits standard.  So I added it.

import starlink.Atl as Atl
import starlink.Ast as Ast
import astropy.io.fits as pyfits
import numpy

# http://fits.gsfc.nasa.gov/registry/tpvwcs/tpv.fits
hdu = pyfits.open('tpv.fits')[0]
fc = Ast.FitsChan(Atl.PyFITSAdapter(hdu))
wcs = fc.read()

# A random test position.  The "true" RA, Dec values are taken from ds9.
'033009.340034', '-284350.811107', 418, 78, 2859.53882
x = 418
y = 78
true_ra = (3 + 30/60. + 9.340034/3600.) * numpy.pi / 12.
true_dec = -(28 + 43/60. + 50.811107/3600.) * numpy.pi / 180.

ra1, dec1 = wcs.tran( numpy.array([ [x], [y] ]))
print 'Initial read of tpv.fits:'
print 'error in ra = ',(ra1-true_ra) * 180.*3600./numpy.pi, 'arcsec'
print 'error in dec = ',(dec1-true_dec) * 180.*3600./numpy.pi, 'arcsec'

# Now cycle through writing and reading to a file

hdu2 = pyfits.PrimaryHDU()
fc2 = Ast.FitsChan(None, Atl.PyFITSAdapter(hdu2, clear=False), "Encoding=FITS-WCS")
success = fc2.write(wcs)
print 'success = ',success
if not success:
    fc2 = Ast.FitsChan(None, Atl.PyFITSAdapter(hdu2, clear=False))
    success = fc2.write(wcs)
    print 'Native encoding: success = ',success
fc2.writefits()
hdu2.writeto('test_tpv.fits', clobber=True)
# This also becomes the tanflip.fits test file, since pyast writes this # file with 
# CTYPE1=DEC--TAN 
# CTYPE2=RA---TAN.  
# So this provides a test of reading fits files with swapped axes.
hdu2.data = hdu.data
for key in hdu2.header.keys():
    # Remove the QV fields to get a pure TAN type.
    # (For some reason pyast removes the PV fields, but not QV here.)
    if 'QV' in key:
        del hdu2.header[key]
hdu2.writeto('tanflip.fits', clobber=True)

hdu3 = pyfits.open('test_tpv.fits')[0]
fc3 = Ast.FitsChan(Atl.PyFITSAdapter(hdu3))
wcs3 = fc3.read()
wcs3 = wcs3.findframe( Ast.SkyFrame() )

ra3, dec3 = wcs3.tran( numpy.array([ [x], [y] ]))
print 'ra1 = ',ra1
print 'ra3 = ',ra3
print 'dec1 = ',dec1
print 'dec3 = ',dec3
print 'After write/read round trip through fits file:'
print 'error in ra = ',(ra3-true_ra) * 180.*3600./numpy.pi, 'arcsec'
print 'error in dec = ',(dec3-true_dec) * 180.*3600./numpy.pi, 'arcsec'

# Make a version identical to tanflip.fits, but not actually flipped.
hdu4 = pyfits.PrimaryHDU()
fc4 = Ast.FitsChan(None, Atl.PyFITSAdapter(hdu4, clear=False),
                  "Encoding=FITS-WCS,FitsAxisOrder=<copy>")
fc4.write(wcs3)
fc4.writefits()
hdu4.data = hdu.data
for key in hdu4.header.keys():
    if 'QV' in key:
        del hdu4.header[key]
hdu4.writeto('tanflip2.fits', clobber=True)

