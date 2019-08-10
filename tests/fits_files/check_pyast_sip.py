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
# PyAst had been failing to write SIP files correctly, but they fixed this in
# v3.9.0.  We override their claim of success regardless, since they aren't
# necessarily accurate enough for our purposes (only accurate to 0.1 pixels).
# Thus, older PyAst versions work correctly in GalSim.

import starlink.Atl as Atl
import starlink.Ast as Ast
import astropy.io.fits as pyfits
import numpy

# http://fits.gsfc.nasa.gov/registry/sip/sipsample.fits
hdu = pyfits.open('sipsample.fits')[0]
fc = Ast.FitsChan(Atl.PyFITSAdapter(hdu))
wcs = fc.read()

# A random test position.  The "true" RA, Dec values are taken from ds9.
x = 242
y = 75
true_ra = (13 + 30/60. + 1.474154/3600. - 24.) * numpy.pi / 12.
true_dec = (47 + 12/60. + 51.794474/3600.) * numpy.pi / 180.

ra1, dec1 = wcs.tran( numpy.array([ [x], [y] ]))
print 'Initial read of sipsample.fits:'
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
hdu2.writeto('test_sip.fits', clobber=True)

hdu3 = pyfits.open('test_sip.fits')[0]
fc3 = Ast.FitsChan(Atl.PyFITSAdapter(hdu3))
wcs3 = fc3.read()

ra3, dec3 = wcs3.tran( numpy.array([ [x], [y] ]))
print 'After write/read round trip through fits file:'
print 'error in ra = ',(ra3-true_ra) * 180.*3600./numpy.pi, 'arcsec'
print 'error in dec = ',(dec3-true_dec) * 180.*3600./numpy.pi, 'arcsec'
