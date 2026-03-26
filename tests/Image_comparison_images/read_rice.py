# Script posted to demonstrate astropy error:
# https://github.com/astropy/astropy/issues/15477
# Fails on astropy 5.3 .. 7.2

from astropy.io import fits

file_name = 'testF.fits.fz'

with fits.open(file_name, 'readonly') as hdu_list:
    hdu = hdu_list[1]
    hdu.data
