"""
This script makes a subcatalog out of a full one for the selection catalog, and writes it to file.
Meant for use for unit tests.
"""
import pyfits
import numpy as np

infile = '/Users/rmandelb/great3/data-23.5/real_galaxy_catalog_23.5_fits.fits'
matchfile = '../../examples/data/real_galaxy_catalog_example.fits'
outfile = '../../examples/data/real_galaxy_catalog_example_fits.fits'

matchdat = pyfits.getdata(matchfile)

dat = pyfits.getdata(infile)
# need to find the stuff to keep
match_ind = np.zeros(len(matchdat)).astype(int)
for ind in range(len(matchdat)):
    match_ind[ind] = list(dat['ident']).index(matchdat['ident'][ind])

new_dat = []
for ind in range(len(dat.columns)):
    new_dat.append(pyfits.Column(name=dat.columns[ind].name,
                                 format=dat.columns[ind].format,
                                 array=dat[dat.columns[ind].name][match_ind]))
new_hdu = pyfits.new_table(pyfits.ColDefs(new_dat))
new_hdu.writeto(outfile,clobber=True)
