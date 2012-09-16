"""
Some example scripts to make multi-object images using the GalSim library.
"""

import sys
import os
import math
import numpy
import logging
import time
import galsim

# Make multiple galaxy images
#
# New features in this demo:
#
# - Output multiple images to a multi-extension fits file
# - Building a config dictionary in Python to specify the image properties.
# - Reading in many of the relevant parameters from an input catalog.

def main(argv):
    """
    Make a fits image cube using parameters from an input catalog
      - The number of images in the cube matches the number of rows in the catalog.
      - Each image size is computed automatically by GalSim based on the Nyquist size.
      - Only galaxies.  No stars.
      - PSF is Moffat
      - Each galaxy is bulge plus disk: deVaucouleurs + Exponential.
      - Parameters taken from the input catalog:
        - PSF beta
        - PSF FWHM
        - PSF e1
        - PSF e2
        - PSF trunc
        - Bulge half-light-radius
        - Bulge e1
        - Bulge e2
        - Bulge flux
        - Disc half-light-radius
        - Disc e1
        - Disc e2
        - Disc flux
        - Galaxy dx (two components have same center)
        - Galaxy dy
      - Applied shear is the same for each file
      - Noise is poisson using a nominal sky value of 1.e6
    """
    logging.basicConfig(format="%(message)s", level=logging.INFO, stream=sys.stdout)
    logger = logging.getLogger("demo4")

    # Define some parameters we'll use below.

    cat_file_name = os.path.join('input','galsim_default_input.asc')
    multi_file_name = os.path.join('output','multi.fits')

    random_seed = 8241573
    sky_level = 1.e6                # ADU / arcsec^2
    pixel_scale = 1.0               # arcsec / pixel  (size units in input catalog are pixels)
    gal_flux = 1.e6                 # arbitrary choise, makes nice (not too) noisy images
    gal_g1 = -0.009                 #
    gal_g2 = 0.011                  #
    xsize = 64                      # pixels
    ysize = 64                      # pixels

    logger.info('Starting demo script 4 using:')
    logger.info('    - parameters taken from catalog %r',cat_file_name)
    logger.info('    - Moffat PSF (parameters from catalog)')
    logger.info('    - pixel scale = %.2f',pixel_scale)
    logger.info('    - Bulge + Disc galaxies (parameters from catalog)')
    logger.info('    - Applied gravitational shear = (%.3f,%.3f)',gal_g1,gal_g2)
    logger.info('    - Poisson noise (sky level = %.1e).', sky_level)

    # Setup the config dict
    config = {}

    # The configuration should set up several top level dictionaries:
    # None of these are technically required, but it is an error to have _neither_
    # psf or gal.
    config['psf'] = {}     # defines the PSF
    config['gal'] = {}     # defines the galaxy
    config['image'] = {}   # defines some information about the images
    config['input'] = {}   # defines any necessary input files
    config['output'] = {}  # defines the output files

    # Each type of profile is specified by a type.  e.g. Moffat:
    config['psf']['type'] = 'Moffat'

    # The various parameters are typically specified as well
    config['psf']['beta'] = 3.5

    # These parameters do not need to be constant.  There are a number of ways to
    # specify variables that might change from object to object.
    # In this case, the parameter specification also has a "type".
    # For now we only have InputCatalog, which means read the value from a catalog:
    config['psf']['fwhm'] = {}
    config['psf']['fwhm']['type'] = 'InputCatalog'

    # InputCatalog requires the extra value of which column to use in the catalog:
    # Note: the first column is called 0, not 1, as per the usual python 
    # 0-based indexing scheme.
    config['psf']['fwhm']['col'] = 5

    # You can also specify both of these on the same line as a dict in the normal python way.
    config['psf']['trunc'] = { 'type' : 'InputCatalog' , 'col' : 8 }

    # You can nest this as deep as you need to
    config['psf']['ellip'] = {
        'type' : 'E1E2',
        'e1' : { 'type' : 'InputCatalog' , 'col' : 6 },
        'e2' : { 'type' : 'InputCatalog' , 'col' : 7 }
    }

    # If you don't specify a parameter, and there is a reasonable default, then it 
    # will be used instead.  If there is no reasonable default, you will get an error.
    #config['psf']['flux'] = 1  # Unnecessary

    # A profile can be the sum of several components, each with its own type and parameters:
    config['gal']['type'] = 'Sum'

    # Sum requires a field called items, which is a list
    config['gal']['items'] = [
        {
            'type' : 'Exponential',
            'half_light_radius' : { 'type' : 'InputCatalog' , 'col' : 9 },
            'ellip' : {
                'type' : 'E1E2',
                'e1' : { 'type' : 'InputCatalog' , 'col' : 10 },
                'e2' : { 'type' : 'InputCatalog' , 'col' : 11 },
            },
            'flux' : 0.6
        },
        {
            'type' : 'DeVaucouleurs',
            'half_light_radius' : { 'type' : 'InputCatalog' , 'col' : 12 },
            'ellip' : {
                'type' : 'E1E2',
                'e1' : { 'type' : 'InputCatalog' , 'col' : 13 },
                'e2' : { 'type' : 'InputCatalog' , 'col' : 14 },
            },
            'flux' : 0.4
        } 
    ]

    # When a composite object (like a Sum) has a flux specified, the "flux" values of the
    # components are taken to be relative fluxes, and the full object's value sets the
    # overall normalization.  If this is omitted, the overall flux is taken to be the
    # sum of the component fluxes.
    config['gal']['flux'] = gal_flux

    # The fields ellip and shear each do the same thing -- shear the profile by some value.
    # Typically ellip refers to the intrinsic shape of the object, while shear refers
    # to the applied graviational shear.  The former is usually specified in terms of 
    # a distortion: e = (a^2-b^2)/(a^2+b^2), while the latter is usually specified in
    # terms of a shear (or "reduced shear"): g = (a-b)/(a+b).  However, either one 
    # may be defined using E1E2 (distortion) or G1G2 (reduced shear).
    # Other possible types for these are EBeta and GBeta (polar coordinates), 
    # and QBeta (using q = b/a).
    config['gal']['shear'] = { 'type' : 'G1G2' , 'g1' : gal_g1 , 'g2' : gal_g2 }

    # The shift field will shift the location of the centroid relative to the image center.
    config['gal']['shift'] = { 
        'type' : 'XY' ,
        'x' : { 'type' : 'InputCatalog' , 'col' : 15 },
        'y' : { 'type' : 'InputCatalog' , 'col' : 16 }
    }

    # Define some other information about the images
    config['image']['pixel_scale'] = pixel_scale
    config['image']['xsize'] = xsize
    config['image']['ysize'] = ysize
    config['image']['noise'] = { 'type' : 'CCDNoise' , 'sky_level' : sky_level }

    # The random seed is a bit special.  We actually set the initial seed for each
    # galaxy in order to be sequential values starting with this one.
    # The reason is so we can have deterministic runs even when we use multiple 
    # processes to build each image.  So this random_seed is the seed for the _first_
    # image.
    config['image']['random_seed'] = random_seed

    # Define the input files -- in this case the catalog file to use.
    config['input']['catalog'] = { 'file_name' : cat_file_name }

    # Define the output format
    config['output']['file_name'] = multi_file_name

    # type = MultiFits means to use a multi-extension fits file
    config['output']['type'] = 'MultiFits'

    # You can specify how many extensions to write to the file with nimages, 
    # but in this case, since we are using an input catalog, the default 
    # value is to do the number of entries in the catalog.
    #config['output']['nimages'] = 100

    # Now the following function will do everything that we specified above:
    # (The logger parameter is optional.)
    galsim.config.Process(config,logger)
    logger.info('Done processing config.')
    logger.info('Images written to multi-extension fits file %r',multi_file_name)


if __name__ == "__main__":
    main(sys.argv)
