"""
Demo #8

The eighth script in our tutorial about using GalSim in python scripts: examples/demo*.py.
(This file is designed to be viewed in a window 100 characters wide.)

In this script, we show how to build a configuration dict from within python, rather
than using a config file.  The parallel tutorial examples/demo*.yaml have shown how to
do the same thing as these demo*.py files using a config file.  Now we turn the tables
and show how to use some of the machinery in the GalSim configuration processing 
from within python itself.  To appreciate this example script, you'll probably want to 
have looked through that series as well up to demo8.yaml and reference that file
as you look through this one.

This could be useful if you want to use the config machinery to build the images, but then
rather than write the images to disk, you want to keep them in memory and do further 
processing with them.  (e.g. Run your shape measurement code on the images from within python.)

New features introduced in this demo:

- galsim.config.Process(config)
- galsim.config.ProcessInput(config)
- galsim.config.ProcessOutput(config)
- galsim.config.BuildFits(file_name, config)
- galsim.config.BuildMultiFits(file_name, config)
- galsim.config.BuildDataCube(file_name, config)
- galsim.config.BuildImage(config)
- galsim.fits.read(file_name)
"""

import sys
import os
import numpy
import logging
import time
import copy
import galsim

def main(argv):
    """
    Make an image containing 10 x 10 postage stamps.
    The galaxies are bulge + disk with parameters drawn from random variates
    Each galaxy is drawn using photon shooting.
    """

    logging.basicConfig(format="%(message)s", level=logging.INFO, stream=sys.stdout)
    logger = logging.getLogger("demo8")

    logger.info('Starting demo script 8')

    # What we think of as the configuration file is really just a python dict:
    config = {}

    # We'll only be using three top-level fields in this file: psf, gal, and image.
    # We don't have any input files, so we don't need input.   And we're only going to 
    # have the config machinery build the images, so we don't need output.
    # And as usual, we'll use a simple square pixel, so we don't need pix.

    # We can define each attribute individually:
    config['psf'] = {}
    config['psf']['type'] = 'Moffat'
    config['psf']['beta'] = 2.4
    config['psf']['fwhm'] = 0.65

    # However, defining each field using the normal python way of specifying a dict is 
    # probably more often going to be the preferred way to do this:
    # (This looks a lot like a JSON file -- see the examples in examples/json/demo*.json.)
    config['gal'] = {
        "type" : "Sum",
        "items" : [
            {
                "type" : "Sersic",
                "n" : 3.6,
                "half_light_radius" : { "type" : "Random" , "min" : 0.3 , "max" : 0.9 },
                "flux" : { "type" : "Random" , "min" : 0.1 , "max" : 0.5 },
                "ellip" : {
                    "type" : "EBeta",
                    "e" : { "type" : "Random" , "min" : 0.0 , "max" : 0.3 },
                    "beta" : { "type" : "Random" }
                }
            },
            {
                "type" : "Sersic",
                "n" : 1.5,
                "half_light_radius" : { "type" : "Random" , "min" : 0.5 , "max" : 1.5 },
                "ellip" : {
                    "type" : "EBeta",
                    "e" : { "type" : "Random" , "min" : 0.2 , "max" : 0.8 },
                    "beta" : { "type" : "Random" }
                }
            }
        ],
        "flux" : { "type" : "Random" , "min" : 1.0e4 , "max" : 1.0e5 }
    }

    config['image'] = {
        'type' : 'Tiled',
        'nx_tiles' : 10,
        'ny_tiles' : 10,
        'stamp_size' : 64,
        'pixel_scale' : 0.28,
        'draw_method' : 'phot',
        'noise' : { 'sky_level' : 1.e4 },
        'random_seed' : 22345921
    }

    # Make a copy of the config dict as it exists now.
    save_config = copy.deepcopy(config)

    # Now that we have the config dict setup, there are a number of functions we can use
    # to process it in various ways.  The simplest is to do the full end-to-end processing
    # as is done by the program galsim_yaml:
    #
    #     galsim.config.Process(config)
    #
    # Since we don't want to have the config machinery output this to a file, we don't want
    # to do that here.  But if you also define the output field appropriately, that's the 
    # simplest way to process a config dict.
    #
    # That function is essentially equivalent to the following two functions:
    #
    #     galsim.config.ProcessInput(config)
    #     galsim.config.ProcessOutput(config)
    #
    # The former in particular may be useful to run separately.  If you are using an input
    # catalog (or other item that requires setup), it will read the file(s) from disk and
    # save the catalog (or whatever) in the config dict in the way that further processing
    # function expect.  However, we don't have any input field, so we don't need it here.
    #
    # The ProcessOutput function reads in the output field and then calls one of the following:
    #
    #     galsim.config.BuildFits(file_name, config)        -- build a regular fits file
    #     galsim.config.BuildMultiFits(file_name, config)   -- build a multi-extension fits file
    #     galsim.config.BuildDataCute(file_name, config)    -- build a fits data cube
    #
    # Finally, these functions all call the following function to process the image field
    # and actually build the images:
    #
    #     galsim.config.BuildImage(config)
    #
    # This returns a tuple of potentially 4 images:
    #
    #     (image, psf_image, weight_image, badpix_image)
    #
    # The default is for the latter 3 to all be None, but you can have the function build those
    # images as well by setting the optional kwargs: make_psf_image=True, make_weight_image=True,
    # and make_badpix_image=True, respectively.
    #

    t1 = time.time()

    # Build the image
    # All of the above functions have an optional kwarg, logger, which can take a 
    # logger object to output diagnostic information if desired.
    image, _, _, _ = galsim.config.BuildImage(config, logger=logger)
    
    # At this point you could do something interesting with the image in memory.
    # However, we're going to be boring and just write it to a file.
    single_file_name = os.path.join('output','bpd_single.fits')
    image.write(single_file_name)

    t2 = time.time()

    # The config processing functions save various things in the dict as they go, so 
    # for the second pass, start with a pristine version of the config dict.
    config = save_config

    # For this pass, we'll use 4 processes to build the image in parallel:
    nproc = 4
    config['image']['nproc'] = nproc

    # This time, let's just combine the two operations above and use the BuildFits function:
    multi_file_name = os.path.join('output','bpd_multi.fits')
    galsim.config.BuildFits(multi_file_name, config, logger=logger)

    t3 = time.time()

    logger.info('Total time taken using a single process = %f',t2-t1)
    logger.info('Total time taken using %d processes = %f',nproc,t3-t2)
    logger.info('Wrote images to %r and %r',single_file_name, multi_file_name)

    # Check that the builds are deterministic, even when using multiple processes.
    image2 = galsim.fits.read(multi_file_name) 
    numpy.testing.assert_array_equal(image.array, image2.array,
                                     err_msg="Images are not equal")
    logger.info('Images created using single and multiple processes are exactly equal.')
    logger.info('')


if __name__ == "__main__":
    main(sys.argv)
