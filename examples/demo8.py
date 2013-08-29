# Copyright 2012, 2013 The GalSim developers:
# https://github.com/GalSim-developers
#
# This file is part of GalSim: The modular galaxy image simulation toolkit.
#
# GalSim is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# GalSim is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with GalSim.  If not, see <http://www.gnu.org/licenses/>
#
"""
Demo #8

The eighth script in our tutorial about using GalSim in python scripts: examples/demo*.py.
(This file is designed to be viewed in a window 100 characters wide.)

In this script, we show how to run the GalSim config processing using a python dict rather
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
- galsim.config.BuildMultiFits(file_name, nimages, config)
- galsim.config.BuildDataCube(file_name, nimages, config)
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
    # Note: This is true regardless of the draw method.  Even if draw_method (below) were fft,
    # the config machinery would automatically create the square pixel for us.

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
    # as is done by the program galsim:
    #
    #     galsim.config.Process(config)
    #
    # Since we don't want to have the config machinery output this to a file, we don't want
    # to do that here.  But if you also define the output field appropriately, that's the 
    # simplest way to process a config dict.
    #
    # That function runs through each output file specified, and for each one, it first
    # processes the input field, then builds the file.  For the former it calls
    #
    #     galsim.config.ProcessInput(config)
    #
    # This function may be useful to call from within a python script.  If you are using an input
    # catalog (or any other item that requires setup), it will read the file(s) from disk and
    # save the catalog (or whatever) in the config dict in the way that further processing
    # functions expect.  However, we don't have any input field, so we don't need it here.
    #
    # To build the files, the Process function then calls one of the following:
    #
    #     galsim.config.BuildFits(file_name, config)               -- build a regular fits file
    #     galsim.config.BuildMultiFits(file_name, nimages, config) -- build a multi-extension fits 
    #                                                                 file
    #     galsim.config.BuildDataCube(file_name, nimages, config)  -- build a fits data cube
    #
    # Again, we'll forego that option here, so we can see how to use the config machinery
    # to produce images that we can use from within python.
    #
    # Each of these functions call the following function to process the image field
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
    # All of the above functions also have an optional kwarg, logger, which can take a 
    # logger object to output diagnostic information if desired.  We'll use that option here
    # to output the progress of the build as we go.  Our logger is set with level=logging.INFO
    # which means it will output a modest amount of text along the way.  Using level=logging.DEBUG 
    # will output a lot of text, useful when diagnosing a mysterious crash.  And using
    # level=logging.WARNING or higher will be pretty silent unless there is a problem.

    t1 = time.time()

    # Build the image
    # Since BuildImage returns a tuple of 4 images (see above) even though the latter
    # three are all returned as None, we still need to deal with the return values.
    # You could take [0] of the return value to just take the first image.  
    # You could also assign them all to an appropriate name and then not use them.
    # Another cute way to do it is to use an underscore for names of returned values
    # that you are planning to ignore:
    image, _, _, _ = galsim.config.BuildImage(config, logger=logger)
    
    # At this point you could do something interesting with the image in memory.
    # After all, that was kind of the point of using BuildImage rather than the other higher
    # level processing functions described above.  So perhaps you could insert your shape
    # measurement code here and pass it the image we just built.
    #
    # However, we're going to be boring and just write it to a file.

    # Make output directory if not already present.
    if not os.path.isdir('output'):
        os.mkdir('output')

    single_file_name = os.path.join('output','bpd_single.fits')
    image.write(single_file_name)

    t2 = time.time()

    # Now let's do it again using multiple processes.  This is really easy to do
    # using the config stuff.  It's just one extra parameter in the dict.

    # The config processing functions save various things in the dict as they go, so 
    # for the second pass, start with a pristine version of the config dict.
    config = save_config

    # Here we use 4 processes to build the image in parallel:
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
