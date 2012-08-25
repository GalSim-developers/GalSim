#!/usr/bin/env python
"""
The main driver program for making images of galaxies whose parameters are specified
in a configuration file.
"""

import sys
import os
import subprocess
import galsim
import yaml
import logging
import time
import json

def main(argv) :

    if len(argv) < 2 : 
        print 'Usage: galsim_yaml config_file'
        print 'See the example configuration files in the examples directory.'
        print 'They are the *.yaml files'
        sys.exit("No configuration file specified")

    # Should have a nice way of specifying a verbosity level...
    # Then we can just pass that verbosity into the logger.
    # Can also have the logging go to a file, etc.
    # But for now, just do a basic setup.
    logging.basicConfig(
        format="%(message)s",
        level=logging.DEBUG,
        stream=sys.stdout
    )
    logger = logging.getLogger('galsim_yaml')

    config_file = argv[1]
    logger.info('Using config file %s',config_file)

    config = yaml.load(open(config_file).read())
    logger.info('config file successfully read in')

    # Initialize the random number generator we will be using.
    if 'random_seed' in config :
        rng = galsim.UniformDeviate(config['random_seed'])
    else :
        rng = galsim.UniformDeviate()

    # Read the catalog
    input_cat = None
    if 'input' in config :
        input = config['input']
        if 'catalog' in input :
            catalog = input['catalog']
            dir = catalog['dir']
            file_name = catalog['file_name']
            full_name = os.path.join(catalog['dir'],catalog['file_name'])
            input_cat = galsim.io.ReadInputCat(config,full_name)
            logger.info('Read %d objects from catalog',input_cat.nobjects)

    # Build the images
    all_images = []
    for i in range(input_cat.nobjects):
        if i is not input_cat.current:
            raise ValueError('i is out of sync with current.')

        t1 = time.time()
        #logger.info('Image %d',input_cat.current)

        psf = galsim.BuildGSObject(config['psf'], input_cat, logger)
        #logger.info('   Made PSF profile')
        t2 = time.time()

        pix = galsim.BuildGSObject(config['pix'], input_cat, logger)
        #logger.info('   Made pixel profile')
        t3 = time.time()

        gal = galsim.BuildGSObject(config['gal'], input_cat, logger)
        #logger.info('   Made galaxy profile')
        t4 = time.time()

        final = galsim.Convolve(psf,pix,gal)
        #im = final.draw(dx=pixel_scale)  # It makes these as 768 x 768 images.  A bit big.
        im = galsim.ImageF(config['image']['xsize'], config['image']['ysize'])
        final.draw(im, dx=config['image']['pixel_scale'])
        xsize, ysize = im.array.shape
        #logger.info('   Drew image: size = %d x %d',xsize,ysize)
        t5 = time.time()

        # Add Poisson noise
        if 'noise' in config['gal'] : 
            noise = config['gal']['noise']
            if noise['type'] == 'poisson' :
                sky_level = float(noise['sky_level'])
                im += sky_level
                im.addNoise(galsim.CCDNoise(rng))
                im -= sky_level
        #logger.info('   Added noise')
        t6 = time.time()

        # Store that into the list of all images
        all_images += [im]
        t7 = time.time()

        # increment the row of the catalog that we should use for the next iteration
        input_cat.current += 1
        #logger.info('   Times: %f, %f, %f, %f, %f, %f', t2-t1, t3-t2, t4-t3, t5-t4, t6-t5, t7-t6)
        logger.info('Image %d: size = %d x %d, total time = %f sec', i, xsize, ysize, t7-t1)

    logger.info('Done making images of galaxies')

    # Now write the image to disk.
    for out in config['output'] :
        if out['type'] == 'tiled_stamps' :
            if not os.path.isdir(out['dir']) :
                os.mkdir(out['dir'])
            file_name = os.path.join(out['dir'],out['file_name'])
            galsim.fits.writeMulti(all_images, file_name, clobber=True)
            logger.info('Wrote images to multi-extension fits file %r',file_name)
        if out['type'] == 'data_cube' :
            if not os.path.isdir(out['dir']) :
                os.mkdir(out['dir'])
            file_name = os.path.join(out['dir'],out['file_name'])
            galsim.fits.writeCube(all_images, file_name, clobber=True)
            logger.info('Wrote image to fits data cube %r',file_name)

if __name__ == "__main__":
    main(sys.argv)
