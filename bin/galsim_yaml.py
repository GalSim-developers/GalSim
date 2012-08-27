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

    # TODO: Should have a nice way of specifying a verbosity level...
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
    if 'random_seed' in config:
        rng = galsim.UniformDeviate(int(config['random_seed']))
    else:
        rng = galsim.UniformDeviate()

    # Read the catalog if we are using one.
    input_cat = None
    nobjects = 1
    if 'input' in config :
        input = config['input']
        if 'catalog' in input :
            catalog = input['catalog']
            dir = catalog['dir']
            file_name = catalog['file_name']
            full_name = os.path.join(catalog['dir'],catalog['file_name'])
            input_cat = galsim.io.ReadInputCat(config,full_name)
            logger.info('Read %d objects from catalog',input_cat.nobjects)
            nobjects = input_cat.nobjects

    # If specified, set the number of objects to draw (default = 1)
    nobjects = config.get('nobjects',nobjects)
    logger.info('nobjects = %d',nobjects)

    # We'll be accessing things from the image field a lot.  So instead of constantly
    # checking "if 'image in config", we do it once and if it's not there, we just
    # make an empty dict for it.
    if not 'image' in config :
        config['image'] = {}

    # Set the size of the postage stamps if desired
    # If not set, the size will be set appropriately to enclose most of the flux.
    if 'xsize' in config['image'] :
        image_xsize = int(config['image']['xsize'])
        image_ysize = int(config['image'].get('ysize',image_xsize))
    elif 'ysize' in config['image'] :
        image_ysize = int(config['image']['ysize'])
        image_xsize = image_ysize
    elif 'size' in config['image'] :
        image_xsize = int(config['image']['size'])
        image_ysize = image_xsize
    else :
        image_xsize = None
        image_ysize = None
    if image_xsize is None:
        logger.info('Automatically sizing images')
    else:
        logger.info('Using image size = %d x %d',image_xsize,image_ysize)

    # Also, set the pixel scale if desired (Default is 1.0)
    pixel_scale = float(config['image'].get('pixel_scale',1.0))
    logger.info('Using pixelscale = %f',pixel_scale)

    # If the output includes either data_cube or tiled_stamps then all images need
    # to be the same size.  We will use the first image's size for all others.
    # This just decides whether this is necessary or not.
    same_sized_images = False
    for out in config['output'] :
        if 'type' in out and nobjects > 1 and (
                out['type'] == 'data_cube' or out['type'] == 'tiled_stamps') :
            same_sized_images = True
            logger.info('All images must be the same size, so will use the automatic' +
                        'size of the first image')

    # Build the images
    all_images = []
    for i in range(nobjects):
        if input_cat and i is not input_cat.current:
            raise ValueError('i is out of sync with current.')

        t1 = time.time()

        fft_list = []
        phot_list = []

        if 'psf' in config :
            psf = galsim.BuildGSObject(config['psf'], input_cat, logger)
            #print 'psf = ',psf
            fft_list.append(psf)
            phot_list.append(psf)
        t2 = time.time()

        if 'pix' in config :
            pix = galsim.BuildGSObject(config['pix'], input_cat, logger)
        else :
            pix = galsim.Pixel(xw=pixel_scale, yw=pixel_scale)
        #print 'pix = ',pix
        fft_list.append(pix)
        t3 = time.time()

        gal = galsim.BuildGSObject(config['gal'], input_cat, logger)
        #print 'gal = ',gal
        fft_list.append(gal)
        phot_list.append(gal)
        t4 = time.time()

        draw_method = config['image'].get('draw_method','fft')
        if draw_method == 'fft' :
            final = galsim.Convolve(fft_list)
            #print 'final = ',final
            if image_xsize is None :
                im = final.draw(dx=pixel_scale)
                # If the output includes either data_cube or tiled_stamps then all images need
                # to be the same size.  Use the first image's size for all others.
                if same_sized_images :
                    image_xsize, image_ysize = im.array.shape
            else:
                im = galsim.ImageF(image_xsize, image_ysize)
                final.draw(im, dx=pixel_scale)
        elif draw_method == 'phot' :
            final = galsim.Convolve(phot_list)
            if image_xsize is None :
                # TODO: Change this once issue #82 is done.
                raise AttributeError(
                    "image size must be specified when doing photon shooting.")
            else:
                im = galsim.ImageF(image_xsize, image_ysize)
                final.drawShoot(im, dx=pixel_scale)
        else :
            raise AttributeError(
                "Unknown draw_method.  Valid values are fft or phot.")
        xsize, ysize = im.array.shape
        t5 = time.time()

        # Add noise
        if 'noise' in config['image'] : 
            noise = config['image']['noise']
            if not 'type' in noise :
                raise AttributeError(
                    "noise needs a type to be specified \n" +
                    "Valid values are Poisson or Gaussian.")
            if noise['type'] == 'Poisson' :
                sky_level = float(noise['sky_level'])
                im += sky_level
                im.addNoise(galsim.CCDNoise(rng))
                im -= sky_level
                #logger.info('   Added Poisson noise for sky_level = %f',sky_level)
            elif noise['type'] == 'Gaussian' :
                if 'sigma' in noise:
                    sigma = noise['sigma']
                elif 'variance' in noise :
                    sigma = math.sqrt(noise['variance'])
                else :
                    raise AttributeError(
                        "Either sigma or variance need to be specified for Gaussian noise")
                im.addNoise(galsim.GaussianDeviate(rng,sigma=sigma))
                #logger.info('   Added Gaussian noise with sigma = %f',sigma)
            else :
                raise AttributeError(
                    "Invalid type for noise \n" +
                    "Valid values are Poisson or Gaussian.")
        t6 = time.time()

        # Store that into the list of all images
        all_images += [im]
        t7 = time.time()

        # increment the row of the catalog that we should use for the next iteration
        if input_cat:
            input_cat.current += 1
        #logger.info('   Times: %f, %f, %f, %f, %f, %f', t2-t1, t3-t2, t4-t3, t5-t4, t6-t5, t7-t6)
        logger.info('Image %d: size = %d x %d, total time = %f sec', i, xsize, ysize, t7-t1)

    logger.info('Done making images of galaxies')

    # Now write the image to disk.
    # We can handle not having an output field.  But as with image, it will be convenient
    # if it exists and is empty.
    if 'output' not in config :
        config['output'] = {}
    #print 'config[output] = ',config['output']

    # We're going to treat output as a list (for multiple file outputs if desired).
    # If it isn't a list, make it one.
    if not isinstance(config['output'],list):
        config['output'] = [ config['output'] ]
    #print 'config[output] => ',config['output']

    # Loop over all output formats:
    for out in config['output'] :
        #print 'out = ',out

        # Get the file_name
        if 'file_name' in out :
            file_name = out['file_name']
        else :
            # If a file_name isn't specified, we use the name of the calling script to
            # generate a fits file name.
            import inspect
            script_name = os.path.basiename(
                inspect.getfile(inspect.currentframe())) # script filename (usually with path)
            # Strip off a final suffix if present.
            file_name = os.path.splitext(script_name)[0]
            logger.info('No output file name specified.  Using %s',file_name)
        #print 'file_name = ',file_name

        # Prepend a dir to the beginning of the filename if requested.
        if 'dir' in out :
            if not os.path.isdir(out['dir']) :
                os.mkdir(out['dir'])
            file_name = os.path.join(out['dir'],file_name)

        # Each kind of output works slightly differently
        output_type = out.get('type','multi_fits')
        #print 'type = ',output_type
        if nobjects == 1 and output_type == 'multi_fits':
            all_images[0].write(file_name, clobber=True)
            logger.info('Wrote image to fits file %r',file_name)
        elif output_type == 'multi_fits' :
            galsim.fits.writeMulti(all_images, file_name, clobber=True)
            logger.info('Wrote images to multi-extension fits file %r',file_name)
        elif output_type == 'data_cube' :
            galsim.fits.writeCube(all_images, file_name, clobber=True)
            logger.info('Wrote image to fits data cube %r',file_name)
        else :
            raise AttributeError(
                "Invalid type for output \n" +
                "Valid values are multi_fits or data_cube")
            

if __name__ == "__main__":
    main(sys.argv)
