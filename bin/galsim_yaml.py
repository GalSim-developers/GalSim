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

    all_config = [ c for c in yaml.load_all(open(config_file).read()) ]
    logger.info('config file successfully read in')
    #print 'all_config = ',all_config

    # If there is only 1 yaml document, then it is of course used for the configuration.
    # If there are multiple yamls documents, then the first one defines a common starting
    # point for the later documents.
    # So the configurations are taken to be:
    #   all_cong[0] + allconfig[1]
    #   all_cong[0] + allconfig[2]
    #   all_cong[0] + allconfig[3]
    #   ...
    base_config = all_config[0]

    if len(all_config) == 1:
        all_config.append({})

    for update_config in all_config[1:]:
        config = base_config
        config.update(update_config)
        #print 'config = ',config

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
 
        # We can handle not having an output field.  But it will be convenient
        # if it exists and is empty.
        if 'output' not in config :
            config['output'] = {}
        #print 'config[output] = ',config['output']
    
        # We're going to treat output as a list (for multiple file outputs if desired).
        # If it isn't a list, make it one.
        if not isinstance(config['output'],list):
            config['output'] = [ config['output'] ]
        #print 'config[output] => ',config['output']
    
        # If the output includes either data_cube or tiled_image then all images need
        # to be the same size.  We will use the first image's size for all others.
        # This just decides whether this is necessary or not.
        same_sized_images = False
        make_psf_images = False
        for output in config['output'] :
            if 'type' in output and (
                    output['type'] == 'data_cube' or output['type'] == 'tiled_image') :
                same_sized_images = True
                logger.info('All images must be the same size, so will use the automatic' +
                            'size of the first image')
            if 'psf' in output:
                make_psf_images = True
    
            # If the output is a tiled_image, then we can figure out nobjects from that:
            if 'type' in output and output['type'] == 'tiled_image':
                if not all (k in output for k in ['nx_tiles','ny_tiles']):
                    raise AttributeError(
                        "parameters nx_tiles and ny_tiles required for tiled_image output")
                nx_tiles = output['nx_tiles']
                ny_tiles = output['ny_tiles']
                nobjects = nx_tiles * ny_tiles

        # If specified, set the number of objects to draw.
        # Of course, nobjects might already be set above, in which case the
        # explicit value here will supersede whatever calculation we may have done.
        # And if nothing else sets nbojects, the default is nobjects=1.
        if 'nobjects' in config:
            nobjects = config['nobjects']
        logger.info('nobjects = %d',nobjects)
    
        # We'll be accessing things from the image field a lot.  So instead of constantly
        # checking "if 'image in config", we do it once and if it's not there, we just
        # make an empty dict for it.
        if 'image' not in config :
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

        # Build the images
        all_images = []
        all_psf_images = []
        for i in range(nobjects):
            if input_cat and i is not input_cat.current:
                raise ValueError('i is out of sync with current.')
    
            t1 = time.time()
    
            fft_list = []
            phot_list = []
            psf_list = []
    
            if 'psf' in config :
                psf, safe_psf = galsim.BuildGSObject(config, 'psf', rng, input_cat)
                #print 'psf = ',psf
                fft_list.append(psf)
                phot_list.append(psf)
                psf_list.append(psf)
            else :
                safe_psf = True
            t2 = time.time()
    
            if 'pix' in config :
                pix, safe_pix = galsim.BuildGSObject(config, 'pix', rng, input_cat)
            else :
                pix = galsim.Pixel(xw=pixel_scale, yw=pixel_scale)
                safe_pix = True
            #print 'pix = ',pix
            fft_list.append(pix)
            psf_list.append(pix)
            t3 = time.time()
    
            # If the image has a WCS, we need to shear the pixel the reverse direction, so the
            # resulting WCS shear later will bring the pixel back to square.
            if 'wcs' in config['image']:
                wcs = config['image']['wcs']
                if 'shear' in wcs:
                    wcs_shear, safe_wcs = galsim.BuildShear(wcs, 'shear', rng)
                    pix.applyShear(-wcs_shear)
                else :
                    raise AttributeError("wcs must specify a shear")
        
            if 'gal' in config:
                # If we are specifying the size according to a resolution, then we 
                # need to get the PSF's half_light_radius.
                if 'resolution' in config['gal']:
                    if not 'psf' in config:
                        raise AttributeError(
                            "Cannot use gal.resolution if no psf is set.")
                    if not 'saved_re' in config['psf']:
                        raise AttributeError(
                            'Cannot use gal.resolution with psf.type = %s'%config['psf']['type'])
                    psf_re = config['psf']['saved_re']
                    resolution = config['gal']['resolution']
                    gal_re = resolution * psf_re
                    config['gal']['half_light_radius'] = gal_re

                gal, safe_gal = galsim.BuildGSObject(config, 'gal', rng, input_cat)
                #print 'gal = ',gal
                fft_list.append(gal)
                phot_list.append(gal)
            else :
                safe_gal = True
            t4 = time.time()

            # Check that we have at least gal or psf.
            if len(phot_list) == 0:
                raise AttributeError("At least one of gal or psf must be specified in config.")

            draw_method = config['image'].get('draw_method','fft')
            if draw_method == 'fft' :
                final = galsim.Convolve(fft_list)
                if 'wcs' in config['image']:
                    final.applyShear(wcs_shear)
                #print 'final = ',final
    
                if image_xsize is None :
                    im = final.draw(dx=pixel_scale)
                    # If the output includes either data_cube or tiled_image then all images need
                    # to be the same size.  Use the first image's size for all others.
                    if same_sized_images :
                        image_xsize, image_ysize = im.array.shape
                else:
                    im = galsim.ImageF(image_xsize, image_ysize)
                    final.draw(im, dx=pixel_scale)

                if 'signal_to_noise' in config['gal']:
                    import math
                    import numpy
                    if 'flux' in config['gal']:
                        raise AttributeError(
                            'Only one of signal_to_noise or flux may be specified for gal')
                    if 'noise' not in config['image'] : 
                        raise AttributeError(
                            'Need to specify noise level when using gal.signal_to_noise')
                        
                    # Now determine what flux we need to get our desired S/N
                    # There are lots of definitions of S/N, but here is the one used by Great08
                    # We use a weighted integral of the flux:
                    # S = sum W(x,y) I(x,y) / sum W(x,y)
                    # N^2 = Var(S) = sum W(x,y)^2 Var(I(x,y)) / (sum W(x,y))^2
                    # Now we assume that Var(I(x,y)) is dominated by the sky noise, so
                    # Var(I(x,y)) = var
                    # We also assume that we are using a matched filter for W, so W(x,y) = I(x,y).
                    # Then a few things cancel and we find that
                    # S/N = sqrt( sum I(x,y)^2 / var )

                    # Get the variance from noise:
                    noise = config['image']['noise']
                    if not 'type' in noise :
                        raise AttributeError("noise needs a type to be specified")
                    if noise['type'] == 'Poisson' :
                        var = float(noise['sky_level'])
                    elif noise['type'] == 'Gaussian' :
                        if 'sigma' in noise:
                            sigma = noise['sigma']
                            var = sigma * sigma
                        elif 'variance' in noise :
                            var = math.sqrt(noise['variance'])
                        else :
                            raise AttributeError(
                                "Either sigma or variance need to be specified for Gaussian noise")
                    elif noise['type'] == 'CCDNoise' :
                        var = float(noise['sky_level'])
                        gain = float(noise.get("gain",1.0))
                        var /= gain
                        read_noise = float(noise.get("read_noise",0.0))
                        var += read_noise * read_noise
                    else :
                        raise AttributeError("Invalid type %s for noise",noise['type'])

                    if var <= 0.:
                        raise ValueError("gal.signal_to_noise requires var(noise) > 0.")
 
                    sn_meas = math.sqrt( numpy.sum(im.array**2) / var )
                    # Now we rescale the flux to get our desired S/N
                    flux = float(config['gal']['signal_to_noise']) / sn_meas
                    im *= flux
                    #print 'sn_meas = ',sn_meas,' flux = ',flux

                    if safe_gal and safe_psf and safe_pix:
                        # If the profile won't be changing, then we can store this 
                        # result for future passes.
                        config['gal']['current'] *= flux
                        config['gal']['flux'] = flux
                        del config['gal']['signal_to_noise']

            elif draw_method == 'phot' :
                final = galsim.Convolve(phot_list)
                if 'wcs' in config['image']:
                    final.applyShear(wcs_shear)
                    
                if image_xsize is None :
                    # TODO: Change this once issue #82 is done.
                    raise AttributeError(
                        "image size must be specified when doing photon shooting.")
                else:
                    im = galsim.ImageF(image_xsize, image_ysize)
                    final.drawShoot(im, dx=pixel_scale)

                if 'signal_to_noise' in config['gal']:
                    raise NotImplementedError(
                        "gal.signal_to_noise not implemented for draw_method = phot")

            else :
                raise AttributeError("Unknown draw_method.")
            xsize, ysize = im.array.shape

            if make_psf_images:
                final_psf = galsim.Convolve(psf_list)
                if 'wcs' in config['image']:
                    final_psf.applyShear(wcs_shear)
                # Special: if the galaxy was shifted, then also shift the psf 
                if 'shift' in config['gal']:
                    final_psf.applyShift(*config['gal']['shift']['current'])
                psf_im = galsim.ImageF(xsize,ysize)
                final_psf.draw(psf_im, dx=pixel_scale)
                all_psf_images += [psf_im]
            t5 = time.time()
    
            # Add noise
            if 'noise' in config['image'] : 
                noise = config['image']['noise']
                if not 'type' in noise :
                    raise AttributeError("noise needs a type to be specified")
                if noise['type'] == 'Poisson' :
                    sky_level = float(noise['sky_level'])
                    im += sky_level
                    im.addNoise(galsim.CCDNoise(rng))
                    im -= sky_level
                    #logger.info('   Added Poisson noise with sky_level = %f',sky_level)
                elif noise['type'] == 'Gaussian' :
                    if 'sigma' in noise:
                        sigma = noise['sigma']
                    elif 'variance' in noise :
                        import math
                        sigma = math.sqrt(noise['variance'])
                    else :
                        raise AttributeError(
                            "Either sigma or variance need to be specified for Gaussian noise")
                    im.addNoise(galsim.GaussianDeviate(rng,sigma=sigma))
                    #logger.info('   Added Gaussian noise with sigma = %f',sigma)
                elif noise['type'] == 'CCDNoise' :
                    sky_level = float(noise['sky_level'])
                    gain = float(noise.get("gain",1.0))
                    read_noise = float(noise.get("read_noise",0.0))
                    im += sky_level
                    im.addNoise(galsim.CCDNoise(rng, gain=gain, read_noise=read_noise))
                    im -= sky_level
                    #logger.info('   Added CCD noise with sky_level = %f, ' +
                                #'gain = %f, read_noise = %f',sky_level,gain,read_noise)
                else :
                    raise AttributeError("Invalid type %s for noise",noise['type'])
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
        # Loop over all output formats:
        for output in config['output'] :
            #print 'output = ',output
    
            # Get the file_name
            if 'file_name' in output :
                file_name = output['file_name']
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
            if 'dir' in output :
                if not os.path.isdir(output['dir']) :
                    os.mkdir(output['dir'])
                file_name = os.path.join(output['dir'],file_name)
    
            if 'psf' in output:
                psf_file_name = None
                output_psf = output['psf']
                if 'file_name' in output_psf:
                    psf_file_name = output_psf['file_name']
                    if 'dir' in output:
                        psf_file_name = os.path.join(output['dir'],psf_file_name)
                else:
                    raise NotImplementedError(
                        "Only the file_name version of psf output is currently implemented.")
    
            # Each kind of output works slightly differently
            if nobjects != 1:
                if not 'type' in output:
                    raise AttributeError(
                        "output type is required when there are multiple images drawn.")
                output_type = output['type']
                #print 'type = ',output_type

            if nobjects == 1:
                all_images[0].write(file_name, clobber=True)
                logger.info('Wrote image to fits file %r',file_name)
                if 'psf' in output:
                    if psf_file_name:
                        all_psf_images[0].write(psf_file_name, clobber=True)
                        logger.info('Wrote psf image to fits file %r',psf_file_name)
            elif output_type == 'multi_fits' :
                galsim.fits.writeMulti(all_images, file_name, clobber=True)
                logger.info('Wrote images to multi-extension fits file %r',file_name)
                if 'psf' in output:
                    if psf_file_name:
                        galsim.fits.writeMulti(all_psf_images, psf_file_name, clobber=True)
                        logger.info('Wrote psf images to multi-extension fits file %r',
                                    psf_file_name)
                    else:
                        raise AttributeError(
                            "Only the file_name version of psf output is possible with multi_fits")
            elif output_type == 'data_cube' :
                galsim.fits.writeCube(all_images, file_name, clobber=True)
                logger.info('Wrote image to fits data cube %r',file_name)
                if 'psf' in output:
                    if psf_file_name:
                        galsim.fits.writeCube(all_psf_images, psf_file_name, clobber=True)
                        logger.info('Wrote psf images to fits data cube %r',psf_file_name)
            elif output_type == 'tiled_image' :
                if not all (k in output for k in ['nx_tiles','ny_tiles']):
                    raise AttributeError(
                        "parameters nx_tiles and ny_tiles required for tiled_image output")
                nx_tiles = output['nx_tiles']
                ny_tiles = output['ny_tiles']
                border = output.get("border",0)
                full_xsize = (image_xsize + border) * nx_tiles
                full_ysize = (image_ysize + border) * ny_tiles
                full_image = galsim.ImageF(full_xsize,full_ysize)
                full_image.setOrigin(0,0) # For convenience, switch to C indexing convention.
                if 'psf' in output:
                    full_psf_image = galsim.ImageF(full_xsize,full_ysize)
                    full_psf_image.setOrigin(0,0) 
                k = 0
                for ix in range(nx_tiles):
                    for iy in range(ny_tiles):
                        if k < len(all_images):
                            xmin = ix * (image_xsize + border)
                            xmax = xmin + image_xsize-1
                            ymin = iy * (image_ysize + border)
                            ymax = ymin + image_ysize-1
                            b = galsim.BoundsI(xmin,xmax,ymin,ymax)
                            full_image[b] = all_images[k]
                            if 'psf' in output:
                                full_psf_image[b] = all_psf_images[k]
                            k = k+1
                full_image.write(file_name, clobber=True)
                logger.info('Wrote tiled image to fits file %r',file_name)
                if 'psf' in output:
                    if psf_file_name:
                        full_psf_image.write(psf_file_name, clobber=True)
                        logger.info('Wrote tled psf images to fits file %r',psf_file_name)
            else :
                raise AttributeError("Invalid type for output: %s",output_type)

    
if __name__ == "__main__":
    main(sys.argv)
