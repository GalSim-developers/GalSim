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
import copy

def MergeConfig(config1, config2, logger=None):
    """
    Merge config2 into config1 sucth that it has all the information from either config1 or 
    config2 including places where both input dicts have some of a field defined.
    e.g. config1 has image.pixel_scale, and config2 has image.noise.
            Then the returned dict will have both.
    For real comflicts (the same value in both cases), config1's value takes precedence
    """
    for (key, value) in config2.items():
        if not key in config1:
            # If this key isn't in config1 yet, just add it
            config1[key] = copy.deepcopy(value)
        elif isinstance(value,dict) and isinstance(config1[key],dict):
            # If they both have a key, first check if the values are dicts
            # If they are, just recurse this process and merge those dicts.
            MergeConfig(config1[key],value)
        else:
            # Otherwise config1 takes precedence
            if logger:
                logger.info("Not merging key %s from the base config, since the later "
                            "one takes precedence",key)
            pass

def ParseConfigInput(config, logger=None):
    """
    Parse the field config['input'], storing the results into the top level of config:

    config['input_cat']  if provided
    config['real_cat']   if provided
    """

    # Make config['input'] exist if it doesn't yet.
    if not 'input' in config:
        config['input'] = {}
    input = config['input']

    # Read the input catalog if provided
    if 'catalog' in input:
        catalog = input['catalog']
        file_name = catalog['file_name']
        if 'dir' in catalog:
            dir = catalog['dir']
            file_name = os.path.join(dir,file_name)
        input_cat = galsim.io.ReadInputCat(config,file_name)
        if logger:
            logger.info('Read %d objects from catalog',input_cat.nobjects)
        # Store input_cat in the config for use by BuildGSObject function.
        config['input_cat'] = input_cat

    # Read the RealGalaxy catalog if provided
    if 'real_catalog' in input:
        catalog = input['real_catalog']
        file_name = catalog['file_name']
        if 'dir' in catalog:
            dir = catalog['dir']
            file_name = os.path.join(dir,file_name)
            image_dir = catalog.get('image_dir',dir)
        else:
            image_dir = catalog.get('image_dir','.')
        real_cat = galsim.RealGalaxyCatalog(file_name, image_dir)
        if logger:
            logger.info('Read %d objects from catalog',real_cat.n)
        if 'preload' in catalog and catalog['preload']:
            real_cat.preload()
            if logger:
                logger.info('Preloaded the real galaxy catalog headers')
        # Store real_cat in the config for use by BuildGSObject function.
        config['real_cat'] = real_cat

def ParseConfigOutput(config, logger=None):
    """
    Parse the field config['output'], storing some values into the top level of config:

    config['same_sized_images']   Must all the images be the same size?
    config['make_psf_images']     Should we make the psf images
    config['nimages']             How many images are we creating
    config['nobj_per_image']      How many objects are we drawing on each image
    config['nobjects']            How many total objects are we drawing
    """

    # Make config['output'] exist if it doesn't yet.
    if 'output' not in config:
        config['output'] = {}
    output = config['output']

    # If the output includes either data_cube or tiled_image then all images need
    # to be the same size.  We will use the first image's size for all others.
    # This just decides whether this is necessary or not.
    config['same_sized_images'] = False
    config['make_psf_images'] = False  

    # Get the file_name
    if 'file_name' in output:
        file_name = output['file_name']
    else:
        # If a file_name isn't specified, we use the name of the calling script to
        # generate a fits file name.
        import inspect
        script_name = os.path.basiename(
            inspect.getfile(inspect.currentframe())) # script filename (usually with path)
        # Strip off a final suffix if present.
        file_name = os.path.splitext(script_name)[0]
        if logger:
            logger.info('No output file name specified.  Using %s',file_name)

    # Prepend a dir to the beginning of the filename if requested.
    if 'dir' in output:
        if not os.path.isdir(output['dir']):
            os.mkdir(output['dir'])
        file_name = os.path.join(output['dir'],file_name)

    # Store the result back in the config:
    output['file_name'] = file_name

    if 'psf' in output:
        config['make_psf_images'] = True
        psf_file_name = None
        output_psf = output['psf']
        if 'file_name' in output_psf:
            psf_file_name = output_psf['file_name']
            if 'dir' in output:
                psf_file_name = os.path.join(output['dir'],psf_file_name)
                output_psf['file_name'] = psf_file_name
        elif 'type' in output and output['type'] == 'multi_fits':
            raise AttributeError(
                    "Only the file_name version of psf output is possible with multi_fits")
        else:
            raise NotImplementedError(
                "Only the file_name version of psf output is currently implemented.")

    # Parse the information in output['image']
    # Accumulate nobj = how many objects per image.
    if not 'image_type' in output:
        output['image_type'] = 'single'
        nobj = 1
    elif output['image_type'] == 'single':
        nobj = 1
    elif output['image_type'] == 'tiled':
        config['same_sized_images'] = True
        if not all (k in output for k in ['nx_tiles','ny_tiles']):
            raise AttributeError(
                "parameters nx_tiles and ny_tiles required for image_type = tiledt")
        nx_tiles = output['nx_tiles']
        ny_tiles = output['ny_tiles']
        nobj = nx_tiles * ny_tiles
    else:
        raise AttributeError("Invalid image_type for output: %s",output['image_type'])

    # Parse the information about the output file type
    # Accumulate nimages = how many images to draw.
    if not 'type' in output:
        output['type'] = 'fits'
        nimages = 1
    elif output['type'] == 'fits':
        nimages = 1
    elif output['type'] == 'multi_fits':
        if 'nimages' in output:
            nimages = output['nimages']
        elif 'input_cat' in config and output['image_type'] == 'single':
            nimages = config['input_cat'].nobjects
        else:
            raise AttributeError(
                "nimages should be specified for output type = multi_fits")
    elif output['type'] == 'data_cube':
        if output['image_type'] == 'single':
            config['same_sized_images'] = True
        if 'nimages' in output:
            nimages = output['nimages']
        elif 'input_cat' in config and output['image_type'] == 'single':
            nimages = config['input_cat'].nobjects
        else:
            raise AttributeError(
                "nimages should be specified for output type = data_cute")
    # TODO: Another output format we'll want is a list of files
    # Need some way to generate multiple filenames: foo_01.fits, foo_02.fits, etc.
    else:
        raise AttributeError("Invalid type for output: %s",output['type'])

    config['nobj_per_image'] = nobj
    config['nimages'] = nimages
    config['nobjects'] = nobj * nimages


def ParseConfigImage(config, logger=None):
    """
    Parse the field config['image'], storing some values into the top level of config:

    config['image_xsize']        Default = 0  (Meaning let galsim determine size)
    config['image_ysize']        Default = 0
    config['pixel_scale']        Default = 1.0
    """
 
    # Make config['image'] exist if it doesn't yet.
    # We'll be accessing things from the image field a lot.  So instead of constantly
    # checking "if 'image in config", we do it once and if it's not there, we just
    # make an empty dict for it.
    if 'image' not in config:
        config['image'] = {}

    # Set the size of the postage stamps if desired
    # If not set, the size will be set appropriately to enclose most of the flux.
    image_size = int(config['image'].get('size',0))
    config['image_xsize'] = int(config['image'].get('xsize',image_size))
    config['image_ysize'] = int(config['image'].get('ysize',image_size))
    
    if ( (config['image_xsize'] == 0) != (config['image_ysize'] == 0) ):
        raise AttributeError(
            "Both (or neither) of image.xsize and image.ysize need to be defined.")

    if not config['image_xsize']:
        if config['same_sized_images']:
            if logger:
                logger.info('All images must be the same size, so will use the automatic ' +
                            'size of the first image only')
        else:
            if logger:
                logger.info('Automatically sizing images')
    else:
        if logger:
            logger.info('Using image size = %d x %d',config['image_xsize'],config['image_ysize'])

    if 'wcs' in config['image']:
        wcs = config['image']['wcs']
        if 'shear' in wcs:
            wcs_shear, safe_wcs = galsim.BuildShear(wcs, 'shear', config)
            config['wcs_shear'] = wcs_shear
        else:
            # TODO: Should add other kinds of WCS specifications.
            # E.g. a full CD matrix and eventually things like TAN and TNX.
            raise AttributeError("wcs must specify a shear")

    # Also, set the pixel scale (Default is 1.0)
    pixel_scale = float(config['image'].get('pixel_scale',1.0))
    config['pixel_scale'] = pixel_scale
    if logger:
        logger.info('Using pixel scale = %f',pixel_scale)

    # Normally, random_seed is just a number, which really means to use that number
    # for the first item and go up sequentially from there for each object.
    # However, we allow for random_seed to be a gettable parameter, so for the 
    # normal case, we just convert it into a Sequence.
    if 'random_seed' in config['image'] and not isinstance(config['image']['random_seed'],dict):
        first_seed = int(config['image']['random_seed'])
        config['image']['random_seed'] = { 'type' : 'Sequence' , 'min' : first_seed }

    if 'draw_method' not in config['image']:
        config['image']['draw_method'] = 'fft'

    # Get the target image variance from noise:
    if 'noise' in config['image']:
        noise = config['image']['noise']

        if not 'type' in noise:
            raise AttributeError("noise needs a type to be specified")
        if noise['type'] == 'Poisson':
            var = float(noise['sky_level']) * pixel_scale**2
        elif noise['type'] == 'Gaussian':
            if 'sigma' in noise:
                sigma = noise['sigma']
                var = sigma * sigma
            elif 'variance' in noise:
                var = noise['variance']
                sigma = math.sqrt(var)
                noise['sigma'] = sigma
            else:
                raise AttributeError(
                    "Either sigma or variance need to be specified for Gaussian noise")
        elif noise['type'] == 'CCDNoise':
            var = float(noise['sky_level']) * pixel_scale**2
            gain = float(noise.get("gain",1.0))
            var /= gain
            read_noise = float(noise.get("read_noise",0.0))
            var += read_noise * read_noise
        else:
            raise AttributeError("Invalid type %s for noise",noise['type'])
        config['noise_var'] = var


def ParseConfigPSF(config, logger=None):
    """
    Parse the field config['psf'] returning the built psf object.

    Also stores config['safe_psf'] marking whether the psf object is marked 
    as safe to reuse.
    """
 
    if 'psf' in config:
        psf, safe_psf = galsim.BuildGSObject(config, 'psf')
    else:
        psf = None
        safe_psf = True
    config['safe_psf'] = safe_psf
    return psf

def ParseConfigPix(config, logger=None):
    """
    Parse the field config['pix'] returning the built pix object.

    Also stores config['safe_pix'] marking whether the pix object is marked 
    as safe to reuse.
    """
 
    if 'pix' in config: 
        pix, safe_pix = galsim.BuildGSObject(config, 'pix')
    else:
        pixel_scale = config['pixel_scale']
        pix = galsim.Pixel(xw=pixel_scale, yw=pixel_scale)
        safe_pix = True

    # If the image has a WCS, we need to shear the pixel the reverse direction, so the
    # resulting WCS shear later will bring the pixel back to square.
    # TODO: Still not sure if this is exactly the right way to deal with a WCS shear.
    # Need to find a good way to test this procedure.
    if 'wcs_shear' in config:
        pix.applyShear(-config['wcs_shear'])

    config['safe_pix'] = safe_pix
    return pix


def ParseConfigGal(config, logger=None):
    """
    Parse the field config['gal'] returning the built gal object.

    Also stores config['safe_gal'] marking whether the gal object is marked 
    as safe to reuse.
    """
 
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

        gal, safe_gal = galsim.BuildGSObject(config, 'gal')
    else:
        gal = None
        safe_gal = True
    config['safe_gal'] = safe_gal
    return gal

def DrawImageFFT(psf, pix, gal, config):
    """
    Draw an image using the given psf, pix and gal profiles (which may be None)
    using the FFT method for doing the convolution.

    @return the resulting image.
    """

    fft_list = [ prof for prof in (psf,pix,gal) if prof is not None ]
    final = galsim.Convolve(fft_list)

    if 'wcs_shear' in config:
        final.applyShear(config['wcs_shear'])
    
    image_xsize = config['image_xsize']
    image_ysize = config['image_ysize']
    pixel_scale = config['pixel_scale']
    if not image_xsize:
        im = final.draw(dx=pixel_scale)
        # If the output includes either data_cube or tiled_image then all images need
        # to be the same size.  Use the first image's size for all others.
        if config['same_sized_images']:
            image_xsize, image_ysize = im.array.shape
            config['image_xsize'] = image_xsize
            config['image_ysize'] = image_ysize
    else:
        im = galsim.ImageF(image_xsize, image_ysize)
        im.setScale(pixel_scale)
        final.draw(im, dx=pixel_scale)

    if 'gal' in config and 'signal_to_noise' in config['gal']:
        import math
        import numpy
        if 'flux' in config['gal']:
            raise AttributeError(
                'Only one of signal_to_noise or flux may be specified for gal')
        if 'noise_var' not in config: 
            raise AttributeError(
                "Need to specify noise level when using gal.signal_to_noise")
        noise_var = config['noise_var']
            
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

        sn_meas = math.sqrt( numpy.sum(im.array**2) / noise_var )
        # Now we rescale the flux to get our desired S/N
        flux = float(config['gal']['signal_to_noise']) / sn_meas
        #print 'sn_meas = ',sn_meas
        #print 'flux = ',flux
        im *= flux
        #print 'sn_meas = ',sn_meas,' flux = ',flux

        if config['safe_gal'] and config['safe_psf'] and config['safe_pix']:
            # If the profile won't be changing, then we can store this 
            # result for future passes so we don't have to recalculate it.
            config['gal']['current'] *= flux
            config['gal']['flux'] = flux
            del config['gal']['signal_to_noise']
    
    # Add noise
    if 'noise' in config['image']: 
        noise = config['image']['noise']
        rng = config['rng']
        if not 'type' in noise:
            raise AttributeError("noise needs a type to be specified")
        if noise['type'] == 'Poisson':
            sky_level_pixel = float(noise['sky_level']) * pixel_scale**2
            im += sky_level_pixel
            #print 'before CCDNoise: ',rng()
            im.addNoise(galsim.CCDNoise(rng))
            #print 'after CCDNoise: ',rng()
            im -= sky_level_pixel
            #if logger:
                #logger.info('   Added Poisson noise with sky_level = %f',sky_level)
        elif noise['type'] == 'Gaussian':
            sigma = noise['sigma']
            im.addNoise(galsim.GaussianDeviate(rng,sigma=sigma))
            #if logger:
                #logger.info('   Added Gaussian noise with sigma = %f',sigma)
        elif noise['type'] == 'CCDNoise':
            sky_level_pixel = float(noise['sky_level']) * pixel_scale**2
            gain = float(noise.get("gain",1.0))
            read_noise = float(noise.get("read_noise",0.0))
            im += sky_level_pixel
            im.addNoise(galsim.CCDNoise(rng, gain=gain, read_noise=read_noise))
            im -= sky_level_pixel
            #if logger:
                #logger.info('   Added CCD noise with sky_level = %f, ' +
                            #'gain = %f, read_noise = %f',sky_level,gain,read_noise)
        else:
            raise AttributeError("Invalid type %s for noise"%noise['type'])
    return im

def DrawImagePhot(psf, gal, config):
    """
    Draw an image using the given psf and gal profiles (which may be None)
    using the photon shooting method.

    @return the resulting image.
    """

    phot_list = [ prof for prof in (psf,gal) if prof is not None ]
    final = galsim.Convolve(phot_list)
    if 'wcs_shear' in config:
        final.applyShear(config['wcs_shear'])
                    
    if 'signal_to_noise' in config['gal']:
        raise NotImplementedError(
            "gal.signal_to_noise not implemented for draw_method = phot")
    if 'noise_var' not in config:
        raise AttributeError(
            "Need to specify noise level when using draw_method = phot")
    noise_var = config['noise_var']
        
    image_xsize = config['image_xsize']
    image_ysize = config['image_ysize']
    pixel_scale = config['pixel_scale']
    rng = config['rng']
    if not image_xsize:
        # TODO: Change this once issue #82 is done.
        raise AttributeError(
            "image size must be specified when doing photon shooting.")
    else:
        im = galsim.ImageF(image_xsize, image_ysize)
        im.setScale(pixel_scale)
        # TODO: Should we make this 100 a variable?
        final.drawShoot(im, noise=noise_var/100, uniform_deviate=rng)
    
    # Add noise
    if 'noise' in config['image']: 
        noise = config['image']['noise']
        if not 'type' in noise:
            raise AttributeError("noise needs a type to be specified")
        if noise['type'] == 'Poisson':
            sky_level_pixel = float(noise['sky_level']) * pixel_scale**2
            # For photon shooting, galaxy already has poisson noise, so we want 
            # to make sure not to add that again!  Just add Poisson with 
            # mean = sky_level_pixel
            im.addNoise(galsim.PoissonDeviate(rng, mean=sky_level_pixel))
            #if logger:
                #logger.info('   Added Poisson noise with sky_level = %f',sky_level)
        elif noise['type'] == 'Gaussian':
            sigma = noise['sigma']
            im.addNoise(galsim.GaussianDeviate(rng,sigma=sigma))
            #if logger:
                #logger.info('   Added Gaussian noise with sigma = %f',sigma)
        elif noise['type'] == 'CCDNoise':
            sky_level_pixel = float(noise['sky_level']) * pixel_scale**2
            gain = float(noise.get("gain",1.0))
            read_noise = float(noise.get("read_noise",0.0))
            # For photon shooting, galaxy already has poisson noise, so we want 
            # to make sure not to add that again!
            im *= gain
            im.addNoise(galsim.PoissonDeviate(rng, mean=sky_level_pixel*gain))
            im /= gain
            im.addNoise(galsim.GaussianDeviate(rng, sigma=read_noise))
            #if logger:
                #logger.info('   Added CCD noise with sky_level = %f, ' +
                            #'gain = %f, read_noise = %f',sky_level,gain,read_noise)
        else:
            raise AttributeError("Invalid type %s for noise",noise['type'])
    return im


def DrawPSFImage(psf, pix, config):
    """
    Draw an image using the given psf and pix profiles.

    @return the resulting image.
    """

    psf_list = [ prof for prof in (psf,pix) if prof is not None ]

    final_psf = galsim.Convolve(psf_list)
    if 'wcs_shear' in config:
        final_psf.applyShear(config['wcs_shear'])
    # Special: if the galaxy was shifted, then also shift the psf 
    if 'shift' in config['gal']:
        final_psf.applyShift(*config['gal']['shift']['current'])

    xsize = config['xsize']
    ysize = config['ysize']
    pixel_scale = config['pixel_scale']
    psf_im = galsim.ImageF(xsize,ysize)
    psf_im.setScale(pixel_scale)
    final_psf.draw(psf_im, dx=pixel_scale)

    return psf_im

def BuildSingleImage(seed, config, logger=None):
    """
    Build a single image using the given seed and config file

    @return image, psf_image, time
    """
    t1 = time.time()

    # Initialize the random number generator we will be using.
    if seed:
        rng = galsim.UniformDeviate(seed)
    else:
        rng = galsim.UniformDeviate()
    # Store the rng in the config for use by BuildGSObject function.
    config['rng'] = rng
    if 'gd' in config:
        del config['gd']  # In case it was set.

    psf = ParseConfigPSF(config,logger)
    t2 = time.time()

    pix = ParseConfigPix(config,logger)
    t3 = time.time()

    gal = ParseConfigGal(config,logger)
    t4 = time.time()

    # Check that we have at least gal or psf.
    if not (gal or psf):
        raise AttributeError("At least one of gal or psf must be specified in config.")

    draw_method = galsim.GetParamValue(config['image'],'draw_method',config,type=str)[0]
    if draw_method == 'fft':
        im = DrawImageFFT(psf,pix,gal,config)
    elif draw_method == 'phot':
        im = DrawImagePhot(psf,gal,config)
    else:
        raise AttributeError("Unknown draw_method %s."%draw_method)
    t5 = time.time()

    # Note: These may be different from image_xsize and image_ysize
    # since the latter may be None.
    # In that case, xsize and ysize are the actual realized sized of the drawn image.
    xsize, ysize = im.array.shape
    config['xsize'] = xsize
    config['ysize'] = ysize

    if config['make_psf_images']:
        psf_im = DrawPSFImage(psf,pix,config)
    else:
        psf_im = None

    t6 = time.time()

    #if logger:
        #logger.info('   Times: %f, %f, %f, %f, %f', t2-t1, t3-t2, t4-t3, t5-t4, t6-t5)
    return im, psf_im, t6-t1


def BuildImages(config, logger=None):
    """
    Build the images specified by the config file

    @return images, psf_images  (Both are lists)
    """
    def worker(input, output):
        """
        input is a queue with (args, info) tuples:
            args are the arguments to pass to BuildSingleImage
            info is passed along to the output queue.
        output is a queue storing (result, info, proc) tuples:
            result is the returned tuple from BuildSingleImage: 
                (image, psf_image, time).
            info is passed through from the input queue.
            proc is the process name.
        """
        for (args, info) in iter(input.get, 'STOP'):
            result = BuildSingleImage(*args)
            output.put( (result, info, current_process().name) )
    
    images = []
    psf_images = []
    nobjects = config['nobjects']

    if 'nproc' not in config['image'] or config['image']['nproc'] == 1:
        for k in range(nobjects):
            if 'random_seed' in config['image']:
                seed = galsim.GetParamValue(config['image'],'random_seed',config,type=int)[0]
            else:
                seed = None
            im, psf_im, t = BuildSingleImage(seed, config, logger)
            images += [im]
            psf_images += [psf_im]
            if logger:
                logger.info('Image %d: size = %d x %d, total time = %f sec', 
                            k, im.array.shape[0], im.array.shape[1], t)
    else:
        from multiprocessing import Process, Queue, current_process, cpu_count
        nproc = config['image']['nproc']
        if nproc <= 0:
            # Try to figure out a good number of processes to use
            try:
                nproc = cpu_count()
                if logger:
                    logger.info('Using ncpu = %d processes',nproc)
            except:
                raise AttributeError(
                    "config.nprof <= 0, but unable to determine number of cpus.")

        # Don't save any 'current' results in the config, so we don't waste time sending
        # pickled versions of things back and forth.
        config['no_save'] = True

        # Initialize the images list to have the correct size.
        # This is important here, since we'll be getting back images in a random order,
        # and we need them to go in the right places (in order to have deterministic
        # output files).  So we initialize the list to be the right size.
        images = [ None for i in range(nobjects) ]
        psf_images = [ None for i in range(nobjects) ]

        # Set up the task list
        task_queue = Queue()
        for k in range(nobjects):
            # Note: we currently pull out the seed from config, since that is always
            # going to get clobbered by the multi-processing, since it involves a state
            # variable.  However, there may be other items in config that have state 
            # variables as well.  So the long term solution will be to construct the 
            # full profile in the main processor, and then send that to each of the 
            # parallel processors to draw the image.  But that will require our GSObjects
            # to be picklable, which they aren't currently.
            if 'random_seed' in config['image']:
                seed = galsim.GetParamValue(config['image'],'random_seed',config,type=int)[0]
            else:
                seed = None
            # Apparently the logger isn't picklable, so can't send that as an arg.
            task_queue.put( ( (seed, config), k) )

        # Run the tasks
        # Each Process command starts up a parallel process that will keep checking the queue 
        # for a new task. If there is one there, it grabs it and does it. If not, it waits 
        # until there is one to grab. When it finds a 'STOP', it shuts down. 
        done_queue = Queue()
        for j in range(nproc):
            Process(target=worker, args=(task_queue, done_queue)).start()

        # In the meanwhile, the main process keeps going.  We pull each image off of the 
        # done_queue and put it in the appropriate place on the main image.  
        # This loop is happening while the other processes are still working on their tasks.
        # You'll see that these logging statements get print out as the stamp images are still 
        # being drawn.  
        for i in range(nobjects):
            result, k, proc = done_queue.get()
            im = result[0]
            psf_im = result[1]
            t = result[2]
            images[k] = im
            psf_images[k] = psf_im
            if logger:
                logger.info('%s: Image %d: size = %d x %d, total time = %f sec', 
                            proc, k, im.array.shape[0], im.array.shape[1], t)

        # Stop the processes
        # The 'STOP's could have been put on the task list before starting the processes, or you
        # can wait.  In some cases it can be useful to clear out the done_queue (as we just did)
        # and then add on some more tasks.  We don't need that here, but it's perfectly fine to do.
        # Once you are done with the processes, putting nproc 'STOP's will stop them all.
        # This is important, because the program will keep running as long as there are running
        # processes, even if the main process gets to the end.  So you do want to make sure to 
        # add those 'STOP's at some point!
        for j in range(nproc):
            task_queue.put('STOP')

    if logger:
        logger.info('Done making images')

    return images, psf_images
 
def BuildSingleFullImage(images, psf_images, config, logger=None):
    """
    Turn some postage stamp images into a single full images according to 
    config['output']['image_type']
    """
    output = config['output']
    if output['image_type'] == 'single':
        return images, psf_images
    elif output['image_type'] == 'tiled':
        nx_tiles = output['nx_tiles']
        ny_tiles = output['ny_tiles']
        border = output.get("border",0)
        xborder = output.get("xborder",border)
        yborder = output.get("yborder",border)
        image_xsize = config['image_xsize']
        image_ysize = config['image_ysize']
        pixel_scale = config['pixel_scale']

        full_xsize = (image_xsize + xborder) * nx_tiles - xborder
        full_ysize = (image_ysize + yborder) * ny_tiles - yborder
        full_image = galsim.ImageF(full_xsize,full_ysize)
        full_image.setScale(pixel_scale)
        if 'psf' in output:
            full_psf_image = galsim.ImageF(full_xsize,full_ysize)
            full_psf_image.setScale(pixel_scale)
        else:
            full_psf_image = None
        k = 0
        for ix in range(nx_tiles):
            for iy in range(ny_tiles):
                if k < len(images):
                    xmin = ix * (image_xsize + xborder) + 1
                    xmax = xmin + image_xsize-1
                    ymin = iy * (image_ysize + yborder) + 1
                    ymax = ymin + image_ysize-1
                    b = galsim.BoundsI(xmin,xmax,ymin,ymax)
                    full_image[b] = images[k]
                    if 'psf' in output:
                        full_psf_image[b] = psf_images[k]
                    k = k+1
        return full_image, full_psf_image
   
def BuildFullImages(images, psf_images, config, logger=None):
    """
    Turn the postage stamp images into full images according to 
    config['output']['image_type']
    """
    output = config['output']
    if output['image_type'] == 'single':
        return images, psf_images
    else:
        full_images = []
        full_psf_images = []
        k1 = 0
        k2 = config['nobj_per_image']
        for i in range(config['nimages']):
            im, psf_im = BuildSingleFullImage(images[k1:k2], psf_images[k1:k2], config, logger)
            full_images += [ im ]
            full_psf_images += [ psf_im ]
        return full_images, full_psf_images
            
   
def WriteImages(images, psf_images, config, logger=None):
    """
    Write the provided images to the files specified in config['output']
    """

    output = config['output']
    file_name = output['file_name']
    if 'psf' in output:
        psf_file_name = output['psf']['file_name']

    if output['type'] == 'fits':
        images[0].write(file_name, clobber=True)
        if logger:
            logger.info('Wrote image to fits file %r',file_name)
        if 'psf' in output:
            psf_images[0].write(psf_file_name, clobber=True)
            if logger:
                logger.info('Wrote psf image to fits file %r',psf_file_name)

    elif output['type'] == 'multi_fits':
        galsim.fits.writeMulti(images, file_name, clobber=True)
        if logger:
            logger.info('Wrote images to multi-extension fits file %r',file_name)
        if 'psf' in output:
            galsim.fits.writeMulti(psf_images, psf_file_name, clobber=True)
            if logger:
                logger.info('Wrote psf images to multi-extension fits file %r',psf_file_name)

    elif output['type'] == 'data_cube':
        galsim.fits.writeCube(images, file_name, clobber=True)
        if logger:
            logger.info('Wrote image to fits data cube %r',file_name)
        if 'psf' in output:
            galsim.fits.writeCube(psf_images, psf_file_name, clobber=True)
            if logger:
                logger.info('Wrote psf images to fits data cube %r',psf_file_name)

    else:
        raise AttributeError("Invalid type for output: %s",output['type'])

def ProcessConfig(config, logger=None):
    """
    Do all processing of the provided configuration dict
    """

    # Parse the input field
    ParseConfigInput(config, logger)

    # Parse the output field
    ParseConfigOutput(config, logger)

    # Parse the image field
    ParseConfigImage(config, logger)

    # Build the postage-stamp images
    images, psf_images = BuildImages(config, logger)

    # Build full images from the postage stamps (if appropriate)
    images, psf_images = BuildFullImages(images, psf_images, config, logger)

    # Write the full images to the appropriate files
    WriteImages(images, psf_images, config, logger)


def main(argv):

    if len(argv) < 2: 
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

    if len(all_config) == 1:
        # If we only have 1, prepend an empty "base_config"
        all_config = [{}] + all_config

    base_config = all_config[0]

    for config in all_config[1:]:

        # Merge the base_config information into this config file.
        MergeConfig(config,base_config)
        #print 'config = ',config

        # Process the configuration
        ProcessConfig(config, logger)
    
if __name__ == "__main__":
    main(sys.argv)
