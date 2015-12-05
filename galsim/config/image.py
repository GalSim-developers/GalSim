# Copyright (c) 2012-2015 by the GalSim developers team on GitHub
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

import galsim
import logging

# This file handles the building of an image by parsing config['image'].
# This file includes the basic functionality, but it calls out to helper functions
# for parts of the process that are different for different image types.  It includes
# those helper functions for the simplest image type, Single.  See image_tiled.py and
# image_scattered.py for the implementation of the Tiled and Scattered image types.


def BuildImages(nimages, config, image_num=0, obj_num=0, nproc=1, logger=None):
    """
    Build a number of postage stamp images as specified by the config dict.

    @param nimages          How many images to build.
    @param config           The configuration dict.
    @param image_num        If given, the current image number. [default: 0]
    @param obj_num          If given, the first object number in the image. [default: 0]
    @param nproc            How many processes to use. [default: 1]
    @param logger           If given, a logger object to log progress. [default: None]

    @returns a list of images
    """
    if logger and logger.isEnabledFor(logging.DEBUG):
        logger.debug('file %d: BuildImages nimages = %d: image, obj = %d,%d',
                     config.get('file_num',0),nimages,image_num,obj_num)

    if nproc != 1:
        # Figure out how many jobs to do per task.
        # Number of images to do in each task should be:
        #  - At most nimages / nproc.
        #  - At least 1 normally, but number in Ring if doing a Ring test
        # Shoot for gemoetric mean of these two values.
        max_nim = nimages / nproc
        min_nim = 1
        if ( ('image' not in config or 'type' not in config['image'] or
                    config['image']['type'] == 'Single') and
                'gal' in config and isinstance(config['gal'],dict) and 'type' in config['gal'] and
                config['gal']['type'] == 'Ring' and 'num' in config['gal'] ):
            min_nim = galsim.config.ParseValue(config['gal'], 'num', config, int)[0]
            if logger and logger.isEnabledFor(logging.DEBUG):
                logger.debug('file %d: Found ring: num = %d',config.get('file_num',0),min_nim)
        if max_nim < min_nim:
            nim_per_task = min_nim
        else:
            import math
            # This formula keeps nim a multiple of min_nim, so Rings are intact.
            nim_per_task = min_nim * int(math.sqrt(float(max_nim) / float(min_nim)))
        if logger and logger.isEnabledFor(logging.DEBUG):
            logger.debug('file %d: nim_per_task = %d',config.get('file_num',0), nim_per_task)
    else:
        nim_per_task = 1  # ignored if MultiProcess gets nproc=1

    jobs = []
    for k in range(nimages):
        kwargs = { 'image_num' : image_num, 'obj_num' : obj_num }
        jobs.append( (kwargs, image_num) )
        obj_num += galsim.config.GetNObjForImage(config, image_num)
        image_num += 1

    def done_func(logger, proc, image_num, image, t):
        if logger and logger.isEnabledFor(logging.INFO):
            # Note: numpy shape is y,x
            ys, xs = image.array.shape
            if proc is None: s0 = ''
            else: s0 = '%s: '%proc
            logger.info(s0 + 'Image %d: size = %d x %d, time = %f sec', image_num, xs, ys, t)

    def except_func(logger, proc, e, tr, image_num):
        if logger:
            if proc is None: s0 = ''
            else: s0 = '%s: '%proc
            logger.error(s0 + 'Exception caught when building image %d', image_num)
            logger.error('%s',tr)
            logger.error('Aborting the rest of this file')

    nproc, images = galsim.config.MultiProcess(nproc, config, BuildImage, jobs, 'stamp', logger,
                                               njobs_per_task = nim_per_task,
                                               done_func = done_func,
                                               except_func = except_func)

    if logger and logger.isEnabledFor(logging.DEBUG):
        logger.debug('file %d: Done making images',config.get('file_num',0))

    return images


def SetupConfigImageNum(config, image_num, obj_num):
    """Do the basic setup of the config dict at the image processing level.

    Includes:
    - Set config['image_num'] = image_num
    - Set config['obj_num'] = obj_num
    - Set config['index_key'] = 'image_num'
    - Make sure config['image'] exists
    - Set config['image']['draw_method'] to 'auto' if not given.

    @param config           The configuration dict.
    @param image_num        The current image number.
    @param obj_num          The first object number in the image.
    """
    config['image_num'] = image_num
    config['obj_num'] = obj_num
    config['index_key'] = 'image_num'

    # Make config['image'] exist if it doesn't yet.
    if 'image' not in config:
        config['image'] = {}
    image = config['image']
    if not isinstance(image, dict):
        raise AttributeError("config.image is not a dict.")

    if 'draw_method' not in image:
        image['draw_method'] = 'auto'
    if 'type' not in image:
        image['type'] = 'Single'


def SetupConfigImageSize(config, xsize, ysize):
    """Do some further setup of the config dict at the image processing level based on
    the provided image size.

    - Set config['image_xsize'], config['image_ysize'] to the size of the image
    - Set config['image_origin'] to the origin of the image
    - Set config['image_center'] to the center of the image
    - Set config['image_bounds'] to the bounds of the image
    - Build the WCS based on either config['image']['wcs'] or config['image']['pixel_scale']
    - Set config['wcs'] to be the built wcs
    - If wcs.isPixelScale(), also set config['pixel_scale'] for convenience.

    @param config       The configuration dict.
    @param xsize        The size of the image in the x-dimension.
    @param ysize        The size of the image in the y-dimension.
    """
    config['image_xsize'] = xsize
    config['image_ysize'] = ysize

    origin = 1 # default
    if 'index_convention' in config['image']:
        convention = galsim.config.ParseValue(config['image'],'index_convention',config,str)[0]
        if convention.lower() in [ '0', 'c', 'python' ]:
            origin = 0
        elif convention.lower() in [ '1', 'fortran', 'fits' ]:
            origin = 1
        else:
            raise AttributeError("Unknown index_convention: %s"%convention)

    config['image_origin'] = galsim.PositionI(origin,origin)
    config['image_center'] = galsim.PositionD( origin + (xsize-1.)/2., origin + (ysize-1.)/2. )
    config['image_bounds'] = galsim.BoundsI(origin, origin+xsize-1, origin, origin+ysize-1)

    # Build the wcs
    wcs = galsim.config.BuildWCS(config)
    config['wcs'] = wcs

    # If the WCS is a PixelScale or OffsetWCS, then store the pixel_scale in base.  The
    # config apparatus does not use it -- we always use the wcs -- but we keep it in case
    # the user wants to use it for an Eval item.  It's one of the variables they are allowed
    # to assume will be present for them.
    if wcs.isPixelScale():
        config['pixel_scale'] = wcs.scale


# Ignore these when parsing the parameters for specific Image types:
image_ignore = [ 'random_seed', 'draw_method', 'noise', 'pixel_scale', 'wcs',
                 'sky_level', 'sky_level_pixel', 'index_convention', 'nproc',
                 'retry_failures', 'n_photons', 'wmult', 'offset', 'gsparams' ]


def BuildImage(config, image_num=0, obj_num=0, logger=None):
    """
    Build an Image according to the information in config.

    @param config           The configuration dict.
    @param image_num        If given, the current image number. [default: 0]
    @param obj_num          If given, the first object number in the image. [default: 0]
    @param logger           If given, a logger object to log progress. [default: None]

    @returns the final image
    """
    if logger and logger.isEnabledFor(logging.DEBUG):
        logger.debug('image %d: BuildImage: image, obj = %d,%d',image_num,image_num,obj_num)

    # Setup basic things in the top-level config dict that we will need.
    SetupConfigImageNum(config,image_num,obj_num)

    image_type = config['image']['type']
    if image_type not in valid_image_types:
        raise AttributeError("Invalid image.type=%s."%image_type)

    # Build the rng to use at the image level.
    seed = galsim.config.SetupConfigRNG(config)
    if logger and logger.isEnabledFor(logging.DEBUG):
        logger.debug('image %d: seed = %d',image_num,seed)
    rng = config['rng'] # Grab this for use later

    # Do the necessary initial setup for this image type.
    setup_func = valid_image_types[image_type]['setup']
    xsize, ysize = setup_func(config, image_num, obj_num, image_ignore, logger)

    # Given this image size (which may be 0,0, in which case it will be set automatically later),
    # do some basic calculations
    SetupConfigImageSize(config,xsize,ysize)
    if logger and logger.isEnabledFor(logging.DEBUG):
        logger.debug('image %d: image_size = %d, %d',image_num,xsize,ysize)
        logger.debug('image %d: image_origin = %s',image_num,config['image_origin'])
        logger.debug('image %d: image_center = %s',image_num,config['image_center'])

    # Sometimes an input field needs to do something special at the start of an image.
    galsim.config.SetupInputsForImage(config,logger)

    # Likewise for the extra output items.
    galsim.config.SetupExtraOutputsForImage(config,logger)

    # Actually build the image now.  This is the main working part of this function.
    # It calls out to the appropriate build function for this image type.
    build_func = valid_image_types[image_type]['build']
    image = build_func(config, image_num, obj_num, logger)

    # Store the current image in the base-level config for reference
    config['current_image'] = image

    # Mark that we are no longer doing a single galaxy by deleting image_pos from config top
    # level, so it cannot be used for things like wcs.pixelArea(image_pos).
    if 'image_pos' in config: del config['image_pos']

    # Put the rng back into config['rng'] for use by the AddNoise function.
    config['rng'] = rng

    # Do whatever processing is required for the extra output items.
    galsim.config.ProcessExtraOutputsForImage(config,logger)

    noise_func = valid_image_types[image_type]['noise']
    if noise_func:
        noise_func(image, config, image_num, obj_num, logger)

    return image


def GetNObjForImage(config, image_num):
    """
    Get the number of objects that will be made for the image number image_num based on
    the information in the config dict.

    @param config           The configuration dict.
    @param image_num        The current image number.

    @returns the number of objects
    """
    if 'image' in config and 'type' in config['image']:
        image_type = config['image']['type']
    else:
        image_type = 'Single'

    # Check that the type is valid
    if image_type not in valid_image_types:
        raise AttributeError("Invalid image.type=%s."%type)

    nobj_func = valid_image_types[image_type]['nobj']

    return nobj_func(config,image_num)


def FlattenNoiseVariance(config, full_image, stamps, current_vars, logger):
    """This is a helper function to bring the noise level up to a constant value
    across the image.  If some of the galaxies are RealGalaxy objects and noise whitening
    (or symmetrizing) is turned on, then there will already be some noise in the
    stamps that get built.  This function goes through and figures out what the maximum
    current variance is anywhere in the full image and adds noise to the other pixels
    to bring everything up to that level.

    @param config           The configuration dict.
    @param full_image       The full image onto which the noise should be added.
    @param stamps           A list of the individual postage stamps.
    @param current_vars     A list of the current variance in each postage stamps.
    @param logger           If given, a logger object to log progress.

    @returns the final variance in the image
    """
    rng = config['rng']
    nobjects = len(stamps)
    max_current_var = max(current_vars)
    if max_current_var > 0:
        if logger and logger.isEnabledFor(logging.DEBUG):
            logger.debug('image %d: maximum noise varance in any stamp is %f',
                         config['image_num'], max_current_var)
        import numpy
        # Then there was whitening applied in the individual stamps.
        # But there could be a different variance in each postage stamp, so the first
        # thing we need to do is bring everything up to a common level.
        noise_image = galsim.ImageF(full_image.bounds)
        for k in range(nobjects):
            b = stamps[k].bounds & full_image.bounds
            if b.isDefined(): noise_image[b] += current_vars[k]
        # Update this, since overlapping postage stamps may have led to a larger
        # value in some pixels.
        max_current_var = numpy.max(noise_image.array)
        if logger and logger.isEnabledFor(logging.DEBUG):
            logger.debug('image %d: maximum noise varance in any pixel is %f',
                         config['image_num'], max_current_var)
        # Figure out how much noise we need to add to each pixel.
        noise_image = max_current_var - noise_image
        # Add it.
        full_image.addNoise(galsim.VariableGaussianNoise(rng,noise_image))
    # Now max_current_var is how much noise is in each pixel.
    return max_current_var


def SetupSingle(config, image_num, obj_num, ignore, logger):
    """
    Do the initialization and setup for building a Single image.

    @param config           The configuration dict.
    @param image_num        The current image number.
    @param obj_num          The first object number in the image.
    @param ignore           A list of parameters that are allowed to be in config['image']
                            that we can ignore here.  i.e. it won't be an error if these
                            parameters are present.
    @param logger           If given, a logger object to log progress.

    @returns xsize, ysize for the image (not built yet)
    """
    if logger and logger.isEnabledFor(logging.DEBUG):
        logger.debug('image %d: BuildSingleImage: image, obj = %d,%d',image_num,image_num,obj_num)

    extra_ignore = [ 'image_pos', 'world_pos' ]
    opt = { 'size' : int , 'xsize' : int , 'ysize' : int }
    params = galsim.config.GetAllParams(config['image'], config, opt=opt,
                                        ignore=ignore+extra_ignore)[0]

    # If image_force_xsize and image_force_ysize were set in config, this overrides the
    # read-in params.
    if 'image_force_xsize' in config and 'image_force_ysize' in config:
        xsize = config['image_force_xsize']
        ysize = config['image_force_ysize']
    else:
        size = params.get('size',0)
        xsize = params.get('xsize',size)
        ysize = params.get('ysize',size)
    if (xsize == 0) != (ysize == 0):
        raise AttributeError(
            "Both (or neither) of image.xsize and image.ysize need to be defined  and != 0.")

    # We allow world_pos to be in config[image], but we don't want it to lead to a final_shift
    # in BuildStamp.  The easiest way to do this is to set image_pos to (0,0).
    if 'world_pos' in config['image']:
        config['image']['image_pos'] = (0,0)

    return xsize, ysize


def BuildSingle(config, image_num, obj_num, logger):
    """
    Build an Image consisting of a single stamp.

    @param config           The configuration dict.
    @param image_num        The current image number.
    @param obj_num          The first object number in the image.
    @param logger           If given, a logger object to log progress.

    @returns the final image
    """
    xsize = config['image_xsize']
    ysize = config['image_ysize']

    image, current_var = galsim.config.BuildStamp(
            config, obj_num=obj_num, xsize=xsize, ysize=ysize,
            do_noise=True, logger=logger)

    return image

def GetNObjSingle(config, image_num):
    """
    Get the number of objects for an Image consisting of a single stamp.

    @param config           The configuration dict.
    @param image_num        The current image number.
    @param obj_num          The first object number in the image.

    @returns 1
    """
    return 1


valid_image_types = {}

def RegisterImageType(image_type, setup_func, build_func, noise_func, nobj_func):
    """Register an image type for use by the config apparatus.

    @param image_type       The name of the type in config['image']
    @param setup_func       The function to call to determine the size of the image and do any
                            other initial setup.
                            The call signature is 
                                xsize, ysize = Setup(config, image_num, obj_num, ignore, logger)
    @param build_func       The function to call for building the image
                            The call signature is:
                                image = Build(config, image_num, obj_num, logger)
    @param noise_func       The function to call to add noise and sky level to the image.
                            The call signature is 
                                AddNoise(image, config, image_num, obj_num, logger)
    @param nobj_func        A function that returns the number of objects that will be built.
                            The call signature is 
                                nobj = GetNObj(config, image_num)
    """
    valid_image_types[image_type] = {
        'setup' : setup_func,
        'build' : build_func,
        'noise' : noise_func,
        'nobj' : nobj_func,
    }

RegisterImageType('Single', SetupSingle, BuildSingle, None, GetNObjSingle)

