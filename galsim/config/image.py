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

from .image_scattered import *
from .image_tiled import *

# The items in each tuple are:
#   - The function to call to build the image
#   - The function to call to get the number of objects that will be built
valid_image_types = { 
    'Single' : ( 'BuildSingleImage', 'GetNObjSingle' ),
    'Tiled' : ( 'BuildTiledImage', 'GetNObjTiled' ),
    'Scattered' : ( 'BuildScatteredImage', 'GetNObjScattered' ),
}


def BuildImages(nimages, config, logger=None, image_num=0, obj_num=0,
                make_psf_image=False, make_weight_image=False, make_badpix_image=False):
    """
    Build a number of postage stamp images as specified by the config dict.

    @param nimages             How many images to build.
    @param config              A configuration dict.
    @param logger              If given, a logger object to log progress. [default: None]
    @param image_num           If given, the current `image_num` [default: 0]
    @param obj_num             If given, the current `obj_num` [default: 0]
    @param make_psf_image      Whether to make `psf_image`. [default: False]
    @param make_weight_image   Whether to make `weight_image`. [default: False]
    @param make_badpix_image   Whether to make `badpix_image`. [default: False]

    @returns the tuple `(images, psf_images, weight_images, badpix_images)`.
             All in tuple are lists.
    """
    config['index_key'] = 'image_num'
    config['image_num'] = image_num
    config['obj_num'] = obj_num

    if logger and logger.isEnabledFor(logging.DEBUG):
        logger.debug('file %d: BuildImages nimages = %d: image, obj = %d,%d',
                     config.get('file_num',0),nimages,image_num,obj_num)

    if ( 'image' in config 
         and 'random_seed' in config['image'] 
         and not isinstance(config['image']['random_seed'],dict) ):
        first = galsim.config.ParseValue(config['image'], 'random_seed', config, int)[0]
        config['image']['random_seed'] = { 'type' : 'Sequence', 'first' : first }

    # Figure out how many processes we will use for building the images.
    if nimages > 1 and 'image' in config and 'nproc' in config['image']:
        nproc = galsim.config.ParseValue(config['image'], 'nproc', config, int)[0]
        # Update this in case the config value is -1
        nproc = galsim.config.UpdateNProc(nproc, nimages, config, logger)
    else:
        nproc = 1

    if nproc != 1:
        # Number of images to do in each task:
        # At least 1 normally, but number in Ring if doing a Ring test
        # Shoot for gemoetric mean of these two.
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
            logger.debug('file %d: nim_per_task = %d',config.get('file_num',0),nim_per_task)
    else:
        nim_per_task = 1

    jobs = []
    for k in range(nimages):
        # The kwargs to pass to BuildImage
        kwargs = {
            'make_psf_image' : make_psf_image,
            'make_weight_image' : make_weight_image,
            'make_badpix_image' : make_badpix_image,
            'image_num' : image_num,
            'obj_num' : obj_num
        }
        obj_num += galsim.config.GetNObjForImage(config, image_num)
        image_num += 1
        jobs.append( (kwargs, image_num) )

    def done_func(logger, proc, image_num, result, t):
        if logger and logger.isEnabledFor(logging.INFO):
            # Note: numpy shape is y,x
            image = result[0]
            ys, xs = image.array.shape
            if proc is None: s0 = ''
            else: s0 = '%s: '%proc
            logger.info(s0 + 'Image %d: size = %d x %d, time = %f sec', image_num, xs, ys, t)
 
    def except_func(logger, proc, e, tr, image_num):
        if logger:
            if proc is None: s0 = ''
            else: s0 = '%s: '%proc
            logger.error(s0 + 'Exception caught when building image %d', image_num)
            #logger.error('%s',tr)
            logger.error('Aborting the rest of this file')

    results = galsim.config.MultiProcess(nproc, config, BuildImage, jobs, 'image', logger,
                                         njobs_per_task = nim_per_task,
                                         done_func = done_func,
                                         except_func = except_func)

    # reshape the results into 4 lists
    images, psf_images, weight_images, badpix_images = zip(*results)

    if logger and logger.isEnabledFor(logging.DEBUG):
        logger.debug('file %d: Done making images',config.get('file_num',0))

    return images, psf_images, weight_images, badpix_images
 

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


image_ignore = [ 'random_seed', 'draw_method', 'noise', 'pixel_scale', 'wcs',
                 'sky_level', 'sky_level_pixel', 'index_convention', 'nproc',
                 'retry_failures', 'n_photons', 'wmult', 'offset', 'gsparams' ]


def BuildImage(config, logger=None, image_num=0, obj_num=0,
               make_psf_image=False, make_weight_image=False, make_badpix_image=False):
    """
    Build an Image according to the information in config.

    @param config              A configuration dict.
    @param logger              If given, a logger object to log progress. [default: None]
    @param image_num           If given, the current `image_num` [default: 0]
    @param obj_num             If given, the current `obj_num` [default: 0]
    @param make_psf_image      Whether to make `psf_image`. [default: False]
    @param make_weight_image   Whether to make `weight_image`. [default: False]
    @param make_badpix_image   Whether to make `badpix_image`. [default: False]

    @returns the tuple `(image, psf_image, weight_image, badpix_image)`.

    Note: All 4 images are always returned in the return tuple,
          but the latter 3 might be None depending on the parameters make_*_image.
    """
    if logger and logger.isEnabledFor(logging.DEBUG):
        logger.debug('image %d: BuildImage: image, obj = %d,%d',image_num,image_num,obj_num)

    # Setup basic things in the top-level config dict that we will need.
    SetupConfigImageNum(config,image_num,obj_num)

    image_type = config['image']['type']
    if image_type not in valid_image_types:
        raise AttributeError("Invalid image.type=%s."%image_type)

    build_func = eval(valid_image_types[image_type][0])
    all_images = build_func(
            config=config, logger=logger,
            image_num=image_num, obj_num=obj_num,
            make_psf_image=make_psf_image, 
            make_weight_image=make_weight_image,
            make_badpix_image=make_badpix_image)

    # The later image building functions build up the weight image as the total variance 
    # in each pixel.  We need to invert this to produce the inverse variance map.
    # Doing it here means it only needs to be done in this one place.
    if all_images[2]:
        all_images[2].invertSelf()

    return all_images


def GetNObjForImage(config, image_num):
    if 'image' in config and 'type' in config['image']:
        image_type = config['image']['type']
    else:
        image_type = 'Single'

    # Check that the type is valid
    if image_type not in valid_image_types:
        raise AttributeError("Invalid image.type=%s."%type)

    nobj_func = eval(valid_image_types[image_type][1])

    return nobj_func(config,image_num)


def FlattenNoiseVariance(config, full_image, stamps, current_vars, logger):
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


def BuildSingleImage(config, logger=None, image_num=0, obj_num=0,
                     make_psf_image=False, make_weight_image=False, make_badpix_image=False):
    """
    Build an Image consisting of a single stamp.

    @param config              A configuration dict.
    @param logger              If given, a logger object to log progress. [default: None]
    @param image_num           If given, the current `image_num` [default: 0]
    @param obj_num             If given, the current `obj_num` [default: 0]
    @param make_psf_image      Whether to make `psf_image`. [default: False]
    @param make_weight_image   Whether to make `weight_image`. [default: False]
    @param make_badpix_image   Whether to make `badpix_image`. [default: False]

    @returns the tuple `(image, psf_image, weight_image, badpix_image)`.

    Note: All 4 Images are always returned in the return tuple,
          but the latter 3 might be None depending on the parameters make_*_image.    
    """
    extra_ignore = [ 'image_pos', 'world_pos' ]
    opt = { 'size' : int , 'xsize' : int , 'ysize' : int }
    params = galsim.config.GetAllParams(config['image'], config, opt=opt, 
                                        ignore=image_ignore+extra_ignore)[0]

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

    SetupConfigImageSize(config, xsize, ysize)
    if logger and logger.isEnabledFor(logging.DEBUG):
        logger.debug('image %d: image_origin = %s',image_num,str(config['image_origin']))
        logger.debug('image %d: image_center = %s',image_num,str(config['image_center']))

    if 'world_pos' in config['image']:
        config['image']['image_pos'] = (0,0)
        # We allow world_pos to be in config[image], but we don't want it to lead to a final_shift
        # in BuildStamp.  The easiest way to do this is to set image_pos to (0,0).

    return galsim.config.BuildStamp(
            config=config, xsize=xsize, ysize=ysize, obj_num=obj_num,
            do_noise=True, logger=logger,
            make_psf_image=make_psf_image, 
            make_weight_image=make_weight_image,
            make_badpix_image=make_badpix_image)[:4] # Required due to `current_var, time` being
                                                     # last two elements of the BuildStamp
                                                     # return tuple

def GetNObjSingle(config, image_num):
    return 1


