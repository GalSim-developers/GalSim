# Copyright (c) 2012-2018 by the GalSim developers team on GitHub
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
import numpy as np

# This file handles the building of an image by parsing config['image'].
# This file includes the basic functionality, but it calls out to helper functions
# for parts of the process that are different for different image types.  It includes
# those helper functions for the simplest image type, Single.  See image_tiled.py and
# image_scattered.py for the implementation of the Tiled and Scattered image types.

# This module-level dict will store all the registered image types.
# See the RegisterImageType function at the end of this file.
# The keys are the (string) names of the image types, and the values will be builder objects
# that will perform the different stages of processing to build each full image.
valid_image_types = {}


def BuildImages(nimages, config, image_num=0, obj_num=0, logger=None):
    """
    Build a number of postage stamp images as specified by the config dict.

    @param nimages          How many images to build.
    @param config           The configuration dict.
    @param image_num        If given, the current image number. [default: 0]
    @param obj_num          If given, the first object number in the image. [default: 0]
    @param logger           If given, a logger object to log progress. [default: None]

    @returns a list of images
    """
    logger = galsim.config.LoggerWrapper(logger)
    logger.debug('file %d: BuildImages nimages = %d: image, obj = %d,%d',
                 config.get('file_num',0),nimages,image_num,obj_num)

    # Figure out how many processes we will use for building the images.
    if 'image' not in config: config['image'] = {}
    image = config['image']
    if nimages > 1 and 'nproc' in image:
        nproc = galsim.config.ParseValue(image, 'nproc', config, int)[0]
        # Update this in case the config value is -1
        nproc = galsim.config.UpdateNProc(nproc, nimages, config, logger)
    else:
        nproc = 1

    jobs = []
    for k in range(nimages):
        kwargs = { 'image_num' : image_num, 'obj_num' : obj_num }
        jobs.append(kwargs)
        obj_num += galsim.config.GetNObjForImage(config, image_num)
        image_num += 1

    def done_func(logger, proc, k, image, t):
        if image is not None:
            # Note: numpy shape is y,x
            ys, xs = image.array.shape
            if proc is None: s0 = ''
            else: s0 = '%s: '%proc
            image_num = jobs[k]['image_num']
            logger.info(s0 + 'Image %d: size = %d x %d, time = %f sec', image_num, xs, ys, t)

    def except_func(logger, proc, k, e, tr):
        if proc is None: s0 = ''
        else: s0 = '%s: '%proc
        image_num = jobs[k]['image_num']
        logger.error(s0 + 'Exception caught when building image %d', image_num)
        logger.debug('%s',tr)
        logger.error('Aborting the rest of this file')

    # Convert to the tasks structure we need for MultiProcess
    tasks = MakeImageTasks(config, jobs, logger)

    images = galsim.config.MultiProcess(nproc, config, BuildImage, tasks, 'image', logger,
                                        done_func = done_func,
                                        except_func = except_func)

    logger.debug('file %d: Done making images',config.get('file_num',0))
    if len(images) == 0:
        logger.error('No images were built.  All were either skipped or had errors.')

    return images

def SetupConfigImageNum(config, image_num, obj_num, logger=None):
    """Do the basic setup of the config dict at the image processing level.

    Includes:
    - Set config['image_num'] = image_num
    - Set config['obj_num'] = obj_num
    - Set config['index_key'] = 'image_num'
    - Make sure config['image'] exists
    - Set default config['image']['type'] to 'Single' if not specified
    - Check that the specified image type is valid.

    @param config           The configuration dict.
    @param image_num        The current image number.
    @param obj_num          The first object number in the image.
    @param logger           If given, a logger object to log progress. [default: None]
    """
    logger = galsim.config.LoggerWrapper(logger)
    config['image_num'] = image_num
    config['obj_num'] = obj_num
    config['index_key'] = 'image_num'

    # Make config['image'] exist if it doesn't yet.
    if 'image' not in config:
        config['image'] = {}
    image = config['image']
    if not isinstance(image, dict):
        raise galsim.GalSimConfigError("config.image is not a dict.")

    if 'file_num' not in config:
        config['file_num'] = 0

    if 'type' not in image:
        image['type'] = 'Single'
    image_type = image['type']
    if image_type not in valid_image_types:
        raise galsim.GalSimConfigValueError("Invalid image.type.", image_type, valid_image_types)

    # In case this hasn't been done yet.
    galsim.config.SetupInput(config, logger)

    # Build the rng to use at the image level.
    seed = galsim.config.SetupConfigRNG(config, logger=logger)
    logger.debug('image %d: seed = %d',image_num,seed)



def SetupConfigImageSize(config, xsize, ysize, logger=None):
    """Do some further setup of the config dict at the image processing level based on
    the provided image size.

    - Set config['image_xsize'], config['image_ysize'] to the size of the image
    - Set config['image_origin'] to the origin of the image
    - Set config['image_center'] to the center of the image
    - Set config['image_bounds'] to the bounds of the image
    - Build the WCS based on either config['image']['wcs'] or config['image']['pixel_scale']
    - Set config['wcs'] to be the built wcs
    - If wcs.isPixelScale(), also set config['pixel_scale'] for convenience.
    - Set config['world_center'] to either a given value or based on wcs and image_center

    @param config       The configuration dict.
    @param xsize        The size of the image in the x-dimension.
    @param ysize        The size of the image in the y-dimension.
    @param logger       If given, a logger object to log progress. [default: None]
    """
    logger = galsim.config.LoggerWrapper(logger)
    config['image_xsize'] = xsize
    config['image_ysize'] = ysize
    image = config['image']

    origin = 1 # default
    if 'index_convention' in image:
        convention = galsim.config.ParseValue(image,'index_convention',config,str)[0]
        if convention.lower() in ('0', 'c', 'python'):
            origin = 0
        elif convention.lower() in ('1', 'fortran', 'fits'):
            origin = 1
        else:
            raise galsim.GalSimConfigValueError("Unknown index_convention", convention,
                                                ('0', 'c', 'python', '1', 'fortran', 'fits'))

    config['image_origin'] = galsim.PositionI(origin,origin)
    config['image_center'] = galsim.PositionD( origin + (xsize-1.)/2., origin + (ysize-1.)/2. )
    config['image_bounds'] = galsim.BoundsI(origin, origin+xsize-1, origin, origin+ysize-1)

    # Build the wcs
    wcs = galsim.config.BuildWCS(image, 'wcs', config, logger)
    config['wcs'] = wcs

    # If the WCS is a PixelScale or OffsetWCS, then store the pixel_scale in base.  The
    # config apparatus does not use it -- we always use the wcs -- but we keep it in case
    # the user wants to use it for an Eval item.  It's one of the variables they are allowed
    # to assume will be present for them.
    if wcs.isPixelScale():
        config['pixel_scale'] = wcs.scale

    # Set world_center
    if 'world_center' in image:
        config['world_center'] = galsim.config.ParseValue(image, 'world_center', config,
                                                          galsim.CelestialCoord)[0]
    else:
        config['world_center'] = wcs.toWorld(config['image_center'])


# Ignore these when parsing the parameters for specific Image types:
from .stamp import stamp_image_keys
image_ignore = [ 'random_seed', 'noise', 'pixel_scale', 'wcs', 'sky_level', 'sky_level_pixel',
                 'world_center', 'index_convention', 'nproc'] + stamp_image_keys

def BuildImage(config, image_num=0, obj_num=0, logger=None):
    """
    Build an Image according to the information in config.

    @param config           The configuration dict.
    @param image_num        If given, the current image number. [default: 0]
    @param obj_num          If given, the first object number in the image. [default: 0]
    @param logger           If given, a logger object to log progress. [default: None]

    @returns the final image
    """
    logger = galsim.config.LoggerWrapper(logger)
    logger.debug('image %d: BuildImage: image, obj = %d,%d',image_num,image_num,obj_num)

    # Setup basic things in the top-level config dict that we will need.
    SetupConfigImageNum(config, image_num, obj_num, logger)

    cfg_image = config['image']  # Use cfg_image to avoid name confusion with the actual image
                                 # we will build later.
    image_type = cfg_image['type']

    # Do the necessary initial setup for this image type.
    builder = valid_image_types[image_type]
    xsize, ysize = builder.setup(cfg_image, config, image_num, obj_num, image_ignore, logger)

    # Given this image size (which may be 0,0, in which case it will be set automatically later),
    # do some basic calculations
    SetupConfigImageSize(config, xsize, ysize, logger)
    logger.debug('image %d: image_size = %d, %d',image_num,xsize,ysize)
    logger.debug('image %d: image_origin = %s',image_num,config['image_origin'])
    logger.debug('image %d: image_center = %s',image_num,config['image_center'])

    # Sometimes an input field needs to do something special at the start of an image.
    galsim.config.SetupInputsForImage(config, logger)

    # Likewise for the extra output items.
    galsim.config.SetupExtraOutputsForImage(config, logger)

    # Actually build the image now.  This is the main working part of this function.
    # It calls out to the appropriate build function for this image type.
    image, current_var = builder.buildImage(cfg_image, config, image_num, obj_num, logger)

    # Store the current image in the base-level config for reference
    config['current_image'] = image

    # Just in case these changed from their initial values, make sure they are correct now:
    if image is not None:
        config['image_origin'] = image.origin
        config['image_center'] = image.true_center
        config['image_bounds'] = image.bounds
    logger.debug('image %d: image_origin => %s',image_num,config['image_origin'])
    logger.debug('image %d: image_center => %s',image_num,config['image_center'])
    logger.debug('image %d: image_bounds => %s',image_num,config['image_bounds'])

    # Mark that we are no longer doing a single galaxy by deleting image_pos from config top
    # level, so it cannot be used for things like wcs.pixelArea(image_pos).
    config.pop('image_pos', None)

    # Go back to using image_num for any indexing.
    config['index_key'] = 'image_num'

    # Do whatever processing is required for the extra output items.
    galsim.config.ProcessExtraOutputsForImage(config,logger)

    builder.addNoise(image, cfg_image, config, image_num, obj_num, current_var, logger)

    return image


def GetNObjForImage(config, image_num):
    """
    Get the number of objects that will be made for the image number image_num based on
    the information in the config dict.

    @param config           The configuration dict.
    @param image_num        The current image number.

    @returns the number of objects
    """
    image = config.get('image',{})
    image_type = image.get('type','Single')
    if image_type not in valid_image_types:
        raise galsim.GalSimConfigValueError("Invalid image.type.", image_type, valid_image_types)
    return valid_image_types[image_type].getNObj(image,config,image_num)


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
    logger = galsim.config.LoggerWrapper(logger)
    rng = config['image_num_rng']
    nobjects = len(stamps)
    max_current_var = max(current_vars)
    if max_current_var > 0:
        logger.debug('image %d: maximum noise varance in any stamp is %f',
                     config['image_num'], max_current_var)
        # Then there was whitening applied in the individual stamps.
        # But there could be a different variance in each postage stamp, so the first
        # thing we need to do is bring everything up to a common level.
        noise_image = galsim.ImageF(full_image.bounds)
        for k in range(nobjects):
            if stamps[k] is None: continue
            b = stamps[k].bounds & full_image.bounds
            if b.isDefined(): noise_image[b] += current_vars[k]
        # Update this, since overlapping postage stamps may have led to a larger
        # value in some pixels.
        max_current_var = np.max(noise_image.array)
        logger.debug('image %d: maximum noise varance in any pixel is %f',
                     config['image_num'], max_current_var)
        # Figure out how much noise we need to add to each pixel.
        noise_image = max_current_var - noise_image
        # Add it.
        full_image.addNoise(galsim.VariableGaussianNoise(rng,noise_image))
    # Now max_current_var is how much noise is in each pixel.
    return max_current_var


def MakeImageTasks(config, jobs, logger):
    """Turn a list of jobs into a list of tasks.

    See the doc string for galsim.config.MultiProcess for the meaning of this distinction.

    For most image types, there is just one job per task, so the tasks list is just:

        tasks = [ [ (job, k) ] for k, job in enumerate(jobs) ]

    But some image types may need groups of jobs to be done sequentially by the same process.
    The image type=Single for instance uses whatever grouping is needed for the stamp type.

    @param config           The configuration dict
    @param jobs             A list of jobs to split up into tasks.  Each job in the list is a
                            dict of parameters that includes 'image_num' and 'obj_num'.
    @param logger           If given, a logger object to log progress.

    @returns a list of tasks
    """
    image = config.get('image', {})
    image_type = image.get('type', 'Single')
    if image_type not in valid_image_types:
        raise galsim.GalSimConfigValueError("Invalid image.type.", image_type, valid_image_types)
    return valid_image_types[image_type].makeTasks(image, config, jobs, logger)


class ImageBuilder(object):
    """A base class for building full images.

    The base class defines the call signatures of the methods that any derived class should follow.
    It also includes the implementation of the default image type: Single.
    """

    def setup(self, config, base, image_num, obj_num, ignore, logger):
        """Do the initialization and setup for building the image.

        This figures out the size that the image will be, but doesn't actually build it yet.

        @param config       The configuration dict for the image field.
        @param base         The base configuration dict.
        @param image_num    The current image number.
        @param obj_num      The first object number in the image.
        @param ignore       A list of parameters that are allowed to be in config that we can
                            ignore here. i.e. it won't be an error if these parameters are present.
        @param logger       If given, a logger object to log progress.

        @returns xsize, ysize
        """
        logger.debug('image %d: Build Single Image: image, obj = %d,%d',
                     image_num,image_num,obj_num)

        extra_ignore = [ 'image_pos', 'world_pos' ]
        opt = { 'size' : int , 'xsize' : int , 'ysize' : int }
        params = galsim.config.GetAllParams(config, base, opt=opt, ignore=ignore+extra_ignore)[0]

        # If image_force_xsize and image_force_ysize were set in base, this overrides the
        # read-in params.
        if 'image_force_xsize' in base and 'image_force_ysize' in base:
            xsize = base['image_force_xsize']
            ysize = base['image_force_ysize']
        else:
            size = params.get('size',0)
            xsize = params.get('xsize',size)
            ysize = params.get('ysize',size)
        if (xsize == 0) != (ysize == 0):
            raise galsim.GalSimConfigError(
                "Both (or neither) of image.xsize and image.ysize need to be defined and != 0.")

        # We allow world_pos to be in config[image], but we don't want it to lead to a final_shift
        # in BuildStamp.  To mark this, we set image_pos to (0,0)
        if 'world_pos' in config and 'image_pos' not in config:
            config['image_pos'] = galsim.PositionD(0,0)

        return xsize, ysize


    def buildImage(self, config, base, image_num, obj_num, logger):
        """Build an Image based on the parameters in the config dict.

        For Single, this is just an image consisting of a single postage stamp.

        @param config       The configuration dict for the image field.
        @param base         The base configuration dict.
        @param image_num    The current image number.
        @param obj_num      The first object number in the image.
        @param logger       If given, a logger object to log progress.

        @returns the final image and the current noise variance in the image as a tuple
        """
        xsize = base['image_xsize']
        ysize = base['image_ysize']
        logger.debug('image %d: Single Image: size = %s, %s',image_num,xsize,ysize)

        # In case there was one set from before, we don't want to confuse the stamp builder
        # thinking that this is the full image onto which we are drawing this object.
        base['current_image'] = None

        image, current_var = galsim.config.BuildStamp(
                base, obj_num=obj_num, xsize=xsize, ysize=ysize, do_noise=True, logger=logger)
        return image, current_var

    def makeTasks(self, config, base, jobs, logger):
        """Turn a list of jobs into a list of tasks.

        Each task is performed separately in multi-processing runs, so this provides a mechanism
        to have multiple jobs depend on each other without being messed up by multi-processing.
        E.g. you could have blends where each task consists of building several overlapping
        galaxies (each of which would be a single job).  Perhaps the first job would include
        a calculation to determine where all the overlapping galaxies should go, and the later
        jobs would use the results of this calculation and just place the later galaxies in the
        appropriate place.
        
        Normally, though, each task is just a single job, in which case, this function is very
        simple.

        For Single, this passes the job onto the MakeStampTasks function (which in turn is
        normally quite simple).  Most other types though probably want one job per task, for which
        the appropriate code would be:

            return [ [ (job, k) ] for k, job in enumerate(jobs) ]

        @param config       The configuration dict for the image field.
        @param base         The base configuration dict.
        @param jobs         A list of jobs to split up into tasks.  Each job in the list is a
                            dict of parameters that includes 'image_num' and 'obj_num'.
        @param logger       If given, a logger object to log progress.

        @returns a list of tasks
        """
        return galsim.config.MakeStampTasks(base, jobs, logger)

    def addNoise(self, image, config, base, image_num, obj_num, current_var, logger):
        """Add the final noise to the image.

        In the base class, this is a no op, since it directs the BuildStamp function to build
        the noise at that level.  But some image types need to do extra work at the end to
        add the noise properly.

        @param image        The image onto which to add the noise.
        @param config       The configuration dict for the image field.
        @param base         The base configuration dict.
        @param image_num    The current image number.
        @param obj_num      The first object number in the image.
        @param current_var  The current noise variance in each postage stamps.
        @param logger       If given, a logger object to log progress.
        """
        pass

    def getNObj(self, config, base, image_num):
        """Get the number of objects that will be built for this image.

        For Single, this is just 1, but other image types would figure this out from the
        configuration parameters.

        @param config       The configuration dict for the image field.
        @param base         The base configuration dict.
        @param image_num    The current image number.

        @returns the number of objects
        """
        return 1

def RegisterImageType(image_type, builder):
    """Register an image type for use by the config apparatus.

    @param image_type       The name of the type in config['image']
    @param builder          A builder object to use for building the images.  It should be an
                            instance of ImageBuilder or a subclass thereof.
    """
    valid_image_types[image_type] = builder

RegisterImageType('Single', ImageBuilder())

