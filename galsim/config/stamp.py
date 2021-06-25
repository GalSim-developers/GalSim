# Copyright (c) 2012-2021 by the GalSim developers team on GitHub
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

import logging
import numpy as np
import math

from .util import LoggerWrapper, GetRNG, UpdateNProc, MultiProcess, SetupConfigRNG, RemoveCurrent
from .input import SetupInput
from .gsobject import UpdateGSParams, SkipThisObject
from .extra import ProcessExtraOutputsForStamp
from .gsobject import BuildGSObject
from .value import ParseValue, CheckAllParams
from .noise import CalculateNoiseVariance, AddSky, AddNoise
from .wcs import BuildWCS
from .photon_ops import BuildPhotonOps
from ..errors import GalSimConfigError, GalSimConfigValueError
from ..image import Image, ImageF
from ..wcs import PixelScale
from ..position import PositionD, PositionI
from ..bounds import _BoundsI
from ..celestial import CelestialCoord
from ..angle import arcsec
from ..gsobject import GSObject
from ..convolve import Convolve
from ..chromatic import ChromaticObject
from ..bandpass import Bandpass

# This file handles the building of postage stamps to place onto a larger image.
# There is only one type of stamp currently, called Basic, which builds a galaxy from
# config['gal'] and a PSF from config['psf'] (either but not both of which may be absent),
# colvolves them together, and draws onto a postage stamp.  This is the default functionality
# and is typically not specified explicitly.  But there are hooks in place to allow for other
# options, either in future versions of GalSim or through user modules.

# This module-level dict will store all the registered stamp types.
# See the RegisterStampType function at the end of this file.
# The keys are the (string) names of the output types, and the values will be builder objects
# that will perform the different stages of processing to build each stamp image.
valid_stamp_types = {}


def BuildStamps(nobjects, config, obj_num=0,
                xsize=0, ysize=0, do_noise=True, logger=None):
    """
    Build a number of postage stamp images as specified by the config dict.

    Parameters:
        nobjects:       How many postage stamps to build.
        config:         A configuration dict.
        obj_num:        If given, the current obj_num. [default: 0]
        xsize:          The size of a single stamp in the x direction. [default: 0,
                        which means to look first for config.stamp.xsize, then for
                        config.image.stamp_xsize, and if neither are given, then use
                        automatic sizing.]
        ysize:          The size of a single stamp in the y direction. [default: 0,
                        which means to look first for config.stamp.ysize, then for
                        config.image.stamp_ysize, and if neither are given, then use
                        automatic sizing.]
        do_noise:       Whether to add noise to the image (according to config['noise']).
                        [default: True]
        logger:         If given, a logger object to log progress. [default: None]

    Returns:
        the tuple (images, current_vars).  Both are themselves tuples.
    """
    logger = LoggerWrapper(logger)
    logger.debug('image %d: BuildStamps nobjects = %d: obj = %d',
                 config.get('image_num',0),nobjects,obj_num)

    if nobjects == 0:
        logger.error("No stamps were built, since nstamps == 0.")
        return (), ()

    # Figure out how many processes we will use for building the stamps:
    if nobjects > 1 and 'image' in config and 'nproc' in config['image']:
        nproc = ParseValue(config['image'], 'nproc', config, int)[0]
        # Update this in case the config value is -1
        nproc = UpdateNProc(nproc, nobjects, config, logger)
    else:
        nproc = 1

    jobs = []
    for k in range(nobjects):
        kwargs = {
            'obj_num' : obj_num + k,
            'xsize' : xsize,
            'ysize' : ysize,
            'do_noise' : do_noise,
        }
        jobs.append(kwargs)

    def done_func(logger, proc, k, result, t):
        if result[0] is not None:
            # Note: numpy shape is y,x
            image = result[0]
            ys, xs = image.array.shape
            if proc is None: s0 = ''
            else: s0 = '%s: '%proc
            obj_num = jobs[k]['obj_num']
            logger.info(s0 + 'Stamp %d: size = %d x %d, time = %f sec', obj_num, xs, ys, t)

    def except_func(logger, proc, k, e, tr):
        if proc is None: s0 = ''
        else: s0 = '%s: '%proc
        obj_num = jobs[k]['obj_num']
        logger.error(s0 + 'Exception caught when building stamp %d', obj_num)
        logger.debug('%s',tr)
        logger.error('Aborting the rest of this image')

    # Convert to the tasks structure we need for MultiProcess.
    # Each task is a list of (job, k) tuples.
    tasks = MakeStampTasks(config, jobs, logger)

    results = MultiProcess(nproc, config, BuildStamp, tasks, 'stamp', logger,
                           done_func = done_func, except_func = except_func)

    images, current_vars = zip(*results)

    logger.debug('image %d: Done making stamps',config.get('image_num',0))
    if all(im is None for im in images):
        logger.error('No stamps were built.  All objects were skipped.')

    return images, current_vars

# A list of keys that really belong in stamp, but are allowed in image both for convenience
# and backwards-compatibility reasons.  Any of these present will be copied over to
# config['stamp'] if they exist in config['image'].
stamp_image_keys = ['offset', 'retry_failures', 'gsparams', 'draw_method',
                    'n_photons', 'max_extra_noise', 'poisson_flux', 'obj_rng']

def SetupConfigObjNum(config, obj_num, logger=None):
    """Do the basic setup of the config dict at the stamp (or object) processing level.

    Includes:

    - Set config['obj_num'] = obj_num
    - Set config['index_key'] = 'obj_num'
    - Make sure config['stamp'] exists
    - Set default config['stamp']['type'] to 'Basic'
    - Copy over values from config['image'] that are allowed there, but really belong
      in config['stamp'].
    - Set config['stamp']['draw_method'] to 'auto' if not given.

    Parameters:
        config:         A configuration dict.
        obj_num:        The current obj_num.
        logger:         If given, a logger object to log progress. [default: None]
    """
    logger = LoggerWrapper(logger)
    config['obj_num'] = obj_num
    config['index_key'] = 'obj_num'

    # Make config['stamp'] exist if it doesn't yet.
    if 'stamp' not in config:
        config['stamp'] = {}
    stamp = config['stamp']
    if '_done' in stamp: return

    # Everything after here only needs to be done once.

    if not isinstance(stamp, dict):
        raise GalSimConfigError("config.stamp is not a dict.")
    if 'type' not in stamp:
        stamp['type'] = 'Basic'

    if 'file_num' not in config:
        config['file_num'] = 0
    if 'image_num' not in config:
        config['image_num'] = 0

    # Copy over some things from config['image'] if they are given there.
    # These are things that we used to advertise as being in the image field, but now that
    # we have a stamp field, they really make more sense here.  But for backwards compatibility,
    # or just because they can make sense in either place, we allow them to be in 'image' still.
    if 'image' in config:
        image = config['image']
        for key in stamp_image_keys:
            if key in image and key not in stamp:
                stamp[key] = image[key]
    else:
        config['image'] = {}

    if 'draw_method' not in stamp:
        stamp['draw_method'] = 'auto'

    # In case this hasn't been done yet.
    SetupInput(config, logger)

    # Mark this as done, so later passes can skip a lot of this.
    stamp['_done'] = True

def SetupConfigStampSize(config, xsize, ysize, image_pos, world_pos, logger=None):
    """Do further setup of the config dict at the stamp (or object) processing level reflecting
    the stamp size and position in either image or world coordinates.

    Note: This is now a StampBuilder method.  So this function just calls StampBuilder.locateStamp.

    Parameters:
        config:         A configuration dict.
        xsize:          The size of the stamp in the x-dimension. [may be 0 if unknown]
        ysize:          The size of the stamp in the y-dimension. [may be 0 if unknown]
        image_pos:      The position of the stamp in image coordinates. [may be None]
        world_pos:      The position of the stamp in world coordinates. [may be None]
        logger:         If given, a logger object to log progress. [default: None]
    """
    stamp = config.get('stamp',{})
    stamp_type = stamp.get('type','Basic')
    builder = valid_stamp_types[stamp_type]
    builder.locateStamp(stamp, config, xsize, ysize, image_pos, world_pos, logger)


# Ignore these when parsing the parameters for specific stamp types:
stamp_ignore = ['xsize', 'ysize', 'size', 'image_pos', 'world_pos', 'sky_pos',
                'offset', 'retry_failures', 'gsparams', 'draw_method',
                'n_photons', 'max_extra_noise', 'poisson_flux', 'photon_ops',
                'skip', 'reject', 'min_flux_frac', 'min_snr', 'max_snr',
                'quick_skip', 'obj_rng', 'index_key', 'rng_index_key', 'rng_num']

valid_draw_methods = ('auto', 'fft', 'phot', 'real_space', 'no_pixel', 'sb')

def BuildStamp(config, obj_num=0, xsize=0, ysize=0, do_noise=True, logger=None):
    """
    Build a single stamp image using the given config file

    Parameters:
        config:         A configuration dict.
        obj_num:        If given, the current obj_num [default: 0]
        xsize:          The xsize of the stamp to build (if known). [default: 0]
        ysize:          The ysize of the stamp to build (if known). [default: 0]
        do_noise:       Whether to add noise to the image (according to config['noise']).
                        [default: True]
        logger:         If given, a logger object to log progress. [default: None]

    Returns:
        the tuple (image, current_var)
    """
    logger = LoggerWrapper(logger)
    SetupConfigObjNum(config, obj_num, logger)

    stamp = config['stamp']
    stamp_type = stamp['type']
    if stamp_type not in valid_stamp_types:
        raise GalSimConfigValueError("Invalid stamp.type.", stamp_type,
                                     list(valid_stamp_types.keys()))
    builder = valid_stamp_types[stamp_type]

    if builder.quickSkip(stamp, config):
        ProcessExtraOutputsForStamp(config, True, logger)
        return None, 0

    builder.setupRNG(stamp, config, logger)

    if 'retry_failures' in stamp:
        ntries = ParseValue(stamp,'retry_failures',config,int)[0]
        # This is how many _re_-tries.  Do at least 1, so ntries is 1 more than this.
        ntries = ntries + 1
    elif ('reject' in stamp or 'min_flux_frac' in stamp or
          'min_snr' in stamp or 'max_snr' in stamp):
        # Still impose a maximum number of tries to prevent infinite loops.
        ntries = 20
    else:
        ntries = 1

    itry = 0
    while True:
        itry += 1  # itry increases from 1..ntries at which point we just reraise the exception.

        # The rest of the stamp generation stage is wrapped in a try/except block.
        # If we catch an exception, we continue the for loop to try again.
        # On the last time through, we reraise any exception caught.
        # If no exception is thrown, we simply break the loop and return.
        try:

            # Do the necessary initial setup for this stamp type.
            xsize, ysize, image_pos, world_pos = builder.setup(
                    stamp, config, xsize, ysize, stamp_ignore, logger)

            # Determine the stamp size and location
            builder.locateStamp(stamp, config, xsize, ysize, image_pos, world_pos, logger)

            # Get the global gsparams kwargs.  Individual objects can add to this.
            gsparams = {}
            if 'gsparams' in stamp:
                gsparams = UpdateGSParams(gsparams, stamp['gsparams'], config)

            # Note: Skip is different from Reject.
            #       Skip means we return None for this stamp image and continue on.
            #       Reject means we retry this object using the same obj_num.
            #       This has implications for the total number of objects as well as
            #       things like ring tests that rely on objects being made in pairs.
            #
            #       Skip is also different from prof = None.
            #       If prof is None, then the user indicated that no object should be
            #       drawn on this stamp, but that a noise image is still desired.
            if builder.getSkip(stamp, config, logger):
                raise SkipThisObject('')

            # Build the object to draw
            psf = builder.buildPSF(stamp, config, gsparams, logger)
            prof = builder.buildProfile(stamp, config, psf, gsparams, logger)

            # Make an empty image
            im = builder.makeStamp(stamp, config, xsize, ysize, logger)

            # Determine which draw method to use
            method = builder.getDrawMethod(stamp, config, logger)

            # Determine the net offset (starting from config['stamp_offset'] typically.
            offset = builder.getOffset(stamp, config, logger)

            # At this point, we may want to update whether the object should be skipped
            # based on other information about the profile and location.
            if builder.updateSkip(prof, im, method, offset, stamp, config, logger):
                raise SkipThisObject('')

            # Draw the object on the postage stamp
            im = builder.draw(prof, im, method, offset, stamp, config, logger)
            # Store the final version of the current profile for reference.
            config['current_prof'] = prof

            # Update the drawn image according to the SNR if desired.
            scale_factor = builder.getSNRScale(im, stamp, config, logger)
            im, prof = builder.applySNRScale(im, prof, scale_factor, method, logger)

            # Set the origin appropriately
            builder.updateOrigin(stamp, config, im)

            # Store the current stamp in the base-level config for reference
            config['current_stamp'] = im
            # This is also information that the weight image calculation needs
            config['do_noise_in_stamps'] = do_noise

            # Check if this object should be rejected.
            reject = builder.reject(stamp, config, prof, psf, im, logger)
            if reject:
                if itry < ntries:
                    logger.warning('Object %d: Rejecting this object and rebuilding', obj_num)
                    builder.reset(config, logger)
                    continue
                else:
                    raise GalSimConfigError(
                            "Rejected an object %d times. If this is expected, "
                            "you should specify a larger stamp.retry_failures."%(ntries))

            ProcessExtraOutputsForStamp(config, False, logger)

            # We always need to do the whiten step here in the stamp processing
            current_var = builder.whiten(prof, im, stamp, config, logger)
            if current_var != 0.:
                logger.debug('obj %d: whitening noise brought current var to %f',
                                config.get('obj_num',0),current_var)

            # Sometimes, depending on the image type, we go on to do the rest of the noise as well.
            if do_noise:
                im, current_var = builder.addNoise(stamp,config,im,current_var,logger)

        except SkipThisObject as e:
            if e.msg != '':
                logger.debug('obj %d: Caught SkipThisObject: e = %s',obj_num,e.msg)
            logger.info('Skipping object %d %s', obj_num, e.msg)
            # If xsize, ysize != 0, then this makes a blank stamp for this object.
            # Otherwise, it's just None here.
            im = builder.makeStamp(stamp, config, xsize, ysize, logger)
            ProcessExtraOutputsForStamp(config, True, logger)
            return im, 0.

        except Exception as e:
            if itry >= ntries:
                # Then this was the last try.  Just re-raise the exception.
                logger.info('Object %d: Caught exception %s',obj_num,str(e))
                if ntries > 1:
                    logger.error(
                        'Object %d: Too many exceptions/rejections for this object. Aborting.',
                        obj_num)
                raise
            else:
                logger.info('Object %d: Caught exception %s',obj_num,str(e))
                logger.info('This is try %d/%d, so trying again.',itry,ntries)
                import traceback
                tr = traceback.format_exc()
                logger.debug('obj %d: Traceback = %s',obj_num,tr)
                # Need to remove the "current"s from the config dict.  Otherwise,
                # the value generators will do a quick return with the cached value.
                builder.reset(config, logger)
                continue

        else:
            # No exception.
            return im, current_var


def MakeStampTasks(config, jobs, logger):
    """Turn a list of jobs into a list of tasks.

    See the doc string for galsim.config.MultiProcess for the meaning of this distinction.

    For the Basic stamp type, there is just one job per task, so the tasks list is just::

        tasks = [ [ (job, k) ] for k, job in enumerate(jobs) ]

    But other stamp types may need groups of jobs to be done sequentially by the same process.
    cf. stamp type=Ring.

    Parameters:
        config:         The configuration dict
        jobs:           A list of jobs to split up into tasks.  Each job in the list is a
                        dict of parameters that includes 'obj_num'.
        logger:         A logger object to log progress.

    Returns:
        a list of tasks
    """
    stamp = config.get('stamp', {})
    stamp_type = stamp.get('type', 'Basic')
    return valid_stamp_types[stamp_type].makeTasks(stamp, config, jobs, logger)


def DrawBasic(prof, image, method, offset, config, base, logger, **kwargs):
    """The basic implementation of the draw command

    This function is provided as a free function, rather than just the base class implementation
    in StampBuilder to make it easier for classes derived from StampBuilder to use to help
    implement their draw functions.  The base class, StampBuilder, just calls this function
    for its draw method.

    This version also allows for additional kwargs, which are passed on to the drawImage function.
    e.g. you can add add_to_image=True or setup_only=True if these are helpful.

    Parameters:
        prof:       The profile to draw.
        image:      The image onto which to draw the profile (which may be None).
        method:     The method to use in drawImage.
        offset:     The offset to apply when drawing.
        config:     The configuration dict for the stamp field.
        base:       The base configuration dict.
        logger:     A logger object to log progress.
        **kwargs:   Any additional kwargs are passed along to the drawImage function.

    Returns:
        the resulting image
    """
    logger = LoggerWrapper(logger)
    # Setup the kwargs to pass to drawImage
    # (Start with any additional kwargs given as extra kwargs to DrawBasic and add to it.)
    kwargs['image'] = image
    kwargs['offset'] = offset
    kwargs['method'] = method
    if 'wcs' not in kwargs and 'scale' not in kwargs:
        kwargs['wcs'] = base['wcs'].local(image_pos = base['image_pos'])
    sensor = base.get('sensor',None)
    if (method == 'phot' or sensor is not None) and 'rng' not in kwargs:
        # Note: use the image.noise rng_num, in case stamp is doing a non-standard cadence.
        # The cadence specified in the noise field is what to use for photon shooting.
        noise = base.get('image',{}).get('noise',{})
        noise = noise if isinstance(noise, dict) else {}
        rng = GetRNG(noise, base, logger, "method='phot'")
        kwargs['rng'] = rng

    # Check validity of extra phot options:
    max_extra_noise = None
    if 'n_photons' in config and 'n_photons' not in kwargs:
        if method != 'phot':
            raise GalSimConfigError('n_photons is invalid with method != phot')
        if 'max_extra_noise' in config:
            logger.warning(
                "Both 'max_extra_noise' and 'n_photons' are set in config dict, "
                "ignoring 'max_extra_noise'.")
        kwargs['n_photons'] = ParseValue(config, 'n_photons', base, int)[0]
    elif 'max_extra_noise' in config:
        max_extra_noise = ParseValue(config, 'max_extra_noise', base, float)[0]
        if method != 'phot' and max_extra_noise is not None:
            raise GalSimConfigError('max_extra_noise is invalid with method != phot')

    if 'poisson_flux' in config and 'poisson_flux' not in kwargs:
        if method != 'phot':
            raise GalSimConfigError('poisson_flux is invalid with method != phot')
        kwargs['poisson_flux'] = ParseValue(config, 'poisson_flux', base, bool)[0]

    if max_extra_noise is not None and 'max_extra_noise' not in kwargs:
        if max_extra_noise < 0.:
            raise GalSimConfigError("image.max_extra_noise cannot be negative")
        if 'image' in base and 'noise' in base['image']:
            noise_var = CalculateNoiseVariance(base)
        else:
            raise GalSimConfigError("Need to specify noise level when using max_extra_noise")
        if noise_var < 0.:
            raise GalSimConfigError("noise_var calculated to be < 0.")
        max_extra_noise *= noise_var
        kwargs['max_extra_noise'] = max_extra_noise

    bandpass = base.get('bandpass', None)
    if bandpass is not None:
        kwargs['bandpass'] = bandpass
    elif isinstance(prof, ChromaticObject):
        raise GalSimConfigError("Drawing chromatic object requires specifying bandpass")

    if 'photon_ops' in config:
        kwargs['photon_ops'] = BuildPhotonOps(config, 'photon_ops', base, logger)

    if sensor is not None:
        sensor.updateRNG(rng)
        kwargs['sensor'] = sensor

    if logger.isEnabledFor(logging.DEBUG):
        # Don't output the full image array.  Use str(image) for that kwarg.  And Bandpass.
        alt_kwargs = dict([(k, str(kwargs[k]) if isinstance(kwargs[k],(Image,Bandpass))
                               else kwargs[k])
                           for k in kwargs])
        logger.debug('obj %d: drawImage kwargs = %s',base.get('obj_num',0), alt_kwargs)
        logger.debug('obj %d: prof = %s',base.get('obj_num',0),prof)
    try:
        image = prof.drawImage(**kwargs)
    except Exception as e:
        logger.debug('obj %d: prof = %r', base.get('obj_num',0), prof)
        raise
    logger.debug('obj %d: added_flux = %s',base.get('obj_num',0), image.added_flux)
    return image

def ParseWorldPos(config, param_name, base, logger):
    """A helper function to parse the 'world_pos' value.

    The world_pos can be specified either as a regular RA, Dec (which in GalSim is known as a
    CelestialCoord) or as Euclidean coordinates in the local tangent plane relative to the
    image center (a PositionD).

    1. For the RA/Dec option, the world_pos field should use the type RADec, which includes two
       values named ra and dec, each of which should be an Angle type.  e.g.::

            world_pos:
                type : RADec
                ra : 37 hours
                dec: -23 degrees

       Technically, any other type that results in a CelestialCoord is valid, but RADec is the
       only one that is defined natively in GalSim.

    2. For the relative position in the local tangent plane (where 0,0 is the position of the
       image center), you can use any PositionD type.  e.g.::

            world_pos:
                type : RandomCircle
                radius : 12       # arcsec
                inner_radius : 3  # arcsec

    Parameters:
        config:     The configuration dict for the stamp field.
        param_name: The name of the field in the config dict to parse as a world_pos.
                    Normally, this is just 'world_pos'.
        base:       The base configuration dict.

    Returns:
        either a CelestialCoord or a PositionD instance.
    """
    param = config[param_name]
    wcs = base.get('wcs', PixelScale(1.0)) # should be here, but just in case...
    if wcs._isCelestial:
        return ParseValue(config, param_name, base, CelestialCoord)[0]
    else:
        return ParseValue(config, param_name, base, PositionD)[0]

class StampBuilder(object):
    """A base class for building stamp images of individual objects.

    The base class defines the call signatures of the methods that any derived class should follow.
    It also includes the implementation of the default stamp type: Basic.
    """

    def setup(self, config, base, xsize, ysize, ignore, logger):
        """
        Do the initialization and setup for building a postage stamp.

        In the base class, we check for and parse the appropriate size and position values in
        config (aka base['stamp'] or base['image'].

        Values given in base['stamp'] take precedence if these are given in both places (which
        would be confusing, so probably shouldn't do that, but there might be a use case where it
        would make sense).

        Parameters:
            config:     The configuration dict for the stamp field.
            base:       The base configuration dict.
            xsize:      The xsize of the image to build (if known).
            ysize:      The ysize of the image to build (if known).
            ignore:     A list of parameters that are allowed to be in config that we can
                        ignore here. i.e. it won't be an error if these parameters are present.
            logger:     A logger object to log progress.

        Returns:
            xsize, ysize, image_pos, world_pos
        """
        # Check for spurious parameters
        CheckAllParams(config, ignore=ignore)

        # Update the size if necessary
        image = base['image']
        if not xsize:
            if 'xsize' in config:
                xsize = ParseValue(config,'xsize',base,int)[0]
            elif 'size' in config:
                xsize = ParseValue(config,'size',base,int)[0]
            elif 'stamp_xsize' in image:
                xsize = ParseValue(image,'stamp_xsize',base,int)[0]
            elif 'stamp_size' in image:
                xsize = ParseValue(image,'stamp_size',base,int)[0]

        if not ysize:
            if 'ysize' in config:
                ysize = ParseValue(config,'ysize',base,int)[0]
            elif 'size' in config:
                ysize = ParseValue(config,'size',base,int)[0]
            elif 'stamp_ysize' in image:
                ysize = ParseValue(image,'stamp_ysize',base,int)[0]
            elif 'stamp_size' in image:
                ysize = ParseValue(image,'stamp_size',base,int)[0]

        # Determine where this object is going to go:
        if 'image_pos' in config:
            image_pos = ParseValue(config, 'image_pos', base, PositionD)[0]
        elif 'image_pos' in image:
            image_pos = ParseValue(image, 'image_pos', base, PositionD)[0]
        else:
            image_pos = None

        if 'world_pos' in config:
            world_pos = ParseWorldPos(config, 'world_pos', base, logger)
        elif 'world_pos' in image:
            world_pos = ParseWorldPos(image, 'world_pos', base, logger)
        else:
            world_pos = None

        return xsize, ysize, image_pos, world_pos

    def quickSkip(self, config, base):
        """Check whether this object should be skipped before doing any work.

        The base class looks for stamp.quick_skip and returns True if it is preset and
        evaluates to True.

        @param config       The configuration dict for the stamp field.
        @param base         The base configuration dict.

        @returns skip
        """
        return ('quick_skip' in config and ParseValue(config, 'quick_skip', base, bool)[0])

    def setupRNG(self, config, base, logger):
        """Setup the RNG for this object.

        @param config       The configuration dict for the stamp field.
        @param base         The base configuration dict.
        """
        if 'obj_rng' in config:
            if not ParseValue(config,'obj_rng',base,bool)[0]:
                # Just use the image_num rng(s).
                base['obj_num_rngs'] = base.get('image_num_rngs',None)
                base['obj_num_rng'] = base.get('image_num_rng',None)
                return

        # Add 1 to the seed here so the first object has a different rng than the file or image.
        seed = SetupConfigRNG(base, seed_offset=1, logger=logger)
        logger.debug('obj %d: seed = %d',base.get('obj_num',0),seed)

    def locateStamp(self, config, base, xsize, ysize, image_pos, world_pos, logger):
        """Determine where and how large the stamp should be.

        The base class version does the followin:

        - If given, set base['stamp_xsize'] = xsize
        - If given, set base['stamp_ysize'] = ysize
        - If only image_pos or world_pos is given, compute the other from base['wcs']
        - Set base['index_pos'] = image_pos
        - Set base['world_pos'] = world_pos
        - Calculate the appropriate value of the center of the stamp, to be used with the
          command: stamp_image.setCenter(stamp_center).  Save this as base['stamp_center']
        - Calculate the appropriate offset for the position of the object from the center of
          the stamp due to just the fractional part of the image position, not including
          any base['stamp']['offset'] item that may be present in the base dict.
          Save this as base['stamp_offset']

        @param config       The configuration dict for the stamp field.
        @param base         The base configuration dict.
        @param xsize        The size of the stamp in the x-dimension. [may be 0 if unknown]
        @param ysize        The size of the stamp in the y-dimension. [may be 0 if unknown]
        @param image_pos    The position of the stamp in image coordinates. [may be None]
        @param world_pos    The position of the stamp in world coordinates. [may be None]
        @param logger       A logger object to log progress.
        """
        logger = LoggerWrapper(logger)

        # Make sure we have a valid wcs in case image-level processing was skipped.
        if 'wcs' not in base:
            base['wcs'] = BuildWCS(base['image'], 'wcs', config, logger)
        wcs = base['wcs']

        if xsize: base['stamp_xsize'] = xsize
        if ysize: base['stamp_ysize'] = ysize

        # If we have either image_pos or world_pos, calculate the other.
        if image_pos is not None and world_pos is None:
            world_pos = wcs.toWorld(image_pos)
        elif world_pos is not None and image_pos is None:
            image_pos = wcs.toImage(world_pos)

        # If the world_pos is a CelestialCoord, then we also call it sky_pos.
        # If the world_pos is not celestial, or the user wants to override it for some reason,
        # then the user may optionally define a sky_pos value, which gets saved as base['sky_pos'].
        # This may be useful for things that need to know where in the sky the pointing is,
        # even if the WCS is not a CelestialWCS.
        if 'sky_pos' in config:
            base['sky_pos'] = ParseValue(config, 'sky_pos', base, CelestialCoord)[0]
        elif isinstance(world_pos, CelestialCoord):
            base['sky_pos'] = world_pos

        # Sometimes we need a "world" position as flat position, rather than a CelestialCoord.
        # This is called uv_pos.  If the WCS is a EuclideanWCS, then uv_pos = world_pos.
        # If the WCS is a CelestialWCS, then sky_pos = world_pos, and we calculate uv_pos as
        # the tangent-plane projection.
        if isinstance(world_pos, CelestialCoord):
            # Then project this position relative to the image center.
            world_center = base.get('world_center', wcs.toWorld(base['image_center']))
            u, v = world_center.project(world_pos, projection='gnomonic')
            base['uv_pos'] = PositionD(u/arcsec, v/arcsec)
        else:
            base['uv_pos'] = world_pos

        if image_pos is not None:
            # The image_pos refers to the location of the true center of the image, which is
            # not necessarily the nominal center we need for adding to the final image.  In
            # particular, even-sized images have their nominal center offset by 1/2 pixel up
            # and to the right.
            # N.B. This works even if xsize,ysize == 0, since the auto-sizing always produces
            # even sized images.
            nominal_x = image_pos.x        # Make sure we don't change image_pos, which is
            nominal_y = image_pos.y        # stored in base['image_pos'].
            if xsize % 2 == 0: nominal_x += 0.5
            if ysize % 2 == 0: nominal_y += 0.5

            stamp_center = PositionI(int(math.floor(nominal_x+0.5)),
                                     int(math.floor(nominal_y+0.5)))
            stamp_offset = PositionD(nominal_x-stamp_center.x,
                                     nominal_y-stamp_center.y)
        else:
            stamp_center = None
            stamp_offset = PositionD(0.,0.)
            # Set the image_pos to the image center in case the wcs needs it.  Probably, if
            # there is no image_pos or world_pos defined, then it is unlikely a
            # non-trivial wcs will have been set.  So anything would actually be fine.
            image_pos = PositionD( (xsize+1.)/2, (ysize+1.)/2 )

        base['stamp_center'] = stamp_center
        base['stamp_offset'] = stamp_offset
        base['image_pos'] = image_pos
        base['world_pos'] = world_pos

        if xsize:
            logger.debug('obj %d: xsize,ysize = %s,%s',base.get('obj_num',0),xsize,ysize)
        logger.debug('obj %d: image_pos = %s',base.get('obj_num',0),image_pos)
        if world_pos:
            logger.debug('obj %d: world_pos = %s',base.get('obj_num',0),world_pos)
        if stamp_center:
            logger.debug('obj %d: stamp_center = %s',base.get('obj_num',0),stamp_center)

    def getSkip(self, config, base, logger):
        """Initial check of whether to skip this object based on the stamp.skip field.

        @param config       The configuration dict for the stamp field.
        @param base         The base configuration dict.
        @param logger       A logger object to log progress.

        @returns skip
        """
        if 'skip' in config:
            skip = ParseValue(config, 'skip', base, bool)[0]
            if skip:
                logger.debug("obj %d: stamp.skip = True",base.get('obj_num',0))
        else:
            skip = False
        return skip

    def buildPSF(self, config, base, gsparams, logger):
        """Build the PSF object.

        For the Basic stamp type, this builds a PSF from the base['psf'] dict, if present,
        else returns None.

        Parameters:
            config:     The configuration dict for the stamp field.
            base:       The base configuration dict.
            gsparams:   A dict of kwargs to use for a GSParams.  More may be added to this
                        list by the galaxy object.
            logger:     A logger object to log progress.

        Returns:
            the PSF
        """
        return BuildGSObject(base, 'psf', gsparams=gsparams, logger=logger)[0]

    def buildProfile(self, config, base, psf, gsparams, logger):
        """Build the surface brightness profile (a GSObject) to be drawn.

        For the Basic stamp type, this builds a galaxy from the base['gal'] dict and convolves
        it with the psf (if given).  If either the psf or the galaxy is None, then the other one
        is returned as is.

        Parameters:
            config:     The configuration dict for the stamp field.
            base:       The base configuration dict.
            psf:        The PSF, if any.  This may be None, in which case, no PSF is convolved.
            gsparams:   A dict of kwargs to use for a GSParams.  More may be added to this
                        list by the galaxy object.
            logger:     A logger object to log progress.

        Returns:
            the final profile
        """
        gal = BuildGSObject(base, 'gal', gsparams=gsparams, logger=logger)[0]

        if psf:
            if gal:
                return Convolve(gal,psf)
            else:
                return psf
        else:
            if gal:
                return gal
            elif 'gal' in base or 'psf' in base:
                return None
            else:
                raise GalSimConfigError(
                    "At least one of gal or psf must be specified in config. "
                    "If you really don't want any object, use gal type = None.")

    def makeStamp(self, config, base, xsize, ysize, logger):
        """Make the initial empty postage stamp image, if possible.

        If we don't know xsize, ysize, return None, in which case the stamp will be created
        automatically by the drawImage command based on the natural size of the profile.

        Parameters:
            config:     The configuration dict for the stamp field.
            base:       The base configuration dict.
            xsize:      The xsize of the image to build (if known).
            ysize:      The ysize of the image to build (if known).
            logger:     A logger object to log progress.

        Returns:
            the image
        """
        if xsize and ysize:
            im = ImageF(xsize, ysize)
            im.setZero()
            return im
        else:
            return None

    def getDrawMethod(self, config, base, logger):
        """Determine the draw method to use.

        @param config       The configuration dict for the stamp field.
        @param base         The base configuration dict.
        @param logger       A logger object to log progress.

        @returns method
        """
        method = ParseValue(config,'draw_method',base,str)[0]
        if method not in valid_draw_methods:
            raise GalSimConfigValueError("Invalid draw_method.", method, valid_draw_methods)
        return method

    def getOffset(self, config, base, logger):
        """Determine the offset to use.

        The base class version adds the stamp_offset, which comes from calculations related to
        world_pos and image_pos, to the field stamp.offset if any.

        @param config       The configuration dict for the stamp field.
        @param base         The base configuration dict.
        @param logger       A logger object to log progress.

        @returns offset
        """
        offset = base['stamp_offset']
        if 'offset' in config:
            offset += ParseValue(config, 'offset', base, PositionD)[0]
        logger.debug('obj %d: stamp_offset = %s, offset = %s',base.get('obj_num',0),
                     base['stamp_offset'], offset)
        return offset

    def updateSkip(self, prof, image, method, offset, config, base, logger):
        """Before drawing the profile, see whether this object can be trivially skipped.

        The base method checks if the object is completely off the main image, so the
        intersection bounds will be undefined.  In this case, don't bother drawing the
        postage stamp for this object.

        Parameters:
            prof:       The profile to draw.
            image:      The image onto which to draw the profile (which may be None).
            method:     The method to use in drawImage.
            offset:     The offset to apply when drawing.
            config:     The configuration dict for the stamp field.
            base:       The base configuration dict.
            logger:     A logger object to log progress.

        Returns:
            whether to skip drawing this object.
        """
        if isinstance(prof,GSObject) and base.get('current_image',None) is not None:
            if image is None:
                prof = base['wcs'].toImage(prof, image_pos=base['image_pos'])
                N = prof.getGoodImageSize(1.)
                N += 2 + int(np.abs(offset.x) + np.abs(offset.y))
                bounds = _BoundsI(1,N,1,N)
            else:
                bounds = image.bounds

            # Set the origin appropriately
            stamp_center = base['stamp_center']
            if stamp_center:
                bounds = bounds.shift(stamp_center - bounds.center)
            else:
                bounds = bounds.shift(base.get('image_origin',PositionI(1,1)) -
                                      PositionI(bounds.xmin, bounds.ymin))

            overlap = bounds & base['current_image'].bounds
            if not overlap.isDefined():
                logger.info('obj %d: skip drawing object because its image will be entirely off '
                            'the main image.', base.get('obj_num',0))
                return True

        return False

    def draw(self, prof, image, method, offset, config, base, logger):
        """Draw the profile on the postage stamp image.

        Parameters:
            prof:       The profile to draw.
            image:      The image onto which to draw the profile (which may be None).
            method:     The method to use in drawImage.
            offset:     The offset to apply when drawing.
            config:     The configuration dict for the stamp field.
            base:       The base configuration dict.
            logger:     A logger object to log progress.

        Returns:
            the resulting image
        """
        if prof is None:
            return image
        else:
            return DrawBasic(prof,image,method,offset,config,base,logger)

    def whiten(self, prof, image, config, base, logger):
        """If appropriate, whiten the resulting image according to the requested noise profile
        and the amount of noise originally present in the profile.

        Parameters:
            prof:       The profile to draw.
            image:      The image onto which to draw the profile.
            config:     The configuration dict for the stamp field.
            base:       The base configuration dict.
            logger:     A logger object to log progress.

        Returns:
            the variance of the resulting whitened (or symmetrized) image.
        """
        # If the object has a noise attribute, then check if we need to do anything with it.
        current_var = 0.  # Default if not overwritten
        if isinstance(prof,GSObject) and prof.noise is not None:
            if 'image' in base and 'noise' in base['image']:
                noise = base['image']['noise']
                whiten = symmetrize = False
                if 'whiten' in noise:
                    whiten = ParseValue(noise, 'whiten', base, bool)[0]
                if 'symmetrize' in noise:
                    symmetrize = ParseValue(noise, 'symmetrize', base, int)[0]
                if whiten and symmetrize:
                    raise GalSimConfigError('Only one of whiten or symmetrize is allowed')
                if whiten or symmetrize:
                    # In case the galaxy was cached, update the rng
                    rng = GetRNG(noise, base, logger, "whiten")
                    prof.noise.rng.reset(rng)
                if whiten:
                    current_var = prof.noise.whitenImage(image)
                if symmetrize:
                    current_var = prof.noise.symmetrizeImage(image, symmetrize)
        return current_var

    def getSNRScale(self, image, config, base, logger):
        """Calculate the factor by which to rescale the image based on a desired S/N level.

        Note: The default implementation does this for the gal or psf field, so if a custom
              stamp builder uses some other way to get the profiles, this method should
              probably be overridden.

        Parameters:
            image:      The current image.
            config:     The configuration dict for the stamp field.
            base:       The base configuration dict.
            logger:     A logger object to log progress.

        Returns:
            scale_factor
        """
        if 'gal' in base and 'signal_to_noise' in base['gal']:
            key = 'gal'
        elif 'gal' not in base and 'psf' in base and 'signal_to_noise' in base['psf']:
            key = 'psf'
        else:
            return 1.

        if 'flux' in base[key]:
            raise GalSimConfigError(
                'Only one of signal_to_noise or flux may be specified for %s'%key)

        if 'image' in base and 'noise' in base['image']:
            noise_var = CalculateNoiseVariance(base)
        else:
            raise GalSimConfigError(
                "Need to specify noise level when using %s.signal_to_noise"%key)
        sn_target = ParseValue(base[key], 'signal_to_noise', base, float)[0]

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

        sn_meas = math.sqrt( np.sum(image.array**2, dtype=float) / noise_var )
        # Now we rescale the flux to get our desired S/N
        scale_factor = sn_target / sn_meas
        return scale_factor

    def applySNRScale(self, image, prof, scale_factor, method, logger):
        """Apply the scale_factor from getSNRScale to the image and profile.

        The default implementaion just multiplies each of them, but if prof is not a regular
        GSObject, then you might need to do something different.

        Parameters:
            image:          The current image.
            prof:           The profile that was drawn.
            scale_factor:   The factor by which to scale both image and prof.
            method:         The method used by drawImage.
            logger:         A logger object to log progress.

        Returns:
            image, prof  (after being properly scaled)
        """
        if scale_factor != 1.0:
            if method == 'phot':
                logger.warning(
                    "signal_to_noise calculation is not accurate for draw_method = phot")
            image *= scale_factor
            prof *= scale_factor
        return image, prof

    def updateOrigin(self, config, base, image):
        """Update the image origin to be centered at the right place

        @param config       The configuration dict for the stamp field.
        @param base         The base configuration dict.
        @param image        The postage stamp image.
        """
        stamp_center = base['stamp_center']
        if stamp_center:
            image.setCenter(stamp_center)
        else:
            image.setOrigin(base.get('image_origin',PositionI(1,1)))


    def reject(self, config, base, prof, psf, image, logger):
        """Check to see if this object should be rejected.

        Parameters:
            config:     The configuration dict for the stamp field.
            base:       The base configuration dict.
            prof:       The profile that was drawn.
            psf:        The psf that was used to build the profile.
            image:      The postage stamp image.  No noise is on it yet at this point.
            logger:     A logger object to log progress.

        Returns:
            whether to reject this object
        """
        # Early exit if no profile
        if prof is None:
            return False

        if 'reject' in config:
            if ParseValue(config, 'reject', base, bool)[0]:
                logger.info('obj %d: reject evaluated to True',base.get('obj_num',0))
                return True
        if 'min_flux_frac' in config:
            if not isinstance(prof, GSObject):
                raise GalSimConfigError(
                    "Cannot apply min_flux_frac for stamp types that do not use "
                    "a single GSObject profile.")
            expected_flux = prof.flux
            measured_flux = np.sum(image.array, dtype=float)
            min_flux_frac = ParseValue(config, 'min_flux_frac', base, float)[0]
            logger.debug('obj %d: flux_frac = %f', base.get('obj_num',0),
                         measured_flux / expected_flux)
            if measured_flux < min_flux_frac * expected_flux:
                logger.warning('Object %d: Measured flux = %f < %s * %f.',
                               base.get('obj_num',0), measured_flux, min_flux_frac, expected_flux)
                return True
        if 'min_snr' in config or 'max_snr' in config:
            if not isinstance(prof, GSObject):
                raise GalSimConfigError(
                    "Cannot apply min_snr for stamp types that do not use "
                    "a single GSObject profile.")
            var = CalculateNoiseVariance(base)
            sumsq = np.sum(image.array**2, dtype=float)
            snr = np.sqrt(sumsq / var)
            logger.debug('obj %d: snr = %f', base.get('obj_num',0), snr)
            if 'min_snr' in config:
                min_snr = ParseValue(config, 'min_snr', base, float)[0]
                if snr < min_snr:
                    logger.warning('Object %d: Measured snr = %f < %s.',
                                   base.get('obj_num',0), snr, min_snr)
                    return True
            if 'max_snr' in config:
                max_snr = ParseValue(config, 'max_snr', base, float)[0]
                if snr > max_snr:
                    logger.warning('Object %d: Measured snr = %f > %s.',
                                   base.get('obj_num',0), snr, max_snr)
                    return True
        return False

    def reset(self, base, logger):
        """Reset some aspects of the config dict so the object can be rebuilt after rejecting the
        current object.

        Parameters:
            base:       The base configuration dict.
            logger:     A logger object to log progress.
        """
        # Clear current values out of psf, gal, and stamp if they are not safe to reuse.
        # This means they are either marked as safe or indexed by something other than obj_num.
        for field in ('psf', 'gal', 'stamp'):
            if field in base:
                RemoveCurrent(base[field], keep_safe=True, index_key='obj_num')

    def addNoise(self, config, base, image, current_var, logger):
        """
        Add the sky level and the noise to the stamp.

        Note: This only gets called if the image type requests that the noise be added to each
            stamp individually, rather than to the full image and the end.

        Parameters:
            config:         The configuration dict for the stamp field.
            base:           The base configuration dict.
            image:          The current image.
            current_var:    The current noise variance present in the image already.
            logger:         A logger object to log progress.

        Returns:
            the new values of image, current_var
        """
        AddSky(base,image)
        base['current_noise_image'] = base['current_stamp']
        current_var = AddNoise(base,image,current_var,logger)
        return image, current_var

    def makeTasks(self, config, base, jobs, logger):
        """Turn a list of jobs into a list of tasks.

        For the Basic stamp type, there is just one job per task, so the tasks list is just:

            tasks = [ [ (job, k) ] for k, job in enumerate(jobs) ]

        Parameters:
            config:     The configuration dict for the stamp field.
            base:       The base configuration dict.
            jobs:       A list of jobs to split up into tasks.  Each job in the list is a
                        dict of parameters that includes 'obj_num'.
            logger:     A logger object to log progress.

        Returns:
            a list of tasks
        """
        return [ [(job, k)] for k, job in enumerate(jobs) ]


def RegisterStampType(stamp_type, builder):
    """Register an image type for use by the config apparatus.

    Parameters:
        stamp_type:     The name of the type in config['stamp']
        builder:        A builder object to use for building the stamp images.  It should be
                        an instance of StampBuilder or a subclass thereof.
    """
    valid_stamp_types[stamp_type] = builder

RegisterStampType('Basic', StampBuilder())
