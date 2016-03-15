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
import numpy

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

    @param nobjects         How many postage stamps to build.
    @param config           A configuration dict.
    @param obj_num          If given, the current obj_num. [default: 0]
    @param xsize            The size of a single stamp in the x direction. [default: 0,
                            which means to look for config.image.stamp_xsize, and if that's
                            not there, use automatic sizing.]
    @param ysize            The size of a single stamp in the y direction. [default: 0,
                            which means to look for config.image.stamp_xsize, and if that's
                            not there, use automatic sizing.]
    @param do_noise         Whether to add noise to the image (according to config['noise']).
                            [default: True]
    @param logger           If given, a logger object to log progress. [default: None]

    @returns the tuple (images, current_vars).  Both are lists.
    """
    if logger:
        logger.debug('image %d: BuildStamp nobjects = %d: obj = %d',
                     config.get('image_num',0),nobjects,obj_num)

    # Figure out how many processes we will use for building the stamps:
    if nobjects > 1 and 'image' in config and 'nproc' in config['image']:
        nproc = galsim.config.ParseValue(config['image'], 'nproc', config, int)[0]
        # Update this in case the config value is -1
        nproc = galsim.config.UpdateNProc(nproc, nobjects, config, logger)
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
        if logger and result[0] is not None:
            # Note: numpy shape is y,x
            image = result[0]
            ys, xs = image.array.shape
            if proc is None: s0 = ''
            else: s0 = '%s: '%proc
            obj_num = jobs[k]['obj_num']
            logger.info(s0 + 'Stamp %d: size = %d x %d, time = %f sec', obj_num, xs, ys, t)

    def except_func(logger, proc, k, e, tr):
        if logger:
            if proc is None: s0 = ''
            else: s0 = '%s: '%proc
            obj_num = jobs[k]['obj_num']
            logger.error(s0 + 'Exception caught when building stamp %d', obj_num)
            #logger.error('%s',tr)
            logger.error('Aborting the rest of this image')

    # Convert to the tasks structure we need for MultiProcess.
    # Each task is a list of (job, k) tuples.
    tasks = MakeStampTasks(config, jobs, logger)

    results = galsim.config.MultiProcess(nproc, config, BuildStamp, tasks, 'stamp', logger,
                                         done_func = done_func,
                                         except_func = except_func)

    if not results:
        images, current_vars = [], []
        if logger:
            logger.error('No images were built.  All were either skipped or had errors.')
    else:
        images, current_vars = zip(*results)
        if logger:
            logger.debug('image %d: Done making stamps',config.get('image_num',0))

    return images, current_vars


def SetupConfigObjNum(config, obj_num):
    """Do the basic setup of the config dict at the stamp (or object) processing level.

    Includes:
    - Set config['obj_num'] = obj_num
    - Set config['index_key'] = 'obj_num'
    - Make sure config['stamp'] exists
    - Set default config['stamp']['type'] to 'Basic'
    - Copy over values from config['image'] that are allowed there, but really belong
      in config['stamp'].
    - Set config['stamp']['draw_method'] to 'auto' if not given.

    @param config           A configuration dict.
    @param obj_num          The current obj_num.
    """
    config['obj_num'] = obj_num
    config['index_key'] = 'obj_num'

    # Make config['stamp'] exist if it doesn't yet.
    if 'stamp' not in config:
        config['stamp'] = {}
    stamp = config['stamp']
    if not isinstance(stamp, dict):
        raise AttributeError("config.stamp is not a dict.")
    if 'type' not in stamp:
        stamp['type'] = 'Basic'

    # Copy over some things from config['image'] if they are given there.
    # These are things that we used to advertise as being in the image field, but now that
    # we have a stamp field, they really make more sense here.  But for backwards compatibility,
    # or just because they can make sense in either place, we allow them to be in 'image' still.
    if '_copied_image_keys_to_stamp' not in config and 'image' in config:
        image = config['image']
        for key in ['offset', 'retry_failures', 'gsparams',
                    'draw_method', 'wmult', 'nphotons', 'max_extra_noise', 'poisson_flux']:
            if key in image and key not in stamp:
                stamp[key] = image[key]
        config['_copied_image_keys_to_stamp'] = True

    if 'draw_method' not in stamp:
        stamp['draw_method'] = 'auto'



def SetupConfigStampSize(config, xsize, ysize, image_pos, world_pos):
    """Do further setup of the config dict at the stamp (or object) processing level reflecting
    the stamp size and position in either image or world coordinates.

    Includes:
    - If given, set config['stamp_xsize'] = xsize
    - If given, set config['stamp_ysize'] = ysize
    - If only image_pos or world_pos is given, compute the other from config['wcs']
    - Set config['index_pos'] = image_pos
    - Set config['world_pos'] = world_pos
    - Calculate the appropriate value of the center of the stamp, to be used with the
      command: stamp_image.setCenter(stamp_center).  Save this as config['stamp_center']
    - Calculate the appropriate offset for the position of the object from the center of
      the stamp due to just the fractional part of the image position, not including
      any config['stamp']['offset'] item that may be present in the config dict.
      Save this as config['stamp_offset']

    @param config           A configuration dict.
    @param xsize            The size of the stamp in the x-dimension. [may be None]
    @param ysize            The size of the stamp in the y-dimension. [may be None]
    @param image_pos        The position of the stamp in image coordinates. [may be None]
    @param world_pos        The position of the stamp in world coordinates. [may be None]
    """

    if xsize: config['stamp_xsize'] = xsize
    if ysize: config['stamp_ysize'] = ysize
    if image_pos is not None and world_pos is None:
        # Calculate and save the position relative to the image center
        world_pos = config['wcs'].toWorld(image_pos)

        # Wherever we use the world position, we expect a Euclidean position, not a
        # CelestialCoord.  So if it is the latter, project it onto a tangent plane at the
        # image center.
        if isinstance(world_pos, galsim.CelestialCoord):
            # Then project this position relative to the image center.
            world_center = config['wcs'].toWorld(config['image_center'])
            world_pos = world_center.project(world_pos, projection='gnomonic')

    elif world_pos is not None and image_pos is None:
        # Calculate and save the position relative to the image center
        image_pos = config['wcs'].toImage(world_pos)

    if image_pos is not None:
        import math
        # The image_pos refers to the location of the true center of the image, which is
        # not necessarily the nominal center we need for adding to the final image.  In
        # particular, even-sized images have their nominal center offset by 1/2 pixel up
        # and to the right.
        # N.B. This works even if xsize,ysize == 0, since the auto-sizing always produces
        # even sized images.
        nominal_x = image_pos.x        # Make sure we don't change image_pos, which is
        nominal_y = image_pos.y        # stored in config['image_pos'].
        if xsize % 2 == 0: nominal_x += 0.5
        if ysize % 2 == 0: nominal_y += 0.5

        stamp_center = galsim.PositionI(int(math.floor(nominal_x+0.5)),
                                        int(math.floor(nominal_y+0.5)))
        config['stamp_center'] = stamp_center
        config['stamp_offset'] = galsim.PositionD(nominal_x-stamp_center.x,
                                                  nominal_y-stamp_center.y)
        config['image_pos'] = image_pos
        config['world_pos'] = world_pos

    else:
        config['stamp_center'] = None
        config['stamp_offset'] = galsim.PositionD(0.,0.)
        # Set the image_pos to (0,0) in case the wcs needs it.  Probably, if
        # there is no image_pos or world_pos defined, then it is unlikely a
        # non-trivial wcs will have been set.  So anything would actually be fine.
        config['image_pos'] = galsim.PositionD(0.,0.)
        config['world_pos'] = world_pos

# Ignore these when parsing the parameters for specific stamp types:
stamp_ignore = ['offset', 'retry_failures', 'gsparams', 'draw_method',
                'wmult', 'nphotons', 'max_extra_noise', 'poisson_flux',
                'reject', 'min_flux_frac', 'min_snr', 'max_snr']

def BuildStamp(config, obj_num=0, xsize=0, ysize=0, do_noise=True, logger=None):
    """
    Build a single stamp image using the given config file

    @param config           A configuration dict.
    @param obj_num          If given, the current obj_num [default: 0]
    @param xsize            The xsize of the image to build (if known). [default: 0]
    @param ysize            The ysize of the image to build (if known). [default: 0]
    @param do_noise         Whether to add noise to the image (according to config['noise']).
                            [default: True]
    @param logger           If given, a logger object to log progress. [default: None]

    @returns the tuple (image, current_var)
    """
    SetupConfigObjNum(config,obj_num)

    stamp = config['stamp']
    stamp_type = stamp['type']
    if stamp_type not in valid_stamp_types:
        raise AttributeErro("Invalid stamp.type=%s."%stamp_type)
    builder = valid_stamp_types[stamp_type]

    # Add 1 to the seed here so the first object has a different rng than the file or image.
    seed = galsim.config.SetupConfigRNG(config, seed_offset=1)
    if logger:
        logger.debug('obj %d: seed = %d',obj_num,seed)

    if 'retry_failures' in stamp:
        ntries = galsim.config.ParseValue(stamp,'retry_failures',config,int)[0]
        # This is how many _re_-tries.  Do at least 1, so ntries is 1 more than this.
        ntries = ntries + 1
    elif 'reject' in stamp or 'min_flux_frac' in stamp:
        # Still impose a maximum number of tries to prevent infinite loops.
        ntries = 20
    else:
        ntries = 1

    for itry in range(ntries):

        # The rest of the stamp generation stage is wrapped in a try/except block.
        # If we catch an exception, we continue the for loop to try again.
        # On the last time through, we reraise any exception caught.
        # If no exception is thrown, we simply break the loop and return.
        try:

            # Do the necessary initial setup for this stamp type.
            xsize, ysize, image_pos, world_pos = builder.setup(
                    stamp, config, xsize, ysize, stamp_ignore, logger)

            # Save these values for possible use in Evals or other modules
            SetupConfigStampSize(config, xsize, ysize, image_pos, world_pos)
            stamp_center = config['stamp_center']
            if logger:
                if xsize:
                    logger.debug('obj %d: xsize,ysize = %s,%s',obj_num,xsize,ysize)
                if image_pos:
                    logger.debug('obj %d: image_pos = %s',obj_num,image_pos)
                if world_pos:
                    logger.debug('obj %d: world_pos = %s',obj_num,world_pos)
                if stamp_center:
                    logger.debug('obj %d: stamp_center = %s',obj_num,stamp_center)

            # Get the global gsparams kwargs.  Individual objects can add to this.
            gsparams = {}
            if 'gsparams' in stamp:
                gsparams = galsim.config.UpdateGSParams(
                    gsparams, stamp['gsparams'], config)

            skip = False
            try :
                psf = galsim.config.BuildGSObject(config, 'psf', gsparams=gsparams,
                                                  logger=logger)[0]
                prof = builder.buildProfile(stamp, config, psf, gsparams, logger)
            except galsim.config.gsobject.SkipThisObject as e:
                if logger:
                    logger.debug('obj %d: Caught SkipThisObject: e = %s',obj_num,e.msg)
                if logger:
                    if e.msg:
                        # If there is a message, upgrade to info level
                        logger.info('Skipping object %d: %s',obj_num,e.msg)
                skip = True
                # Note: Skip is different from Reject.
                #       Skip means we return None for this stamp image and continue on.
                #       Reject means we retry this object using the same obj_num.
                #       This has implications for the total number of objects as well as
                #       things like ring tests that rely on objects being made in pairs.

            im = builder.makeStamp(stamp, config, xsize, ysize, logger)

            if not skip:
                if 'draw_method' in stamp:
                    method = galsim.config.ParseValue(stamp,'draw_method',config,str)[0]
                else:
                    method = 'auto'
                if method not in ['auto', 'fft', 'phot', 'real_space', 'no_pixel', 'sb']:
                    raise AttributeError("Invalid draw_method: %s"%method)

                offset = config['stamp_offset']
                if 'offset' in stamp:
                    offset += galsim.config.ParseValue(stamp, 'offset', config, galsim.PositionD)[0]
                if logger:
                    logger.debug('obj %d: offset = %s',obj_num,offset)

                im = builder.draw(prof, im, method, offset, stamp, config, logger)

                scale_factor = builder.getSNRScale(im, stamp, config, logger)
                im, prof = builder.applySNRScale(im, prof, scale_factor, method, logger)

            # Set the origin appropriately
            if im is None:
                # Note: im might be None here if the stamp size isn't given and skip==True.
                pass
            elif stamp_center:
                im.setCenter(stamp_center)
            else:
                im.setOrigin(config['image_origin'])

            # Store the current stamp in the base-level config for reference
            config['current_stamp'] = im
            # This is also information that the weight image calculation needs
            config['do_noise_in_stamps'] = do_noise

            # Check if this object should be rejected.
            reject = builder.reject(stamp, config, prof, psf, im, logger)
            if reject:
                if itry+1 < ntries:
                    if logger and logger.isEnabledFor(logging.WARN):
                        logger.warn('Object %d: Rejecting this object and rebuilding',obj_num)
                    builder.reset(config, logger)
                    continue
                else:
                    if logger:
                        logger.error('Object %d: Too many rejections for this object. Aborting.',
                                     obj_num)
                    raise RuntimeError("Rejected an object %d times. If this is expected, "%ntries+
                                       "you should specify a larger retry_failures.")

            galsim.config.ProcessExtraOutputsForStamp(config, logger)

            # We always need to do the whiten step here in the stamp processing
            if not skip:
                current_var = builder.whiten(prof, im, stamp, config, logger)
                if current_var != 0.:
                    if logger:
                        logger.debug('obj %d: whitening noise brought current var to %f',
                                     config['obj_num'],current_var)
            else:
                current_var = 0.

            # Sometimes, depending on the image type, we go on to do the rest of the noise as well.
            if do_noise:
                im, current_var = builder.addNoise(stamp,config,skip,im,current_var,logger)

            return im, current_var

        except KeyboardInterrupt:
            raise
        except Exception as e:
            if itry == ntries-1:
                # Then this was the last try.  Just re-raise the exception.
                raise
            else:
                if logger:
                    logger.info('Object %d: Caught exception %s',obj_num,str(e))
                    logger.info('This is try %d/%d, so trying again.',itry+1,ntries)
                if logger and logger.isEnabledFor(logging.DEBUG):
                    import traceback
                    tr = traceback.format_exc()
                    logger.debug('obj %d: Traceback = %s',obj_num,tr)
                # Need to remove the "current_val"s from the config dict.  Otherwise,
                # the value generators will do a quick return with the cached value.
                builder.reset(config, logger)
                continue

def MakeStampTasks(config, jobs, logger):
    """Turn a list of jobs into a list of tasks.

    See the doc string for galsim.config.MultiProcess for the meaning of this distinction.

    For the Basic stamp type, there is just one job per task, so the tasks list is just:

        tasks = [ [ (job, k) ] for k, job in enumerate(jobs) ]

    But other stamp types may need groups of jobs to be done sequentially by the same process.
    cf. stamp type=Ring.

    @param config           The configuration dict
    @param jobs             A list of jobs to split up into tasks.  Each job in the list is a
                            dict of parameters that includes 'obj_num'.
    @param logger           If given, a logger object to log progress.

    @returns a list of tasks
    """
    stamp = config.get('stamp', {})
    stamp_type = stamp.get('type', 'Basic')
    return valid_stamp_types[stamp_type].makeTasks(stamp, config, jobs, logger)


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

        @param config       The configuration dict for the stamp field.
        @param base         The base configuration dict.
        @param xsize        The xsize of the image to build (if known).
        @param ysize        The ysize of the image to build (if known).
        @param ignore       A list of parameters that are allowed to be in config that we can
                            ignore here. i.e. it won't be an error if these parameters are present.
        @param logger       If given, a logger object to log progress.

        @returns xsize, ysize, image_pos, world_pos
        """
        # Check for spurious parameters
        galsim.config.CheckAllParams(config, ignore=ignore)

        # Update the size if necessary
        image = base['image']
        if not xsize:
            if 'xsize' in config:
                xsize = galsim.config.ParseValue(config,'xsize',base,int)[0]
            elif 'size' in config:
                xsize = galsim.config.ParseValue(config,'size',base,int)[0]
            elif 'stamp_xsize' in image:
                xsize = galsim.config.ParseValue(image,'stamp_xsize',base,int)[0]
            elif 'stamp_size' in image:
                xsize = galsim.config.ParseValue(image,'stamp_size',base,int)[0]

        if not ysize:
            if 'ysize' in config:
                ysize = galsim.config.ParseValue(config,'ysize',base,int)[0]
            elif 'size' in config:
                ysize = galsim.config.ParseValue(config,'size',base,int)[0]
            elif 'stamp_ysize' in image:
                ysize = galsim.config.ParseValue(image,'stamp_ysize',base,int)[0]
            elif 'stamp_size' in image:
                ysize = galsim.config.ParseValue(image,'stamp_size',base,int)[0]

        # Determine where this object is going to go:
        if 'image_pos' in config:
            image_pos = galsim.config.ParseValue(config, 'image_pos', base, galsim.PositionD)[0]
        elif 'image_pos' in image:
            image_pos = galsim.config.ParseValue(image, 'image_pos', base, galsim.PositionD)[0]
        else:
            image_pos = None

        if 'world_pos' in config:
            world_pos = galsim.config.ParseValue(config, 'world_pos', base, galsim.PositionD)[0]
        elif 'world_pos' in image:
            world_pos = galsim.config.ParseValue(image, 'world_pos', base, galsim.PositionD)[0]
        else:
            world_pos = None

        return xsize, ysize, image_pos, world_pos

    def buildProfile(self, config, base, psf, gsparams, logger):
        """Build the surface brightness profile (a GSObject) to be drawn.
 
        For the Basic stamp type, this builds a galaxy from the base['gal'] dict and convolves
        it with the psf (if given).  If either the psf or the galaxy is None, then the other one
        is returned as is.

        @param config       The configuration dict for the stamp field.
        @param base         The base configuration dict.
        @param psf          The PSF, if any.  This may be None, in which case, no PSF is convolved.
        @param gsparams     A dict of kwargs to use for a GSParams.  More may be added to this
                            list by the galaxy object.
        @param logger       If given, a logger object to log progress.

        @returns the final profile
        """
        gal = galsim.config.BuildGSObject(base, 'gal', gsparams=gsparams, logger=logger)[0]

        if psf:
            if gal:
                return galsim.Convolve(gal,psf)
            else:
                return psf
        else:
            if gal:
                return gal
            else:
                raise AttributeError("At least one of gal or psf must be specified in config.")

    def makeStamp(self, config, base, xsize, ysize, logger):
        """Make the initial empty postage stamp image, if possible.

        If we don't know xsize, ysize, return None, in which case the stamp will be created
        automatically by the drawImage command based on the natural size of the profile.

        @param config       The configuration dict for the stamp field.
        @param base         The base configuration dict.
        @param xsize        The xsize of the image to build (if known).
        @param ysize        The ysize of the image to build (if known).
        @param logger       If given, a logger object to log progress.

        @returns the image
        """
        if xsize and ysize:
            # If the size is set, we need to do something reasonable to return this size.
            im = galsim.ImageF(xsize, ysize)
            im.setZero()
            return im
        else:
            return None

    def draw(self, prof, image, method, offset, config, base, logger):
        """Draw the profile on the postage stamp image.

        @param prof         The profile to draw.
        @param image        The image onto which to draw the profile (which may be None).
        @param method       The method to use in drawImage.
        @param offset       The offset to apply when drawing.
        @param config       The configuration dict for the stamp field.
        @param base         The base configuration dict.
        @param logger       If given, a logger object to log progress.

        @returns the resulting image
        """
        # Setup the kwargs to pass to drawImage
        kwargs = {}
        kwargs['image'] = image
        kwargs['offset'] = offset
        kwargs['method'] = method
        if 'wmult' in config:
            kwargs['wmult'] = galsim.config.ParseValue(config, 'wmult', base, float)[0]
        kwargs['wcs'] = base['wcs'].local(image_pos = base['image_pos'])
        if method == 'phot':
            kwargs['rng'] = base['rng']

        # Check validity of extra phot options:
        max_extra_noise = None
        if 'n_photons' in config:
            if method != 'phot':
                raise AttributeError('n_photons is invalid with method != phot')
            if 'max_extra_noise' in config:
                if logger:
                    logger.warn(
                        "Both 'max_extra_noise' and 'n_photons' are set in config dict, "+
                        "ignoring 'max_extra_noise'.")
            kwargs['n_photons'] = galsim.config.ParseValue(config, 'n_photons', base, int)[0]
        elif 'max_extra_noise' in config:
            if method != 'phot':
                raise AttributeError('max_extra_noise is invalid with method != phot')
            max_extra_noise = galsim.config.ParseValue(config, 'max_extra_noise', base, float)[0]
        elif method == 'phot':
            max_extra_noise = 0.01

        if 'poisson_flux' in config:
            if method != 'phot':
                raise AttributeError('poisson_flux is invalid with method != phot')
            kwargs['poisson_flux'] = galsim.config.ParseValue(config, 'poisson_flux', base, bool)[0]

        if max_extra_noise is not None:
            if max_extra_noise < 0.:
                raise ValueError("image.max_extra_noise cannot be negative")
            if max_extra_noise > 0.:
                if 'image' in base and 'noise' in base['image']:
                    noise_var = galsim.config.CalculateNoiseVar(base)
                else:
                    raise AttributeError("Need to specify noise level when using max_extra_noise")
                if noise_var < 0.:
                    raise ValueError("noise_var calculated to be < 0.")
                max_extra_noise *= noise_var
                kwargs['max_extra_noise'] = max_extra_noise

        image = prof.drawImage(**kwargs)
        return image

    def whiten(self, prof, image, config, base, logger):
        """If appropriate, whiten the resulting image according to the requested noise profile
        and the amount of noise originally present in the profile.

        @param prof         The profile to draw.
        @param image        The image onto which to draw the profile.
        @param config       The configuration dict for the stamp field.
        @param base         The base configuration dict.
        @param logger       If given, a logger object to log progress.

        @returns the variance of the resulting whitened (or symmetrized) image.
        """
        # If the object has a noise attribute, then check if we need to do anything with it.
        current_var = 0.  # Default if not overwritten
        if hasattr(prof,'noise'):
            if 'image' in base and 'noise' in base['image']:
                noise = base['image']['noise']
                if 'whiten' in noise:
                    if 'symmetrize' in noise:
                        raise AttributeError('Only one of whiten or symmetrize is allowed')
                    whiten, safe = galsim.config.ParseValue(noise, 'whiten', base, bool)
                    current_var = prof.noise.whitenImage(image)
                elif 'symmetrize' in noise:
                    symmetrize, safe = galsim.config.ParseValue(noise, 'symmetrize', base, int)
                    current_var = prof.noise.symmetrizeImage(image, symmetrize)
        return current_var

    def getSNRScale(self, image, config, base, logger):
        """Calculate the factor by which to rescale the image based on a desired S/N level.

        @param image        The current image.
        @param config       The configuration dict for the stamp field.
        @param base         The base configuration dict.
        @param logger       If given, a logger object to log progress.

        @returns scale_factor
        """
        if (('gal' in base and 'signal_to_noise' in base['gal']) or
            ('gal' not in base and 'psf' in base and 'signal_to_noise' in base['psf'])):
            import math
            import numpy
            if 'gal' in base: root_key = 'gal'
            else: root_key = 'psf'

            if 'flux' in base[root_key]:
                raise AttributeError(

                    'Only one of signal_to_noise or flux may be specified for %s'%root_key)

            if 'image' in base and 'noise' in base['image']:
                noise_var = galsim.config.CalculateNoiseVar(base)
            else:
                raise AttributeError(
                    "Need to specify noise level when using %s.signal_to_noise"%root_key)
            sn_target = galsim.config.ParseValue(base[root_key], 'signal_to_noise', base, float)[0]

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

            sn_meas = math.sqrt( numpy.sum(image.array**2) / noise_var )
            # Now we rescale the flux to get our desired S/N
            scale_factor = sn_target / sn_meas
            return scale_factor
        else:
            return 1.

    def applySNRScale(self, image, prof, scale_factor, method, logger):
        """Apply the scale_factor from getSNRScale to the image and profile.

        The default implementaion just multiplies each of them, but if prof is not a regular
        GSObject, then you might need to do something different.

        @param image        The current image.
        @param prof         The profile that was drawn.
        @param scale_factor The factor by which to scale both image and prof.
        @param method       The method used by drawImage.
        @param logger       If given, a logger object to log progress.

        @returns image, prof  (after being properly scaled)
        """
        if scale_factor != 1.0:
            if method == 'phot':
                logger.warn(
                    "signal_to_noise calculation is not accurate for draw_method = phot")
            image *= scale_factor
            prof *= scale_factor
        return image, prof

    def reject(self, config, base, prof, psf, image, logger):
        """Check to see if this object should be rejected.

        @param config       The configuration dict for the stamp field.
        @param base         The base configuration dict.
        @param prof         The profile that was drawn.
        @param psf          The psf that was used to build the profile.
        @param image        The postage stamp image.  No noise is on it yet at this point.
        @param logger       If given, a logger object to log progress.

        @returns whether to reject this object
        """
        # Check that we aren't on a second or later item in a Ring.
        # This check can be removed once we do not need to support the deprecated gsobject
        # type=Ring.
        if 'gal' in base:
            block_size = galsim.config.gsobject._GetMinimumBlock(base['gal'], base)
            if base['obj_num'] % block_size != 0:
                # Don't reject, since the first item passed.
                # If we reject now, it will mess things up.
                return False

        reject = False
        if 'reject' in config:
            reject = galsim.config.ParseValue(config, 'reject', base, bool)[0]
        if 'min_flux_frac' in config:
            if not isinstance(prof, galsim.GSObject):
                raise ValueError("Cannot apply min_flux_frac for stamp types that do not use "+
                                "a single GSObject profile.")
            expected_flux = prof.flux
            measured_flux = numpy.sum(image.array)
            min_flux_frac = galsim.config.ParseValue(config, 'min_flux_frac', base, float)[0]
            if measured_flux < min_flux_frac * expected_flux:
                if logger and logger.isEnabledFor(logging.WARN):
                    logger.warn('Object %d: Measured flux = %f < %s * %f.',
                                base['obj_num'], measured_flux, min_flux_frac, expected_flux)
                reject = True
        if 'min_snr' in config or 'max_snr' in config:
            if not isinstance(prof, galsim.GSObject):
                raise ValueError("Cannot apply min_snr for stamp types that do not use "+
                                "a single GSObject profile.")
            var = galsim.config.CalculateNoiseVar(base)
            sumsq = numpy.sum(image.array**2)
            snr = numpy.sqrt(sumsq / var)
            if 'min_snr' in config:
                min_snr = galsim.config.ParseValue(config, 'min_snr', base, float)[0]
                if snr < min_snr:
                    if logger and logger.isEnabledFor(logging.WARN):
                        logger.warn('Object %d: Measured snr = %f < %s.',
                                    base['obj_num'], snr, min_snr)
                    reject = True
            if 'max_snr' in config:
                max_snr = galsim.config.ParseValue(config, 'max_snr', base, float)[0]
                if snr > max_snr:
                    if logger and logger.isEnabledFor(logging.WARN):
                        logger.warn('Object %d: Measured snr = %f > %s.',
                                    base['obj_num'], snr, max_snr)
                    reject = True
        return reject

    def reset(self, base, logger):
        """Reset some aspects of the config dict so the object can be rebuilt after rejecting the
        current object.

        @param base         The base configuration dict.
        @param logger       If given, a logger object to log progress.
        """
        # Clear current values out of psf, gal, and stamp if they are not safe to reuse.
        # This means they are either marked as safe or indexed by something other than obj_num.
        for field in ['psf', 'gal', 'stamp']:
            galsim.config.RemoveCurrent(base[field], keep_safe=True, index_key='obj_num')

    def addNoise(self, config, base, skip, image, current_var, logger):
        """
        Add the sky level and the noise to the stamp.

        Note: This only gets called if the image type requests that the noise be added to each
            stamp individually, rather than to the full image and the end.

        @param config       The configuration dict for the stamp field.
        @param base         The base configuration dict.
        @param skip         Are we skipping this image? (Usually means to add sky, but not noise.)
        @param image        The current image.
        @param current_var  The current noise variance present in the image already.
        @param logger       If given, a logger object to log progress.

        @returns the new values of image, current_var
        """
        galsim.config.AddSky(base,image)
        if not skip:
            current_var = galsim.config.AddNoise(base,image,current_var,logger)
        return image, current_var

    def makeTasks(self, config, base, jobs, logger):
        """Turn a list of jobs into a list of tasks.

        For the Basic stamp type, there is just one job per task, so the tasks list is just:

            tasks = [ [ (job, k) ] for k, job in enumerate(jobs) ]

        @param config       The configuration dict for the stamp field.
        @param base         The base configuration dict.
        @param jobs         A list of jobs to split up into tasks.  Each job in the list is a
                            dict of parameters that includes 'obj_num'.
        @param logger       If given, a logger object to log progress.

        @returns a list of tasks
        """
        # For backwards compatibility with the old Ring gsobject type, we actually check for that
        # here and make a list of tasks that work for that.  This is undocumented though, since the
        # gsobject type=Ring is now deprecated.
        block_size = 1
        if 'gal' in base:
            block_size = galsim.config.gsobject._GetMinimumBlock(base['gal'], base)
        tasks = [ [ (jobs[j], j) for j in range(k,min(k+block_size,len(jobs))) ]
                    for k in range(0, len(jobs), block_size) ]
        return tasks



def RegisterStampType(stamp_type, builder):
    """Register an image type for use by the config apparatus.

    @param stamp_type       The name of the type in config['stamp']
    @param builder          A builder object to use for building the stamp images.  It should be
                            an instance of StampBuilder or a subclass thereof.
    """
    valid_stamp_types[stamp_type] = builder

RegisterStampType('Basic', StampBuilder())

