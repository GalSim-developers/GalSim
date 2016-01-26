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

# This file handles the building of postage stamps to place onto a larger image.
# There is only one type of stamp currently, called Basic, which builds a galaxy from
# config['gal'] and a PSF from config['psf'] (either but not both of which may be absent),
# colvolves them together, and draws onto a postage stamp.  This is the default functionality
# and is typically not specified explicitly.  But there are hooks in place to allow for other
# options, either in future versions of GalSim or through user modules.


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

    nobj_per_task = galsim.config.CalculateNObjPerTask(nproc, nobjects, config)
    if logger:
        logger.debug('image %d: nobj_per_task = %d',config.get('image_num',0), nobj_per_task)

    jobs = []
    for k in range(nobjects):
        kwargs = {
            'obj_num' : obj_num + k,
            'xsize' : xsize,
            'ysize' : ysize,
            'do_noise' : do_noise,
        }
        jobs.append( (kwargs, obj_num+k) )

    def done_func(logger, proc, obj_num, result, t):
        if logger:
            # Note: numpy shape is y,x
            image = result[0]
            ys, xs = image.array.shape
            if proc is None: s0 = ''
            else: s0 = '%s: '%proc
            logger.info(s0 + 'Stamp %d: size = %d x %d, time = %f sec', obj_num, xs, ys, t)

    def except_func(logger, proc, e, tr, obj_num):
        if logger:
            if proc is None: s0 = ''
            else: s0 = '%s: '%proc
            logger.error(s0 + 'Exception caught when building stamp %d', obj_num)
            #logger.error('%s',tr)
            logger.error('Aborting the rest of this image')

    results = galsim.config.MultiProcess(nproc, config, BuildStamp, jobs, 'stamp', logger,
                                         njobs_per_task = nobj_per_task,
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
                'wmult', 'nphotons', 'max_extra_noise', 'poisson_flux']

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

    stamp_type = config['stamp']['type']
    if stamp_type not in valid_stamp_types:
        raise AttributeErro("Invalid stamp.type=%s."%stamp_type)

    # Add 1 to the seed here so the first object has a different rng than the file or image.
    seed = galsim.config.SetupConfigRNG(config, seed_offset=1)
    if logger:
        logger.debug('obj %d: seed = %d',obj_num,seed)

    if 'retry_failures' in config['stamp']:
        ntries = galsim.config.ParseValue(config['stamp'],'retry_failures',config,int)[0]
        # This is how many _re_-tries.  Do at least 1, so ntries is 1 more than this.
        ntries = ntries + 1
    else:
        ntries = 1

    for itry in range(ntries):

        # The rest of the stamp generation stage is wrapped in a try/except block.
        # If we catch an exception, we continue the for loop to try again.
        # On the last time through, we reraise any exception caught.
        # If no exception is thrown, we simply break the loop and return.
        try:

            # Do the necessary initial setup for this stamp type.
            setup_func = valid_stamp_types[stamp_type]['setup']
            xsize, ysize, image_pos, world_pos = setup_func(
                    config, xsize, ysize, stamp_ignore, logger)

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
            if 'gsparams' in config['stamp']:
                gsparams = galsim.config.UpdateGSParams(
                    gsparams, config['stamp']['gsparams'], config)

            skip = False
            try :
                psf = galsim.config.BuildGSObject(config, 'psf', gsparams=gsparams,
                                                  logger=logger)[0]

                profile_func = valid_stamp_types[stamp_type]['prof']
                prof = profile_func(config, psf, gsparams, logger)

            except galsim.config.gsobject.SkipThisObject, e:
                if logger:
                    logger.debug('obj %d: Caught SkipThisObject: e = %s',obj_num,e.msg)
                if logger:
                    if e.msg:
                        # If there is a message, upgrade to info level
                        logger.info('Skipping object %d: %s',obj_num,e.msg)
                skip = True

            stamp_func = valid_stamp_types[stamp_type]['stamp']
            im = stamp_func(config, xsize, ysize)

            if not skip:
                if 'draw_method' in config['stamp']:
                    method = galsim.config.ParseValue(config['stamp'],'draw_method',config,str)[0]
                else:
                    method = 'auto'
                if method not in ['auto', 'fft', 'phot', 'real_space', 'no_pixel', 'sb']:
                    raise AttributeError("Invalid draw_method: %s"%method)

                offset = config['stamp_offset']
                if 'offset' in config['stamp']:
                    offset += galsim.config.ParseValue(config['stamp'], 'offset', config,
                                                       galsim.PositionD)[0]
                if logger:
                    logger.debug('obj %d: offset = %s',obj_num,offset)

                draw_func = valid_stamp_types[stamp_type]['draw']
                im = draw_func(prof, im, method, offset, config)

                snr_func = valid_stamp_types[stamp_type]['snr']
                scale_factor = snr_func(im, config)
                if scale_factor != 1.0:
                    if method == 'phot':
                        logger.error(
                            "signal_to_noise calculation is not accurate for draw_method = phot")
                    im *= scale_factor
                    prof *= scale_factor

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

            galsim.config.ProcessExtraOutputsForStamp(config, logger)

            # We always need to do the whiten step here in the stamp processing
            if not skip:
                whiten_func = valid_stamp_types[stamp_type]['whiten']
                current_var = whiten_func(prof, im, config)
                if current_var != 0.:
                    if logger:
                        logger.debug('obj %d: whitening noise brought current var to %f',
                                     config['obj_num'],current_var)
            else:
                current_var = 0.

            # Sometimes, depending on the image type, we go on to do the rest of the noise as well.
            if do_noise:
                galsim.config.AddSky(config,im)
                if not skip:
                    galsim.config.AddNoise(config,im,current_var,logger)

            return im, current_var

        except Exception as e:

            if itry == ntries-1:
                # Then this was the last try.  Just re-raise the exception.
                raise
            else:
                if logger:
                    logger.info('Object %d: Caught exception %s',obj_num,str(e))
                    logger.info('This is try %d/%d, so trying again.',itry+1,ntries)
                # Need to remove the "current_val"s from the config dict.  Otherwise,
                # the value generators will do a quick return with the cached value.
                galsim.config.RemoveCurrent(config, keep_safe=True)
                continue


def SetupBasic(config, xsize, ysize, ignore, logger):
    """
    Do the initialization and setup for building a Basic postage stamp.  In this case
    we check for and parse the appropriate size and position values in config['stamp']
    or config['image'].

    Values given in config['stamp'] take precedence if these are given in both places (which would
    be confusing, so probably shouldn't do that, but there might be a use case where it would make
    sense).

    @param config           The configuration dict.
    @param xsize            The xsize of the image to build (if known).
    @param ysize            The ysize of the image to build (if known).
    @param ignore           A list of parameters that are allowed to be in config['stamp']
                            that we can ignore here.  i.e. it won't be an error if these
                            parameters are present.
    @param logger           If given, a logger object to log progress.

    @returns xsize, ysize, image_pos, world_pos
    """
    # Check for spurious parameters
    galsim.config.CheckAllParams(config['stamp'], ignore=ignore)

    # Update the size if necessary
    if not xsize:
        if 'xsize' in config['stamp']:
            xsize = galsim.config.ParseValue(config['stamp'],'xsize',config,int)[0]
        elif 'size' in config['stamp']:
            xsize = galsim.config.ParseValue(config['stamp'],'size',config,int)[0]
        elif 'stamp_xsize' in config['image']:
            xsize = galsim.config.ParseValue(config['image'],'stamp_xsize',config,int)[0]
        elif 'stamp_size' in config['image']:
            xsize = galsim.config.ParseValue(config['image'],'stamp_size',config,int)[0]

    if not ysize:
        if 'ysize' in config['stamp']:
            ysize = galsim.config.ParseValue(config['stamp'],'ysize',config,int)[0]
        elif 'size' in config['stamp']:
            ysize = galsim.config.ParseValue(config['stamp'],'size',config,int)[0]
        elif 'stamp_ysize' in config['image']:
            ysize = galsim.config.ParseValue(config['image'],'stamp_ysize',config,int)[0]
        elif 'stamp_size' in config['image']:
            ysize = galsim.config.ParseValue(config['image'],'stamp_size',config,int)[0]

    # Determine where this object is going to go:
    if 'image_pos' in config['stamp']:
        image_pos = galsim.config.ParseValue(
            config['stamp'], 'image_pos', config, galsim.PositionD)[0]
    elif 'image_pos' in config['image']:
        image_pos = galsim.config.ParseValue(
            config['image'], 'image_pos', config, galsim.PositionD)[0]
    else:
        image_pos = None

    if 'world_pos' in config['stamp']:
        world_pos = galsim.config.ParseValue(
            config['stamp'], 'world_pos', config, galsim.PositionD)[0]
    elif 'world_pos' in config['image']:
        world_pos = galsim.config.ParseValue(
            config['image'], 'world_pos', config, galsim.PositionD)[0]
    else:
        world_pos = None

    return xsize, ysize, image_pos, world_pos


def ProfileBasic(config, psf, gsparams, logger):
    """
    Build the object to be drawn.  For the Basic stamp type, this builds a galaxy from
    the config['gal'] dict and convolves it with the psf (if given).  If either the psf or
    the galaxy is None, then the other one is returned as is.

    @param config           The configuration dict.
    @param psf              The PSF, if any.  This may be None, in which case, no PSF is convolved.
    @param gsparams         A dict of kwargs to use for a GSParams.  More may be added to this
                            list by the galaxy object.
    @param logger           If given, a logger object to log progress.

    @returns the final profile
    """
    gal = galsim.config.BuildGSObject(config, 'gal', gsparams=gsparams, logger=logger)[0]

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

def StampBasic(config, xsize, ysize):
    """
    Returns the postage stamp image onto which we will draw the profile if we can before the
    drawImage command.  If we don't know xsize, ysize, return None.

    @param config           The configuration dict.
    @param xsize            The xsize of the image to build (if known).
    @param ysize            The ysize of the image to build (if known).

    @returns the image
    """
    if xsize and ysize:
        # If the size is set, we need to do something reasonable to return this size.
        im = galsim.ImageF(xsize, ysize)
        im.setZero()
        return im
    else:
        return None

def DrawBasic(prof, image, method, offset, config):
    """
    Draw the profile on the image.

    @param prof         The profile to draw.
    @param image        The image onto which to draw the profile.
    @param method       The method to use in drawImage.
    @param offset       The offset to apply when drawing.
    @param config       The configuration dict.

    @returns the resulting image
    """
    # Setup the kwargs to pass to drawImage
    kwargs = {}
    kwargs['image'] = image
    kwargs['offset'] = offset
    kwargs['method'] = method
    if 'wmult' in config['stamp']:
        kwargs['wmult'] = galsim.config.ParseValue(config['stamp'], 'wmult', config, float)[0]
    kwargs['wcs'] = config['wcs'].local(image_pos = config['image_pos'])
    if method == 'phot':
        kwargs['rng'] = config['rng']

    # Check validity of extra phot options:
    max_extra_noise = None
    if 'n_photons' in config['stamp']:
        if method != 'phot':
            raise AttributeError('n_photons is invalid with method != phot')
        if 'max_extra_noise' in config['stamp']:
            if logger:
                logger.warn(
                    "Both 'max_extra_noise' and 'n_photons' are set in config dict, "+
                    "ignoring 'max_extra_noise'.")
        kwargs['n_photons'] = galsim.config.ParseValue(config['stamp'], 'n_photons', config, int)[0]
    elif 'max_extra_noise' in config['stamp']:
        if method != 'phot':
            raise AttributeError('max_extra_noise is invalid with method != phot')
        max_extra_noise = galsim.config.ParseValue(
            config['stamp'], 'max_extra_noise', config, float)[0]
    elif method == 'phot':
        max_extra_noise = 0.01

    if 'poisson_flux' in config['stamp']:
        if method != 'phot':
            raise AttributeError('poisson_flux is invalid with method != phot')
        kwargs['poisson_flux'] = galsim.config.ParseValue(
                config['stamp'], 'poisson_flux', config, bool)[0]

    if max_extra_noise is not None:
        if max_extra_noise < 0.:
            raise ValueError("image.max_extra_noise cannot be negative")
        if max_extra_noise > 0.:
            if 'image' in config and 'noise' in config['image']:
                noise_var = galsim.config.CalculateNoiseVar(config)
            else:
                raise AttributeError(
                    "Need to specify noise level when using draw_method = phot")
            if noise_var < 0.:
                raise ValueError("noise_var calculated to be < 0.")
            max_extra_noise *= noise_var
            kwargs['max_extra_noise'] = max_extra_noise

    image = prof.drawImage(**kwargs)
    return image

def WhitenBasic(prof, image, config):
    """
    If appropriate, whiten the resulting image according to the requested noise profile
    and the amount of noise originally present in the profile.

    @param prof         The profile to draw.
    @param image        The image onto which to draw the profile.
    @param config       The configuration dict.

    @returns the variance of the resulting whitened (or symmetrized) image.
    """
    # If the object has a noise attribute, then check if we need to do anything with it.
    current_var = 0.  # Default if not overwritten
    if hasattr(prof,'noise'):
        if 'image' in config and 'noise' in config['image']:
            noise = config['image']['noise']
            if 'whiten' in noise:
                if 'symmetrize' in noise:
                    raise AttributeError('Only one of whiten or symmetrize is allowed')
                whiten, safe = galsim.config.ParseValue(noise, 'whiten', config, bool)
                current_var = prof.noise.whitenImage(image)
            elif 'symmetrize' in noise:
                symmetrize, safe = galsim.config.ParseValue(noise, 'symmetrize', config, int)
                current_var = prof.noise.symmetrizeImage(image, symmetrize)
    return current_var


def SNRBasic(image, config):
    """
    Calculate the factor by which to rescale the image based on a desired S/N level.

    @param image            The current image.
    @param config           The configuration dict.

    @returns scale_factor
    """
    if (('gal' in config and 'signal_to_noise' in config['gal']) or
        ('gal' not in config and 'psf' in config and 'signal_to_noise' in config['psf'])):
        import math
        import numpy
        if 'gal' in config: root_key = 'gal'
        else: root_key = 'psf'

        if 'flux' in config[root_key]:
            raise AttributeError(
                'Only one of signal_to_noise or flux may be specified for %s'%root_key)

        if 'image' in config and 'noise' in config['image']:
            noise_var = galsim.config.CalculateNoiseVar(config)
        else:
            raise AttributeError(
                "Need to specify noise level when using %s.signal_to_noise"%root_key)
        sn_target = galsim.config.ParseValue(config[root_key], 'signal_to_noise', config, float)[0]

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

valid_stamp_types = {}

def RegisterStampType(stamp_type, setup_func=None, prof_func=None, stamp_func=None,
                      draw_func=None, whiten_func=None, snr_func=None):
    """Register an image type for use by the config apparatus.

    You only need to specify the functions that you want to change from the Basic stamp
    functionality.

    @param stamp_type       The name of the type in config['stamp']
    @param setup_func       The function to call to determine the size of the stamp and do any
                            other initial setup.
                            The call signature is
                                xsize, ysize, image_pos, world_pos = \
                                    Setup(config, xsize, ysize, ignore, logger)
    @param prof_func        The function to call to build the profile
                            The call signature is
                                prof = Profile(config, psf, gsparams, logger)
    @param stamp_func       The function to call to build the postage stamp image
                            The call signature is:
                                image = Stamp(config, xsize, ysize)
    @param draw_func        The function to call to draw the image.
                            The call signature is
                                image = Draw(prof, image, method, offset, config)
    @param whiten_func      The function to call to whiten the image.
                            The call signature is
                                current_var = Whiten(prof, image, config)
    @param snr_func         The function to call to rescale the image according to the desired
                            signal-to-noise.
                            The call signature is
                                scale_factor = SNR(image, config)
    """
    if setup_func is None: setup_func = SetupBasic
    if prof_func is None: prof_func = ProfileBasic
    if stamp_func is None: stamp_func = StampBasic
    if draw_func is None: draw_func = DrawBasic
    if whiten_func is None: whiten_func = WhitenBasic
    if snr_func is None: snr_func = SNRBasic

    valid_stamp_types[stamp_type] = {
        'setup' : setup_func,
        'prof' : prof_func,
        'stamp' : stamp_func,
        'draw' : draw_func,
        'whiten' : whiten_func,
        'snr' : snr_func,
    }

RegisterStampType('Basic', SetupBasic, ProfileBasic, StampBasic, DrawBasic, WhitenBasic, SNRBasic)

