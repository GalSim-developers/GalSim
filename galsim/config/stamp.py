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

from .extra_psf import DrawPSFStamp

def BuildStamps(nobjects, config, nproc=1, logger=None, obj_num=0,
                xsize=0, ysize=0, do_noise=True,
                make_psf_image=False, make_weight_image=False, make_badpix_image=False):
    """
    Build a number of postage stamp images as specified by the config dict.

    @param nobjects         How many postage stamps to build.
    @param config           A configuration dict.
    @param nproc            How many processes to use. [default: 1]
    @param logger           If given, a logger object to log progress. [default: None]
    @param obj_num          If given, the current obj_num. [default: 0]
    @param xsize            The size of a single stamp in the x direction. [default: 0,
                            which means to look for config.image.stamp_xsize, and if that's
                            not there, use automatic sizing.]
    @param ysize            The size of a single stamp in the y direction. [default: 0,
                            which means to look for config.image.stamp_xsize, and if that's
                            not there, use automatic sizing.]
    @param do_noise         Whether to add noise to the image (according to config['noise']).
                            [default: True]
    @param make_psf_image   Whether to make psf_image. [default: False]
    @param make_weight_image  Whether to make weight_image. [default: False]
    @param make_badpix_image  Whether to make badpix_image. [default: False]

    @returns the tuple (images, psf_images, weight_images, badpix_images, current_vars).
             All in tuple are lists.
    """
    config['obj_num'] = obj_num

    # Update nproc in case the config value is -1
    nproc = galsim.config.UpdateNProc(nproc, nobjects, config, logger)
    
    if nproc > 1:
        # Number of objects to do in each task:
        # At most nobjects / nproc.
        # At least 1 normally, but number in Ring if doing a Ring test
        # Shoot for geometric mean of these two.
        max_nobj = nobjects / nproc
        min_nobj = 1
        if ( 'gal' in config and isinstance(config['gal'],dict) and 'type' in config['gal'] and
             config['gal']['type'] == 'Ring' and 'num' in config['gal'] ):
            min_nobj = galsim.config.ParseValue(config['gal'], 'num', config, int)[0]
        if max_nobj < min_nobj: 
            nobj_per_task = min_nobj
        else:
            import math
            # This formula keeps nobj a multiple of min_nobj, so Rings are intact.
            nobj_per_task = min_nobj * int(math.sqrt(float(max_nobj) / float(min_nobj)))
    else:
        nobj_per_task = 1

    jobs = []
    for k in range(nobjects):
        kwargs = {
            'obj_num' : obj_num + k,
            'xsize' : xsize,
            'ysize' : ysize, 
            'do_noise' : do_noise,
            'make_psf_image' : make_psf_image,
            'make_weight_image' : make_weight_image,
            'make_badpix_image' : make_badpix_image
        }
        jobs.append( (kwargs, obj_num+k) )

    def done_func(logger, proc, obj_num, result, t):
        if logger and logger.isEnabledFor(logging.INFO):
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
        images, psf_images, weight_images, badpix_images, current_vars = [], [], [], [], []
        if logger:
            logger.error('No images were built.  All were either skipped or had errors.')
    else:
        images, psf_images, weight_images, badpix_images, current_vars, time = zip(*results)
        if logger and logger.isEnabledFor(logging.DEBUG):
            logger.debug('image %d: Done making stamps',config.get('image_num',0))

    return images, psf_images, weight_images, badpix_images, current_vars


def SetupConfigObjNum(config, obj_num):
    """Do the basic setup of the config dict at the stamp (or object) processing level.
    Includes:
    - Set config['obj_num'] = obj_num
    - Set config['index_key'] = 'obj_num'

    @param config           A configuration dict.
    @param obj_num          The current obj_num.
    """
    config['obj_num'] = obj_num
    config['index_key'] = 'obj_num'


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
      any config['image']['offset'] item that may be present in the config dict.
      Save this as config['stamp_offset']

    @param config           A configuration dict.
    @param xsize            The size of the stamp in the x-dimension. [may be None]
    @param ysize            The size of the stamp in the y-dimension. [may be None]
    @param image_pos        The posotion of the stamp in image coordinates. [may be None]
    @param world_pos        The posotion of the stamp in world coordinates. [may be None]
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
        world_pos = galsim.config.ParseValue(
            config['image'], 'world_pos', config, galsim.PositionD)[0]
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


def BuildStamp(config, xsize=0, ysize=0,
               obj_num=0, do_noise=True, logger=None,
               make_psf_image=False, make_weight_image=False, make_badpix_image=False):
    """
    Build a single stamp image using the given config file

    @param config           A configuration dict.
    @param xsize            The xsize of the image to build (if known). [default: 0]
    @param ysize            The ysize of the image to build (if known). [default: 0]
    @param obj_num          If given, the current obj_num [default: 0]
    @param do_noise         Whether to add noise to the image (according to config['noise']).
                            [default: True]
    @param logger           If given, a logger object to log progress. [default: None]
    @param make_psf_image   Whether to make psf_image. [default: False]
    @param make_weight_image  Whether to make weight_image. [default: False]
    @param make_badpix_image  Whether to make badpix_image. [default: False]

    @returns the tuple (image, psf_image, weight_image, badpix_image, current_var, time)
    """
    import time
    t1 = time.time()

    SetupConfigObjNum(config, obj_num)

    seed = galsim.config.SetupConfigRNG(config)
    if logger and logger.isEnabledFor(logging.DEBUG):
        logger.debug('obj %d: seed = %d',obj_num,seed)

    if 'image' in config and 'retry_failures' in config['image']:
        ntries = galsim.config.ParseValue(config['image'],'retry_failures',config,int)[0]
        # This is how many _re_-tries.  Do at least 1, so ntries is 1 more than this.
        ntries = ntries + 1
    else:
        ntries = 1

    for itry in range(ntries):

        try:  # The rest of the stamp generation stage is wrapped in a try/except block.
              # If we catch an exception, we continue the for loop to try again.
              # On the last time through, we reraise any exception caught.
              # If no exception is thrown, we simply break the loop and return.

            # Determine the size of this stamp
            if not xsize:
                if 'stamp_xsize' in config['image']:
                    xsize = galsim.config.ParseValue(config['image'],'stamp_xsize',config,int)[0]
                elif 'stamp_size' in config['image']:
                    xsize = galsim.config.ParseValue(config['image'],'stamp_size',config,int)[0]
            if not ysize:
                if 'stamp_ysize' in config['image']:
                    ysize = galsim.config.ParseValue(config['image'],'stamp_ysize',config,int)[0]
                elif 'stamp_size' in config['image']:
                    ysize = galsim.config.ParseValue(config['image'],'stamp_size',config,int)[0]

            # Determine where this object is going to go:
            if 'image_pos' in config['image']:
                image_pos = galsim.config.ParseValue(
                    config['image'], 'image_pos', config, galsim.PositionD)[0]
            else:
                image_pos = None
            if 'world_pos' in config['image']:
                world_pos = galsim.config.ParseValue(
                    config['image'], 'world_pos', config, galsim.PositionD)[0]
            else:
                world_pos = None

            # Save these values for possible use in Evals or other modules
            SetupConfigStampSize(config, xsize, ysize, image_pos, world_pos)
            stamp_center = config['stamp_center']
            if logger and logger.isEnabledFor(logging.DEBUG):
                if xsize:
                    logger.debug('obj %d: xsize,ysize = %s,%s',obj_num,xsize,ysize)
                if image_pos:
                    logger.debug('obj %d: image_pos = %s',obj_num,image_pos)
                if world_pos:
                    logger.debug('obj %d: world_pos = %s',obj_num,world_pos)
                if stamp_center:
                    logger.debug('obj %d: stamp_center = %s',obj_num,stamp_center)

            gsparams = {}
            if 'gsparams' in config['image']:
                gsparams = galsim.config.UpdateGSParams(
                    gsparams, config['image']['gsparams'], config)

            skip = False
            try :
                t4=t3=t2=t1  # in case we throw.
        
                psf = galsim.config.BuildGSObject(config, 'psf', config, gsparams, logger)[0]
                t2 = time.time()

                gal = galsim.config.BuildGSObject(config, 'gal', config, gsparams, logger)[0]
                t4 = time.time()

                # Check that we have at least gal or psf.
                if not (gal or psf):
                    raise AttributeError("At least one of gal or psf must be specified in config.")

            except galsim.config.gsobject.SkipThisObject, e:
                if logger and logger.isEnabledFor(logging.DEBUG):
                    logger.debug('obj %d: Caught SkipThisObject: e = %s',obj_num,e.msg)
                if logger and logger.isEnabledFor(logging.INFO):
                    if e.msg:
                        # If there is a message, upgrade to info level
                        logger.info('Skipping object %d: %s',obj_num,e.msg)
                skip = True

            offset = config['stamp_offset']
            if not skip and 'offset' in config['image']:
                offset1 = galsim.config.ParseValue(config['image'], 'offset', config,
                                                   galsim.PositionD)[0]
                offset += offset1

            if 'image' in config and 'draw_method' in config['image']:
                method = galsim.config.ParseValue(config['image'],'draw_method',config,str)[0]
            else:
                method = 'auto'
            if method not in ['auto', 'fft', 'phot', 'real_space', 'no_pixel', 'sb']:
                raise AttributeError("Invalid draw_method: %s"%method)

            if skip: 
                if xsize and ysize:
                    # If the size is set, we need to do something reasonable to return this size.
                    im = galsim.ImageF(xsize, ysize)
                    im.setOrigin(config['image_origin'])
                    im.setZero()
                    if do_noise:
                        galsim.config.AddSky(config,im)
                else:
                    # Otherwise, we don't set the bounds, so it will be noticed as invalid upstream.
                    im = galsim.ImageF()

                if make_weight_image:
                    weight_im = galsim.ImageF(im.bounds, wcs=im.wcs)
                    weight_im.setZero()
                else:
                    weight_im = None
                current_var = 0

            else:
                im, current_var = DrawStamp(psf,gal,config,xsize,ysize,offset,method,logger)

                # Set the origin appropriately
                if stamp_center:
                    im.setCenter(stamp_center)
                else:
                    im.setOrigin(config['image_origin'])

                if make_weight_image:
                    weight_im = galsim.ImageF(im.bounds, wcs=im.wcs)
                    weight_im.setZero()
                else:
                    weight_im = None
                if do_noise:
                    # The default indexing for the noise is image_num, not obj_num
                    config['index_key'] = 'image_num'
                    galsim.config.AddSky(config,im)
                    galsim.config.AddNoise(config,im,weight_im,current_var,logger)
                    config['index_key'] = 'obj_num'

            if make_badpix_image:
                badpix_im = galsim.ImageS(im.bounds, wcs=im.wcs)
                badpix_im.setZero()
            else:
                badpix_im = None

            t5 = time.time()

            if make_psf_image:
                psf_im = DrawPSFStamp(psf,config,im.bounds,offset,method)
                if ('output' in config and 'psf' in config['output'] and 
                        'signal_to_noise' in config['output']['psf'] and
                        'noise' in config['image']):
                    config['index_key'] = 'image_num'
                    galsim.config.AddNoise(config,psf_im,None,0,logger)
                    config['index_key'] = 'obj_num'
            else:
                psf_im = None

            t6 = time.time()

            if logger and logger.isEnabledFor(logging.DEBUG):
                logger.debug('obj %d: Times: %f, %f, %f, %f, %f',
                             obj_num, t2-t1, t3-t2, t4-t3, t5-t4, t6-t5)

        except Exception as e:

            if itry == ntries-1:
                # Then this was the last try.  Just re-raise the exception.
                raise
            else:
                if logger and logger.isEnabledFor(logging.INFO):
                    logger.info('Object %d: Caught exception %s',obj_num,str(e))
                    logger.info('This is try %d/%d, so trying again.',itry+1,ntries)
                # Need to remove the "current_val"s from the config dict.  Otherwise,
                # the value generators will do a quick return with the cached value.
                galsim.config.RemoveCurrent(config, keep_safe=True)
                continue

    return im, psf_im, weight_im, badpix_im, current_var, t6-t1


def DrawStamp(psf, gal, config, xsize, ysize, offset, method, logger):
    """
    Draw an image using the given psf and gal profiles (which may be None)
    using the FFT method for doing the convolution.

    @returns the resulting image.
    """

    # Setup the object to draw:
    prof_list = [ prof for prof in (gal,psf) if prof is not None ]
    assert len(prof_list) > 0  # Should have already been checked.
    if len(prof_list) > 1:
        final = galsim.Convolve(prof_list)
    else:
        final = prof_list[0]

    # Setup the kwargs to pass to drawImage
    kwargs = {}
    if xsize:
        kwargs['image'] = galsim.ImageF(xsize, ysize)
    kwargs['offset'] = offset
    kwargs['method'] = method
    if 'image' in config and 'wmult' in config['image']:
        kwargs['wmult'] = galsim.config.ParseValue(config['image'], 'wmult', config, float)[0]
    kwargs['wcs'] = config['wcs'].local(image_pos = config['image_pos'])
    if method == 'phot':
        kwargs['rng'] = config['rng']

    # Check validity of extra phot options:
    max_extra_noise = None
    if 'image' in config and 'n_photons' in config['image']:
        if method != 'phot':
            raise AttributeError('n_photons is invalid with method != phot')
        if 'max_extra_noise' in config['image']:
            if logger and logger.isEnabledFor(logging.WARN):
                logger.warn(
                    "Both 'max_extra_noise' and 'n_photons' are set in config['image'], "+
                    "ignoring 'max_extra_noise'.")
        kwargs['n_photons'] = galsim.config.ParseValue(config['image'], 'n_photons', config, int)[0]
    elif 'image' in config and 'max_extra_noise' in config['image']:
        if method != 'phot':
            raise AttributeError('max_extra_noise is invalid with method != phot')
        max_extra_noise = galsim.config.ParseValue(
            config['image'], 'max_extra_noise', config, float)[0]
    elif method == 'phot':
        max_extra_noise = 0.01

    if 'image' in config and 'poisson_flux' in config['image']:
        if method != 'phot':
            raise AttributeError('poisson_flux is invalid with method != phot')
        kwargs['poisson_flux'] = galsim.config.ParseValue(
                config['image'], 'poisson_flux', config, bool)[0]

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

    im = final.drawImage(**kwargs)
    im.setOrigin(config['image_origin'])

    # If the object has a noise attribute, then check if we need to do anything with it.
    current_var = 0.  # Default if not overwritten
    if hasattr(final,'noise'):
        if 'noise' in config['image']:
            noise = config['image']['noise']
            if 'whiten' in noise:
                if 'symmetrize' in noise:
                    raise AttributeError('Only one of whiten or symmetrize is allowed')
                whiten, safe = galsim.config.ParseValue(noise, 'whiten', config, bool)
                current_var = final.noise.whitenImage(im)
                if logger and logger.isEnabledFor(logging.DEBUG):
                    logger.debug('obj %d: whitening noise brought current var to %f',
                                 config['obj_num'],current_var)

            elif 'symmetrize' in noise:
                symmetrize, safe = galsim.config.ParseValue(noise, 'symmetrize', config, int)
                current_var = final.noise.symmetrizeImage(im, symmetrize)
                if logger and logger.isEnabledFor(logging.DEBUG):
                    logger.debug('obj %d: symmetrizing noise brought current var to %f',
                                 config['obj_num'],current_var)

    if (('gal' in config and 'signal_to_noise' in config['gal']) or
        ('gal' not in config and 'psf' in config and 'signal_to_noise' in config['psf'])):
        if method == 'phot':
            raise NotImplementedError(
                "signal_to_noise option not implemented for draw_method = phot")
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

        sn_meas = math.sqrt( numpy.sum(im.array**2) / noise_var )
        # Now we rescale the flux to get our desired S/N
        flux = sn_target / sn_meas
        im *= flux
        if hasattr(final,'noise'):
            current_var *= flux**2

    return im, current_var

