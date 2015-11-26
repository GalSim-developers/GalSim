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

valid_noise_types = { 
    # The values in the tuple are:
    # - A function to add noise after drawing
    # - A function that returns the variance of the noise
    'Gaussian' : ('AddNoiseGaussian', 'NoiseVarGaussian'),
    'Poisson' : ('AddNoisePoisson', 'NoiseVarPoisson'),
    'CCD' : ('AddNoiseCCD', 'NoiseVarCCD'),
    'COSMOS' : ('AddNoiseCOSMOS', 'NoiseVarCOSMOS'),
}

# items that are parsed separately from the normal noise function
noise_ignore = [ 'whiten', 'symmetrize' ]

def _get_sky(config, base, wcs=None):

    if 'sky_level' in config:
        if 'sky_level_pixel' in config:
            raise AttributeError("Cannot specify both sky_level and sky_level_pixel")
        sky_level = galsim.config.ParseValue(config,'sky_level',base,float)[0]
        if not wcs:
            wcs = config['wcs']
        if wcs.isUniform():
            return sky_level * wcs.pixelArea()
        elif 'image_pos' in base:
            return sky_level * wcs.pixelArea(base['image_pos'])
        else:
            # We should never get to this point unless im != None.
            sky = galsim.Image(im.bounds, wcs=wcs)
            wcs.makeSkyImage(sky, sky_level)
            return sky
    elif 'sky_level_pixel' in config:
        sky_level_pixel = galsim.config.ParseValue(config,'sky_level_pixel',base,float)[0]
        return sky_level_pixel
    else:
        return 0.


#
# First the driver functions:
#

def AddNoise(config, draw_method, im, weight_im, current_var, logger, add_sky=True):
    """
    Add noise to an image according to the noise specifications in the noise dict
    appropriate for an image that has been drawn using the specified method.
    """
    if 'noise' in config['image']:
        noise = config['image']['noise']
        if not isinstance(noise, dict):
            raise AttributeError("image.noise is not a dict.")
    else:
        # No noise.  Equivalent to draw_method = skip.
        draw_method = 'skip'
    rng = config['rng']

    # Add the overall sky level, if desired
    if add_sky:
        sky = _get_sky(config['image'], config, wcs=im.wcs)
        im += sky
    else:
        sky = 0.

    # Add the noise specified
    if draw_method is not 'skip':

        if 'type' in noise:
            type = noise['type']
        else:
            type = 'Poisson'  # Default is Poisson
        if type not in valid_noise_types:
            raise AttributeError("Invalid type %s for noise",type)

        noise_func = eval(valid_noise_types[type][0])
        noise_func(noise, config, draw_method, rng, im, weight_im, current_var, sky, logger)


def CalculateNoiseVar(config):
    """
    Calculate the noise variance from the noise specified in the noise dict.
    """
    noise = config['image']['noise']
    if not isinstance(noise, dict):
        raise AttributeError("image.noise is not a dict.")

    if 'type' in noise:
        type = noise['type']
    else:
        type = 'Poisson'  # Default is Poisson
    if type not in valid_noise_types:
        raise AttributeError("Invalid type %s for noise",type)

    noisevar_func = eval(valid_noise_types[type][1])
    return noisevar_func(noise, config)

#
# Gaussian
#

def AddNoiseGaussian(noise, config, draw_method, rng, im, weight_im, current_var, sky, logger):
    # NB: Identical for fft and phot

    # The noise level can be specified either as a sigma or a variance.  Here we just calculate
    # the value of the variance from either one.
    single = [ { 'sigma' : float , 'variance' : float } ]
    params = galsim.config.GetAllParams(noise, 'noise', config, single=single,
                                        ignore=noise_ignore)[0]
    if 'sigma' in params:
        sigma = params['sigma']
        var = sigma**2
    else:
        var = params['variance']

    # If we are saving the noise level in a weight image, do that now.
    if weight_im:
        weight_im += var

    # If we already have some variance in the image (from whitening), then we subtract this much
    # from sigma**2.
    if current_var: 
        if logger and logger.isEnabledFor(logging.DEBUG):
            logger.debug('image %d, obj %d: Target variance is %f, current variance is %f',
                         config['image_num'],config['obj_num'],var,current_var)
        if var < current_var:
            raise RuntimeError(
                "Whitening already added more noise than requested Gaussian noise.")
        var -= current_var

    # Now apply the noise.
    import math
    sigma = math.sqrt(var)
    im.addNoise(galsim.GaussianNoise(rng,sigma=sigma))

    if logger and logger.isEnabledFor(logging.DEBUG):
        logger.debug('image %d, obj %d: Added Gaussian noise with sigma = %f',
                     config['image_num'],config['obj_num'],sigma)

def NoiseVarGaussian(noise, config):

    # The noise variance is just sigma^2 or variance
    single = [ { 'sigma' : float , 'variance' : float } ]
    params = galsim.config.GetAllParams(noise, 'noise', config, single=single,
                                        ignore=noise_ignore)[0]
    if 'sigma' in params:
        sigma = params['sigma']
        return sigma * sigma
    else:
        return params['variance']


#
# Poisson
#

def AddNoisePoisson(noise, config, draw_method, rng, im, weight_im, current_var, sky, logger):

    # Get how much extra sky to assume from the image.noise attribute.
    if sky:
        opt = { 'sky_level' : float, 'sky_level_pixel' : float }
        single = []
    else:
        opt = {}
        single = [ { 'sky_level' : float , 'sky_level_pixel' : float } ]
    params = galsim.config.GetAllParams(noise, 'noise', config, opt=opt, single=single,
                                        ignore=noise_ignore)[0]
    if 'sky_level' in params:
        if 'sky_level_pixel' in params:
            raise AttributeError("Only one of sky_level and sky_level_pixel is allowed for "
                                 "noise.type = Poisson")
        sky_level = params['sky_level']
        if im.wcs.isUniform():
            extra_sky = sky_level * im.wcs.pixelArea()
        elif 'image_pos' in config:
            extra_sky = sky_level * im.wcs.pixelArea(config['image_pos'])
        else:
            extra_sky = galsim.Image(im.bounds, wcs=im.wcs)
            im.wcs.makeSkyImage(extra_sky, sky_level)
    elif 'sky_level_pixel' in params:
        extra_sky = params['sky_level_pixel']
    else:
        extra_sky = 0.

    # If we are saving the noise level in a weight image, do that now.
    if weight_im:
        # Check if a weight image should include the object variance.
        # Note: For the phot case, we don't actually have an exact value for the variance in each 
        # pixel, but the drawn image before adding the Poisson noise is our best guess for the 
        # variance from the object's flux, so if we want the object variance included, this is 
        # still the best we can do.
        include_obj_var = False
        if ('output' in config and 'weight' in config['output'] and 
            'include_obj_var' in config['output']['weight']):
            include_obj_var = galsim.config.ParseValue(
                config['output']['weight'], 'include_obj_var', config, bool)[0]
        if include_obj_var:
            # The image right now has the object variance in each pixel.  So before going on with 
            # the noise, copy these over to the weight image.  (We invert this later...)
            weight_im.copyFrom(im)
        else:
            # Otherwise, just add in the current sky noise:
            if sky: weight_im += sky
        # And add in the extra sky noise:
        if extra_sky: weight_im += extra_sky

    # If we already have some variance in the image (from whitening), then we subtract this much
    # off of the sky level.  It's not precisely accurate, since the existing variance is Gaussian,
    # rather than Poisson, but it's the best we can do.
    if current_var:
        if logger and logger.isEnabledFor(logging.DEBUG):
            logger.debug('image %d, obj %d: Target variance is %f, current variance is %f',
                         config['image_num'],config['obj_num'],extra_sky, current_var)
        if isinstance(sky, galsim.Image) or isinstance(extra_sky, galsim.Image):
            test = ((sky+extra_sky).image.array < current_var).any()
        else:
            test = (sky+extra_sky < current_var)
        if test:
            raise RuntimeError(
                "Whitening already added more noise than requested Poisson noise.")
        extra_sky -= current_var

    # At this point, there is a slight difference between fft and phot. For photon shooting, the 
    # galaxy already has Poisson noise, so we want to make sure not to add that again!
    if draw_method == 'phot':
        # Only add in the noise from the sky.
        if isinstance(sky, galsim.Image) or isinstance(extra_sky, galsim.Image):
            noise_im = sky + extra_sky
            noise_im.addNoise(galsim.PoissonNoise(rng))
            if sky:
                noise_im -= sky
            if extra_sky:
                noise_im -= extra_sky
            # noise_im should now have zero mean, but with the noise of the total sky level.
            im += noise_im
        else:
            total_sky = sky + extra_sky
            if total_sky > 0.:
                im.addNoise(galsim.DeviateNoise(galsim.PoissonDeviate(rng, mean=total_sky)))
                # This deviate adds a noisy version of the sky, so need to subtract the mean back 
                # off.
                im -= total_sky
    else:
        im += extra_sky
        # Do the normal PoissonNoise calculation.
        im.addNoise(galsim.PoissonNoise(rng))
        im -= extra_sky

    if logger and logger.isEnabledFor(logging.DEBUG):
        logger.debug('image %d, obj %d: Added Poisson noise with sky = %f',
                     config['image_num'],config['obj_num'],sky)


def NoiseVarPoisson(noise, config):
    # The noise variance is the net sky level per pixel

    # Start with the background sky level for the image
    sky = _get_sky(config['image'], config)

    # And add in any extra sky level request for the noise
    if sky:
        opt = { 'sky_level' : float, 'sky_level_pixel' : float }
        single = []
    else:
        opt = {}
        single = [ { 'sky_level' : float , 'sky_level_pixel' : float } ]
    params = galsim.config.GetAllParams(noise, 'noise', config, opt=opt, single=single,
                                        ignore=noise_ignore)[0]
    if 'sky_level' in params:
        if 'sky_level_pixel' in params:
            raise AttributeError("Only one of sky_level and sky_level_pixel is allowed for "
                                 "noise.type = Poisson")
        sky += params['sky_level'] * config['wcs'].pixelArea(config['image_pos'])
    elif 'sky_level_pixel' in params:
        sky += params['sky_level_pixel']

    return sky


#
# CCD
#

def AddNoiseCCD(noise, config, draw_method, rng, im, weight_im, current_var, sky, logger):

    # This process goes a lot like the Poisson routine.  There are just two differences.
    # The Poisson noise is in the electron, not ADU, and now we allow for a gain = e-/ADU,
    # so we need to account for that properly.  And we also allow for an additional Gaussian
    # read noise.

    # Get how much extra sky to assume from the image.noise attribute.
    opt = { 'gain' : float , 'read_noise' : float }
    # The noise sky_level is only required here if the image doesn't have any.
    if sky:
        opt['sky_level'] = float
        opt['sky_level_pixel'] = float
        single = []
    else:
        single = [ { 'sky_level' : float , 'sky_level_pixel' : float } ]
    params = galsim.config.GetAllParams(noise, 'noise', config, opt=opt, single=single,
                                        ignore=noise_ignore)[0]
    gain = params.get('gain',1.0)
    read_noise = params.get('read_noise',0.0)
    read_noise_var = read_noise**2
    if 'sky_level' in params:
        if 'sky_level_pixel' in params:
            raise AttributeError("Only one of sky_level and sky_level_pixel is allowed for "
                                 "noise.type = CCD")
        sky_level = params['sky_level']
        if im.wcs.isUniform():
            extra_sky = sky_level * im.wcs.pixelArea()
        elif 'image_pos' in config:
            extra_sky = sky_level * im.wcs.pixelArea(config['image_pos'])
        else:
            extra_sky = galsim.Image(im.bounds, wcs=im.wcs)
            im.wcs.makeSkyImage(extra_sky, sky_level)
    elif 'sky_level_pixel' in params:
        extra_sky = params['sky_level_pixel']
    else:
        extra_sky = 0.

    # If we are saving the noise level in a weight image, do that now.
    if weight_im:
        # Check if a weight image should include the object variance.
        # Note: For the phot case, we don't actually have an exact value for the variance in each 
        # pixel, but the drawn image before adding the Poisson noise is our best guess for the 
        # variance from the object's flux, so if we want the object variance included, this is 
        # still the best we can do.
        include_obj_var = False
        if ('output' in config and 'weight' in config['output'] and 
            'include_obj_var' in config['output']['weight']):
            include_obj_var = galsim.config.ParseValue(
                config['output']['weight'], 'include_obj_var', config, bool)[0]
        if include_obj_var:
            # The image right now has the object variance in each pixel.  So before going on with 
            # the noise, copy these over to the weight image.  (We invert this later...)
            weight_im.copyFrom(im)

            # Account for the gain and read noise
            if gain != 1.0:
                import math
                weight_im /= math.sqrt(gain)
            if read_noise != 0.0:
                weight_im += read_noise_var
        else:
            # Otherwise, just add in the current sky noise:
            if sky or read_noise != 0.0:
                weight_im += sky / gain + read_noise_var

        # And add in the extra sky noise:
        if extra_sky: weight_im += extra_sky

    # If we already have some variance in the image (from whitening), then we try to subtract it 
    # from the read noise if possible.  If now, we subtract the rest off of the sky level.  It's 
    # not precisely accurate, since the existing variance is Gaussian, rather than Poisson, but 
    # it's the best we can do.
    if current_var:
        if logger and logger.isEnabledFor(logging.DEBUG):
            logger.debug('image %d, obj %d: Target variance is %f, current variance is %f',
                         config['image_num'],config['obj_num'],
                         read_noise_var+extra_sky, current_var)
        if isinstance(sky, galsim.Image) or isinstance(extra_sky, galsim.Image):
            test = ((sky+extra_sky).image.array/gain + read_noise_var < current_var).any()
        else:
            test = (sky+extra_sky) / gain + read_noise_var < current_var
        if test:
            raise RuntimeError(
                "Whitening already added more noise than requested CCD noise.")
        if read_noise_var >= current_var:
            # First try to take away from the read_noise, since this one is actually Gaussian.
            import math
            read_noise_var -= current_var
            read_noise = math.sqrt(read_noise_var)
        else:
            # Take read_noise down to zero, since already have at least that much already.
            current_var -= read_noise_var
            read_noise = 0
            read_noise_var = 0
            # Take the rest away from the sky level
            extra_sky -= current_var * gain

    # At this point, there is a slight difference between fft and phot. For photon shooting, the 
    # galaxy already has Poisson noise, so we want to make sure not to add that again!
    if draw_method == 'phot':
        # Add in the noise from the sky.
        if isinstance(sky, galsim.Image) or isinstance(extra_sky, galsim.Image):
            noise_im = sky + extra_sky
            if gain != 1.0: noise_im *= gain
            noise_im.addNoise(galsim.PoissonNoise(rng))
            if gain != 1.0: noise_im /= gain
            if sky:
                noise_im -= sky
            if extra_sky:
                noise_im -= extra_sky
            # noise_im should now have zero mean, but with the noise of the total sky level.
            im += noise_im
        else:
            total_sky = sky + extra_sky
            if total_sky > 0.:
                if gain != 1.0: im *= gain
                im.addNoise(galsim.DeviateNoise(galsim.PoissonDeviate(rng, mean=total_sky*gain)))
                if gain != 1.0: im /= gain
                im -= total_sky
        # And add the read noise
        if read_noise != 0.:
            im.addNoise(galsim.GaussianNoise(rng, sigma=read_noise))
    else:
        # Do the normal CCDNoise calculation.
        im += extra_sky
        im.addNoise(galsim.CCDNoise(rng, gain=gain, read_noise=read_noise))
        im -= extra_sky

    if logger and logger.isEnabledFor(logging.DEBUG):
        logger.debug('image %d, obj %d: Added CCD noise with sky = %f, ' +
                     'gain = %f, read_noise = %f',
                     config['image_num'],config['obj_num'],sky,gain,read_noise)

def NoiseVarCCD(noise, config):
    # The noise variance is sky / gain + read_noise^2

    # Start with the background sky level for the image
    sky = _get_sky(config['image'], config)

    # Add in any extra sky level request for the noise
    opt = { 'gain' : float , 'read_noise' : float }
    if sky:
        opt['sky_level'] = float
        opt['sky_level_pixel'] = float
        single = []
    else:
        single = [ { 'sky_level' : float , 'sky_level_pixel' : float } ]
    params = galsim.config.GetAllParams(noise, 'noise', config, opt=opt, single=single,
                                        ignore=noise_ignore)[0]
    if 'sky_level' in params:
        if 'sky_level_pixel' in params:
            raise AttributeError("Only one of sky_level and sky_level_pixel is allowed for "
                                 "noise.type = CCD")
        sky += params['sky_level'] * config['wcs'].pixelArea(config['image_pos'])
    elif 'sky_level_pixel' in params:
        sky += params['sky_level_pixel']

    # Account for the gain and read_noise
    gain = params.get('gain',1.0)
    read_noise = params.get('read_noise',0.0)
    return sky / gain + read_noise * read_noise

#
# COSMOS
#

def AddNoiseCOSMOS(noise, config, draw_method, rng, im, weight_im, current_var, sky, logger):
    # NB: Identical for fft and phot

    req = { 'file_name' : str }
    opt = { 'cosmos_scale' : float, 'variance' : float }
        
    kwargs = galsim.config.GetAllParams(noise, 'noise', config, req=req, opt=opt,
                                        ignore=noise_ignore)[0]

    # Build the correlated noise 
    cn = galsim.correlatednoise.getCOSMOSNoise(rng, **kwargs)
    var = cn.getVariance()

    # If we are saving the noise level in a weight image, do that now.
    if weight_im: 
        weight_im += var

    # Subtract off the current variance if any
    if current_var:
        if logger and logger.isEnabledFor(logging.DEBUG):
            logger.debug('image %d, obj %d: Target variance is %f, current variance is %f',
                         config['image_num'],config['obj_num'], var, current_var)
        if var < current_var:
            raise RuntimeError(
                "Whitening already added more noise than requested COSMOS noise.")
        cn -= galsim.UncorrelatedNoise(rng, im.wcs, current_var)

    # Add the noise to the image
    im.addNoise(cn)

    if logger and logger.isEnabledFor(logging.DEBUG):
        logger.debug('image %d, obj %d: Added COSMOS correlated noise with variance = %f',
                     config['image_num'],config['obj_num'],var)

def NoiseVarCOSMOS(noise, config):
    # The variance is given by the getVariance function.

    req = { 'file_name' : str }
    opt = { 'cosmos_scale' : float, 'variance' : float }
    kwargs = galsim.config.GetAllParams(noise, 'noise', config, req=req, opt=opt,
                                        ignore=noise_ignore)[0]

    # Build and add the correlated noise (lets the cn internals handle dealing with the options
    # for default variance: quick and ensures we don't needlessly duplicate code) 
    # Note: the rng being passed here is arbitrary, since we don't need it to calculate the
    # variance.  Building a BaseDeviate with a particular seed is the fastest option.
    cn = galsim.correlatednoise.getCOSMOSNoise(galsim.BaseDeviate(123), **kwargs)

    # zero distance correlation function value returned as variance
    return cn.getVariance()


