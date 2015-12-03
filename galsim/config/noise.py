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

# This file handles the functionality for adding noise and the sky to an image after
# drawing the objects.


#
# First the driver functions:
#

def AddSky(config, im):
    """Add the sky level to the image
    """
    sky = GetSky(config['image'], config)
    if sky:
        im += sky


def AddNoise(config, im, current_var=0., logger=None):
    """
    Add noise to an image according to the noise specifications in the noise dict.

    @param config           The configuration dict
    @param im               The image onto which to add the noise
    @param current_var      The current noise variance present in the image already [default: 0]
    @param logger           If given, a logger object to log progress. [default: None]
    """
    if 'noise' in config['image']:
        noise = config['image']['noise']
    else:
        # No noise.
        return

    if 'type' in noise:
        type = noise['type']
    else:
        type = 'Poisson'  # Default is Poisson
    if type not in valid_noise_types:
        raise AttributeError("Invalid type %s for noise",type)

    func = valid_noise_types[type][0]
    func(noise, config, im, current_var, logger)

def CalculateNoiseVar(config):
    """
    Calculate the noise variance from the noise specified in the noise dict.

    @param config           The configuration dict

    @returns the noise variance
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

    func = valid_noise_types[type][1]
    return func(noise, config)

def AddNoiseVariance(config, im, include_obj_var=False, logger=None):
    """
    Add the noise variance to an image according to the noise specifications in the noise dict.
    Typically, this is used for buliding a weight map, which is typically the inverse variance.

    @param config           The configuration dict
    @param im               The image onto which to add the variance values
    @param include_obj_var  Whether to add the variance from the object photons for noise
                            models that have a component based on the number of photons.
                            [default: False]
    @param logger           If given, a logger object to log progress. [default: None]
    """
    if 'noise' in config['image']:
        noise = config['image']['noise']
    else:
        # No noise.
        return

    if 'type' in noise:
        type = noise['type']
    else:
        type = 'Poisson'  # Default is Poisson
    if type not in valid_noise_types:
        raise AttributeError("Invalid type %s for noise",type)

    func = valid_noise_types[type][2]
    func(noise, config, im, include_obj_var, logger)


def GetSky(config, base):
    """Parse the sky information and return either a float value for the sky level per pixel
    or an image, as needed.
    
    If an image is required (because wcs is not uniform) then it will use the presence of
    base['image_pos'] to determine what size image to return (stamp or full).  If there is
    a current image_pos, then we are doing a stamp.  Otherwise a full image.
    """
    if 'sky_level' in config:
        if 'sky_level_pixel' in config:
            raise AttributeError("Cannot specify both sky_level and sky_level_pixel")
        sky_level = galsim.config.ParseValue(config,'sky_level',base,float)[0]
        wcs = base['wcs']
        if wcs.isUniform():
            return sky_level * wcs.pixelArea()
        elif 'image_pos' in base:
            return sky_level * wcs.pixelArea(base['image_pos'])
        else:
            # This calculation is non-trivial, so store this in case we need it again.
            tag = (base['file_num'], base['image_num'])
            if config.get('current_sky_tag',None) == tag:
                return config['current_sky']
            else:
                bounds = base['current_image'].bounds
                sky = galsim.Image(bounds, wcs=wcs)
                wcs.makeSkyImage(sky, sky_level)
                config['current_sky_tag'] = tag
                config['current_sky'] = sky
                return sky
    elif 'sky_level_pixel' in config:
        sky_level_pixel = galsim.config.ParseValue(config,'sky_level_pixel',base,float)[0]
        return sky_level_pixel
    else:
        return 0.


# items that are parsed separately from the normal noise function
noise_ignore = [ 'whiten', 'symmetrize' ]

#
# Gaussian
#

def AddNoiseGaussian(config, base, im, current_var, logger):

    var = NoiseVarGaussian(config, base)

    # If we already have some variance in the image (from whitening), then we subtract this much
    # from sigma**2.
    if current_var: 
        if logger and logger.isEnabledFor(logging.DEBUG):
            logger.debug('image %d, obj %d: Target variance is %f, current variance is %f',
                         base['image_num'],base['obj_num'],var,current_var)
        if var < current_var:
            raise RuntimeError(
                "Whitening already added more noise than the requested Gaussian noise.")
        var -= current_var

    # Now apply the noise.
    import math
    sigma = math.sqrt(var)
    rng = base['rng']
    im.addNoise(galsim.GaussianNoise(rng,sigma=sigma))

    if logger and logger.isEnabledFor(logging.DEBUG):
        logger.debug('image %d, obj %d: Added Gaussian noise with var = %f',
                     base['image_num'],base['obj_num'],var)

def NoiseVarGaussian(config, base):

    # The noise level can be specified either as a sigma or a variance.  Here we just calculate
    # the value of the variance from either one.
    single = [ { 'sigma' : float , 'variance' : float } ]
    params = galsim.config.GetAllParams(config, base, single=single, ignore=noise_ignore)[0]
    if 'sigma' in params:
        sigma = params['sigma']
        return sigma * sigma
    else:
        return params['variance']

def AddNoiseVarianceGaussian(config, base, im, include_obj_var, logger):
    im += NoiseVarGaussian(config, base)


#
# Poisson
#

def AddNoisePoisson(config, base, im, current_var, logger):

    # Get how much extra sky to assume from the image.noise attribute.
    sky = GetSky(base['image'], base)
    extra_sky = GetSky(config, base)
    if not sky and not extra_sky:
        raise AttributeError(
            "Must provide either sky_level or sky_level_pixel for noise.type = Poisson")

    # If we already have some variance in the image (from whitening), then we subtract this much
    # off of the sky level.  It's not precisely accurate, since the existing variance is Gaussian,
    # rather than Poisson, but it's the best we can do.
    if current_var:
        if logger and logger.isEnabledFor(logging.DEBUG):
            logger.debug('image %d, obj %d: Target variance is %f, current variance is %f',
                         base['image_num'],base['obj_num'],extra_sky, current_var)
        if isinstance(sky, galsim.Image) or isinstance(extra_sky, galsim.Image):
            test = ((sky+extra_sky).image.array < current_var).any()
        else:
            test = (sky+extra_sky < current_var)
        if test:
            raise RuntimeError(
                "Whitening already added more noise than the requested Poisson noise.")
        extra_sky -= current_var

    # At this point, there is a slight difference between fft and phot. For photon shooting, the 
    # galaxy already has Poisson noise, so we want to make sure not to add that again!
    draw_method = galsim.config.GetCurrentValue('image.draw_method',base,str)
    rng = base['rng']
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
        logger.debug('image %d, obj %d: Added Poisson noise', base['image_num'],base['obj_num'])


def NoiseVarPoisson(config, base):
    # The noise variance is the net sky level per pixel

    # Start with the background sky level for the image
    sky = GetSky(base['image'], base)
    sky += GetSky(config, base)
    return sky


def AddNoiseVariancePoisson(config, base, im, include_obj_var, logger):
    if include_obj_var:
        # The current image at this point should be the noise-free, sky-free image,
        # which is the object variance in each pixel.
        im += base['current_image']

        # Note: For the phot case, we don't actually have an exact value for the variance in each 
        # pixel, but the drawn image before adding the Poisson noise is our best guess for the 
        # variance from the object's flux, so if we want the object variance included, this is 
        # still the best we can do.

    # Add the total sky level
    im += NoiseVarPoisson(config, base)


#
# CCD
#

def _GetCCDNoiseParams(config, base):
    opt = { 'gain' : float , 'read_noise' : float }
    ignore = ['sky_level', 'sky_level_pixel']
    params = galsim.config.GetAllParams(config, base, opt=opt, ignore=noise_ignore + ignore)[0]
    gain = params.get('gain',1.0)
    read_noise = params.get('read_noise',0.0)
    read_noise_var = read_noise**2

    return gain, read_noise, read_noise_var

def AddNoiseCCD(config, base, im, current_var, logger):

    # This process goes a lot like the Poisson routine.  There are just two differences.
    # First, the Poisson noise is in electrons, not ADU, and now we allow for a gain = e-/ADU,
    # so we need to account for that properly.  Second, we also allow for an additional Gaussian
    # read noise.
    gain, read_noise, read_noise_var = _GetCCDNoiseParams(config, base)

    # Get how much extra sky to assume from the image.noise attribute.
    sky = GetSky(base['image'], base)
    extra_sky = GetSky(config, base)
    if not sky and not extra_sky:
        raise AttributeError(
            "Must provide either sky_level or sky_level_pixel for noise.type = Poisson")

    # If we already have some variance in the image (from whitening), then we try to subtract it 
    # from the read noise if possible.  If now, we subtract the rest off of the sky level.  It's 
    # not precisely accurate, since the existing variance is Gaussian, rather than Poisson, but 
    # it's the best we can do.
    if current_var:
        if logger and logger.isEnabledFor(logging.DEBUG):
            logger.debug('image %d, obj %d: Target variance is %f, current variance is %f',
                         base['image_num'],base['obj_num'],
                         read_noise_var+extra_sky, current_var)
        if isinstance(sky, galsim.Image) or isinstance(extra_sky, galsim.Image):
            test = ((sky+extra_sky).image.array/gain + read_noise_var < current_var).any()
        else:
            test = (sky+extra_sky) / gain + read_noise_var < current_var
        if test:
            raise RuntimeError(
                "Whitening already added more noise than the requested CCD noise.")
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
    draw_method = galsim.config.GetCurrentValue('image.draw_method',base,str)
    rng = base['rng']
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
        logger.debug('image %d, obj %d: Added CCD noise with gain = %f, read_noise = %f',
                     base['image_num'],base['obj_num'],sky,gain,read_noise)

def NoiseVarCCD(config, base):
    # The noise variance is sky / gain + read_noise^2
    gain, read_noise, read_noise_var = _GetCCDNoiseParams(config, base)

    # Start with the background sky level for the image
    sky = GetSky(base['image'], base)
    sky += GetSky(config, base)

    # Account for the gain and read_noise
    return sky / gain + read_noise_var

def AddNoiseVarianceCCD(config, base, im, include_obj_var, logger):
    gain, read_noise, read_noise_var = _GetCCDNoiseParams(config, base)
    if include_obj_var:
        # The current image at this point should be the noise-free, sky-free image,
        # which is the object variance in each pixel.
        im += base['current_image']

        # Account for the gain and read noise
        if gain != 1.0:
            import math
            im /= math.sqrt(gain)
        if read_noise_var != 0.0:
            im += read_noise_var

    # Otherwise, just add in the current sky noise and read noise:
    sky = GetSky(base['image'], base)
    sky += GetSky(config, base)

    if sky or read_noise_var != 0.0:
        im += sky / gain + read_noise_var


#
# COSMOS
#

def _GetCOSMOSNoise(config, base):
    # Save the constructed CorrelatedNoise object, since we might need it again.
    tag = (base['file_num'], base['image_num'])
    if config.get('current_cn_tag',None) == tag:
        return config['current_cn']
    else:
        req = { 'file_name' : str }
        opt = { 'cosmos_scale' : float, 'variance' : float }
        
        kwargs = galsim.config.GetAllParams(config, base, req=req, opt=opt, ignore=noise_ignore)[0]
        rng = base['rng']
        cn = galsim.correlatednoise.getCOSMOSNoise(rng, **kwargs)
        config['current_cn'] = cn
        config['current_cn_tag'] = tag
        return cn

def AddNoiseCOSMOS(config, base, im, current_var, logger):

    # Build the correlated noise 
    cn = _GetCOSMOSNoise(config,base)
    var = cn.getVariance()

    # Subtract off the current variance if any
    if current_var:
        if logger and logger.isEnabledFor(logging.DEBUG):
            logger.debug('image %d, obj %d: Target variance is %f, current variance is %f',
                         base['image_num'],base['obj_num'], var, current_var)
        if var < current_var:
            raise RuntimeError(
                "Whitening already added more noise than the requested COSMOS noise.")
        cn -= galsim.UncorrelatedNoise(rng, im.wcs, current_var)

    # Add the noise to the image
    im.addNoise(cn)

    if logger and logger.isEnabledFor(logging.DEBUG):
        logger.debug('image %d, obj %d: Added COSMOS correlated noise with variance = %f',
                     base['image_num'],base['obj_num'],var)

def NoiseVarCOSMOS(config, base):
    cn = _GetCOSMOSNoise(config,base)
    return cn.getVariance()

def AddNoiseVarianceCOSMOS(config, base, im, include_obj_var, logger):
    im += NoiseVarCOSMOS(config, base)



# valid_noise_type is a dict that defines how to process each noise type.
# The values in the tuple are:
# - A function to add noise to an image
#   The call signature is AddNoise(config, base, im, current_var, logger)
# - A function that returns the variance of the noise
#   The call signature is NoiseVar(config, base)
# - A function to add the noise variance to an image
#   The call signature is AddNoiseVariance(config, base, im, include_obj_var, logger)

valid_noise_types = { 
    'Gaussian' : (AddNoiseGaussian, NoiseVarGaussian, AddNoiseVarianceGaussian),
    'Poisson' : (AddNoisePoisson, NoiseVarPoisson, AddNoiseVariancePoisson),
    'CCD' : (AddNoiseCCD, NoiseVarCCD, AddNoiseVarianceCCD),
    'COSMOS' : (AddNoiseCOSMOS, NoiseVarCOSMOS, AddNoiseVarianceCOSMOS),
}

