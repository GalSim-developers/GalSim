# Copyright 2012, 2013 The GalSim developers:
# https://github.com/GalSim-developers
#
# This file is part of GalSim: The modular galaxy image simulation toolkit.
#
# GalSim is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# GalSim is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with GalSim.  If not, see <http://www.gnu.org/licenses/>
#

import galsim

valid_noise_types = { 
    # The values in the tuple are:
    # - A function to add noise after drawing
    # - A function that returns the variance of the noise
    'Gaussian' : ('AddNoiseGaussian', 'NoiseVarGaussian'),
    'Poisson' : ('AddNoisePoisson', 'NoiseVarPoisson'),
    'CCD' : ('AddNoiseCCD', 'NoiseVarCCD'),
    'COSMOS' : ('AddNoiseCOSMOS', 'NoiseVarCOSMOS'),
}


def _get_sky_level_pixel(config):
    image_pos = config['image_pos']

    image = config['image']
    if 'sky_level' in image and 'sky_level_pixel' in image:
        raise AttributeError("Only one of sky_level and sky_level_pixel is allowed for "
            "noise.type = %s"%type)
    if 'sky_level_pixel' in image:
        sky_level_pixel = galsim.config.ParseValue(image,'sky_level_pixel',config,float)[0]
    elif 'sky_level' in image:
        sky_level = galsim.config.ParseValue(image,'sky_level',config,float)[0]
        sky_level_pixel = sky_level * config['wcs'].pixelArea(image_pos)
    else:
        sky_level_pixel = 0.
    return sky_level_pixel

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

    if add_sky:
        sky_level_pixel = _get_sky_level_pixel(config)
    else:
        sky_level_pixel = 0.

    # Add the noise specified
    if draw_method is not 'skip':

        if 'type' in noise:
            type = noise['type']
        else:
            type = 'Poisson'  # Default is Poisson
        if type not in valid_noise_types:
            raise AttributeError("Invalid type %s for noise",type)

        noise_func = eval(valid_noise_types[type][0])
        noise_func(noise, config, draw_method, rng, im, weight_im, current_var, sky_level_pixel, logger)

    # Then add the overall sky level, if desired
    if sky_level_pixel != 0.:
        im += sky_level_pixel


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

def AddNoiseGaussian(noise, config, draw_method, rng, im, weight_im, current_var, sky_level_pixel, 
                     logger):
    # NB: Identical for fft and phot

    # The noise level can be specified either as a sigma or a variance.  Here we just calculate
    # the value of the variance from either one.
    single = [ { 'sigma' : float , 'variance' : float } ]
    params = galsim.config.GetAllParams(noise, 'noise', config, single=single)[0]
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
        if var < current_var:
            raise RuntimeError(
                "Whitening already added more noise than requested Gaussian noise.")
        var -= current_var

    # Now apply the noise.
    import math
    sigma = math.sqrt(var)
    im.addNoise(galsim.GaussianNoise(rng,sigma=sigma))

    if logger:
        logger.debug('image %d, obj %d: Added Gaussian noise with sigma = %f',
                     config['image_num'],config['obj_num'],sigma)

def NoiseVarGaussian(noise, config):

    # The noise variance is just sigma^2 or variance
    single = [ { 'sigma' : float , 'variance' : float } ]
    params = galsim.config.GetAllParams(noise, 'noise', config, single=single)[0]
    if 'sigma' in params:
        sigma = params['sigma']
        return sigma * sigma
    else:
        return params['variance']


#
# Poisson
#

def AddNoisePoisson(noise, config, draw_method, rng, im, weight_im, current_var, sky_level_pixel, logger):

    # We need to calculate the net sky level for the purpos of the noise.
    # We start off with the overall sky level, taken as an input parameter.

    # Here we add in how much sky to assume from the image.noise attribute.
    opt = {}
    single = []
    if sky_level_pixel:
        # The noise sky_level is only required here if the image doesn't have any.
        opt['sky_level'] = float
        opt['sky_level_pixel'] = float
    else:
        single = [ { 'sky_level' : float , 'sky_level_pixel' : float } ]
    params = galsim.config.GetAllParams(noise, 'noise', config, opt=opt, single=single)[0]
    if 'sky_level' in params and 'sky_level_pixel' in params:
        raise AttributeError("Only one of sky_level and sky_level_pixel is allowed for "
            "noise.type = %s"%type)
    if 'sky_level' in params:
        sky_level_pixel += params['sky_level'] * im.wcs.pixelArea(config['image_pos'])
    if 'sky_level_pixel' in params:
        sky_level_pixel += params['sky_level_pixel']

    # If we are saving the noise level in a weight image, do that now.
    if weight_im:
        # Check if a weight image should include the object variance.
        # Note: For the phot case, we don't actually have an exact value for the variance in each pixel,
        # but the drawn image before adding the Poisson noise is our best guess for the variance from the 
        # object's flux, so if we want the object variance included, this is still the best we can do.
        include_obj_var = False
        if ('output' in config and 'weight' in config['output'] and 
            'include_obj_var' in config['output']['weight']):
            include_obj_var = galsim.config.ParseValue(
                config['output']['weight'], 'include_obj_var', config, bool)[0]
        if include_obj_var:
            # The image right now has the object variance in each pixel.  So before going on with the 
            # noise, copy these over to the weight image.  (We invert this later...)
            weight_im.copyFrom(im)

        # And add in the sky noise:
        if sky_level_pixel != 0.:
            weight_im += sky_level_pixel

    # If we already have some variance in the image (from whitening), then we subtract this much
    # off of the sky level.  It's not precisely accurate, since the existing variance is Gaussian,
    # rather than Poisson, but it's the best we can do.
    if current_var:
        if sky_level_pixel < current_var:
            raise RuntimeError(
                "Whitening already added more noise than requested Poisson noise.")
        sky_level_pixel -= current_var

    # At this point, there is a slight difference between fft and phot. For photon shooting, the 
    # galaxy already has Poisson noise, so we want to make sure not to add that again!
    if draw_method == 'fft':
        # Do the normal PoissonNoise calculation.
        im.addNoise(galsim.PoissonNoise(rng, sky_level=sky_level_pixel))
    else:
        # Only add in the noise from the sky.
        if sky_level_pixel > 0.:
            im.addNoise(galsim.DeviateNoise(galsim.PoissonDeviate(rng, mean=sky_level_pixel)))
            im -= sky_level_pixel

    if logger:
        logger.debug('image %d, obj %d: Added Poisson noise with sky_level_pixel = %f',
                     config['image_num'],config['obj_num'],sky_level_pixel)


def NoiseVarPoisson(noise, config):
    # The noise variance is the net sky level per pixel

    # Start with the background sky level for the image
    sky_level_pixel = _get_sky_level_pixel(config)

    # And add in any extra sky level request for the noise
    opt = {}
    single = []
    if sky_level_pixel:
        opt['sky_level'] = float
        opt['sky_level_pixel'] = float
    else:
        single = [ { 'sky_level' : float , 'sky_level_pixel' : float } ]
        sky_level_pixel = 0. # Switch from None to 0.
    params = galsim.config.GetAllParams(noise, 'noise', config, opt=opt, single=single)[0]
    if 'sky_level' in params and 'sky_level_pixel' in params:
        raise AttributeError("Only one of sky_level and sky_level_pixel is allowed for "
            "noise.type = %s"%type)
    if 'sky_level' in params:
        sky_level_pixel += params['sky_level'] * config['wcs'].pixelArea(config['image_pos'])
    if 'sky_level_pixel' in params:
        sky_level_pixel += params['sky_level_pixel']

    return sky_level_pixel


#
# CCD
#

def AddNoiseCCD(noise, config, draw_method, rng, im, weight_im, current_var, sky_level_pixel, logger):

    # This process goes a lot like the Poisson routine.  There are just two differences.
    # The Poisson noise is in the electron, not ADU, and now we allow for a gain = e-/ADU,
    # so we need to account for that properly.  And we also allow for an additional Gaussian
    # read noise.j

    # Figure out our net sky level, gain, and read noise:
    opt = { 'gain' : float , 'read_noise' : float }
    single = []
    if sky_level_pixel:
        # The noise sky_level is only required here if the image doesn't have any.
        opt['sky_level'] = float
        opt['sky_level_pixel'] = float
    else:
        single = [ { 'sky_level' : float , 'sky_level_pixel' : float } ]
    params = galsim.config.GetAllParams(noise, 'noise', config, opt=opt, single=single)[0]
    gain = params.get('gain',1.0)
    read_noise = params.get('read_noise',0.0)
    read_noise_var = read_noise**2
    if 'sky_level' in params and 'sky_level_pixel' in params:
        raise AttributeError("Only one of sky_level and sky_level_pixel is allowed for "
            "noise.type = %s"%type)
    if 'sky_level' in params:
        sky_level_pixel += params['sky_level'] * im.wcs.pixelArea(config['image_pos'])
    if 'sky_level_pixel' in params:
        sky_level_pixel += params['sky_level_pixel']

    # If we are saving the noise level in a weight image, do that now.
    if weight_im:
        # Check if a weight image should include the object variance.
        # Note: For the phot case, we don't actually have an exact value for the variance in each pixel,
        # but the drawn image before adding the Poisson noise is our best guess for the variance from the 
        # object's flux, so if we want the object variance included, this is still the best we can do.
        include_obj_var = False
        if ('output' in config and 'weight' in config['output'] and 
            'include_obj_var' in config['output']['weight']):
            include_obj_var = galsim.config.ParseValue(
                config['output']['weight'], 'include_obj_var', config, bool)[0]
        if include_obj_var:
            # The image right now has the object variance in each pixel.  So before going on with the 
            # noise, copy these over to the weight image.  (We invert this later...)
            weight_im.copyFrom(im)

            # Add in the sky noise:
            if sky_level_pixel != 0.0:
                weight_im += sky_level_pixel

            # Account for the gain and read noise
            if gain != 1.0:
                import math
                weight_im /= math.sqrt(gain)
            if read_noise != 0.0:
                weight_im += read_noise_var
        else:
            # Otherwise, can do the sky and read noise all at once, so more efficient.
            if sky_level_pixel != 0.0 or read_noise != 0.0:
                weight_im += sky_level_pixel / gain + read_noise_var

    # If we already have some variance in the image (from whitening), then we try to subtract it from
    # the read noise if possible.  If now, we subtract the rest off of the sky level.  It's not 
    # precisely accurate, since the existing variance is Gaussian, rather than Poisson, but it's the 
    # best we can do.
    if current_var:
        if sky_level_pixel / gain + read_noise_var < current_var:
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
            sky_level_pixel -= current_var * gain

    # At this point, there is a slight difference between fft and phot. For photon shooting, the 
    # galaxy already has Poisson noise, so we want to make sure not to add that again!
    if draw_method == 'fft':
        # Do the normal CCDNoise calculation.
        im.addNoise(galsim.CCDNoise(rng, sky_level=sky_level_pixel, gain=gain,
                                    read_noise=read_noise))
    else:
        # Add in the noise from the sky.
        if sky_level_pixel:
            if gain != 1.0: im *= gain
            im.addNoise(galsim.DeviateNoise(galsim.PoissonDeviate(rng, mean=sky_level_pixel*gain)))
            if gain != 1.0: im /= gain
            im -= sky_level_pixel*gain
        # Add and the read noise
        if read_noise != 0.:
            im.addNoise(galsim.GaussianNoise(rng, sigma=read_noise))

    if logger:
        logger.debug('image %d, obj %d: Added CCD noise with sky_level_pixel = %f, ' +
                     'gain = %f, read_noise = %f',
                     config['image_num'],config['obj_num'],sky_level_pixel,gain,read_noise)


def NoiseVarCCD(noise, config):
    # The noise variance is sky_level_pixel / gain + read_noise^2

    # Start with the background sky level for the image
    sky_level_pixel = _get_sky_level_pixel(config)

    # Add in any extra sky level request for the noise
    opt = { 'gain' : float , 'read_noise' : float }
    single = []
    if sky_level_pixel:
        opt['sky_level'] = float
        opt['sky_level_pixel'] = float
    else:
        single = [ { 'sky_level' : float , 'sky_level_pixel' : float } ]
        sky_level_pixel = 0. # Switch from None to 0.
    params = galsim.config.GetAllParams(noise, 'noise', config, opt=opt, single=single)[0]
    if 'sky_level' in params and 'sky_level_pixel' in params:
        raise AttributeError("Only one of sky_level and sky_level_pixel is allowed for "
            "noise.type = %s"%type)
    if 'sky_level' in params:
        sky_level_pixel += params['sky_level'] * config['wcs'].pixelArea(config['image_pos'])
    if 'sky_level_pixel' in params:
        sky_level_pixel += params['sky_level_pixel']

    # Account for the gain and read_noise
    gain = params.get('gain',1.0)
    read_noise = params.get('read_noise',0.0)
    return sky_level_pixel / gain + read_noise * read_noise

#
# COSMOS
#

def AddNoiseCOSMOS(noise, config, draw_method, rng, im, weight_im, current_var, sky_level_pixel, logger):
    # NB: Identical for fft and phot

    req = { 'file_name' : str }
    opt = { 'cosmos_scale' : float, 'variance' : float }
        
    kwargs = galsim.config.GetAllParams(noise, 'noise', config, req=req, opt=opt)[0]

    # Build the correlated noise 
    cn = galsim.correlatednoise.getCOSMOSNoise(rng, **kwargs)
    var = cn.getVariance()

    # If we are saving the noise level in a weight image, do that now.
    if weight_im: 
        weight_im += var

    # Subtract off the current variance if any
    if current_var:
        if var < current_var:
            raise RuntimeError(
                "Whitening already added more noise than requested COSMOS noise.")
        cn -= galsim.UncorrelatedNoise(rng, im.wcs, current_var)

    # Add the noise to the image
    im.addNoise(cn)

    if logger:
        logger.debug('image %d, obj %d: Added COSMOS correlated noise with variance = %f',
                     config['image_num'],config['obj_num'],var)

def NoiseVarCOSMOS(noise, config):
    # The variance is given by the getVariance function.

    req = { 'file_name' : str }
    opt = { 'cosmos_scale' : float, 'variance' : float }
    kwargs = galsim.config.GetAllParams(noise, 'noise', config, req=req, opt=opt)[0]

    # Build and add the correlated noise (lets the cn internals handle dealing with the options
    # for default variance: quick and ensures we don't needlessly duplicate code) 
    # Note: the rng being passed here is arbitrary, since we don't need it to calculate the
    # variance.  Building a BaseDeviate with a particular seed is the fastest option.
    cn = galsim.correlatednoise.getCOSMOSNoise(galsim.BaseDeviate(123), **kwargs)

    # zero distance correlation function value returned as variance
    return cn.getVariance()


