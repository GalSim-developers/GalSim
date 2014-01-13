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

def _get_sky_level_pixel(config, image_pos=None):
    if image_pos is None:
        image_pos = config['image_center']

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


def AddNoiseFFT(config, im, weight_im, current_var, sky_level_pixel, logger=None):
    """
    Add noise to an image according to the noise specifications in the noise dict
    appropriate for an image that has been drawn using the FFT method.
    """
    noise = config['image']['noise']
    if not isinstance(noise, dict):
        raise AttributeError("image.noise is not a dict.")
    rng = config['rng']

    if 'type' not in noise:
        noise['type'] = 'Poisson'  # Default is Poisson
    type = noise['type']

    # First add the background sky level, if provided
    if sky_level_pixel:
        im += sky_level_pixel

    # Check if a weight image should include the object variance.
    if weight_im:
        include_obj_var = False
        if ('output' in config and 'weight' in config['output'] and 
            'include_obj_var' in config['output']['weight']):
            include_obj_var = galsim.config.ParseValue(
                config['output']['weight'], 'include_obj_var', config, bool)[0]

    # Then add the correct kind of noise
    if type == 'Gaussian':
        single = [ { 'sigma' : float , 'variance' : float } ]
        params = galsim.config.GetAllParams(noise, 'noise', config, single=single)[0]

        if 'sigma' in params:
            sigma = params['sigma']
            if current_var: 
                var = sigma**2
                if var < current_var:
                    raise RuntimeError(
                        "Whitening already added more noise than requested Gaussian noise.")
                sigma = sqrt(var - current_var)
        else:
            import math
            var = params['variance']
            if current_var:
                if var < current_var: 
                    raise RuntimeError(
                        "Whitening already added more noise than requested Gaussian noise.")
                var -= current_var
            sigma = math.sqrt(var)
        im.addNoise(galsim.GaussianNoise(rng,sigma=sigma))

        if weight_im:
            weight_im += sigma*sigma + current_var
        if logger:
            logger.debug('image %d, obj %d: Added Gaussian noise with sigma = %f',
                         config['image_num'],config['obj_num'],sigma)

    elif type == 'Poisson':
        opt = {}
        single = []
        if sky_level_pixel:
            # The noise sky_level is only required here if the image doesn't have any.
            opt['sky_level'] = float
            opt['sky_level_pixel'] = float
        else:
            single = [ { 'sky_level' : float , 'sky_level_pixel' : float } ]
            sky_level_pixel = 0.
        params = galsim.config.GetAllParams(noise, 'noise', config, opt=opt, single=single)[0]
        if 'sky_level' in params and 'sky_level_pixel' in params:
            raise AttributeError("Only one of sky_level and sky_level_pixel is allowed for "
                "noise.type = %s"%type)
        extra_sky_level_pixel = 0.
        if 'sky_level' in params:
            extra_sky_level_pixel = params['sky_level'] * im.wcs.pixelArea(config['image_pos'])
        if 'sky_level_pixel' in params:
            extra_sky_level_pixel = params['sky_level_pixel']
        sky_level_pixel += extra_sky_level_pixel

        if current_var:
            if sky_level_pixel < current_var:
                raise RuntimeError(
                    "Whitening already added more noise than requested Poisson noise.")
            extra_sky_level_pixel -= current_var

        if weight_im:
            if include_obj_var:
                # The image right now has the variance in each pixel.  So before going on with the 
                # noise, copy these over to the weight image.  (We invert this later...)
                weight_im.copyFrom(im)
            else:
                # Otherwise, just add the sky noise:
                weight_im += sky_level_pixel

        im.addNoise(galsim.PoissonNoise(rng, sky_level=extra_sky_level_pixel))
        if logger:
            logger.debug('image %d, obj %d: Added Poisson noise with sky_level_pixel = %f',
                         config['image_num'],config['obj_num'],sky_level_pixel)

    elif type == 'CCD':
        opt = { 'gain' : float , 'read_noise' : float }
        single = []
        if sky_level_pixel:
            # The noise sky_level is only required here if the image doesn't have any.
            opt['sky_level'] = float
            opt['sky_level_pixel'] = float
        else:
            single = [ { 'sky_level' : float , 'sky_level_pixel' : float } ]
            sky_level_pixel = 0.
        params = galsim.config.GetAllParams(noise, 'noise', config, opt=opt, single=single)[0]
        gain = params.get('gain',1.0)
        read_noise = params.get('read_noise',0.0)
        if 'sky_level' in params and 'sky_level_pixel' in params:
            raise AttributeError("Only one of sky_level and sky_level_pixel is allowed for "
                "noise.type = %s"%type)
        extra_sky_level_pixel = 0.
        if 'sky_level' in params:
            extra_sky_level_pixel = params['sky_level'] * im.wcs.pixelArea(config['image_pos'])
        if 'sky_level_pixel' in params:
            extra_sky_level_pixel = params['sky_level_pixel']
        sky_level_pixel += extra_sky_level_pixel
        read_noise_var = read_noise**2

        if weight_im:
            if include_obj_var:
                # The image right now has the variance in each pixel.  So before going on with the 
                # noise, copy these over to the weight image and invert.
                weight_im.copyFrom(im)
                if gain != 1.0:
                    import math
                    weight_im /= math.sqrt(gain)
                if read_noise != 0.0:
                    weight_im += read_noise_var
            else:
                # Otherwise, just add the sky and read_noise:
                weight_im += sky_level_pixel / gain + read_noise_var

        if current_var:
            if sky_level_pixel / gain + read_noise_var < current_var:
                raise RuntimeError(
                    "Whitening already added more noise than requested CCD noise.")
            if read_noise_var >= current_var:
                import math
                # First try to take away from the read_noise, since this one is actually Gaussian.
                read_noise_var -= current_var
                read_noise = math.sqrt(read_noise_var)
            else:
                # Take read_noise down to zero, since already have at least that much already.
                current_var -= read_noise_var
                read_noise = 0
                # Take the rest away from the sky level
                extra_sky_level_pixel -= current_var * gain

        im.addNoise(galsim.CCDNoise(rng, sky_level=extra_sky_level_pixel, gain=gain,
                                    read_noise=read_noise))
        if logger:
            logger.debug('image %d, obj %d: Added CCD noise with sky_level_pixel = %f, ' +
                         'gain = %f, read_noise = %f',
                         config['image_num'],config['obj_num'],extra_sky_level_pixel,gain,read_noise)

    elif type == 'COSMOS':
        req = { 'file_name' : str }
        opt = { 'cosmos_scale' : float, 'variance' : float }
        
        kwargs = galsim.config.GetAllParams(noise, 'noise', config, req=req, opt=opt)[0]

        # Build the correlated noise 
        cn = galsim.correlatednoise.getCOSMOSNoise(rng, **kwargs)
        cn_var = cn.getVariance()

        # Subtract off the current variance if any
        if current_var:
            if cn_var < current_var:
                raise RuntimeError(
                    "Whitening already added more noise than requested COSMOS noise.")
            cn -= galsim.UncorrelatedNoise(rng, im.wcs, current_var)

        # Add the noise to the image
        im.addNoise(cn)

        # Then add the variance to the weight image, using the zero-lag correlation function value
        if weight_im: weight_im += cn_var

        if logger:
            logger.debug('image %d, obj %d: Added COSMOS correlated noise with variance = %f',
                         config['image_num'],config['obj_num'],cn_var)

    else:
        raise AttributeError("Invalid type %s for noise"%type)


def AddNoisePhot(config, im, weight_im, current_var, sky_level_pixel, logger=None):
    """
    Add noise to an image according to the noise specifications in the noise dict
    appropriate for an image that has been drawn using the photon-shooting method.
    """
    noise = config['image']['noise']
    if not isinstance(noise, dict):
        raise AttributeError("image.noise is not a dict.")
    rng = config['rng']

    if 'type' not in noise:
        noise['type'] = 'Poisson'  # Default is Poisson
    type = noise['type']

    # First add the sky noise, if provided
    if sky_level_pixel:
        im += sky_level_pixel

    if type == 'Gaussian':
        single = [ { 'sigma' : float , 'variance' : float } ]
        params = galsim.config.GetAllParams(noise, 'noise', config, single=single)[0]

        if 'sigma' in params:
            sigma = params['sigma']
            if current_var: 
                var = sigma**2
                if var < current_var:
                    raise RuntimeError(
                        "Whitening already added more noise than requested Gaussian noise.")
                sigma = sqrt(var - current_var)
        else:
            import math
            var = params['variance']
            if current_var:
                if var < current_var:
                    raise RuntimeError(
                        "Whitening already added more noise than requested Gaussian noise.")
                var -= current_var
            sigma = math.sqrt(var)
        im.addNoise(galsim.GaussianNoise(rng,sigma=sigma))

        if weight_im:
            weight_im += sigma*sigma + current_var
        if logger:
            logger.debug('image %d, obj %d: Added Gaussian noise with sigma = %f',
                         config['image_num'],config['obj_num'],sigma)

    elif type == 'Poisson':
        opt = {}
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
            sky_level_pixel += params['sky_level'] * im.wcs.pixelArea(config['image_pos'])
        if 'sky_level_pixel' in params:
            sky_level_pixel += params['sky_level_pixel']
        if current_var:
            if sky_level_pixel < current_var:
                raise RuntimeError(
                    "Whitening already added more noise than requested Poisson noise.")
            sky_level_pixel -= current_var

        # We don't have an exact value for the variance in each pixel, but the drawn image
        # before adding the Poisson noise is our best guess for the variance from the 
        # object's flux, so just use that for starters.
        if weight_im and include_obj_var:
            weight_im.copyFrom(im)

        # For photon shooting, galaxy already has Poisson noise, so we want 
        # to make sure not to add that again!
        if sky_level_pixel != 0.:
            im.addNoise(galsim.DeviateNoise(galsim.PoissonDeviate(rng, mean=sky_level_pixel)))
            im -= sky_level_pixel
            if weight_im:
                weight_im += sky_level_pixel + current_var

        if logger:
            logger.debug('image %d, obj %d: Added Poisson noise with sky_level_pixel = %f',
                         config['image_num'],config['obj_num'],sky_level_pixel)

    elif type == 'CCD':
        opt = { 'gain' : float , 'read_noise' : float }
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
            sky_level_pixel += params['sky_level'] * im.wcs.pixelArea(config['image_pos'])
        if 'sky_level_pixel' in params:
            sky_level_pixel += params['sky_level_pixel']
        gain = params.get('gain',1.0)
        read_noise = params.get('read_noise',0.0)
        read_noise_var = read_noise**2

        if weight_im:
            # We don't have an exact value for the variance in each pixel, but the drawn image
            # before adding the Poisson noise is our best guess for the variance from the 
            # object's flux, so just use that for starters.
            if include_obj_var: weight_im.copyFrom(im)
            if sky_level_pixel != 0.0 or read_noise != 0.0:
                weight_im += sky_level_pixel / gain + read_noise_var

        if current_var:
            if sky_level_pixel / gain + read_noise_var < current_var:
                raise RuntimeError(
                    "Whitening already added more noise than requested CCD noise.")
            if read_noise_var >= current_var:
                import math
                # First try to take away from the read_noise, since this one is actually Gaussian.
                read_noise_var -= current_var
                read_noise = math.sqrt(read_noise_var)
            else:
                # Take read_noise down to zero, since already have at least that much already.
                current_var -= read_noise_var
                read_noise = 0
                # Take the rest away from the sky level
                sky_level_pixel -= current_var * gain
 
        # For photon shooting, galaxy already has Poisson noise, so we want 
        # to make sure not to add that again!
        if sky_level_pixel != 0.:
            if gain != 1.0: im *= gain
            im.addNoise(galsim.DeviateNoise(galsim.PoissonDeviate(rng, mean=sky_level_pixel*gain)))
            if gain != 1.0: im /= gain
            im -= sky_level_pixel*gain
        if read_noise != 0.:
            im.addNoise(galsim.GaussianNoise(rng, sigma=read_noise))

        if logger:
            logger.debug('image %d, obj %d: Added CCD noise with sky_level_pixel = %f, ' +
                         'gain = %f, read_noise = %f',
                         config['image_num'],config['obj_num'],sky_level_pixel,gain,read_noise)

    elif type == 'COSMOS':
        req = { 'file_name' : str }
        opt = { 'cosmos_scale' : float, 'variance' : float }
        
        kwargs = galsim.config.GetAllParams(noise, 'noise', config, req=req, opt=opt)[0]

        # Build and add the correlated noise 
        cn = galsim.correlatednoise.getCOSMOSNoise(rng, **kwargs)
        cn_var = cn.getVariance()

        # Subtract off the current variance if any
        if current_var:
            if cn_var < current_var:
                raise RuntimeError(
                    "Whitening already added more noise than requested COSMOS noise.")
            cn -= galsim.UncorrelatedNoise(rng, im.wcs, current_var)

        # Add the noise to the image
        im.addNoise(cn)

        # Then add the variance to the weight image, using the zero-lag correlation function value
        if weight_im: weight_im += cn_var

        if logger:
            logger.debug('image %d, obj %d: Added COSMOS correlated noise with variance = %f',
                         config['image_num'],config['obj_num'],cn_var)

    else:
        raise AttributeError("Invalid type %s for noise",type)


def CalculateNoiseVar(config, sky_level_pixel):
    """
    Calculate the noise variance from the noise specified in the noise dict.
    """
    noise = config['image']['noise']
    if not isinstance(noise, dict):
        raise AttributeError("image.noise is not a dict.")

    if 'type' not in noise:
        noise['type'] = 'Poisson'  # Default is Poisson
    type = noise['type']

    if type == 'Gaussian':
        single = [ { 'sigma' : float , 'variance' : float } ]
        params = galsim.config.GetAllParams(noise, 'noise', config, single=single)[0]
        if 'sigma' in params:
            sigma = params['sigma']
            var = sigma * sigma
        else:
            var = params['variance']

    elif type == 'Poisson':
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
        var = sky_level_pixel

    elif type == 'CCD':
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
        gain = params.get('gain',1.0)
        read_noise = params.get('read_noise',0.0)
        var = sky_level_pixel / gain + read_noise * read_noise

    elif type == 'COSMOS':
        req = { 'file_name' : str }
        opt = { 'cosmos_scale' : float, 'variance' : float }
        
        kwargs = galsim.config.GetAllParams(noise, 'noise', config, req=req, opt=opt)[0]

        # Build and add the correlated noise (lets the cn internals handle dealing with the options
        # for default variance: quick and ensures we don't needlessly duplicate code) 
        # Note: the rng being passed here is arbitrary, since we don't need it to calculate the
        # variance.  Building a BaseDeviate with a particular seed is the fastest option.
        cn = galsim.correlatednoise.getCOSMOSNoise(galsim.BaseDeviate(123), **kwargs)

        # zero distance correlation function value returned as variance
        var = cn.getVariance()

    else:
        raise AttributeError("Invalid type %s for noise",type)

    return var


