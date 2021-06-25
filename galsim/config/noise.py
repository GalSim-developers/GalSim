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

from .util import LoggerWrapper, GetIndex, GetRNG
from .value import ParseValue, GetCurrentValue, GetAllParams
from .input import RegisterInputConnectedType
from ..errors import GalSimConfigError, GalSimConfigValueError
from ..image import Image
from ..position import PositionI
from ..random import PoissonDeviate
from ..noise import GaussianNoise, PoissonNoise, DeviateNoise, CCDNoise
from ..correlatednoise import getCOSMOSNoise, UncorrelatedNoise

# This file handles the functionality for adding noise and the sky to an image after
# drawing the objects.

# This module-level dict will store all the registered noise types.
# See the RegisterNoiseType function at the end of this file.
# The keys are the (string) names of the noise types, and the values will be builder objects
# that will perform the different functions related to adding noise to images.
valid_noise_types = {}

#
# First the driver functions:
#

def AddSky(config, im):
    """Add the sky level to the image

    Parameters:
        config:         The (base) configuration dict
        im:             The image onto which to add the sky
    """
    index, orig_index_key = GetIndex(config['image'], config)
    config['index_key'] = 'image_num'
    sky = GetSky(config['image'], config, full=True)
    if sky:
        im += sky
    config['index_key'] = orig_index_key


def AddNoise(config, im, current_var=0., logger=None):
    """
    Add noise to an image according to the noise specifications in the noise dict.

    Parameters:
        config:         The (base) configuration dict
        im:             The image onto which to add the noise
        current_var:    The current noise variance present in the image already [default: 0]
        logger:         If given, a logger object to log progress. [default: None]
    """
    from .stamp import SetupConfigObjNum

    logger = LoggerWrapper(logger)
    if 'noise' in config['image']:
        noise = config['image']['noise']
    else: # No noise.
        return
    if not isinstance(noise, dict):
        raise GalSimConfigError("image.noise is not a dict.")

    # Default is Poisson
    noise_type = noise.get('type', 'Poisson')
    if noise_type not in valid_noise_types:
        raise GalSimConfigValueError("Invalid noise.type.", noise_type,
                                     list(valid_noise_types.keys()))

    # We need to use image_num for the index_key, but if we are running this from the stamp
    # building phase, then we want to use obj_num_rng for the noise rng.  So get the rng now
    # before we change config['index_key'].
    index, orig_index_key = GetIndex(noise, config)
    rng = GetRNG(noise, config)

    # This makes sure draw_method is properly copied over and given a default value.
    SetupConfigObjNum(config, config.get('obj_num',0), logger)
    draw_method = GetCurrentValue('draw_method', config['stamp'], str, config)

    builder = valid_noise_types[noise_type]
    config['index_key'] = 'image_num'
    var = builder.addNoise(noise, config, im, rng, current_var, draw_method, logger)
    config['index_key'] = orig_index_key

    return var

def CalculateNoiseVariance(config, full=False):
    """
    Calculate the noise variance from the noise specified in the noise dict.

    Parameters:
        config:     The (base) configuration dict
        full:       If the noise is variable across the image, return the full image with the
                    noise variance at every pixel.  Otherwise, just return the value at the center.

    Returns:
        the noise variance
    """
    noise = config['image']['noise']
    if not isinstance(noise, dict):
        raise GalSimConfigError("image.noise is not a dict.")

    noise_type = noise.get('type', 'Poisson')
    if noise_type not in valid_noise_types:
        raise GalSimConfigValueError("Invalid noise.type.", noise_type,
                                     list(valid_noise_types.keys()))

    index, orig_index_key = GetIndex(noise, config)
    config['index_key'] = 'image_num'

    builder = valid_noise_types[noise_type]
    var = builder.getNoiseVariance(noise, config, full=full)
    config['index_key'] = orig_index_key

    return var

def AddNoiseVariance(config, im, include_obj_var=False, logger=None):
    """
    Add the noise variance to an image according to the noise specifications in the noise dict.
    Typically, this is used for building a weight map, which is typically the inverse variance.

    Parameters:
        config:             The (base) configuration dict
        im:                 The image onto which to add the variance values
        include_obj_var:    Whether to add the variance from the object photons for noise
                            models that have a component based on the number of photons.
                            Note: if this is True, the returned variance will not include this
                            contribution to the noise variance.  [default: False]
        logger:             If given, a logger object to log progress. [default: None]

    Returns:
        the variance in the image
    """
    logger = LoggerWrapper(logger)
    if 'noise' in config['image']:
        noise = config['image']['noise']
    else: # No noise.
        return
    if not isinstance(noise, dict):
        raise GalSimConfigError("image.noise is not a dict.")

    noise_type = noise.get('type', 'Poisson')
    if noise_type not in valid_noise_types:
        raise GalSimConfigValueError("Invalid noise.type.", noise_type,
                                     list(valid_noise_types.keys()))

    index, orig_index_key = GetIndex(noise, config)
    config['index_key'] = 'image_num'

    builder = valid_noise_types[noise_type]
    builder.addNoiseVariance(noise, config, im, include_obj_var, logger)
    config['index_key'] = orig_index_key

def GetSky(config, base, logger=None, full=False):
    """Parse the sky information and return either a float value for the sky level per pixel
    or an image, as needed.

    If an image is required (because wcs is not uniform) then it will use the presence of
    base['image_pos'] to determine what size image to return (stamp or full).  If there is
    a current image_pos, then we are doing a stamp.  Otherwise a full image.

    Parameters:
        config:             The configuration field with the sky specification, which can be either
                            base['image'] or base['image']['noise']
        base:               The base configuration dict
        logger:             If given, a logger object to log progress. [default: None]
        full:               If the sky level is variable across the image, return the full
                            image with the sky at every pixel.  Otherwise, just return the
                            sky at the image center.

    Returns:
        sky, either a float value or an Image.  (The latter only if full=True)
    """
    logger = LoggerWrapper(logger)
    if 'sky_level' in config:
        if 'sky_level_pixel' in config:
            raise GalSimConfigValueError(
                "Cannot specify both sky_level and sky_level_pixel",
                (config['sky_level'], config['sky_level_pixel']))
        sky_level = ParseValue(config,'sky_level',base,float)[0]
        logger.debug('image %d, obj %d: sky_level = %f',
                     base.get('image_num',0),base.get('obj_num',0), sky_level)
        wcs = base['wcs']
        if wcs._isUniform:
            sky_level_pixel = sky_level * wcs.pixelArea()
            logger.debug('image %d, obj %d: Uniform: sky_level_pixel = %f',
                         base.get('image_num',0),base.get('obj_num',0), sky_level_pixel)
            return sky_level_pixel
        elif full:
            # This calculation is non-trivial, so store this in case we need it again.
            tag = (id(base), base['file_num'], base['image_num'])
            if config.get('_current_sky_tag',None) == tag:
                logger.debug('image %d, obj %d: Using saved sky image',
                             base.get('image_num',0),base.get('obj_num',0))
                return config['_current_sky']
            else:
                logger.debug('image %d, obj %d: Non-uniform wcs.  Making sky image',
                             base.get('image_num',0),base.get('obj_num',0))
                bounds = base['current_noise_image'].bounds
                sky = Image(bounds, wcs=wcs)
                wcs.makeSkyImage(sky, sky_level)
                sensor = base.get('sensor', None)
                if sensor is not None:
                    center = base.get('image_origin', PositionI(1,1)) - sky.origin
                    use_flux = config.get('use_flux_sky_areas', False)
                    # TODO: If use_flux_sky_areas = True, then we should really build this up
                    #       in steps.  E.g. for a flat field.
                    #       This one step calcualtion isn't right.
                    area = sensor.calculate_pixel_areas(sky, orig_center=center, use_flux=use_flux)
                    sky *= area
                config['_current_sky_tag'] = tag
                config['_current_sky'] = sky
                return sky
        else:
            center = base['current_noise_image'].bounds.true_center
            return wcs.local(image_pos=center).pixelArea() * sky_level
    elif 'sky_level_pixel' in config:
        sky_level_pixel = ParseValue(config,'sky_level_pixel',base,float)[0]
        logger.debug('image %d, obj %d: sky_level_pixel = %f',
                     base.get('image_num',0),base.get('obj_num',0), sky_level_pixel)
        return sky_level_pixel
    else:
        return 0.


# items that are parsed separately from the normal noise function
noise_ignore = [ 'whiten', 'symmetrize', 'use_flux_sky_areas' ]

class NoiseBuilder(object):
    """A base class for building noise objects and applying the noise to images.

    The base class doesn't do anything, but it defines the call signatures of the methods
    that derived classes should use for the different specific noise types.
    """
    def addNoise(self, config, base, im, rng, current_var, draw_method, logger):
        """Read the noise parameters from the config dict and add the appropriate noise to the
        given image.

        Parameters:
            config:         The configuration dict for the noise field.
            base:           The base configuration dict.
            im:             The image onto which to add the noise
            rng:            The random number generator to use for adding the noise.
            current_var:    The current noise variance present in the image already.
            draw_method:    The method that was used to draw the objects on the image.
            logger:         If given, a logger object to log progress.
        """
        raise NotImplementedError("The %s class has not overridden addNoise"%self.__class__)

    def getNoiseVariance(self, config, base, full=False):
        """Read the noise parameters from the config dict and return the variance.

        Parameters:
            config:     The configuration dict for the noise field.
            base:       The base configuration dict.
            full:       If the noise is variable across the image, return the full image with the
                        noise variance at every pixel.  Otherwise, just return the value at the
                        center.

        Returns:
            the variance of the noise model
        """
        raise NotImplementedError("The %s class has not overridden addNoise"%self.__class__)

    def addNoiseVariance(self, config, base, im, include_obj_var, logger):
        """Read the noise parameters from the config dict and add the appropriate noise variance
        to the given image.

        This is used for constructing the weight map iamge.  It doesn't add a random value to
        each pixel.  Rather, it adds the variance of the noise that was used in the main image to
        each pixel in this image.

        This method has a default implemenation that is appropriate for noise models that have
        a constant noise variance.  It just gets the variance from getNoiseVariance and adds
        that constant value to every pixel.

        Parameters:
            config:             The configuration dict for the noise field.
            base:               The base configuration dict.
            im:                 The image onto which to add the noise variance
            include_obj_var:    Whether the noise variance values should the photon noise from
                                object flux in addition to the sky flux.  Only relevant for
                                noise models that are based on the image flux values such as
                                Poisson and CCDNoise.
            logger:             If given, a logger object to log progress.
        """
        im += self.getNoiseVariance(config, base, full=True)

#
# Gaussian
#

class GaussianNoiseBuilder(NoiseBuilder):

    def addNoise(self, config, base, im, rng, current_var, draw_method, logger):

        # Read the noise variance
        var = self.getNoiseVariance(config, base)
        ret = var  # save for the return value

        # If we already have some variance in the image (from whitening), then we subtract this much
        # from sigma**2.
        if current_var:
            logger.debug('image %d, obj %d: Target variance is %f, current variance is %f',
                        base.get('image_num',0),base.get('obj_num',0),var,current_var)
            if var < current_var:
                raise GalSimConfigError(
                    "Whitening already added more noise than the requested Gaussian noise.")
            var -= current_var

        # Now apply the noise.
        import math
        sigma = math.sqrt(var)
        im.addNoise(GaussianNoise(rng,sigma=sigma))

        logger.debug('image %d, obj %d: Added Gaussian noise with var = %f',
                     base.get('image_num',0),base.get('obj_num',0),var)

        return ret

    def getNoiseVariance(self, config, base, full=False):

        # The noise level can be specified either as a sigma or a variance.  Here we just calculate
        # the value of the variance from either one.
        single = [ { 'sigma' : float , 'variance' : float } ]
        params = GetAllParams(config, base, single=single, ignore=noise_ignore)[0]
        if 'sigma' in params:
            sigma = params['sigma']
            return sigma * sigma
        else:
            return params['variance']


#
# Poisson
#

class PoissonNoiseBuilder(NoiseBuilder):

    def addNoise(self, config, base, im, rng, current_var, draw_method, logger):

        # Get how much extra sky to assume from the image.noise attribute.
        sky = GetSky(base['image'], base, logger, full=True)
        extra_sky = GetSky(config, base, logger, full=True)
        total_sky = sky + extra_sky # for the return value
        if isinstance(total_sky, Image):
            var = np.mean(total_sky.array)
        else:
            var = total_sky
        # (This could be zero, in which case we only add poisson noise for the object photons)

        # If we already have some variance in the image (from whitening), then we subtract this
        # much off of the sky level.  It's not precisely accurate, since the existing variance is
        # Gaussian, rather than Poisson, but it's the best we can do.
        if current_var:
            logger.debug('image %d, obj %d: Target variance is %f, current variance is %f',
                         base.get('image_num',0),base.get('obj_num',0), var, current_var)
            if isinstance(total_sky, Image):
                test = np.any(total_sky.array < current_var)
            else:
                test = (total_sky < current_var)
            if test:
                raise GalSimConfigError(
                    "Whitening already added more noise than the requested Poisson noise.")
            total_sky -= current_var
            extra_sky -= current_var

        # At this point, there is a slight difference between fft and phot. For photon shooting,
        # the galaxy already has Poisson noise, so we want to make sure not to add that again!
        if draw_method == 'phot':
            # Only add in the noise from the sky.
            if isinstance(total_sky, Image):
                noise_im = total_sky.copy()
                noise_im.addNoise(PoissonNoise(rng))
                noise_im -= total_sky
                # total_sky should now have zero mean, but with the noise of the total sky level.
                im += noise_im
            else:
                im.addNoise(DeviateNoise(PoissonDeviate(rng, mean=total_sky)))
                # This deviate adds a noisy version of the sky, so need to subtract the mean
                # back off.
                im -= total_sky
        else:
            # Do the normal PoissonNoise calculation.
            if isinstance(total_sky, Image):
                im += extra_sky
                im.addNoise(PoissonNoise(rng))
                im -= extra_sky
            else:
                im.addNoise(PoissonNoise(rng, sky_level=extra_sky))

        logger.debug('image %d, obj %d: Added Poisson noise',
                     base.get('image_num',0),base.get('obj_num',0))
        return var

    def getNoiseVariance(self, config, base, full=False):
        # The noise variance is the net sky level per pixel
        sky = GetSky(base['image'], base, full=full)
        sky += GetSky(config, base, full=full)
        return sky

    def addNoiseVariance(self, config, base, im, include_obj_var, logger):
        if include_obj_var:
            # The current image at this point should be the noise-free, sky-free image,
            # which is the object variance in each pixel.
            im += base['current_noise_image']

            # Note: For the phot case, we don't actually have an exact value for the variance in
            # each pixel, but the drawn image before adding the Poisson noise is our best guess for
            # the variance from the object's flux, so if we want the object variance included, this
            # is still the best we can do.

        # Add the total sky level
        im += self.getNoiseVariance(config, base, full=True)


#
# CCD
#

class CCDNoiseBuilder(NoiseBuilder):

    def getCCDNoiseParams(self, config, base):
        opt = { 'gain' : float , 'read_noise' : float }
        ignore = ['sky_level', 'sky_level_pixel']
        params = GetAllParams(config, base, opt=opt, ignore=noise_ignore + ignore)[0]
        gain = params.get('gain',1.0)
        read_noise = params.get('read_noise',0.0)
        read_noise_var = read_noise**2

        return gain, read_noise, read_noise_var

    def addNoise(self, config, base, im, rng, current_var, draw_method, logger):

        # This process goes a lot like the Poisson routine.  There are just two differences.
        # First, the Poisson noise is in electrons, not ADU, and now we allow for a gain = e-/ADU,
        # so we need to account for that properly.  Second, we also allow for an additional Gaussian
        # read noise.
        gain, read_noise, read_noise_var = self.getCCDNoiseParams(config, base)

        # Get how much extra sky to assume from the image.noise attribute.
        sky = GetSky(base['image'], base, logger, full=True)
        extra_sky = GetSky(config, base, logger, full=True)
        total_sky = sky + extra_sky # for the return value
        if isinstance(total_sky, Image):
            var = np.mean(total_sky.array) + read_noise_var
        else:
            var = total_sky + read_noise_var

        # If we already have some variance in the image (from whitening), then we try to subtract
        # it from the read noise if possible.  If not, we subtract the rest off of the sky level.
        # It's not precisely accurate, since the existing variance is Gaussian, rather than
        # Poisson, but it's the best we can do.
        if current_var:
            logger.debug('image %d, obj %d: Target variance is %f, current variance is %f',
                        base.get('image_num',0),base.get('obj_num',0), var, current_var)
            read_noise_var_adu = read_noise_var / gain**2
            if isinstance(total_sky, Image):
                test = np.any(total_sky.array/gain + read_noise_var_adu < current_var)
            else:
                target_var = total_sky / gain + read_noise_var_adu
                logger.debug('image %d, obj %d: Target variance is %f, current variance is %f',
                             base.get('image_num',0),base.get('obj_num',0),
                             target_var, current_var)
                test = target_var < current_var
            if test:
                raise GalSimConfigError(
                    "Whitening already added more noise than the requested CCD noise.")
            if read_noise_var_adu >= current_var:
                # First try to take away from the read_noise, since this one is actually Gaussian.
                import math
                read_noise_var -= current_var * gain**2
                read_noise = math.sqrt(read_noise_var)
            else:
                # Take read_noise down to zero, since already have at least that much already.
                current_var -= read_noise_var_adu
                read_noise = 0
                read_noise_var = 0
                # Take the rest away from the sky level
                total_sky -= current_var * gain
                extra_sky -= current_var * gain

        # At this point, there is a slight difference between fft and phot. For photon shooting,
        # the galaxy already has Poisson noise, so we want to make sure not to add that again!
        if draw_method == 'phot':
            # Add in the noise from the sky.
            if isinstance(total_sky, Image):
                noise_im = total_sky.copy()
                if gain != 1.0: noise_im *= gain
                noise_im.addNoise(PoissonNoise(rng))
                if gain != 1.0: noise_im /= gain
                noise_im -= total_sky
                # total_sky should now have zero mean, but with the noise of the total sky level.
                im += noise_im
            else:
                if gain != 1.0: im *= gain
                pd = PoissonDeviate(rng, mean=total_sky*gain)
                im.addNoise(DeviateNoise(pd))
                if gain != 1.0: im /= gain
                im -= total_sky
            # And add the read noise
            if read_noise != 0.:
                im.addNoise(GaussianNoise(rng, sigma=read_noise/gain))
        else:
            # Do the normal CCDNoise calculation.
            if isinstance(total_sky, Image):
                im += extra_sky
                im.addNoise(CCDNoise(rng, gain=gain, read_noise=read_noise))
                im -= extra_sky
            else:
                im.addNoise(CCDNoise(rng, gain=gain, read_noise=read_noise, sky_level=extra_sky))

        logger.debug('image %d, obj %d: Added CCD noise with gain = %f, read_noise = %f',
                     base.get('image_num',0),base.get('obj_num',0), gain, read_noise)

        return var

    def getNoiseVariance(self, config, base, full=False):
        # The noise variance is sky / gain + (read_noise/gain)**2
        gain, read_noise, read_noise_var = self.getCCDNoiseParams(config, base)

        # Start with the background sky level for the image
        sky = GetSky(base['image'], base, full=full)
        sky += GetSky(config, base, full=full)

        # Account for the gain and read_noise
        read_noise_var_adu = read_noise_var / gain**2
        return sky / gain + read_noise_var_adu

    def addNoiseVariance(self, config, base, im, include_obj_var, logger):
        gain, read_noise, read_noise_var = self.getCCDNoiseParams(config, base)
        if include_obj_var:
            # The current image at this point should be the noise-free, sky-free image,
            # which is the object variance in each pixel.
            if gain != 1.0:
                im += base['current_noise_image']/gain
            else:
                im += base['current_noise_image']

        # Now add on the regular CCDNoise from the sky and read noise.
        im += self.getNoiseVariance(config, base, full=True)


#
# COSMOS
#

class COSMOSNoiseBuilder(NoiseBuilder):

    def getCOSMOSNoise(self, config, base, rng=None):
        # Save the constructed CorrelatedNoise object, since we might need it again.
        tag = (id(base), base['file_num'], base['image_num'])
        if base.get('_current_cn_tag',None) == tag:
            return base['_current_cn']
        else:
            opt = { 'file_name' : str, 'cosmos_scale' : float, 'variance' : float }

            kwargs = GetAllParams(config, base, opt=opt, ignore=noise_ignore)[0]
            if rng is None:
                rng = GetRNG(config, base)
            cn = getCOSMOSNoise(rng=rng, **kwargs)
            base['_current_cn'] = cn
            base['_current_cn_tag'] = tag
            return cn

    def addNoise(self, config, base, im, rng, current_var, draw_method, logger):

        # Build the correlated noise
        cn = self.getCOSMOSNoise(config,base,rng)
        var = cn.getVariance()

        # Subtract off the current variance if any
        if current_var:
            logger.debug('image %d, obj %d: Target variance is %f, current variance is %f',
                         base.get('image_num',0),base.get('obj_num',0), var, current_var)
            if var < current_var:
                raise GalSimConfigError(
                    "Whitening already added more noise than the requested COSMOS noise.")
            cn -= UncorrelatedNoise(current_var, rng=rng, wcs=cn.wcs)

        # Add the noise to the image
        im.addNoise(cn)

        logger.debug('image %d, obj %d: Added COSMOS correlated noise with variance = %f',
                     base.get('image_num',0),base.get('obj_num',0), var)
        return var

    def getNoiseVariance(self, config, base, full=False):
        cn = self.getCOSMOSNoise(config,base)
        return cn.getVariance()


def RegisterNoiseType(noise_type, builder, input_type=None):
    """Register a noise type for use by the config apparatus.

    Parameters:
        noise_type:     The name of the type in config['image']['noise']
        builder:        A builder object to use for building the noise.  It should be an
                        instance of a subclass of NoiseBuilder.
        input_type:     If the builder utilises an input object, give the key name of the
                        input type here.  (If it uses more than one, this may be a list.)
                        [default: None]
    """
    valid_noise_types[noise_type] = builder
    RegisterInputConnectedType(input_type, noise_type)


RegisterNoiseType('Gaussian', GaussianNoiseBuilder())
RegisterNoiseType('Poisson', PoissonNoiseBuilder())
RegisterNoiseType('CCD', CCDNoiseBuilder())
RegisterNoiseType('COSMOS', COSMOSNoiseBuilder())

