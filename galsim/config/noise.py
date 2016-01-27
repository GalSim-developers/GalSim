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
    """
    orig_index = config.get('index_key','image_num')
    if orig_index == 'obj_num':
        config['index_key'] = 'image_num'

    if im:
        sky = GetSky(config['image'], config)
        if sky:
            im += sky

    if orig_index == 'obj_num':
        config['index_key'] = 'obj_num'


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
        noise_type = noise['type']
    else:
        noise_type = 'Poisson'  # Default is Poisson
    if noise_type not in valid_noise_types:
        raise AttributeError("Invalid type %s for noise",noise_type)

    # We need to use image_num for the index_key, but if we are in the stamp processing
    # make sure to reset it back when we are done.  Also, we want to use obj_num_rng in this
    # case for the noise.  The easiest way to make sure this doesn't get messed up by any
    # value parsing is to copy it over to a new item in the dict.
    orig_index = config.get('index_key','image_num')
    if orig_index == 'obj_num':
        config['index_key'] = 'image_num'
        rng = config.get('obj_num_rng', config['rng'])
    else:
        rng = config['rng']

    builder = valid_noise_types[noise_type]
    builder.addNoise(noise, config, im, rng, current_var, logger)

    if orig_index == 'obj_num':
        config['index_key'] = 'obj_num'

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
        noise_type = noise['type']
    else:
        noise_type = 'Poisson'  # Default is Poisson
    if noise_type not in valid_noise_types:
        raise AttributeError("Invalid type %s for noise",noise_type)

    orig_index = config.get('index_key','image_num')
    if orig_index == 'obj_num':
        config['index_key'] = 'image_num'

    builder = valid_noise_types[noise_type]
    var = builder.getNoiseVariance(noise, config)

    if orig_index == 'obj_num':
        config['index_key'] = 'obj_num'

    return var

def AddNoiseVariance(config, im, include_obj_var=False, logger=None):
    """
    Add the noise variance to an image according to the noise specifications in the noise dict.
    Typically, this is used for building a weight map, which is typically the inverse variance.

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
        noise_type = noise['type']
    else:
        noise_type = 'Poisson'  # Default is Poisson
    if noise_type not in valid_noise_types:
        raise AttributeError("Invalid type %s for noise",noise_type)

    orig_index = config.get('index_key','image_num')
    if orig_index == 'obj_num':
        config['index_key'] = 'image_num'

    builder = valid_noise_types[noise_type]
    builder.addNoiseVariance(noise, config, im, include_obj_var, logger)

    if orig_index == 'obj_num':
        config['index_key'] = orig_index

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

class NoiseBuilder(object):
    """A base class for building noise objects and applying the noise to images.

    The base class doesn't do anything, but it defines the call signatures of the methods
    that derived classes should use for the different specific noise types.
    """
    def addNoise(self, config, base, im, rng, current_var, logger):
        """Read the noise parameters from the config dict and add the appropriate noise to the
        given image.

        @param config           The configuration dict for the noise field.
        @param base             The base configuration dict.
        @param im               The image onto which to add the noise
        @param rng              The random number generator to use for adding the noise.
        @param current_var      The current noise variance present in the image already [default: 0]
        @param logger           If given, a logger object to log progress.
        """
        raise NotImplemented("The %s class has not overridden addNoise"%self.__class__)

    def getNoiseVariance(self, config, base):
        """Read the noise parameters from the config dict and return the variance.

        @param config           The configuration dict for the noise field.
        @param base             The base configuration dict.

        @returns the variance of the noise model
        """
        raise NotImplemented("The %s class has not overridden addNoise"%self.__class__)

    def addNoiseVariance(self, config, base, im, include_obj_var, logger):
        """Read the noise parameters from the config dict and add the appropriate noise variance
        to the given image.

        This is used for constructing the weight map iamge.  It doesn't add a random value to
        each pixel.  Rather, it adds the variance of the noise that was used in the main image to
        each pixel in this image.

        This method has a default implemenation that is appropriate for noise models that have
        a constant noise variance.  It just gets the variance from getNoiseVariance and adds
        that constant value to every pixel.

        @param config           The configuration dict for the noise field.
        @param base             The base configuration dict.
        @param im               The image onto which to add the noise variance
        @param include_obj_var  Whether the noise variance values should the photon noise from
                                object flux in addition to the sky flux.  Only relevant for
                                noise models that are based on the image flux values such as
                                Poisson and CCDNoise.
        @param logger           If given, a logger object to log progress.
        """
        im += self.getNoiseVariance(config, base)

#
# Gaussian
#

class GaussianNoiseBuilder(NoiseBuilder):

    def addNoise(self, config, base, im, rng, current_var, logger):

        # Read the noise variance
        var = self.getNoiseVariance(config, base)

        # If we already have some variance in the image (from whitening), then we subtract this much
        # from sigma**2.
        if current_var:
            if logger:
                logger.debug('image %d, obj %d: Target variance is %f, current variance is %f',
                            base['image_num'],base['obj_num'],var,current_var)
            if var < current_var:
                raise RuntimeError(
                    "Whitening already added more noise than the requested Gaussian noise.")
            var -= current_var

        # Now apply the noise.
        import math
        sigma = math.sqrt(var)
        im.addNoise(galsim.GaussianNoise(rng,sigma=sigma))

        if logger:
            logger.debug('image %d, obj %d: Added Gaussian noise with var = %f',
                        base['image_num'],base['obj_num'],var)

    def getNoiseVariance(self, config, base):

        # The noise level can be specified either as a sigma or a variance.  Here we just calculate
        # the value of the variance from either one.
        single = [ { 'sigma' : float , 'variance' : float } ]
        params = galsim.config.GetAllParams(config, base, single=single, ignore=noise_ignore)[0]
        if 'sigma' in params:
            sigma = params['sigma']
            return sigma * sigma
        else:
            return params['variance']


#
# Poisson
#

class PoissonNoiseBuilder(NoiseBuilder):

    def addNoise(self, config, base, im, rng, current_var, logger):

        # Get how much extra sky to assume from the image.noise attribute.
        sky = GetSky(base['image'], base)
        extra_sky = GetSky(config, base)
        if not sky and not extra_sky:
            raise AttributeError(
                "Must provide either sky_level or sky_level_pixel for noise.type = Poisson")

        # If we already have some variance in the image (from whitening), then we subtract this
        # much off of the sky level.  It's not precisely accurate, since the existing variance is
        # Gaussian, rather than Poisson, but it's the best we can do.
        if current_var:
            if logger:
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

        # At this point, there is a slight difference between fft and phot. For photon shooting,
        # the galaxy already has Poisson noise, so we want to make sure not to add that again!
        draw_method = galsim.config.GetCurrentValue('stamp.draw_method',base,str)
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
                    # This deviate adds a noisy version of the sky, so need to subtract the mean
                    # back off.
                    im -= total_sky
        else:
            im += extra_sky
            # Do the normal PoissonNoise calculation.
            im.addNoise(galsim.PoissonNoise(rng))
            im -= extra_sky

        if logger:
            logger.debug('image %d, obj %d: Added Poisson noise', base['image_num'],base['obj_num'])

    def getNoiseVariance(self, config, base):
        # The noise variance is the net sky level per pixel
        sky = GetSky(base['image'], base)
        sky += GetSky(config, base)
        return sky

    def addNoiseVariance(self, config, base, im, include_obj_var, logger):
        if include_obj_var:
            # The current image at this point should be the noise-free, sky-free image,
            # which is the object variance in each pixel.
            im += base['current_image']

            # Note: For the phot case, we don't actually have an exact value for the variance in
            # each pixel, but the drawn image before adding the Poisson noise is our best guess for
            # the variance from the object's flux, so if we want the object variance included, this
            # is still the best we can do.

        # Add the total sky level
        im += self.getNoiseVariance(config, base)


#
# CCD
#

class CCDNoiseBuilder(NoiseBuilder):

    def getCCDNoiseParams(self, config, base):
        opt = { 'gain' : float , 'read_noise' : float }
        ignore = ['sky_level', 'sky_level_pixel']
        params = galsim.config.GetAllParams(config, base, opt=opt, ignore=noise_ignore + ignore)[0]
        gain = params.get('gain',1.0)
        read_noise = params.get('read_noise',0.0)
        read_noise_var = read_noise**2

        return gain, read_noise, read_noise_var

    def addNoise(self, config, base, im, rng, current_var, logger):

        # This process goes a lot like the Poisson routine.  There are just two differences.
        # First, the Poisson noise is in electrons, not ADU, and now we allow for a gain = e-/ADU,
        # so we need to account for that properly.  Second, we also allow for an additional Gaussian
        # read noise.
        gain, read_noise, read_noise_var = self.getCCDNoiseParams(config, base)

        # Get how much extra sky to assume from the image.noise attribute.
        sky = GetSky(base['image'], base)
        extra_sky = GetSky(config, base)
        if not sky and not extra_sky:
            raise AttributeError(
                "Must provide either sky_level or sky_level_pixel for noise.type = Poisson")

        # If we already have some variance in the image (from whitening), then we try to subtract
        # t from the read noise if possible.  If now, we subtract the rest off of the sky level.
        # It's not precisely accurate, since the existing variance is Gaussian, rather than
        # Poisson, but it's the best we can do.
        if current_var:
            if logger:
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

        # At this point, there is a slight difference between fft and phot. For photon shooting,
        # the galaxy already has Poisson noise, so we want to make sure not to add that again!
        draw_method = galsim.config.GetCurrentValue('stamp.draw_method',base,str)
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
                    pd = galsim.PoissonDeviate(rng, mean=total_sky*gain)
                    im.addNoise(galsim.DeviateNoise(pd))
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

        if logger:
            logger.debug('image %d, obj %d: Added CCD noise with gain = %f, read_noise = %f',
                        base['image_num'],base['obj_num'],gain,read_noise)

    def getNoiseVariance(self, config, base):
        # The noise variance is sky / gain + read_noise^2
        gain, read_noise, read_noise_var = self.getCCDNoiseParams(config, base)

        # Start with the background sky level for the image
        sky = GetSky(base['image'], base)
        sky += GetSky(config, base)

        # Account for the gain and read_noise
        return sky / gain + read_noise_var

    def addNoiseVariance(self, config, base, im, include_obj_var, logger):
        gain, read_noise, read_noise_var = self.getCCDNoiseParams(config, base)
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

class COSMOSNoiseBuilder(NoiseBuilder):

    def getCOSMOSNoise(self, config, base, rng=None):
        # Save the constructed CorrelatedNoise object, since we might need it again.
        tag = (base['file_num'], base['image_num'])
        if config.get('current_cn_tag',None) == tag:
            return config['current_cn']
        else:
            req = { 'file_name' : str }
            opt = { 'cosmos_scale' : float, 'variance' : float }

            kwargs = galsim.config.GetAllParams(config, base, req=req, opt=opt,
                                                ignore=noise_ignore)[0]
            if rng is None:
                rng = base['rng']
            cn = galsim.correlatednoise.getCOSMOSNoise(rng, **kwargs)
            config['current_cn'] = cn
            config['current_cn_tag'] = tag
            return cn

    def addNoise(self, config, base, im, rng, current_var, logger):

        # Build the correlated noise
        cn = self.getCOSMOSNoise(config,base,rng)
        var = cn.getVariance()

        # Subtract off the current variance if any
        if current_var:
            if logger:
                logger.debug('image %d, obj %d: Target variance is %f, current variance is %f',
                            base['image_num'],base['obj_num'], var, current_var)
            if var < current_var:
                raise RuntimeError(
                    "Whitening already added more noise than the requested COSMOS noise.")
            cn -= galsim.UncorrelatedNoise(rng, im.wcs, current_var)

        # Add the noise to the image
        im.addNoise(cn)

        if logger:
            logger.debug('image %d, obj %d: Added COSMOS correlated noise with variance = %f',
                        base['image_num'],base['obj_num'],var)

    def getNoiseVariance(self, config, base):
        cn = self.getCOSMOSNoise(config,base)
        return cn.getVariance()


def RegisterNoiseType(noise_type, builder):
    """Register a noise type for use by the config apparatus.

    @param noise_type       The name of the type in config['image']['noise']
    @param builder          A builder object to use for building the noise.  It should be an
                            instance of a subclass of NoiseBuilder.
    """
    valid_noise_types[noise_type] = builder

RegisterNoiseType('Gaussian', GaussianNoiseBuilder())
RegisterNoiseType('Poisson', PoissonNoiseBuilder())
RegisterNoiseType('CCD', CCDNoiseBuilder())
RegisterNoiseType('COSMOS', COSMOSNoiseBuilder())

