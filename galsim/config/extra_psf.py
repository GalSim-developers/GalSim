# Copyright (c) 2012-2018 by the GalSim developers team on GitHub
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
import math
import numpy as np
import logging

# The psf extra output type builds an Image of the PSF at the same locations as the galaxies.
from .stamp import valid_draw_methods

# The code the actually draws the PSF on a postage stamp.
def DrawPSFStamp(psf, config, base, bounds, offset, method, logger):
    """
    Draw an image using the given psf profile.

    @returns the resulting image.
    """
    if 'draw_method' in config:
        method = galsim.config.ParseValue(config,'draw_method',base,str)[0]
        if method not in valid_draw_methods:
            raise galsim.GalSimConfigValueError("Invalid draw_method.", method, valid_draw_methods)
    else:
        method = 'auto'

    if 'flux' in config:
        flux = galsim.config.ParseValue(config,'flux',base,float)[0]
        psf = psf.withFlux(flux)

    if method == 'phot':
        rng = galsim.config.GetRNG(config, base)
        n_photons = psf.flux
    else:
        rng = None
        n_photons = 0

    wcs = base['wcs'].local(base['image_pos'])
    im = galsim.ImageF(bounds, wcs=wcs)
    im = psf.drawImage(image=im, offset=offset, method=method, rng=rng, n_photons=n_photons)

    if 'signal_to_noise' in config:
        if 'flux' in config:
            raise galsim.GalSimConfigError(
                "Cannot specify both flux and signal_to_noise for psf output")
        if method == 'phot':
            raise galsim.GalSimConfigError(
                "signal_to_noise option not implemented for draw_method = phot")

        if 'image' in base and 'noise' in base['image']:
            noise_var = galsim.config.CalculateNoiseVariance(base)
        else:
            raise galsim.GalSimConfigError(
                "Need to specify noise level when using psf.signal_to_noise")

        sn_target = galsim.config.ParseValue(config, 'signal_to_noise', base, float)[0]

        sn_meas = math.sqrt( np.sum(im.array**2, dtype=float) / noise_var )
        flux = sn_target / sn_meas
        im *= flux

    return im


# The function to call at the end of building each stamp
from .extra import ExtraOutputBuilder
class ExtraPSFBuilder(ExtraOutputBuilder):
    """Build an image that draws the PSF at the same location as each object on the main image.

    This makes the most sense when the main image consists of non-overlapping stamps, such as
    a TiledImage, since you wouldn't typically want the PSF images to overlap.  But it just
    follows whatever pattern of stamp locations the main image has.
    """
    def processStamp(self, obj_num, config, base, logger):
        # If this doesn't exist, an appropriate exception will be raised.
        psf = base['psf']['current'][0]
        draw_method = galsim.config.GetCurrentValue('draw_method', base['stamp'], str, base)
        bounds = base['current_stamp'].bounds

        # Check if we should shift the psf:
        if 'shift' in config:
            # Special: output.psf.shift = 'galaxy' means use the galaxy shift.
            if config['shift'] == 'galaxy':
                # This shift value might be in either stamp or gal.
                if 'shift' in base['stamp']:
                    shift = galsim.config.GetCurrentValue('shift', base['stamp'],
                                                          galsim.PositionD, base)
                else:
                    # This will raise an appropriate error if there is no gal.shift or stamp.shift.
                    shift = galsim.config.GetCurrentValue('shift', base['gal'],
                                                          galsim.PositionD, base)
            else:
                shift = galsim.config.ParseValue(config, 'shift', base, galsim.PositionD)[0]
            logger.debug('obj %d: psf shift: %s',base.get('obj_num',0),str(shift))
            psf = psf.shift(shift)

        # Start with the offset required just due to the stamp size/shape.
        offset = base['stamp_offset']
        # Check if we should apply any additional offset:
        if 'offset' in config:
            # Special: output.psf.offset = 'galaxy' means use the same offset as in the galaxy
            #          image, which is actually in config.stamp, not config.gal.
            if config['offset'] == 'galaxy':
                offset += galsim.config.GetCurrentValue('offset', base['stamp'],
                                                        galsim.PositionD, base)
            else:
                offset += galsim.config.ParseValue(config, 'offset', base, galsim.PositionD)[0]
            logger.debug('obj %d: psf offset: %s',base.get('obj_num',0),str(offset))

        psf_im = DrawPSFStamp(psf,config,base,bounds,offset,draw_method,logger)
        if 'signal_to_noise' in config:
            base['current_noise_image'] = base['current_stamp']
            galsim.config.AddNoise(base,psf_im,current_var=0,logger=logger)
        self.scratch[obj_num] = psf_im

    # The function to call at the end of building each image
    def processImage(self, index, obj_nums, config, base, logger):
        image = galsim.ImageF(base['image_bounds'], wcs=base['wcs'], init_value=0.)
        # Make sure to only use the stamps for objects in this image.
        for obj_num in obj_nums:
            stamp = self.scratch[obj_num]
            b = stamp.bounds & image.bounds
            logger.debug('image %d: psf image at b = %s = %s & %s',
                         base['image_num'],b,stamp.bounds,image.bounds)
            if b.isDefined(): # pragma: no branch  (We normally guard against this already.)
                image[b] += stamp[b]
                logger.debug('obj %d: added psf image to main image',base.get('obj_num',0))
        self.data[index] = image


# Register this as a valid extra output
from .extra import RegisterExtraOutput
RegisterExtraOutput('psf', ExtraPSFBuilder())
