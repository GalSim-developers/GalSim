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

def DrawPSFStamp(psf, config, bounds, offset, method):
    """
    Draw an image using the given psf profile.

    @returns the resulting image.
    """

    if not psf:
        raise AttributeError("DrawPSFStamp requires psf to be provided.")

    if ('output' in config and 'psf' in config['output'] and 
        'draw_method' in config['output']['psf'] ):
        method = galsim.config.ParseValue(config['output']['psf'],'draw_method',config,str)[0]
        if method not in ['auto', 'fft', 'phot', 'real_space', 'no_pixel', 'sb']:
            raise AttributeError("Invalid draw_method: %s"%method)
    else:
        method = 'auto'

    # Special: if the galaxy was shifted, then also shift the psf 
    if 'shift' in config['gal']:
        gal_shift = galsim.config.GetCurrentValue(config['gal'],'shift')
        if False:
            logger.debug('obj %d: psf shift (1): %s',config['obj_num'],str(gal_shift))
        psf = psf.shift(gal_shift)

    wcs = config['wcs'].local(config['image_pos'])
    im = galsim.ImageF(bounds, wcs=wcs)
    im = psf.drawImage(image=im, offset=offset, method=method)

    if (('output' in config and 'psf' in config['output'] 
            and 'signal_to_noise' in config['output']['psf']) or
        ('gal' not in config and 'psf' in config and 'signal_to_noise' in config['psf'])):
        if method == 'phot':
            raise NotImplementedError(
                "signal_to_noise option not implemented for draw_method = phot")
        import math
        import numpy

        if 'image' in config and 'noise' in config['image']:
            noise_var = galsim.config.CalculateNoiseVar(config)
        else:
            raise AttributeError(
                "Need to specify noise level when using psf.signal_to_noise")

        if ('output' in config and 'psf' in config['output'] 
                and 'signal_to_noise' in config['output']['psf']):
            cf = config['output']['psf']
        else:
            cf = config['psf']
        sn_target = galsim.config.ParseValue(cf, 'signal_to_noise', config, float)[0]
            
        sn_meas = math.sqrt( numpy.sum(im.array**2) / noise_var )
        flux = sn_target / sn_meas
        im *= flux

    return im
           

