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
"""
This code checks whether a galaxy in a particular postage stamp passes the cuts imposed in GREAT3.
It includes cuts on resolution and S/N, and can easily be changed to define resolution or S/N in a
different way.
"""
import galsim
import numpy as np

def Great3Reject(config, base, value_type):
    """See if a drawn postage stamp should be rejected based on Great3 cuts.
    """
    stamp = base['current_stamp']
    im = config['current_image']
    gal = base['gal']['current_val']
    psf = base['psf']['current_val']

    reject = False

    # Check the resolution.  This depends on the PSF-convolved galaxy image and the PSF image.  The
    # latter will be the same for all galaxies in the same image, so we can draw the PSF image and
    # cache it, or retrieve it from the cache if it's already there.
    if 'cached_psf_image' in base and psf is base['cached_psf']:
        psf_image = base['cached_psf_image']
    else:
        psf_image = psf.drawImage(scale=im.scale)
        base['cached_psf_image'] = psf_image
        base['cached_psf'] = psf
    # We will define resolution using the resolution factor from re-Gaussianization, which uses a
    # trace-based size estimate of the post-seeing object.  For simplicity, we run
    # re-Gaussianization directly.
    res = galsim.hsm.EstimateShearHSM(im, psf_image, strict='False')
    # Cut based on the resolution failure, or if the measurement simply failed.
    if res.error_message != "" or res.resolution_factor < 1./3:
        reject = True

    # Next we check the SNR using the idealized SNR estimator used for GREAT3.  For a derivation of
    # this calculation, see the docstring for the addNoiseSNR method.  This result is too optimistic
    # compared to one that might be used in reality.  Also, for realistic galaxies, it will be
    # confused by the noise in the image.  In GREAT3, we dealt with this by precomputing SNR values
    # using parametric galaxies models and using them even when calculating SNR for realistic
    # galaxies.  It is hard to imagine how to implement this here, but it would at least be better
    # to use some kind of Gaussian-weighted SNR estimator for the realistic galaxy branches.  Or,
    # subtract off the expected contribution given the value of 'current_var' in the postage stamp?
    #
    # First get the sum of pixel values squared:
    sumsq = np.sum(im.array**2)
    # Get the noise variance for this stamp:
    var = galsim.config.noise.NoiseVarGaussian(config, base)
    snr = np.sqrt(sumsq / var)
    # We nominally used a cut at SNR=17 (in an attempt to compensate for the SNR estimator being
    # overly optimistic) and also eliminated SNR>100.
    if snr < 17. or snr > 100.:
        reject = True

    return reject


galsim.config.RegisterValueType('Great3Reject', Great3Reject, [ bool ])

