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
"""
@file wfirst_detectors.py

Part of the WFIRST module.  This file includes helper routines to apply image defects that are
specific to WFIRST.
"""

import galsim
import galsim.wfirst
import numpy as np
import os

from . import exptime as default_exptime


def applyNonlinearity(img):
    """
    Applies the WFIRST nonlinearity function to the supplied image `im`.

    For more information about nonlinearity, see the docstring for galsim.Image.applyNonlinearity.
    Unlike that routine, this one does not require any arguments, since it uses the nonlinearity
    function defined within the WFIRST module.

    After calling this method, the Image instance `img` is transformed to include the nonlinearity.

    @param img              The Image to be transformed.
    """
    img.applyNonlinearity(NLfunc=galsim.wfirst.NLfunc)

def addReciprocityFailure(img, exptime=default_exptime):
    """
    Accounts for the reciprocity failure for the WFIRST directors and includes it in the original
    Image `img` directly.

    For more information about reciprocity failure, see the docstring for
    galsim.Image.addReciprocityFailure.  Unlike that routine, this one does not need the parameters
    for reciprocity failure to be provided, though it still takes exposure time as an optional
    argument.

    @param img              The Image to be transformed.
    @param exptime          The exposure time (t) in seconds, which goes into the expression for
                            reciprocity failure given in the docstring.  If None, then the routine
                            will use the default WFIRST exposure time in galsim.wfirst.exptime.
                            [default: {exptime}]
    """.format(exptime=default_exptime)
    img.addReciprocityFailure(exp_time=exptime, alpha=galsim.wfirst.reciprocity_alpha,
                              base_flux=1.0)

def applyIPC(img, edge_treatment='extend', fill_value=None):
    """
    Applies the effect of interpixel capacitance (IPC) to the Image instance.

    For more information about IPC, see the docstring for galsim.Image.applyIPC.  Unlike that
    routine, this one does not need the IPC kernel to be specified, since it uses the IPC kernel
    defined within the WFIRST module.

    @param img                     The Image to be transformed.
    @param edge_treatment          Specifies the method of handling edges and should be one of
                                   'crop', 'extend' or 'wrap'. See galsim.Image.applyIPC docstring
                                   for more information.
                                   [default: 'extend']
    @param fill_value              Specifies the value (including nan) to fill the edges with when
                                   edge_treatment is 'crop'. If unspecified or set to 'None', the
                                   original pixel values are retained at the edges. If
                                   edge_treatment is not 'crop', then this is ignored.
    """
    img.applyIPC(galsim.wfirst.ipc_kernel, edge_treatment=edge_treatment, fill_value=fill_value)

def applyPersistence(img, prev_exposures):
    """
    Applies the persistence effect to the Image instance by adding a small fraction of the previous
    exposures (up to {ncoeff}) supplied as the 'prev_exposures' argument.

    For more information about persistence, see the docstring for galsim.Image.applyPersistence.
    Unlike that routine, this one does not need the coefficients to be specified. However, the list
    of previous eight exposures will have to be supplied. Earlier exposures, if supplied, will be
    ignored.

    @param img               The Image to be transformed.
    @param prev_exposures    List of up to {ncoeff} Image instances in the order of exposures, with the
                             recent exposure being the first element.
    """.format(ncoeff=len(galsim.wfirst.persistence_coefficients))
    if not hasattr(prev_exposures,'__iter__'):
        raise TypeError("In wfirst.applyPersistence, prev_exposures must be a list of Image "
                        "instances")

    n_exp = min(len(prev_exposures),len(galsim.wfirst.persistence_coefficients))
    img.applyPersistence(prev_exposures[:n_exp],galsim.wfirst.persistence_coefficients[:n_exp])

def allDetectorEffects(img, prev_exposures=(), rng=None, exptime=default_exptime):
    """
    This utility applies all sources of noise and detector effects for WFIRST that are implemented
    in GalSim.  In terms of noise, this includes the Poisson noise due to the signal (sky +
    background), dark current, and read noise.  The detector effects that are included are
    reciprocity failure, quantization, persistence, nonlinearity, and interpixel capacitance. It
    also includes the necessary factors of gain.  In short, the user should be able to pass in an
    Image with all sources of signal (background plus astronomical objects), and the Image will be
    modified to include all subsequent steps in the image generation process for WFIRST that are
    implemented in GalSim. However, to include the effect of persistence, the user needs to provide
    a list of up to {ncoeff} recent exposures (without the readout effects such nonlinearity and
    interpixel capacitance included) and the routine returns an updated list of up to {ncoeff}
    recent exposures.

    @param img               The Image to be modified.
    @param prev_exposures    List of up to {ncoeff} Image instances in the order of exposures, with
                             the recent exposure being the first element. [default: []]
    @param rng               An optional galsim.BaseDeviate to use for the addition of noise.  If
                             None, a new one will be initialized.  [default: None]
    @param exptime           The exposure time, in seconds.  If None, then the WFIRST default
                             exposure time will be used.  [default: {exptime}]

    @returns prev_exposures  Updated list of previous exposures containing up to {ncoeff} Image
                             instances.
    """.format(ncoeff=len(galsim.wfirst.persistence_coefficients), exptime=default_exptime)

    # Make sure we don't have any negative values.
    img.replaceNegative(0.)

    # Add Poisson noise.
    rng = galsim.BaseDeviate(rng)
    poisson_noise = galsim.PoissonNoise(rng)
    img.addNoise(poisson_noise)

    # Reciprocity failure (use WFIRST routine, with the supplied exposure time).
    addReciprocityFailure(img, exptime=exptime)

    # Quantize.
    img.quantize()

    # Dark current (use exposure time).
    dark_current = galsim.wfirst.dark_current*exptime
    dark_noise = galsim.DeviateNoise(galsim.PoissonDeviate(rng, dark_current))
    img.addNoise(dark_noise)

    # Persistence (use WFIRST coefficients)
    prev_exposures = list(prev_exposures)
    applyPersistence(img,prev_exposures)

    # Update the 'prev_exposures' queue
    ncoeff = len(galsim.wfirst.persistence_coefficients)
    prev_exposures = [img.copy()] + prev_exposures[:ncoeff-1]

    # Nonlinearity (use WFIRST routine).
    applyNonlinearity(img)

    # IPC (use WFIRST routine).
    applyIPC(img)

    # Read noise.
    read_noise = galsim.GaussianNoise(rng, sigma=galsim.wfirst.read_noise)
    img.addNoise(read_noise)

    # Gain.
    img /= galsim.wfirst.gain

    # Quantize.
    img.quantize()

    return prev_exposures
