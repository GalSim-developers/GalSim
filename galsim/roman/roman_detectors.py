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
"""
@file roman_detectors.py

Part of the Roman Space Telescope module.  This file includes helper routines to apply image
defects that are specific to Roman.
"""

import numpy as np
import os

from . import exptime, persistence_coefficients, nonlinearity_beta

def NLfunc(x):
    return x + nonlinearity_beta*(x**2)

def applyNonlinearity(img):
    """
    Applies the Roman nonlinearity function to the supplied image ``im``.

    For more information about nonlinearity, see the docstring for galsim.Image.applyNonlinearity.
    Unlike that routine, this one does not require any arguments, since it uses the nonlinearity
    function defined within the Roman module.

    After calling this method, the Image instance ``img`` is transformed to include the
    nonlinearity.

    Parameters:
        img:        The Image to be transformed.
    """
    img.applyNonlinearity(NLfunc=NLfunc)

def addReciprocityFailure(img, exptime=exptime):
    from . import reciprocity_alpha
    img.addReciprocityFailure(exp_time=exptime, alpha=reciprocity_alpha, base_flux=1.0)

# Note: Formatted doc strings don't work if put in the normal place.  Unless the function is
# actually called, the formatting statement is never executed.  So put it here instead.
addReciprocityFailure.__doc__ = """
Accounts for the reciprocity failure for the Roman directors and includes it in the original
Image ``img`` directly.

For more information about reciprocity failure, see the docstring for
galsim.Image.addReciprocityFailure.  Unlike that routine, this one does not need the parameters
for reciprocity failure to be provided, though it still takes exposure time as an optional
argument.

Parameters:
    img:            The Image to be transformed.
    exptime:        The exposure time (t) in seconds, which goes into the expression for
                    reciprocity failure given in the docstring.  If None, then the routine
                    will use the default Roman exposure time in galsim.roman.exptime.
                    [default: {exptime}]
""".format(exptime=exptime)


def applyIPC(img, edge_treatment='extend', fill_value=None):
    """
    Applies the effect of interpixel capacitance (IPC) to the Image instance.

    For more information about IPC, see the docstring for galsim.Image.applyIPC.  Unlike that
    routine, this one does not need the IPC kernel to be specified, since it uses the IPC kernel
    defined within the Roman module.

    Parameters:
        img:                The Image to be transformed.
        edge_treatment:     Specifies the method of handling edges and should be one of
                            'crop', 'extend' or 'wrap'. See galsim.Image.applyIPC docstring
                            for more information.
                            [default: 'extend']
        fill_value:         Specifies the value (including nan) to fill the edges with when
                            edge_treatment is 'crop'. If unspecified or set to 'None', the
                            original pixel values are retained at the edges. If
                            edge_treatment is not 'crop', then this is ignored.
    """
    from . import ipc_kernel
    img.applyIPC(ipc_kernel, edge_treatment=edge_treatment, fill_value=fill_value)

def applyPersistence(img, prev_exposures, method='fermi'):
    from .. import GalSimValueError

    if not hasattr(prev_exposures,'__iter__'):
        raise TypeError("In roman.applyPersistence, prev_exposures must be a list of Image instances")

    if method == 'linear':

        n_exp = min(len(prev_exposures),len(persistence_coefficients))
        img.applyPersistence(prev_exposures[:n_exp], persistence_coefficients[:n_exp])

    elif method == 'fermi':

        n_exp = len(prev_exposures)
        for i in range(n_exp):
            # The slew/settle time and the reset time should be specified.
            # Now we simply assume them as 0 and take the persitence current at the mid-time of
            # exposures as the average persistence until we get more information about the
            # observation timeline.
            img.array[:,:] += fermi_linear(prev_exposures[i].array, (0.5+i)*exptime)*exptime

    else:
        raise GalSimValueError("applyPersistence only accepts 'linear' or 'fermi' methods, got",
                               method)

# Again, need to put the doc outside the function to get formatting to work.
applyPersistence.__doc__ = """
This method applies either of the two different persistence models: 'linear' and 'fermi'.
Slew between pointings and consecutive resets after illumination are not considered.

'linear' persistence model
    Applies the persistence effect to the Image instance by adding a small fraction of the
    previous exposures (up to {ncoeff}) supplied as the 'prev_exposures' argument.
    For more information about persistence, see `galsim.Image.applyPersistence`.
    Unlike that routine, this one does not need the coefficients to be specified. However,
    the list of previous {ncoeff} exposures will have to be supplied. Earlier exposures, if
    supplied, will be ignored.

'fermi' persistence model
    Applies the persistence effect to the Image instance by adding the accumulated persistence
    dark current of previous exposures supplied as the 'prev_exposures' argument.
    Unlike galsim.Image.applyPersistence, this one does not use constant coefficients but a
    fermi model plus a linear tail below half of saturation.

    For more info about the fermi model, see:

    http://www.stsci.edu/hst/wfc3/ins_performance/persistence/

Parameters:
    img:                The Image to be transformed.
    prev_exposures:     List of Image instances in the order of exposures, with the recent
                        exposure being the first element. In the linear model, the exposures
                        exceeding the limit ({ncoeff} exposures) will be ignored.
    method:             The persistence model ('linear' or 'fermi') to be applied.
                        [default: 'fermi']
""".format(ncoeff=len(persistence_coefficients))


def fermi_linear(x, t):
    """
    The fermi model for persistence: A* (x/x0)**a * (t/1000.)**(-r) / (exp( -(x-x0)/dx ) +1. )
    For influence level below the half well, the persistence is linear in x.

    Parameters:
        x:      Array of pixel influence levels in unit of electron counts.
        t:      Time (in seconds) since reset.

    Returns:
        The persistence signal of the input exposure x.
    """
    from . import persistence_fermi_parameters

    x = np.asarray(x)
    y = np.zeros_like(x)

    A, x0, dx, a, r, half_well = persistence_fermi_parameters
    ps    = A* (    x    /x0)**a * (t/1000.)**(-r)/(np.exp( -(x-x0)/dx) +1.)
    ps_hf = A* (half_well/x0)**a * (t/1000.)**(-r)/(np.exp( -(half_well-x0)/dx) +1.)

    mask1 = x > half_well
    mask2 = (x > 0.) & (x <= half_well)

    y[mask1] += ps[mask1]
    y[mask2] += ps_hf*x[mask2]/half_well

    return y

# Again, need to put the doc outside the function to get formatting to work.
def allDetectorEffects(img, prev_exposures=(), rng=None, exptime=exptime):
    from . import exptime, dark_current, read_noise, gain
    from .. import BaseDeviate, PoissonNoise, DeviateNoise, GaussianNoise, PoissonDeviate

    # Make sure we don't have any negative values.
    img.replaceNegative(0.)

    # Add Poisson noise.
    rng = BaseDeviate(rng)
    poisson_noise = PoissonNoise(rng)
    img.addNoise(poisson_noise)

    # Quantize: have an integer number of photons in every pixel after inclusion of sky noise.
    img.quantize()

    # Reciprocity failure (use Roman routine, with the supplied exposure time).
    addReciprocityFailure(img, exptime=exptime)

    # Dark current (use exposure time).
    total_dark_current = dark_current*exptime
    dark_noise = DeviateNoise(PoissonDeviate(rng, total_dark_current))
    img.addNoise(dark_noise)

    # Persistence (use Roman H4RG-lo fermi model)
    prev_exposures = list(prev_exposures)
    applyPersistence(img, prev_exposures, method='fermi')
    # Update the 'prev_exposures' queue.
    prev_exposures = [img.copy()] + prev_exposures[:]

    # Nonlinearity (use Roman routine).
    applyNonlinearity(img)

    # IPC (use Roman routine).
    applyIPC(img)

    # Read noise.
    gn = GaussianNoise(rng, sigma=read_noise)
    img.addNoise(gn)

    # Gain.
    img /= gain

    # Quantize.
    img.quantize()

    return prev_exposures

allDetectorEffects.__doc__ = """
This utility applies all sources of noise and detector effects for Roman that are implemented
in GalSim.  In terms of noise, this includes the Poisson noise due to the signal (sky +
background), dark current, and read noise.  The detector effects that are included are
reciprocity failure, quantization, persistence, nonlinearity, and interpixel capacitance. It
also includes the necessary factors of gain.  In short, the user should be able to pass in an
Image with all sources of signal (background plus astronomical objects), and the Image will be
modified to include all subsequent steps in the image generation process for Roman that are
implemented in GalSim. However, to include the effect of persistence, the user needs to provide
a list of recent exposures (without the readout effects) and the routine
returns an updated list of recent exposures.

Parameters:
    img:            The Image to be modified.
    prev_exposures: List of Image instances in the order of exposures, with
                    the recent exposure being the first element. [default: ()]
    rng:            An optional galsim.BaseDeviate to use for the addition of noise.  If
                    None, a new one will be initialized.  [default: None]
    exptime:        The exposure time, in seconds.  If None, then the Roman default
                    exposure time will be used.  [default: {exptime}]

Returns:
    prev_exposures: Updated list of previous exposures Image instances.
""".format(exptime=exptime)

