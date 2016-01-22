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

# This file adds input type nfw_halo and value types PowerSpectrumShear and
# PowerSpectrumMagnification.

# A PowerSpectrum input type requires a special initialization at the start of each image
# to build the shear grid.  This is done in SetupPowerSpecrum.  There are also a couple of
# parameters that are specific to that step, which we want to ignore when getting the 
# initialization kwargs, so we define a special GetPowerSpectrumKwargs function here.

def GetPowerSpectrumKwargs(config, base):
    """Get the initialization kwargs for PowerSpectrum from the config dict.

    @param config       The configuration dict for 'power_spectrum'
    @param base         The base configuration dict.
    """
    # Ignore these parameters here, since they are for the buildGrid step, not the initialization
    # of the PowerSpectrum object.
    ignore = ['grid_spacing', 'interpolant']
    opt = galsim.PowerSpectrum._opt_params
    return galsim.config.GetAllParams(config, base, opt=opt, ignore=ignore)

def SetupPowerSpectrum(ps, config, base, logger=None):
    """Set up the PowerSpectrum input object's gridded values based on the
    size of the image and the grid spacing.

    @param ps           The PowerSpectrum object to use
    @param config       The configuration dict for 'power_spectrum'
    @param base         The base configuration dict.
    @param logger       If given, a logger object to log progress.  [default: None]
    """
    if 'grid_spacing' in config:
        grid_spacing = galsim.config.ParseValue(config, 'grid_spacing', base, float)[0]
    elif 'tile_xsize' in base:
        # Then we have a tiled image.  Can use the tile spacing as the grid spacing.
        stamp_size = min(base['tile_xsize'], base['tile_ysize'])
        # Note: we use the (max) pixel scale at the image center.  This isn't
        # necessarily optimal, but it seems like the best choice.
        scale = base['wcs'].maxLinearScale(base['image_center'])
        grid_spacing = stamp_size * scale
    else:
        raise AttributeError("power_spectrum.grid_spacing required for non-tiled images")

    if 'tile_xsize' in base and base['tile_xsize'] == base['tile_ysize']:
        # PowerSpectrum can only do a square FFT, so make it the larger of the two n's.
        ngrid = max(base['nx_tiles'], base['ny_tiles'])
        # Normally that's good, but if tiles aren't square, need to drop through to the
        # second option.
    else:
        import math
        image_size = max(base['image_xsize'], base['image_ysize'])
        scale = base['wcs'].maxLinearScale(base['image_center'])
        ngrid = int(math.ceil(image_size * scale / grid_spacing))

    if 'interpolant' in config:
        interpolant = galsim.config.ParseValue(config, 'interpolant', base, str)[0]
    else:
        interpolant = None

    # We don't care about the output here.  This just builds the grid, which we'll
    # access for each object using its position.
    if base['wcs'].isCelestial():
        world_center = galsim.PositionD(0,0)
    else:
        world_center = base['wcs'].toWorld(base['image_center'])
    ps.buildGrid(grid_spacing=grid_spacing, ngrid=ngrid, center=world_center,
                 rng=base['rng'], interpolant=interpolant)

    # Make sure this process gives consistent results regardless of the number of processes
    # being used.
    if not isinstance(ps, galsim.PowerSpectrum):
        # Then ps is really a proxy, which means the rng was pickled, so we need to
        # discard the same number of random calls from the one in the config dict.
        base['rng'].discard(ps.nRandCallsForBuildGrid())

# Register this as a valid input type
from .input import RegisterInputType
RegisterInputType('power_spectrum', galsim.PowerSpectrum,
                  types=['PowerSpectrumShear','PowerSpectrumMagnification'],
                  kwargs_func=GetPowerSpectrumKwargs,
                  setup_func=SetupPowerSpectrum)


# There are two value types associated with this: PowerSpectrumShear and 
# PowerSpectrumMagnification.

def _GenerateFromPowerSpectrumShear(config, base, value_type):
    """@brief Return a shear calculated from a PowerSpectrum object.
    """
    power_spectrum = galsim.config.GetInputObj('power_spectrum', config, base, 'PowerSpectrumShear')

    if 'world_pos' not in base:
        raise ValueError("PowerSpectrumShear requested, but no position defined.")
    pos = base['world_pos']

    # There aren't any parameters for this, so just make sure num is the only (optional)
    # one present.
    galsim.config.CheckAllParams(config, opt={ 'num' : int })

    try:
        g1,g2 = power_spectrum.getShear(pos)
        shear = galsim.Shear(g1=g1,g2=g2)
    except Exception as e:
        import warnings
        warnings.warn("Warning: PowerSpectrum shear is invalid -- probably strong lensing!  " +
                      "Using shear = 0.")
        shear = galsim.Shear(g1=0,g2=0)

    #print base['obj_num'],'PS shear = ',shear
    return shear, False

def _GenerateFromPowerSpectrumMagnification(config, base, value_type):
    """@brief Return a magnification calculated from a PowerSpectrum object.
    """
    power_spectrum = galsim.config.GetInputObj('power_spectrum', config, base,
                                               'PowerSpectrumMagnification')

    if 'world_pos' not in base:
        raise ValueError("PowerSpectrumMagnification requested, but no position defined.")
    pos = base['world_pos']

    opt = { 'max_mu' : float, 'num' : int }
    kwargs = galsim.config.GetAllParams(config, base, opt=opt)[0]

    mu = power_spectrum.getMagnification(pos)

    max_mu = kwargs.get('max_mu', 25.)
    if not max_mu > 0.: 
        raise ValueError(
            "Invalid max_mu=%f (must be > 0) for type = PowerSpectrumMagnification"%max_mu)

    if mu < 0 or mu > max_mu:
        import warnings
        warnings.warn("Warning: PowerSpectrum mu = %f means strong lensing!  Using mu=%f"%(
            mu,max_mu))
        mu = max_mu

    #print base['obj_num'],'PS mu = ',mu
    return mu, False

# Register these as valid value types
from .value import RegisterValueType
RegisterValueType('PowerSpectrumShear', _GenerateFromPowerSpectrumShear, [ galsim.Shear ])
RegisterValueType('PowerSpectrumMagnification', _GenerateFromPowerSpectrumMagnification, [ float ])
