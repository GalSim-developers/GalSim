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

def PowerSpectrumInit(ps, config, base):
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
        raise ValueError("Invalid max_mu=%f (must be > 0) for PowerSpectrumMagnification"%max_mu)

    if mu < 0 or mu > max_mu:
        import warnings
        warnings.warn("Warning: PowerSpectrum mu = %f means strong lensing!  Using mu=%f"%(
            mu,max_mu))
        mu = max_mu

    #print base['obj_num'],'PS mu = ',mu
    return mu, False

