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

def _GenerateFromNFWHaloShear(config, base, value_type):
    """@brief Return a shear calculated from an NFWHalo object.
    """
    nfw_halo = galsim.config.GetInputObj('nfw_halo', config, base, 'NFWHaloShear')

    if 'world_pos' not in base:
        raise ValueError("NFWHaloShear requested, but no position defined.")
    pos = base['world_pos']

    if 'gal' not in base or 'redshift' not in base['gal']:
        raise ValueError("NFWHaloShear requested, but no gal.redshift defined.")
    redshift = galsim.config.GetCurrentValue(base['gal'],'redshift')

    # There aren't any parameters for this, so just make sure num is the only (optional)
    # one present.
    galsim.config.CheckAllParams(config, opt={ 'num' : int })

    try:
        g1,g2 = nfw_halo.getShear(pos,redshift)
        shear = galsim.Shear(g1=g1,g2=g2)
    except Exception as e:
        import warnings
        warnings.warn("Warning: NFWHalo shear is invalid -- probably strong lensing!  " +
                      "Using shear = 0.")
        shear = galsim.Shear(g1=0,g2=0)

    #print base['obj_num'],'NFW shear = ',shear
    return shear, False


def _GenerateFromNFWHaloMagnification(config, base, value_type):
    """@brief Return a magnification calculated from an NFWHalo object.
    """
    nfw_halo = galsim.config.GetInputObj('nfw_halo', config, base, 'NFWHaloMagnification')

    if 'world_pos' not in base:
        raise ValueError("NFWHaloMagnification requested, but no position defined.")
    pos = base['world_pos']

    if 'gal' not in base or 'redshift' not in base['gal']:
        raise ValueError("NFWHaloMagnification requested, but no gal.redshift defined.")
    redshift = galsim.config.GetCurrentValue(base['gal'],'redshift')

    opt = { 'max_mu' : float, 'num' : int }
    kwargs = galsim.config.GetAllParams(config, base, opt=opt)[0]

    mu = nfw_halo.getMagnification(pos,redshift)

    max_mu = kwargs.get('max_mu', 25.)
    if not max_mu > 0.: 
        raise ValueError("Invalid max_mu=%f (must be > 0) for NFWHaloMagnification"%max_mu)

    if mu < 0 or mu > max_mu:
        import warnings
        warnings.warn("Warning: NFWHalo mu = %f means strong lensing!  Using mu=%f"%(mu,max_mu))
        mu = max_mu

    #print base['obj_num'],'NFW mu = ',mu
    return mu, False


