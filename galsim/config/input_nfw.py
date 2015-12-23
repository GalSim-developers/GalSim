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

def _GenerateFromNFWHaloShear(param, param_name, base, value_type):
    """@brief Return a shear calculated from an NFWHalo object.
    """
    if 'world_pos' not in base:
        raise ValueError("NFWHaloShear requested, but no position defined.")
    pos = base['world_pos']

    if 'gal' not in base or 'redshift' not in base['gal']:
        raise ValueError("NFWHaloShear requested, but no gal.redshift defined.")
    redshift = galsim.config.GetCurrentValue(base['gal'],'redshift')

    if 'nfw_halo' not in base['input_objs']:
        raise ValueError("NFWHaloShear requested, but no input.nfw_halo defined.")

    opt = { 'num' : int }
    kwargs = galsim.config.GetAllParams(param, param_name, base, opt=opt)[0]

    num = kwargs.get('num',0)
    if num < 0:
        raise ValueError("Invalid num < 0 supplied for NFWHalowShear: num = %d"%num)
    if num >= len(base['input_objs']['nfw_halo']):
        raise ValueError("Invalid num supplied for NFWHaloShear (too large): num = %d"%num)
    nfw_halo = base['input_objs']['nfw_halo'][num]

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


def _GenerateFromNFWHaloMagnification(param, param_name, base, value_type):
    """@brief Return a magnification calculated from an NFWHalo object.
    """
    if 'world_pos' not in base:
        raise ValueError("NFWHaloMagnification requested, but no position defined.")
    pos = base['world_pos']

    if 'gal' not in base or 'redshift' not in base['gal']:
        raise ValueError("NFWHaloMagnification requested, but no gal.redshift defined.")
    redshift = galsim.config.GetCurrentValue(base['gal'],'redshift')

    if 'nfw_halo' not in base['input_objs']:
        raise ValueError("NFWHaloMagnification requested, but no input.nfw_halo defined.")
 
    opt = { 'max_mu' : float, 'num' : int }
    kwargs = galsim.config.GetAllParams(param, param_name, base, opt=opt)[0]

    num = kwargs.get('num',0)
    if num < 0:
        raise ValueError("Invalid num < 0 supplied for NFWHaloMagnification: num = %d"%num)
    if num >= len(base['input_objs']['nfw_halo']):
        raise ValueError("Invalid num supplied for NFWHaloMagnification (too large): num = %d"%num)
    nfw_halo = base['input_objs']['nfw_halo'][num]

    mu = nfw_halo.getMagnification(pos,redshift)

    max_mu = kwargs.get('max_mu', 25.)
    if not max_mu > 0.: 
        raise ValueError(
            "Invalid max_mu=%f (must be > 0) for %s.type = NFWHaloMagnification"%(
                max_mu,param_name))

    if mu < 0 or mu > max_mu:
        import warnings
        warnings.warn("Warning: NFWHalo mu = %f means strong lensing!  Using mu=%f"%(mu,max_mu))
        mu = max_mu

    #print base['obj_num'],'NFW mu = ',mu
    return mu, False


