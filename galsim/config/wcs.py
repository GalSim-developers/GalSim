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

# This file handles the construction of wcs types in config['image']['wcs'].

def BuildWCS(config):
    """Read the wcs from the config dict, writing both it and, if it is well-defined, the 
    pixel_scale to the config.  If the wcs does not have a well-defined pixel_scale, it will 
    be stored as None.
    """
    image = config['image']

    # If there is a wcs field, read it and update the wcs variable.
    if 'wcs' in image:
        image_wcs = image['wcs']
        if 'type' in image_wcs:
            wcs_type = image_wcs['type']
        else:
            wcs_type = 'PixelScale'

        # Special case: origin == center means to use image_center for the wcs origin
        if 'origin' in image_wcs and image_wcs['origin'] == 'center':
            origin = config['image_center']
            image_wcs['origin'] = origin

        if wcs_type not in valid_wcs_types:
            raise AttributeError("Invalid image.wcs.type=%s."%wcs_type)

        if ( valid_wcs_types[wcs_type]['local'] is not None and
             ('origin' in image_wcs or 'world_origin' in image_wcs) ):
            build_func = valid_wcs_types[wcs_type]['local']
        else:
            build_func = valid_wcs_types[wcs_type]['init']

        kwargs_func = valid_wcs_types[wcs_type]['kwargs']
        if kwargs_func is not None:
            kwargs = kwargs_func(image_wcs, config)
        else:
            req = build_func._req_params
            opt = build_func._opt_params
            single = build_func._single_params

            # Pull in the image layer pixel_scale as a scale item if necessary.
            if ( ('scale' in req or 'scale' in opt) and 'scale' not in image_wcs and 
                'pixel_scale' in image ):
                image_wcs['scale'] = image['pixel_scale']

            kwargs, safe = galsim.config.GetAllParams(image_wcs, config, req, opt, single)

            # This would be weird, but might as well check...
            if build_func._takes_rng:
                if 'rng' not in config:
                    raise ValueError("No config['rng'] available for %s.type = %s"%(key,wcs_type))
                kwargs['rng'] = config['rng']

        wcs = build_func(**kwargs) 

    else:
        # Default if no wcs is to use PixelScale
        if 'pixel_scale' in image:
            scale = galsim.config.ParseValue(image, 'pixel_scale', config, float)[0]
        else:
            scale = 1.0
        wcs = galsim.PixelScale(scale)

    return wcs

def GetTanWCSKwargs(config, base):
    """The TanWCS type needs special handling to get the kwargs, since the TanWCS function
    takes an AffineTransform as one of the arguments, so we need to build that from 
    dudx, dudy, etc.  We also need to construct a CelestialCoord object for the world_origin,
    which we make from ra, dec paramters.
    """
    req = { "dudx" : float, "dudy" : float, "dvdx" : float, "dvdy" : float,
            "ra" : galsim.Angle, "dec" : galsim.Angle }
    opt = { "units" : str, "origin" : galsim.PositionD }
    params, safe = galsim.config.GetAllParams(config, base, req=req, opt=opt)

    dudx = params['dudx']
    dudy = params['dudy']
    dvdx = params['dvdx']
    dvdy = params['dvdy']
    ra = params['ra']
    dec = params['dec']
    units = params.get('units', 'arcsec')
    origin = params.get('origin', None)

    affine = galsim.AffineTransform(dudx, dudy, dvdx, dvdy, origin)
    world_origin = galsim.CelestialCoord(ra, dec)
    units = galsim.angle.get_angle_unit(units)

    return { 'affine' : affine,
             'world_origin' : world_origin,
             'units' : units }


valid_wcs_types = {}

def RegisterWCSType(wcs_type, init_func, local_func=None, kwargs_func=None):
    """Register a wcs type for use by the config apparatus.

    @param wcs_type         The name of the type in config['image']['wcs']
    @param init_func        A function or class name to use to build the wcs.
    @param local_func       If given, use a different class when origin is not given as
                            one of the parameters.
    @param kwargs_func      A function to get the initialization kwargs if the regular
                            _req, _opt, etc. kind of initialization will not work. The call
                            signature is:
                                kwargs, safe = GetKwargs(config, base)
                            [default: None, which means use the regular initialization]
    """
    valid_wcs_types[wcs_type] = {
        'init' : init_func,
        'local' : local_func,
        'kwargs' : kwargs_func
    }

RegisterWCSType('PixelScale', galsim.PixelScale, local_func=galsim.OffsetWCS)
RegisterWCSType('Shear', galsim.ShearWCS, local_func=galsim.OffsetShearWCS)
RegisterWCSType('Jacobian', galsim.JacobianWCS, local_func=galsim.AffineTransform)
RegisterWCSType('Affine', galsim.JacobianWCS, local_func=galsim.AffineTransform)
RegisterWCSType('UVFunction', galsim.UVFunction)
RegisterWCSType('RaDecFunction', galsim.RaDecFunction)
RegisterWCSType('Fits', galsim.FitsWCS)
RegisterWCSType('Tan', galsim.TanWCS, kwargs_func=GetTanWCSKwargs )

