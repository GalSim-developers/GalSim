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

# We distinguish some classes according to whether they have an origin parameter.
# The first item in the tuple is the builder class or function that does not take an 
# offset parameter.  The second item is the version that does.
# Most WCS types can use the normal _req_params, etc.  Currently, only Tan has a custom builder.
valid_wcs_types = { 
    'PixelScale' : ( 'galsim.PixelScale', 'galsim.OffsetWCS' ),
    'Shear' : ( 'galsim.ShearWCS', 'galsim.OffsetShearWCS' ),
    'Jacobian' : ( 'galsim.JacobianWCS', 'galsim.AffineTransform' ),
    'Affine' : ( 'galsim.JacobianWCS', 'galsim.AffineTransform', ),
    'UVFunction' : ( 'galsim.UVFunction', 'galsim.UVFunction' ),
    'RaDecFunction' : ( 'galsim.RaDecFunction', 'galsim.RaDecFunction' ),
    'Fits' : ( 'galsim.FitsWCS', 'galsim.FitsWCS' ),
    'Tan' : ( 'TanWCSBuilder', 'TanWCSBuilder' ),
}

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
            type = image_wcs['type']
        else:
            type = 'PixelScale'

        # Special case: origin == center means to use image_center for the wcs origin
        if 'origin' in image_wcs and image_wcs['origin'] == 'center':
            origin = config['image_center']
            image_wcs['origin'] = origin

        if type not in valid_wcs_types:
            raise AttributeError("Invalid image.wcs.type=%s."%type)

        if 'origin' in image_wcs or 'world_origin' in image_wcs:
            build_func = eval(valid_wcs_types[type][1])
        else:
            build_func = eval(valid_wcs_types[type][0])

        req = build_func._req_params
        opt = build_func._opt_params
        single = build_func._single_params

        # Pull in the image layer pixel_scale as a scale item if necessary.
        if ( ('scale' in req or 'scale' in opt) and 'scale' not in image_wcs and 
             'pixel_scale' in image ):
            image_wcs['scale'] = image['pixel_scale']

        kwargs, safe = galsim.config.GetAllParams(image_wcs, type, config, req, opt, single)

        # This would be weird, but might as well check...
        if build_func._takes_rng:
            if 'rng' not in config:
                raise ValueError("No config['rng'] available for %s.type = %s"%(key,type))
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

def TanWCSBuilder(dudx, dudy, dvdx, dvdy, ra, dec, units='arcsec', origin=galsim.PositionD(0,0)):
    # The TanWCS uses a custom builder because the normal function takes an AffineTransform, which
    # we need to construct.  It also takes a CelestialCoord for its world_origin parameter, so we
    # make that out of ra and dec parameters.
    affine = galsim.AffineTransform(dudx, dudy, dvdx, dvdy, origin)
    world_origin = galsim.CelestialCoord(ra, dec)
    units = galsim.angle.get_angle_unit(units)
    return galsim.TanWCS(affine, world_origin, units)

TanWCSBuilder._req_params = { "dudx" : float, "dudy" : float, "dvdx" : float, "dvdy" : float,
                              "ra" : galsim.Angle, "dec" : galsim.Angle }
TanWCSBuilder._opt_params = { "units" : str, "origin" : galsim.PositionD }
TanWCSBuilder._single_params = []
TanWCSBuilder._takes_rng = False

