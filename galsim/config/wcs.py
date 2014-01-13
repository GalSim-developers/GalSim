# Copyright 2012, 2013 The GalSim developers:
# https://github.com/GalSim-developers
#
# This file is part of GalSim: The modular galaxy image simulation toolkit.
#
# GalSim is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# GalSim is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with GalSim.  If not, see <http://www.gnu.org/licenses/>
#

import galsim


# The only item in the tuple is the name of the WCS class (or build function).
# We made it a tuple to make it easier to add extra features later.
# Most WCS types can use the normal _req_params, etc.  Currently, only TanWCS has a custom builder.
valid_wcs_types = { 
    'PixelScale' : ( 'galsim.PixelScale', ),
    'Offset' : ( 'galsim.OffsetWCS', ),
    'Shear' : ( 'galsim.ShearWCS', ),
    'OffsetShear' : ( 'galsim.OffsetShearWCS', ),
    'Jacobian' : ( 'galsim.JacobianWCS', ),
    'AffineTransform' : ( 'galsim.AffineTransform', ),
    'UVFunction' : ( 'galsim.UVFunction', ),
    # TODO: Not everything works with the celestial wcs classes.  There are a few places
    # where we assume that the world coordinates are Euclidean (u,v) coordinates. It needs a 
    # bit of work to make sure everything is working correctly with celestial coordinates.
    'RaDecFunction' : ( 'galsim.RaDecFunction', ),
    'Fits' : ( 'galsim.FitsWCS', ),
    'Tan' : ( 'TanWCSBuilder', ),
}

def BuildWCS(config, logger=None):
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
        elif 'origin' in image_wcs:
            type = 'Offset'
        else:
            type = 'PixelScale'

        # Special case: origin == center means to use image_center for the wcs origin
        if 'origin' in image_wcs and image_wcs['origin'] == 'center':
            origin = config['image_center']
            if logger:
                logger.debug('image %d: Using origin = %s',config['image_num'],str(origin))
            image_wcs['origin'] = origin

        if type not in valid_wcs_types:
            raise AttributeError("Invalid image.wcs.type=%s."%type)

        build_func = eval(valid_wcs_types[type][0])

        if logger:
            logger.debug('image %d: Build WCS for type = %s using %s',
                         config['image_num'],type,str(build_func))

        req = build_func._req_params
        opt = build_func._opt_params
        single = build_func._single_params

        # Pull in the image layer pixel_scale as a scale item if necessary.
        if ( ('scale' in req or 'scale' in opt) and 'scale' not in image_wcs and 
             'pixel_scale' in image ):
            image_wcs['scale'] = image['pixel_scale']

        kwargs, safe = galsim.config.GetAllParams(image_wcs, type, config, req, opt, single)

        if logger and build_func._takes_logger: 
            kwargs['logger'] = logger

        # This would be weird, but might as well check...
        if build_func._takes_rng:
            if 'rng' not in config:
                raise ValueError("No config['rng'] available for %s.type = %s"%(key,type))
            kwargs['rng'] = config['rng']

        if logger:
            logger.debug('image %d: kwargs = %s',config['image_num'],str(kwargs))
        wcs = build_func(**kwargs) 

    else:
        # Default if no wcs is to use PixelScale
        if 'pixel_scale' in image:
            scale = galsim.config.ParseValue(image, 'pixel_scale', config, float)[0]
        else:
            scale = 1.0
        wcs = galsim.PixelScale(scale)

    # Write it to the config dict and also return it.
    config['wcs'] = wcs
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
TanWCSBuilder._takes_logger = False
