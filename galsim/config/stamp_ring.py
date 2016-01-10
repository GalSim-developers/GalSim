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

# This file adds stamp type Ring which builds an object once every n times, and then
# rotates it in a ring for the other n-1 times per per group.

def SetupRing(config, xsize, ysize, ignore, logger):
    """Do the initialization and setup for the Ring type.

    Most of the work is done by SetupBasic, but we do need to check that the required parameters
    are present, and also that no additional parameters are present.

    @param config           The configuration dict.
    @param xsize            The xsize of the image to build (if known).
    @param ysize            The ysize of the image to build (if known).
    @param ignore           A list of parameters that are allowed to be in config['stamp']
                            that we can ignore here.  i.e. it won't be an error if these
                            parameters are present.
    @param logger           If given, a logger object to log progress.

    @returns xsize, ysize, image_pos, world_pos
    """
    req = { 'num' : int }
    opt = { 'full_rotation' : galsim.Angle , 'index' : int }
    ignore = ignore + []

    stamp = config['stamp']
    params = galsim.config.GetAllParams(stamp, config, req=req, opt=opt, ignore=ignore)[0]

    num = params['num']
    if num <= 0:
        raise ValueError("Attribute num for gal.type == Ring must be > 0")

    # Setup the indexing sequence if it hasn't been specified using the number of items.
    galsim.config.SetDefaultIndex(stamp, num)

    # Set the default full_rotation to pi radians
    if 'full_rotation' not in params:
        import math
        stamp['full_rotation'] = math.pi * galsim.radians

    ignore = ignore + req.keys() + opt.keys()
    return galsim.config.SetupBasic(config, xsize, ysize, ignore, logger)

def ProfileRing(config, psf, gsparams, logger):
    """
    Build the object to be drawn.

    For the first item in the ring, this is the same as ProfileBasic. It stores the galaxy object
    created on the first time.  Then for later stamps in the ring, it retrieves the stored
    first galaxy and just rotates it before convolving by the psf.

    @param config           The configuration dict.
    @param psf              The PSF, if any.  This may be None, in which case, no PSF is convolved.
    @param gsparams         A dict of kwargs to use for a GSParams.  More may be added to this
                            list by the galaxy object.
    @param logger           If given, a logger object to log progress.

    @returns the final profile
    """
    stamp = config['stamp']

    # These have all already been checked to exist in SetupRing.
    num = galsim.config.ParseValue(stamp, 'num', config, int)[0]
    index = galsim.config.ParseValue(stamp, 'index', config, int)[0]
    if index < 0 or index >= num:
        raise AttributeError("index %d out of bounds for Ring"%index)

    if index % num == 0:
        # Then we are on the first item in the ring, so make it normally.
        gal = galsim.config.BuildGSObject(config, 'gal', gsparams=gsparams, logger=logger)[0]
        if gal is None:
            raise AttributeError("The gal field must define a valid galaxy for stamp type=Ring.")
        # Save the galaxy profile for next time.
        config['stamp_ring_first'] = gal
    else:
        # Grab the saved first galaxy.
        if 'stamp_ring_first' not in config:
            raise RuntimeError("Building Ring after the first item, but no first gal stored.")
        gal = config['stamp_ring_first']
        full_rotation = galsim.config.ParseValue(stamp, 'full_rotation', config, galsim.Angle)[0]
        dtheta = full_rotation / num
        gal = gal.rotate(index*dtheta)

    if psf is not None:
        return galsim.Convolve(gal,psf)
    else:
        return gal

def RejectRing(config, prof, psf, image, logger):
    """Check to see if this object should be rejected.

    This is the same as RejectBasic for the first item in the ring.  Later items are not checked
    though, since rejecting them would mess up the ring.

    @param config       The configuration dict.
    @param prof         The profile that was drawn.
    @param psf          The psf that was used to build the profile.
    @param image        The postage stamp image.  No noise is on it yet at this point.
    @param logger       If given, a logger object to log progress.

    @returns whether the galaxy was rejected.
    """
    index = galsim.config.ParseValue(stamp, 'index', config, int)[0]
    if index == 0:
        return RejectBasic(config, prof, psf, image, logger)
    else:
        return False


# Register this as a valid stamp type
# Use the regular Basic functions for most things.  Just specialize the setup, profile, and reject
# functions.
from .stamp import RegisterStampType
RegisterStampType('Ring',
                  setup_func = SetupRing,
                  prof_func = ProfileRing,
                  reject_func = RejectRing)

