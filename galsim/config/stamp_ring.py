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

from .stamp import StampBuilder, RegisterStampType
from .value import ParseValue, GetAllParams, SetDefaultIndex
from .gsobject import BuildGSObject, TransformObject
from ..errors import GalSimConfigError, GalSimConfigValueError
from ..convolve import Convolve
from ..angle import Angle, radians

# This file adds stamp type Ring which builds an object once every n times, and then
# rotates it in a ring for the other n-1 times per per group.

class RingBuilder(StampBuilder):
    """This performs the tasks necessary for building a Ring stamp type.

    It uses the regular Basic functions for most things.
    It specializes the setup, buildProfile, reject, and makeTasks functions.
    """

    def setup(self, config, base, xsize, ysize, ignore, logger):
        """Do the initialization and setup for the Ring type.

        Most of the work is done by SetupBasic, but we do need to check that the required parameters
        are present, and also that no additional parameters are present.

        Parameters:
            config:     The configuration dict for the stamp field.
            base:       The base configuration dict.
            xsize:      The xsize of the image to build (if known).
            ysize:      The ysize of the image to build (if known).
            ignore:     A list of parameters that are allowed to be in config that we can
                        ignore here. i.e. it won't be an error if these parameters are present.
            logger:     If given, a logger object to log progress.

        Returns:
            xsize, ysize, image_pos, world_pos
        """
        req = { 'num' : int }
        opt = { 'full_rotation' : Angle , 'index' : int }
        # Ignore the transformation specifications that are allowed in stamp for Ring types.
        ignore = ignore + [
            'dilate', 'dilation', 'ellip', 'rotate', 'rotation', 'scale_flux',
            'magnify', 'magnification', 'shear', 'shift' ]

        params = GetAllParams(config, base, req=req, opt=opt, ignore=ignore)[0]

        num = params['num']
        if num <= 0:
            raise GalSimConfigValueError("Attribute num for gal.type == Ring must be > 0", num)

        # Setup the indexing sequence if it hasn't been specified using the number of items.
        SetDefaultIndex(config, num)

        # Set the default full_rotation to pi radians
        if 'full_rotation' not in params:
            import math
            config['full_rotation'] = math.pi * radians

        # Now go on and do the base class setup.
        ignore = ignore + list(req) + list(opt)
        return super(RingBuilder, self).setup(config, base, xsize, ysize, ignore, logger)

    def buildProfile(self, config, base, psf, gsparams, logger):
        """
        Build the object to be drawn.

        For the first item in the ring, this is the same as Basic. It stores the galaxy object
        created on the first time.  Then for later stamps in the ring, it retrieves the stored
        first galaxy and just rotates it before convolving by the psf.

        Parameters:
            config:     The configuration dict for the stamp field.
            base:       The base configuration dict.
            psf:        The PSF, if any.  This may be None, in which case, no PSF is convolved.
            gsparams:   A dict of kwargs to use for a GSParams.  More may be added to this
                        list by the galaxy object.
            logger:     If given, a logger object to log progress.

        Returns:
            the final profile
        """
        # These have all already been checked to exist in SetupRing.
        num = ParseValue(config, 'num', base, int)[0]
        index = ParseValue(config, 'index', base, int)[0]
        if index < 0 or index >= num:
            raise GalSimConfigError("index %d out of bounds for Ring"%index)

        if index % num == 0:
            # Then we are on the first item in the ring, so make it normally.
            gal = BuildGSObject(base, 'gal', gsparams=gsparams, logger=logger)[0]
            if gal is None:
                raise GalSimConfigError(
                    "The gal field must define a valid galaxy for stamp type=Ring.")
            # Save the galaxy profile for next time.
            self.first = gal
        else:
            # Grab the saved first galaxy.
            if not hasattr(self, 'first'):
                raise GalSimConfigError(
                    "Building Ring after the first item, but no first gal stored.")
            gal = self.first
            full_rot = ParseValue(config, 'full_rotation', base, Angle)[0]
            dtheta = full_rot / num
            gal = gal.rotate(index*dtheta)

        # Apply any transformations that are given in the stamp field.
        gal = TransformObject(gal, config, base, logger)[0]

        if psf is not None:
            return Convolve(gal,psf)
        else:
            return gal

    def reject(self, config, base, prof, psf, image, logger):
        """Check to see if this object should be rejected.

        This is the same as base class reject for the first item in the ring.  Later items are not
        checked though, since rejecting them would mess up the ring.

        Parameters:
            config:     The configuration dict for the stamp field.
            base:       The base configuration dict.
            prof:       The profile that was drawn.
            psf:        The psf that was used to build the profile.
            image:      The postage stamp image.  No noise is on it yet at this point.
            logger:     If given, a logger object to log progress.

        Returns:
            whether the galaxy was rejected.
        """
        index = ParseValue(config, 'index', base, int)[0]
        if index == 0:
            return super(RingBuilder,self).reject(config, base, prof, psf, image, logger)
        else:
            return False

    def makeTasks(self, config, base, jobs, logger):
        """Turn a list of jobs into a list of tasks.

        For the Ring stamp type, we group jobs into sets of num.  If there are extra jobs that
        don't fit into a full ring, the last task will be a partial ring.

        Parameters:
            config:     The configuration dict for the stamp field.
            base:       The base configuration dict.
            jobs:       A list of jobs to split up into tasks.  Each job in the list is a
                        dict of parameters that includes 'obj_num'.
            logger:     If given, a logger object to log progress.

        Returns:
            a list of tasks
        """
        if 'num' not in config:
            raise GalSimConfigError("Attribute num is required for type = Ring")
        num = ParseValue(config, 'num', base, int)[0]
        ntot = len(jobs)
        tasks = [ [ (jobs[j], j) for j in range(k,min(k+num,ntot)) ] for k in range(0, ntot, num) ]
        return tasks


# Register this as a valid stamp type
RegisterStampType('Ring', RingBuilder())

