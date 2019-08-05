# Copyright (c) 2012-2018 by the GalSim developers team on GitHub
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
import os

default_bad_ccds=[2,31,61]
def get_random_chipnum(ud, bad_ccds):
    while True:
        chipnum = int(ud() * 62) + 1
        if chipnum not in bad_ccds and chipnum <= 62:
            break
    return chipnum

# This class works, but it's pretty slow.  I'm leaving it here as an example of a relatively
# straightforward wcs builder.  But the better class that uses an input field is below.

class DES_SlowLocalWCSBuilder(galsim.config.WCSBuilder):

    def buildWCS(self, config, base, logger):
        """Build a local WCS from the given location in a DES focal plane, given a directory
        with the image files.  By default, it will pick a random chipnum and image position,
        but these can be optionally specified.
        """

        req = { "dir" : str,  # The directory with the files. We'll use 'des_data' here.
                "root" : str,  # The root name of the files. We'll use 'DECam_00154912' here.
            }
        opt = { "chipnum" : int,  # Which chip to use: 1-62.  Default is to pick a random chip.
                "image_pos" : galsim.PositionD,  # The position in the chip.  Default is random.
                "ext" : str,  # The file extension.  Default is ".fits.fz"
                "bad_ccds" : list,  # A list of ccds to skip.  Default is default_bad_ccds.
            }
        params, safe = galsim.config.GetAllParams(config, base, req=req, opt=opt)

        # These will already have been checked to be present, since they are in the req dict.
        dir = params['dir']
        root = params['root']

        rng = base['rng']
        ud = galsim.UniformDeviate(rng)  # Will give float values between 0 and 1.

        if 'chipnum' in params:
            chipnum = params['chipnum']
        else:
            bad_ccds = params.get('bad_ccds', default_bad_ccds)
            chipnum=get_random_chipnum(ud, bad_ccds)

        ext = params.get('ext','.fits.fz')

        # Build the full path of the file to use.
        file_name = os.path.join(dir, "%s_%02d%s"%(root,chipnum,ext))

        # Read the full WCS as a regular FitsWCS.
        full_wcs = galsim.FitsWCS(file_name)

        # Determine where in the image we will get the local WCS
        if 'image_pos' in params:
            image_pos = params['image_pos']
        else:
            x = ud() * 2048. + 0.5  # The pixel centers go from 1-2048, so edges are 0.5-2048.5
            y = ud() * 4096. + 0.5
            image_pos = galsim.PositionD(x,y)

        # Finally, return the local wcs at this location.
        local_wcs = full_wcs.local(image_pos)
        return local_wcs

# Register this with GalSim:
galsim.config.RegisterWCSType('DES_SlowLocal', DES_SlowLocalWCSBuilder())

# The above class works, but it's slow, since it reads in a file for every stamp.  (Hence
# the name SlowLocal.)  Now we'll do a version that reads in all the wcs files at the start
# using an input field and selects from them randomly for each stamp.

class DES_FullFieldWCS(object):
    """A class for storing a set of WCS objects read from DES images.

    @param dir      The directory with the image files
    @param root     The root name for the files
    @param ext      The extension of the files.  [default: '.fits.fz']
    """
    # The normal way to tell GalSim what parameters are required and/or optional is
    # through some class attributes given here:
    _req_params = {
        "dir" : str,    # The directory with the files. We'll use 'des_data' here.
        "root" : str,   # The root name of the files. We'll use 'DECam_00154912' here.
    }
    _opt_params = {
        "ext" : str,    # The file extension.  Default is ".fits.fz"
        "bad_ccds" : list,  # A list of ccds to skip.  Default is default_bad_ccds.
    }

    # And some other attributes that are required to be present if you do the above.
    _single_params = [] # If there are sets of parameters where one and only one is required,
                        # put them here.  We don't have any for this class.

    _takes_rng = False  # Does the constructor take an rng parameter?  No, in this case.

    def __init__(self, dir, root, ext='.fits.fz', bad_ccds=default_bad_ccds):
        self.bad_ccds = bad_ccds
        # Read all the wcs objects indexed by their chipnum
        self.all_wcs = {}
        for chipnum in range(1,63):
            # skip bad ccds
            if chipnum not in bad_ccds:
                file_name = os.path.join(dir, "%s_%02d%s"%(root,chipnum,ext))
                wcs = galsim.FitsWCS(file_name)
                self.all_wcs[chipnum] = wcs

    # A slight subtlety that you should be aware of with respect to using input objects.
    # Because of at least the possibiltiy of doing multiprocessing, we use proxy objects
    # to communicate between processes, which are mostly equivalent to using a regular
    # instance of the class with one important exception.  You cannot directly use attributes
    # of the input object.
    # In this case, we want to access self.bad_ccds below in DES_LocalWCSBuilder.
    # To make that possible, we need to add a getter method that accesses the attribute
    # we want to use.
    def get_bad_ccds(self):
        """Return the list of bad ccds.
        """
        return self.bad_ccds

    def get_chip_wcs(self, chipnum):
        """Return the wcs to use for a given chipnum
        """
        return self.all_wcs[chipnum]


class DES_LocalWCSBuilder(galsim.config.WCSBuilder):

    def buildWCS(self, config, base, logger):
        """Build a local WCS from the given location in a DES focal plane.

        This function is used in conjunction with the des_wcs input field, which loads all the
        files at the start.

        By default, it will pick a random chipnum and image position, but these can be optionally
        specified.
        """

        opt = { "chipnum" : int,  # Which chip to use: 1-62.  Default is to pick a random chip.
                "image_pos" : galsim.PositionD,  # The position in the chip.  Default is random.
            }
        params, safe = galsim.config.GetAllParams(config, base, opt=opt)

        rng = base['rng']
        ud = galsim.UniformDeviate(rng)  # Will give float values between 0 and 1.

        # Get the input des_wcs object.  The last parameter is just used for error reporting, and
        # should be the name of the current type being processed.
        des_wcs = galsim.config.GetInputObj('des_wcs', config, base, 'DES_LocalWCS')

        if 'chipnum' in params:
            chipnum = params['chipnum']
        else:
            chipnum=get_random_chipnum(ud, des_wcs.get_bad_ccds())

        full_wcs = des_wcs.get_chip_wcs(chipnum)

        # Determine where in the image we will get the local WCS
        if 'image_pos' in params:
            image_pos = params['image_pos']
        else:
            x = ud() * 2048. + 0.5  # The pixel centers go from 1-2048, so edges are 0.5-2048.5
            y = ud() * 4096. + 0.5
            image_pos = galsim.PositionD(x,y)

        # Finally, return the local wcs at this location.
        local_wcs = full_wcs.local(image_pos)
        return local_wcs

# Register these with GalSim:
galsim.config.RegisterInputType('des_wcs', galsim.config.InputLoader(DES_FullFieldWCS))
galsim.config.RegisterWCSType('DES_Local', DES_LocalWCSBuilder(), input_type='des_wcs')

