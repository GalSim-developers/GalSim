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
import os

# This class works, but it's pretty slow.  I'm leaving it here as an example of a relatively
# straightforward wcs builder.  But the better class that uses an input field is below.

def DES_SlowLocalWCS(config, base):
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
        chipnum = int(ud() * 62) + 1
        if chipnum == 63: chipnum = 62  # Just in case.

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
galsim.config.RegisterWCSType('DES_SlowLocal', DES_SlowLocalWCS)

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
    }

    # And some other attributes that are required to be present if you do the above.
    _single_params = [] # If there are sets of parameters where one and only one is required,
                        # put them here.  We don't have any for this class.

    _takes_rng = False  # Does the constructor take an rng parameter?  No, in this case.

    def __init__(self, dir, root, ext='.fits.fz'):
        # Read all the wcs objects indexed by their chipnum
        self.all_wcs = {}
        for chipnum in range(1,63):
            file_name = os.path.join(dir, "%s_%02d%s"%(root,chipnum,ext))
            wcs = galsim.FitsWCS(file_name)
            self.all_wcs[chipnum] = wcs

    def get_chip_wcs(self, chipnum):
        """Return the wcs to use for a given chipnum
        """
        return self.all_wcs[chipnum]


def DES_LocalWCS(config, base):
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
        chipnum = int(ud() * 62) + 1
        if chipnum == 63: chipnum = 62  # Just in case.

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
galsim.config.RegisterInputType('des_wcs', DES_FullFieldWCS, ['DES_LocalWCS'])
galsim.config.RegisterWCSType('DES_Local', DES_LocalWCS)


