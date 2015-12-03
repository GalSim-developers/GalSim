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

import numpy
import galsim

# The badpix extra output type is currently just a placeholder for when we eventually add 
# defects, saturation, etc.  Now it always just builds an Image with all 0's.

# The function that returns the kwargs for constructing the Image
def GetBadPixKwargs(config, base, logger=None):
    return {'dtype' : numpy.int16 }

# The function called at the start of each image.
def SetupBadPix(image, scratch, config, base, logger=None):
    image.resize(base['image_bounds'], wcs=base['wcs'])
    image.setZero()
    scratch.clear()

# The function to call at the end of building each stamp
def ProcessBadPixStamp(image, scratch, config, base, obj_num, logger=None):
    # Note: This is just a placeholder for now.  Once we implement defects, saturation, etc.,
    # these features should be marked in the badpix mask.  For now though, all pixels = 0.
    if base['do_noise_in_stamps']:
        badpix_im = galsim.ImageS(base['current_stamp'].bounds, wcs=base['wcs'], init_value=0)
        scratch[obj_num] = badpix_im

# The function to call at the end of building each image
def ProcessBadPixImage(image, scratch, config, base, logger=None):
    if len(scratch) > 0.:
        # If we have been accumulating the variance on the stamps, build the total from them.
        for stamp in scratch.values():
            b = stamp.bounds & image.getBounds()
            if b.isDefined():
                # This next line is equivalent to:
                #    image[b] |= stamp[b]
                # except that this doesn't work through the proxy.  We can only call methods
                # that don't start with _.  Hence using the more verbose form here.
                image.setSubImage(b, image.subImage(b) | stamp[b])
    else:
        # Otherwise, build the bad pixel mask here.
        # Again, nothing here yet.
        pass

# Register this as a valid extra output
from .extra import valid_extra_outputs
valid_extra_outputs['badpix'] = (
    galsim.Image, GetBadPixKwargs,
    SetupBadPix, ProcessBadPixStamp, ProcessBadPixImage, 
    galsim.Image.write, galsim.Image.view 
)

