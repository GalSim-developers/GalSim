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

# The weight extra output type builds an ImageF of the inverse noise variance in the image.
# It builds up the variance either from the stamp information if noise is being added then
# or at the end from the full image if that is when noise is added.  Then at the end of 
# the image processing, it inverts the image to get the appropriate weight map.

# The function to call at the end of building each stamp
def ProcessWeightStamp(images, scratch, config, base, obj_num, logger=None):
    if base['do_noise_in_stamps']:
        weight_im = galsim.ImageF(base['current_stamp'].bounds, wcs=base['wcs'], init_value=0.)
        if 'include_obj_var' in base['output']['weight']:
            include_obj_var = galsim.config.ParseValue(
                    base['output']['weight'], 'include_obj_var', config, bool)[0]
        else:
            include_obj_var = False
        galsim.config.AddNoiseVariance(base,weight_im,include_obj_var,logger)
        scratch[obj_num] = weight_im

# The function to call at the end of building each image
def ProcessWeightImage(images, scratch, config, base, logger=None):
    image = galsim.ImageF(base['image_bounds'], wcs=base['wcs'], init_value=0.)
    if len(scratch) > 0.:
        # If we have been accumulating the variance on the stamps, build the total from them.
        for stamp in scratch.values():
            b = stamp.bounds & image.getBounds()
            if b.isDefined():
                # This next line is equivalent to:
                #    image[b] += stamp[b]
                # except that this doesn't work through the proxy.  We can only call methods
                # that don't start with _.  Hence using the more verbose form here.
                image.setSubImage(b, image.subImage(b) + stamp[b])
    else:
        # Otherwise, build the variance map now.
        if 'include_obj_var' in base['output']['weight']:
            include_obj_var = galsim.config.ParseValue(
                    base['output']['weight'], 'include_obj_var', config, bool)[0]
        else:
            include_obj_var = False
        if isinstance(image, galsim.Image):
            galsim.config.AddNoiseVariance(base,image,include_obj_var,logger)
        else:
            # If we are using a Proxy for the image, the code in AddNoiseVar won't work properly.
            # The easiest workaround is to build a new image here and copy it over.
            im2 = galsim.ImageF(image.getBounds(), wcs=base['wcs'], init_value=0.)
            galsim.config.AddNoiseVariance(base,im2,include_obj_var,logger)
            image.copyFrom(im2)
 
    # Now invert the variance image to get weight map.
    # Note that any zeros present in the image are maintained as zeros after inversion.
    # So it is ok to set bad pixels to have zero variance above, and they will invert to have
    # zero weight.
    image.invertSelf()
    k = base['image_num'] - base['start_image_num']
    images[k] = image

# For the hdu, just return the first element
def HDUWeight(images):
    n = len(images)
    if n == 0:
        raise RuntimeError("No weight images were created.")
    elif n > 1:
        raise RuntimeError("%d weight images were created, but expecting only 1."%n)
    return images[0]



# Register this as a valid extra output
from .extra import RegisterExtraOutput
RegisterExtraOutput('weight',
                    stamp_func = ProcessWeightStamp,
                    image_func = ProcessWeightImage,
                    write_func = galsim.fits.writeMulti,
                    hdu_func = HDUWeight)
 
