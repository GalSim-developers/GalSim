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

# The psf extra output type builds an Image of the PSF at the same locations as the galaxies.

# The function called at the start of each image.
def SetupExtraPSF(image, scratch, config, base, logger=None):
    image.resize(base['image_bounds'], wcs=base['wcs'])
    image.setZero()
    scratch.clear()

# The function to call at the end of building each stamp
def ProcessExtraPSFStamp(image, scratch, config, base, obj_num, logger=None):
    # If this doesn't exist, an appropriate exception will be raised.
    psf = base['psf']['current_val']
    draw_method = galsim.config.GetCurrentValue('image.draw_method',base,str)
    bounds = base['current_stamp'].bounds
    offset = base['stamp_offset']
    if 'offset' in base['image']:
        offset += galsim.config.ParseValue(base['image'], 'offset', base, galsim.PositionD)[0]
    psf_im = galsim.config.DrawPSFStamp(psf,base,bounds,offset,draw_method,logger)
    if 'signal_to_noise' in config:
        base['index_key'] = 'image_num'
        galsim.config.AddNoise(base,psf_im,0,logger)
        base['index_key'] = 'obj_num'
    scratch[obj_num] = psf_im

# The function to call at the end of building each image
def ProcessExtraPSFImage(image, scratch, config, base, logger=None):
    for stamp in scratch.values():
        b = stamp.bounds & image.getBounds()
        if b.isDefined():
            # This next line is equivalent to:
            #    image[b] += stamp[b]
            # except that this doesn't work through the proxy.  We can only call methods
            # that don't start with _.  Hence using the more verbose form here.
            image.setSubImage(b, image.subImage(b) + stamp[b])

# Normally import galsim would give us access to galsim.config.valid_extra_outputs.
# However, since galsim.config imports this file, it doesn't exist yet, so we need to
# get it directly from the extra module.
from .extra import valid_extra_outputs

# Register psf as an extra output type in config
valid_extra_outputs['psf'] = (
    # The values are tuples with:
    # - the class name to build, if any.
    # - a function to get the initialization kwargs if building something.
    # - a function to call at the start of each image
    # - a function to call at the end of building each stamp
    # - a function to call at the end of building each image
    # - a function to call to write the output file
    # - a function to call to build either a FITS HDU or an Image to put in an HDU
    galsim.Image, None,
    SetupExtraPSF, ProcessExtraPSFStamp, ProcessExtraPSFImage, 
    galsim.Image.write, galsim.Image.view 
)

