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


from .extra import ExtraOutputBuilder
class WeightBuilder(ExtraOutputBuilder):
    """This builds a bad pixel mask image to go along with each regular data image.

    There's not much here currently, since GalSim doesn't yet have any image artifacts that
    would be appropriate to do something with here.  So this is mostly just a placeholder for
    when we eventually add defects, saturation, etc.
    """

    # The function to call at the end of building each stamp
    def processStamp(self, obj_num, config, base, logger):
        if base['do_noise_in_stamps']:
            weight_im = galsim.ImageF(base['current_stamp'].bounds, wcs=base['wcs'], init_value=0.)
            if 'include_obj_var' in base['output']['weight']:
                include_obj_var = galsim.config.ParseValue(
                        base['output']['weight'], 'include_obj_var', config, bool)[0]
            else:
                include_obj_var = False
            galsim.config.AddNoiseVariance(base,weight_im,include_obj_var,logger)
            self.scratch[obj_num] = weight_im

    # The function to call at the end of building each image
    def processImage(self, index, obj_nums, config, base, logger):
        image = galsim.ImageF(base['image_bounds'], wcs=base['wcs'], init_value=0.)
        if len(self.scratch) > 0.:
            # If we have been accumulating the variance on the stamps, build the total from them.
            for obj_num in obj_nums:
                stamp = self.scratch[obj_num]
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
                # If we are using a Proxy for the image, the code in AddNoiseVar won't work
                # properly.  The easiest workaround is to build a new image here and copy it over.
                im2 = galsim.ImageF(image.getBounds(), wcs=base['wcs'], init_value=0.)
                galsim.config.AddNoiseVariance(base,im2,include_obj_var,logger)
                image.copyFrom(im2)

        # Now invert the variance image to get weight map.
        # Note that any zeros present in the image are maintained as zeros after inversion.
        # So it is ok to set bad pixels to have zero variance above, and they will invert to have
        # zero weight.
        image.invertSelf()
        self.data[index] = image

    # Write the image(s) to a file
    def writeFile(self, file_name, config, base, logger):
        galsim.fits.writeMulti(self.data, file_name)

    # For the hdu, just return the first element
    def writeHdu(self, config, base, logger):
        n = len(self.data)
        if n == 0:
            raise RuntimeError("No weight images were created.")
        elif n > 1:
            raise RuntimeError("%d weight images were created, but expecting only 1."%n)
        return self.data[0]


# Register this as a valid extra output
from .extra import RegisterExtraOutput
RegisterExtraOutput('weight', WeightBuilder)
