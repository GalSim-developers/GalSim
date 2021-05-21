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

from .extra import ExtraOutputBuilder, RegisterExtraOutput
from .value import ParseValue, GetCurrentValue
from .noise import AddNoiseVariance
from ..image import ImageF

# The weight extra output type builds an ImageF of the inverse noise variance in the image.
# It builds up the variance either from the stamp information if noise is being added then
# or at the end from the full image if that is when noise is added.  Then at the end of
# the image processing, it inverts the image to get the appropriate weight map.

class WeightBuilder(ExtraOutputBuilder):
    """This builds a weight map image to go along with each regular data image.

    The weight is the inverse variance of the noise in the image.

    There is a option called 'include_obj_var' that governs whether the weight should include the
    Poisson variance of the signal.  In real data, you don't know the true signal, and estimating
    the Poisson noise from the realized image can lead to biases.  As such, different applications
    may or may not want this included.
    """

    # The function to call at the end of building each stamp
    def processStamp(self, obj_num, config, base, logger):
        if base['do_noise_in_stamps']:
            weight_im = ImageF(base['current_stamp'].bounds, wcs=base['wcs'], init_value=0.)
            if 'include_obj_var' in config:
                include_obj_var = ParseValue(config, 'include_obj_var', base, bool)[0]
            else:
                include_obj_var = False
            base['current_noise_image'] = base['current_stamp']
            AddNoiseVariance(base,weight_im,include_obj_var,logger)
            self.scratch[obj_num] = weight_im

    # The function to call at the end of building each image
    def processImage(self, index, obj_nums, config, base, logger):
        image = ImageF(base['image_bounds'], wcs=base['wcs'], init_value=0.)
        if len(self.scratch) > 0.:
            # If we have been accumulating the variance on the stamps, build the total from them.
            for obj_num in obj_nums:
                stamp = self.scratch[obj_num]
                b = stamp.bounds & image.bounds
                if b.isDefined(): # pragma: no branch
                    # This next line is equivalent to:
                    #    image[b] += stamp[b]
                    # except that this doesn't work through the proxy.  We can only call methods
                    # that don't start with _.  Hence using the more verbose form here.
                    image.setSubImage(b, image.subImage(b) + stamp[b])
        else:
            # Otherwise, build the variance map now.
            if 'include_obj_var' in config:
                include_obj_var = ParseValue(config, 'include_obj_var', base, bool)[0]
            else:
                include_obj_var = False
            base['current_noise_image'] = base['current_image']
            AddNoiseVariance(base,image,include_obj_var,logger)

        # Now invert the variance image to get weight map.
        # Note that any zeros present in the image are maintained as zeros after inversion.
        # So it is ok to set bad pixels to have zero variance above, and they will invert to have
        # zero weight.
        image.invertSelf()
        self.data[index] = image


# Register this as a valid extra output
RegisterExtraOutput('weight', WeightBuilder())
