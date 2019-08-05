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

# This custom "Extra Output" builder simply copies over the image before noise is added
# to a second image that will be written out to either another hdu or a separate file.
# It is based heavily on the 'weight' output builder.  cf. galsim/config/extra_weight.py.

class NoiseFreeBuilder(galsim.config.ExtraOutputBuilder):
    """This builds a noise-free image to go along with each regular data image.
    """

    # The function to call at the end of building each stamp
    def processStamp(self, obj_num, config, base, logger):
        if base['do_noise_in_stamps']:
            noise_free_im = base['current_stamp'].copy()
            self.scratch[obj_num] = noise_free_im

    # The function to call at the end of building each image
    def processImage(self, index, obj_nums, config, base, logger):
        if len(self.scratch) > 0.:
            # If we have been accumulating the stamp images, build the total from them.
            image = galsim.ImageF(base['image_bounds'], wcs=base['wcs'], init_value=0.)
            for obj_num in obj_nums:
                stamp = self.scratch[obj_num]
                b = stamp.bounds & image.bounds
                if b.isDefined():
                    image.setSubImage(b, image.subImage(b) + stamp[b])
        else:
            # Otherwise, just copy the current image, which is the image we want without noise.
            image = base['current_image'].copy()

        self.data[index] = image

# Register this as a valid extra output
galsim.config.RegisterExtraOutput('noise_free', NoiseFreeBuilder())
