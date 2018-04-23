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

import os
import galsim
import logging

from .output import OutputBuilder

class DataCubeBuilder(OutputBuilder):
    """Builder class for constructing and writing DataCube output types.
    """

    def buildImages(self, config, base, file_num, image_num, obj_num, ignore, logger):
        """Build the images

        A point of attention for DataCubes is that they must all be the same size.
        This function builds the first image alone, finds out its size and then forces
        all subsequent images to be the same size.

        @param config           The configuration dict for the output field.
        @param base             The base configuration dict.
        @param file_num         The current file_num.
        @param image_num        The current image_num.
        @param obj_num          The current obj_num.
        @param ignore           A list of parameters that are allowed to be in config that we can
                                ignore here.  i.e. it won't be an error if they are present.
        @param logger           If given, a logger object to log progress.

        @returns a list of the images built
        """
        import time
        nimages = self.getNImages(config, base, file_num)

        # The above call sets up a default nimages if appropriate.  Now, check that there are no
        # invalid parameters in the config dict.
        req = { 'nimages' : int }
        ignore += [ 'file_name', 'dir', 'nfiles' ]
        galsim.config.CheckAllParams(config, ignore=ignore, req=req)

        # All images need to be the same size for a data cube.
        # Enforce this by building the first image outside the below loop and setting
        # config['image_force_xsize'] and config['image_force_ysize'] to be the size of the first
        # image.
        t1 = time.time()
        base1 = galsim.config.CopyConfig(base)
        image0 = galsim.config.BuildImage(base1, image_num, obj_num, logger=logger)
        t2 = time.time()
        # Note: numpy shape is y,x
        ys, xs = image0.array.shape
        logger.info('Image %d: size = %d x %d, time = %f sec', image_num, xs, ys, t2-t1)

        # Note: numpy shape is y,x
        image_ysize, image_xsize = image0.array.shape
        base['image_force_xsize'] = image_xsize
        base['image_force_ysize'] = image_ysize

        images = [ image0 ]

        if nimages > 1:
            obj_num += galsim.config.GetNObjForImage(base, image_num)
            images += galsim.config.BuildImages(nimages-1, base, logger=logger,
                                                image_num=image_num+1, obj_num=obj_num)

        return images

    def getNImages(self, config, base, file_num):
        """Returns the number of images to be built.

        @param config           The configuration dict for the output field.
        @param base             The base configuration dict.
        @param file_num         The current file number.

        @returns the number of images to build.
        """
        # Allow nimages to be automatic based on input catalog if image type is Single
        if ( 'nimages' not in config and
            ( 'image' not in base or 'type' not in base['image'] or
            base['image']['type'] == 'Single' ) ):
            nimages = galsim.config.ProcessInputNObjects(base)
            if nimages:
                config['nimages'] = nimages
        if 'nimages' not in config:
            raise galsim.GalSimConfigError(
                "Attribute output.nimages is required for output.type = MultiFits")
        return galsim.config.ParseValue(config,'nimages',base,int)[0]

    def writeFile(self, data, file_name, config, base, logger):
        """Write the data to a file.

        @param data             The data to write.  Usually a list of images returned by
                                buildImages, but possibly with extra HDUs tacked onto the end
                                from the extra output items.
        @param file_name        The file_name to write to.
        @param config           The configuration dict for the output field.
        @param base             The base configuration dict.
        @param logger           If given, a logger object to log progress.
        """
        galsim.fits.writeCube(data,file_name)

    def canAddHdus(self):
        """Returns whether it is permissible to add extra HDUs to the end of the data list.

        False for DataCube.
        """
        return False


# Register this as a valid output type
from .output import RegisterOutputType
RegisterOutputType('DataCube', DataCubeBuilder())
