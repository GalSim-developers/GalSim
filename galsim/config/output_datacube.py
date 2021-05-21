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

import os
import logging

from .output import OutputBuilder, RegisterOutputType
from .util import CopyConfig
from .image import BuildImages, BuildImage, GetNObjForImage
from .input import ProcessInputNObjects
from .value import ParseValue, CheckAllParams
from ..errors import GalSimConfigError
from ..fits import writeCube

class DataCubeBuilder(OutputBuilder):
    """Builder class for constructing and writing DataCube output types.
    """

    def buildImages(self, config, base, file_num, image_num, obj_num, ignore, logger):
        """Build the images

        A point of attention for DataCubes is that they must all be the same size.
        This function builds the first image alone, finds out its size and then forces
        all subsequent images to be the same size.

        Parameters:
            config:         The configuration dict for the output field.
            base:           The base configuration dict.
            file_num:       The current file_num.
            image_num:      The current image_num.
            obj_num:        The current obj_num.
            ignore:         A list of parameters that are allowed to be in config that we can
                            ignore here.  i.e. it won't be an error if they are present.
            logger:         If given, a logger object to log progress.

        Returns:
            a list of the images built
        """
        import time
        nimages = self.getNImages(config, base, file_num, logger=logger)

        # The above call sets up a default nimages if appropriate.  Now, check that there are no
        # invalid parameters in the config dict.
        req = { 'nimages' : int }
        ignore += [ 'file_name', 'dir', 'nfiles' ]
        CheckAllParams(config, ignore=ignore, req=req)

        # All images need to be the same size for a data cube.
        # Enforce this by building the first image outside the below loop and setting
        # config['image_force_xsize'] and config['image_force_ysize'] to be the size of the first
        # image.
        t1 = time.time()
        base1 = CopyConfig(base)
        image0 = BuildImage(base1, image_num, obj_num, logger=logger)
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
            obj_num += GetNObjForImage(base, image_num, logger=logger)
            images += BuildImages(nimages-1, base, logger=logger,
                                  image_num=image_num+1, obj_num=obj_num)

        return images

    def getNImages(self, config, base, file_num, logger=None):
        """Returns the number of images to be built.

        Parameters:
            config:         The configuration dict for the output field.
            base:           The base configuration dict.
            file_num:       The current file number.
            logger:         If given, a logger object to log progress.

        Returns:
            the number of images to build.
        """
        # Allow nimages to be automatic based on input catalog if image type is Single
        if ( 'nimages' not in config and
            ( 'image' not in base or 'type' not in base['image'] or
            base['image']['type'] == 'Single' ) ):
            nimages = ProcessInputNObjects(base)
            if nimages:
                config['nimages'] = nimages
        if 'nimages' not in config:
            raise GalSimConfigError(
                "Attribute output.nimages is required for output.type = MultiFits")
        return ParseValue(config,'nimages',base,int)[0]

    def writeFile(self, data, file_name, config, base, logger):
        """Write the data to a file.

        Parameters:
            data:           The data to write.  Usually a list of images returned by
                            buildImages, but possibly with extra HDUs tacked onto the end
                            from the extra output items.
            file_name:      The file_name to write to.
            config:         The configuration dict for the output field.
            base:           The base configuration dict.
            logger:         If given, a logger object to log progress.
        """
        writeCube(data,file_name)

    def canAddHdus(self):
        """Returns whether it is permissible to add extra HDUs to the end of the data list.

        False for DataCube.
        """
        return False


# Register this as a valid output type
RegisterOutputType('DataCube', DataCubeBuilder())
