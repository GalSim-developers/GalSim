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
from .image import BuildImages
from .input import ProcessInputNObjects
from .value import ParseValue, CheckAllParams
from ..errors import GalSimConfigError

class MultiFitsBuilder(OutputBuilder):
    """Builder class for constructing and writing MultiFits output types.
    """

    def buildImages(self, config, base, file_num, image_num, obj_num, ignore, logger):
        """Build the images

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
        nimages = self.getNImages(config, base, file_num, logger=logger)

        # The above call sets up a default nimages if appropriate.  Now, check that there are no
        # invalid parameters in the config dict.
        req = { 'nimages' : int }
        ignore += [ 'file_name', 'dir', 'nfiles' ]
        CheckAllParams(config, ignore=ignore, req=req)

        return BuildImages(nimages, base, image_num, obj_num, logger=logger)

    def getNImages(self, config, base, file_num, logger=None):
        """
        Get the number of images for a MultiFits file type.

        Parameters:
            config:         The configuration dict for the output field.
            base:           The base configuration dict.
            file_num:       The current file number.
            logger:         If given, a logger object to log progress.

        Returns:
            the number of images
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


# Register this as a valid output type
RegisterOutputType('MultiFits', MultiFitsBuilder())
