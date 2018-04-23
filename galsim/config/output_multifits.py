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
class MultiFitsBuilder(OutputBuilder):
    """Builder class for constructing and writing MultiFits output types.
    """

    def buildImages(self, config, base, file_num, image_num, obj_num, ignore, logger):
        """Build the images

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
        nimages = self.getNImages(config, base, file_num)

        # The above call sets up a default nimages if appropriate.  Now, check that there are no
        # invalid parameters in the config dict.
        req = { 'nimages' : int }
        ignore += [ 'file_name', 'dir', 'nfiles' ]
        galsim.config.CheckAllParams(config, ignore=ignore, req=req)

        return galsim.config.BuildImages(nimages, base, image_num, obj_num, logger=logger)

    def getNImages(self, config, base, file_num):
        """
        Get the number of images for a MultiFits file type.

        @param config           The configuration dict for the output field.
        @param base             The base configuration dict.
        @param file_num         The current file number.

        @returns the number of images
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


# Register this as a valid output type
from .output import RegisterOutputType
RegisterOutputType('MultiFits', MultiFitsBuilder())
