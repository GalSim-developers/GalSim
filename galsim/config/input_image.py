# Copyright (c) 2012-2023 by the GalSim developers team on GitHub
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


from .input import InputLoader, RegisterInputType
from .value import GetAllParams
from ..fits import read

# This file adds input type initial_image.

class InitialImageLoader(InputLoader):

    def __init__(self):
        self.init_func = read
        self.has_nobj = False
        self.file_scope = False
        self.takes_logger = False
        self.use_proxy = False

    def getKwargs(self, config, base, logger):
        """Parse the config dict and return the kwargs needed to build the PowerSpectrum object.

        Parameters:
            config:     The configuration dict for 'power_spectrum'
            base:       The base configuration dict
            logger:     If given, a logger object to log progress.

        Returns:
            kwargs, safe
        """
        req = { 'file_name': str }
        opt = { 'dir': str, 'read_header': bool }
        return GetAllParams(config, base, req=req, opt=opt)

    def setupImage(self, input_obj, config, base, logger=None):
        """Set up the PowerSpectrum input object's gridded values based on the
        size of the image and the grid spacing.

        Parameters:
            input_obj:  The PowerSpectrum object to use
            config:     The configuration dict for 'power_spectrum'
            base:       The base configuration dict.
            logger:     If given, a logger object to log progress.
        """
        base['current_image'] = input_obj.copy()

# Register this as a valid input type
RegisterInputType('initial_image', InitialImageLoader())
