# Copyright (c) 2012-2016 by the GalSim developers team on GitHub
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
"""@file sensor.py

The Sensor classes implement the process of turning a set of photons incident at the surface
of the detector in the focal plane into an image with counts of electrons in each pixel.

The Sensor class itself implements the simplest possible sensor model, which just converts each
photon into an electron in whatever pixel is below the location where the photon hits.
However, it also serves as a base class for other more classes that implement more sophisticated
treatments of the photon to electron conversion and the drift from the conversion layer to the
bottom of the detector.
"""

import numpy as np
import galsim

class Sensor(object):
    """
    The base class for other sensor models, and also an implementation of the simplest possible
    sensor model that just converts each photon into an electron and drops it in the appropriate
    pixel.
    """

    def __init__(self):
        pass

    def accumulate(self, photons, image):
        """Accumulate the photons incident at the surface of the sensor into the appropriate
        pixels in the image.

        Each photon has a position, which corresponds to the (x,y) position at the top of the
        sensor.  In general, they may also have incidence directions and wavelengths, although
        these are not used by the base class implementation.

        The base class implementation simply accumulates the photons above each pixel into that
        pixel.

        @param photons      A PhotonArray instance describing the incident photons
        @param image        The image into which the photons should be accumuated.
        """
        return photons.addTo(image.image)


class SiliconSensor(Sensor):
    """
    A model of a silicon-based CCD sensor that converts photons to electrons at a wavelength-
    dependent depth (probabilistically) and drifts them down to the wells, properly taking
    into account the repulsion of previously accumulated electros (known as the brighter-fatter
    effect).

    @param config_file      A configuration file with all the details about this sensor model.
                            TODO: makde these configuration parameters settable from Python.
    @param rng              A BaseDeviate object to use for the random number generation
                            for the stochastic aspects of the electron production and drift.
                            [default: None, in which case one will be made for you]
    """
    def __init__(self, config_file, rng):
        self._silicon = galsim._galsim.Silicon(config_file)

        if rng is None:
            self.rng = galsim.UniformDeviate()
        elif not isinstance(rng, galsim.BaseDeviate):
            raise TypeError("rng is not a BaseDeviate")
        else:
            self.rng = galsim.UniformDeviate(rng)


    def accumulate(self, photons, image):
        """Accumulate the photons incident at the surface of the sensor into the appropriate
        pixels in the image.

        @param photons      A PhotonArray instance describing the incident photons
        @param image        The image into which the photons should be accumuated.
        """
        return self._silicon.accumulate(photons, self.rng, image.image)

