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
import glob
import os

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
    into account the repulsion of previously accumulated electrons (known as the brighter-fatter
    effect).

    @param dir              The name of the directory which contains the sensor information,
                            presumably calculated from the Poisson_CCD simulator.  This directory
                            may be specified either as an absolute path or as a subdirectory of
                            share_dir/sensors/, where share_dir is `galsim.meta_data.share_dir`.
                            It must contain, at a minimum, the *.cfg file used to simulate the
                            pixel distortions, and the *_Vertices.dat file which carries the
                            distorted pixel information.  [default: 'lsst_itl']
    @param num_elec         This parameter is the number of electrons in the central pixel in the
                            Poisson simulation that generated the vertex_file.  Depending how the
                            simulation was done, it may depend on different parameters in the
                            config_file, so can be entered manually. Note that you can also
                            use this parameter to adjust the strength of the brighter-fatter
                            effect.  For example, if *_Vertices.dat file was generated with
                            80,000 e- in the reference pixel, and you enter 40,000 in num_elec,
                            you will basically be doubling the strength of the brighter-fatter
                            effect.  [default: None, in which case the code reads the
                            CollectedCharge_0_0 parameter from  the .cfg file]
    @param rng              A BaseDeviate object to use for the random number generation
                            for the stochastic aspects of the electron production and drift.
                            [default: None, in which case one will be made for you]
    @param diffusion_factor A factor by which to multiple the diffusion.  Use 0.0 to turn off the
                            effect of diffusion entirely. [default: 1.0]
    @param qdist            The maximum number of pixels away to calculate the distortion due to
                            the charge accumulation. A large value will increase accuracy but
                            take more time. If it is increased larger than 4, the size of the
                            Poisson simulation must be increased to match. [default: 3]
    @param nrecalc          The number of electrons to accumulate before recalculating the
                            distortion of the pixel shapes. [default: 10000]

    """
    def __init__(self, dir='lsst_itl', num_elec=None, rng=None, diffusion_factor=1.0, qdist=3,
                 nrecalc=10000):
        self.rng = galsim.UniformDeviate(rng)
        if not os.path.isdir(dir):
            dir1 = os.path.join(galsim.meta_data.share_dir, 'sensors', dir)
            if not os.path.isdir(dir1):
                raise IOError("Cannot locate directory %s or %s"%(dir,dir1))
            dir = dir1

        config_files = glob.glob(os.path.join(dir,'*.cfg'))
        if len(config_files) == 0:
            raise IOError("No .cfg file found in dir %s"%dir)
        elif len(config_files) > 1:
            raise IOError("Multiple .cfg files found in dir %s"%dir)
        else:
            config_file = config_files[0]

        config = self._read_config_file(config_file)
        diff_step = self._calculate_diff_step(config, diffusion_factor)
        NumVertices = config['NumVertices']
        Nx = config['PixelBoundaryNx']
        Ny = config['PixelBoundaryNy']
        PixelSize = config['PixelSize']
        SensorThickness = config['SensorThickness']
        vertex_file = os.path.join(dir,config['outputfilebase'] + '_0_Vertices.dat')
        if num_elec == None:
            num_elec = config['CollectedCharge_0_0']
        vertex_data = np.loadtxt(vertex_file, skiprows = 1)

        if vertex_data.size != 5 * Nx * Ny * (4 * NumVertices + 4):
            raise IOError("Vertex file %s does not match config file %s"%(vertex_file, config_file))

        self._silicon = galsim._galsim.Silicon(NumVertices, num_elec, Nx, Ny, qdist, nrecalc,
                                               diff_step, PixelSize, SensorThickness, vertex_data)

    def accumulate(self, photons, image):
        """Accumulate the photons incident at the surface of the sensor into the appropriate
        pixels in the image.

        @param photons      A PhotonArray instance describing the incident photons
        @param image        The image into which the photons should be accumuated.
        """
        return self._silicon.accumulate(photons, self.rng, image.image)

    def _read_config_file(self, filename):
        # This reads the Poisson simulator config file for
        # the settings that were run
        # and returns a dictionary with the values

        with open(filename,'r') as file:
            lines=file.readlines()
        lines = [ l.strip() for l in lines ]
        lines = [ l.split() for l in lines if len(l) > 0 and l[0] != '#' ]
        if any([l[1] != '=' for l in lines]):
            raise IOError("Error reading config file %s"%filename)
        config = dict([(l[0], l[2]) for l in lines])
        # convert strings to int or float values when appropriate
        for k in config:
            try:
                config[k] = eval(config[k])
            except:
                pass
        return config

    def _calculate_diff_step(self, config, diffusion_factor):
        CollectingPhases = config['CollectingPhases']
        PixelSize = config['PixelSize']
        SensorThickness = config['SensorThickness']
        ChannelStopWidth = config['ChannelStopWidth']
        Vbb = config['Vbb']
        Vparallel_lo = config['Vparallel_lo']
        Vparallel_hi = config['Vparallel_hi']
        CCDTemperature = config['CCDTemperature']
        # This calculates the diffusion step size given the detector
        # parameters.  The diffusion step size is the mean radius of diffusion
        # assuming the electron propagates the full width of the sensor.
        # It depends on the temperature, the sensor voltages, and
        # the diffusion_factor parameter.

        # Set up the diffusion step size at 100 C
        if CollectingPhases == 1: # pragma: no cover
            # This is one collecting gate
            Vdiff = (2.0 * Vparallel_lo + Vparallel_hi) / 3.0 - Vbb
        elif CollectingPhases == 2: # This is the only value we have now, so the only one tested
            #This is two collecting gates
            Vdiff = (Vparallel_lo + 2.0 * Vparallel_hi) / 3.0 - Vbb
        else: # pragma: no cover
            return 0.0;

        # 0.026 is kT/q at room temp (298 K)
        diff_step = np.sqrt(2 * 0.026 * CCDTemperature / 298 / Vdiff)
        diff_step *= SensorThickness * diffusion_factor

        return diff_step
