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
However, it also serves as a base class for other classes that implement more sophisticated
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

        @param photons      A PhotonArray instance describing the incident photons.
        @param image        The image into which the photons should be accumuated.
        """
        return photons.addTo(image.image.view())

    def __repr__(self):
        return 'galsim.Sensor()'

    def __eq__(self, other):
        return (isinstance(other, Sensor) and
                repr(self) == repr(other))  # Checks that neither is a subclass

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self): return hash(repr(self))


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
    @param strength         Set the strength of the brighter-fatter effect relative to the
                            amount specified by the Poisson simulation results.  [default: 1]
    @param rng              A BaseDeviate object to use for the random number generation
                            for the stochastic aspects of the electron production and drift.
                            [default: None, in which case one will be made for you]
    @param diffusion_factor A factor by which to multiply the diffusion.  Use 0.0 to turn off the
                            effect of diffusion entirely. [default: 1.0]
    @param qdist            The maximum number of pixels away to calculate the distortion due to
                            the charge accumulation. A large value will increase accuracy but
                            take more time. If it is increased larger than 4, the size of the
                            Poisson simulation must be increased to match. [default: 3]
    @param nrecalc          The number of electrons to accumulate before recalculating the
                            distortion of the pixel shapes. [default: 10000]

    """
    def __init__(self, dir='lsst_itl', strength=1.0, rng=None, diffusion_factor=1.0, qdist=3,
                 nrecalc=10000):
        self.dir = dir
        self.strength = strength
        self.rng = galsim.UniformDeviate(rng)
        self.diffusion_factor = diffusion_factor
        self.qdist = qdist
        self.nrecalc = nrecalc

        if not os.path.isdir(dir):
            self.full_dir = os.path.join(galsim.meta_data.share_dir, 'sensors', dir)
            if not os.path.isdir(self.full_dir):
                raise IOError("Cannot locate directory %s or %s"%(dir,self.full_dir))
        else:
            self.full_dir = dir

        config_files = glob.glob(os.path.join(self.full_dir,'*.cfg'))
        if len(config_files) == 0:
            raise IOError("No .cfg file found in dir %s"%self.full_dir)
        elif len(config_files) > 1:
            raise IOError("Multiple .cfg files found in dir %s"%self.full_dir)
        else:
            self.config_file = config_files[0]

        self.config = self._read_config_file(self.config_file)
        self._init_silicon()

    def _init_silicon(self):
        diff_step = self._calculate_diff_step() * self.diffusion_factor
        NumVertices = self.config['NumVertices']
        Nx = self.config['PixelBoundaryNx']
        Ny = self.config['PixelBoundaryNy']
        PixelSize = self.config['PixelSize']
        SensorThickness = self.config['SensorThickness']
        num_elec = float(self.config['CollectedCharge_0_0']) / self.strength
        # Scale this too, especially important if strength >> 1
        nrecalc = float(self.nrecalc) / self.strength
        vertex_file = os.path.join(self.full_dir,self.config['outputfilebase'] + '_0_Vertices.dat')
        vertex_data = np.loadtxt(vertex_file, skiprows = 1)

        if vertex_data.size != 5 * Nx * Ny * (4 * NumVertices + 4):
            raise IOError("Vertex file %s does not match config file %s"
                          % (vertex_file, self.config_file))

        self._silicon = galsim._galsim.Silicon(NumVertices, num_elec, Nx, Ny, self.qdist, nrecalc,
                                               diff_step, PixelSize, SensorThickness, vertex_data)

    def __str__(self):
        s = 'galsim.SiliconSensor(%r'%self.dir
        if self.strength != 1.: s += ', strength=%f'%self.strength
        if self.diffusion_factor != 1.: s += ', diffusion_factor=%f'%self.diffusion_factor
        s += ')'
        return s

    def __repr__(self):
        return ('galsim.SiliconSensor(dir=%r, strength=%f, rng=%r, diffusion_factor=%f, '
                'qdist=%d, nrecalc=%f'%(self.full_dir, self.strength, self.rng,
                                        self.diffusion_factor, self.qdist, self.nrecalc))

    def __eq__(self, other):
        return (isinstance(other, SiliconSensor) and
                self.config == other.config and
                self.strength == other.strength and
                self.rng == other.rng and
                self.diffusion_factor == other.diffusion_factor and
                self.qdist == other.qdist and
                self.nrecalc == other.nrecalc)

    __hash__ = None

    def __getstate__(self):
        d = self.__dict__.copy()
        del d['_silicon']
        return d

    def __setstate__(self, d):
        self.__dict__ = d
        self._init_silicon()  # Build the _silicon object.

    def accumulate(self, photons, image):
        """Accumulate the photons incident at the surface of the sensor into the appropriate
        pixels in the image.

        @param photons      A PhotonArray instance describing the incident photons
        @param image        The image into which the photons should be accumuated.
        """
        return self._silicon.accumulate(photons, self.rng, image.image.view())

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
            except (SyntaxError, NameError):
                pass
        return config

    def _calculate_diff_step(self):
        CollectingPhases = self.config['CollectingPhases']
        PixelSize = self.config['PixelSize']
        SensorThickness = self.config['SensorThickness']
        ChannelStopWidth = self.config['ChannelStopWidth']
        Vbb = self.config['Vbb']
        Vparallel_lo = self.config['Vparallel_lo']
        Vparallel_hi = self.config['Vparallel_hi']
        CCDTemperature = self.config['CCDTemperature']
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
            return 0.0

        # 0.026 is kT/q at room temp (298 K)
        diff_step = np.sqrt(2 * 0.026 * CCDTemperature / 298 / Vdiff) * SensorThickness

        return diff_step
