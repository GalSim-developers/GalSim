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

    def accumulate(self, photons, image, orig_center=None):
        """Accumulate the photons incident at the surface of the sensor into the appropriate
        pixels in the image.

        Each photon has a position, which corresponds to the (x,y) position at the top of the
        sensor.  In general, they may also have incidence directions and wavelengths, although
        these are not used by the base class implementation.

        The base class implementation simply accumulates the photons above each pixel into that
        pixel.

        @param photons      A PhotonArray instance describing the incident photons.
        @param image        The image into which the photons should be accumuated.
        @param orig_center  The position of the image center in the original image coordinates.
                            [default: (0,0)]
        """
        return photons.addTo(image)

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

    There are currently three sensors shipped with GalSim, which you can specify as the `name`
    parameter mentioned below.

        lsst_itl_8      The ITL sensor being used for LSST, using 8 points along each side of the
                        pixel boundaries.
        lsst_itl_32     The ITL sensor being used for LSST, using 32 points along each side of the
                        pixel boundaries.  (This is more accurate than the lsst_itl_8, but slower.)
        lsst_etv_32     The ETV sensor being used for LSST, using 32 points along each side of the
                        pixel boundaries.  (This file is still somewhat preliminary and may be
                        updated in the future.)

    There is also an option to include "tree rings" in the Silicon model, which add small
    distortions to the sensor pixel positions due to non-uniform background doping in the silicon
    sensor.  The tree rings are defined by a center and a radial amplitude function.  The radial
    function needs to be a galsim.LookupTable instance.  Note that if you just want a simple cosine
    radial function, you can use the helper class method `SiliconSensor.simple_treerings` to
    build the LookupTable for you.

    @param name             The base name of the files which contains the sensor information,
                            presumably calculated from the Poisson_CCD simulator, which may
                            be specified either as an absolute path or as one of the above names
                            that are in the `galsim.meta_data.share_dir/sensors` directory.
                            name.cfg should be the file used to simulate the pixel distortions,
                            and name.dat should containt the distorted pixel information.
                            [default: 'lsst_itl_8']
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
    @param treering_func    A galsim.LookupTable giving the tree ring pattern f(r). [default: None]
    @param treering_center  A PositionD object with the center of the tree ring pattern in pixel
                            coordinates, which may be outside the pixel region. [default: None;
                            required if treering_func is provided]
    """
    def __init__(self, name='lsst_itl_8', strength=1.0, rng=None, diffusion_factor=1.0, qdist=3,
                 nrecalc=10000, treering_func=None, treering_center=galsim.PositionD(0,0)):
        self.name = name
        self.strength = strength
        self.rng = galsim.UniformDeviate(rng)
        self.diffusion_factor = diffusion_factor
        self.qdist = qdist
        self.nrecalc = nrecalc
        self.treering_func = treering_func
        self.treering_center = treering_center

        self.config_file = name + '.cfg'
        self.vertex_file = name + '.dat'
        if not os.path.isfile(self.config_file):
            cfg_file = os.path.join(galsim.meta_data.share_dir, 'sensors', self.config_file)
            if not os.path.isfile(cfg_file):
                raise IOError("Cannot locate file %s or %s"%(self.config_file, cfg_file))
            self.config_file = cfg_file
            self.vertex_file = os.path.join(galsim.meta_data.share_dir, 'sensors', self.vertex_file)
        if not os.path.isfile(self.vertex_file):
            raise IOError("Cannot locate vertex file %s"%(self.vertex_file))

        self.config = self._read_config_file(self.config_file)

        # Get the Tree ring radial function, if it exists
        if treering_func is None:
            # This is a dummy table in the case where no function is specified
            # A bit kludgy, but it works
            self.treering_func = galsim.LookupTable(x=[0.0,1.0], f=[0.0,0.0], interpolant='linear')
        elif not isinstance(treering_func, galsim.LookupTable):
            raise ValueError("treering_func must be a galsim.LookupTable")
        if not isinstance(treering_center, galsim.PositionD):
            raise ValueError("treering_center must be a galsim.PositionD")

        # Now we read in the absorption length table:
        abs_file = os.path.join(galsim.meta_data.share_dir, 'sensors', 'abs_length.dat')
        self._read_abs_length(abs_file)
        self._init_silicon()

    def _init_silicon(self):
        diff_step = self._calculate_diff_step() * self.diffusion_factor
        NumVertices = self.config['NumVertices']
        Nx = self.config['PixelBoundaryNx']
        Ny = self.config['PixelBoundaryNy']
        # This parameter may be either PixelSize or PixelSizeX.
        PixelSize = self.config['PixelSizeX']
        SensorThickness = self.config['SensorThickness']
        num_elec = float(self.config['CollectedCharge_0_0']) / self.strength
        # Scale this too, especially important if strength >> 1
        nrecalc = float(self.nrecalc) / self.strength
        vertex_data = np.loadtxt(self.vertex_file, skiprows = 1)

        if vertex_data.size != 5 * Nx * Ny * (4 * NumVertices + 4):
            raise IOError("Vertex file %s does not match config file %s"%(
                          self.vertex_file, self.config_file))

        self._silicon = galsim._galsim.Silicon(NumVertices, num_elec, Nx, Ny, self.qdist, nrecalc,
                                               diff_step, PixelSize, SensorThickness, vertex_data,
                                               self.treering_func.table, self.treering_center,
                                               self.abs_length_table.table)

    def __str__(self):
        s = 'galsim.SiliconSensor(%r'%self.name
        if self.strength != 1.: s += ', strength=%f'%self.strength
        if self.diffusion_factor != 1.: s += ', diffusion_factor=%f'%self.diffusion_factor
        s += ')'
        return s

    def __repr__(self):
        return ('galsim.SiliconSensor(name=%r, strength=%f, rng=%r, diffusion_factor=%f, '
                'qdist=%d, nrecalc=%f, treering_func=%r, treering_center=%r)')%(
                        self.name, self.strength, self.rng,
                        self.diffusion_factor, self.qdist, self.nrecalc,
                        self.treering_func, self.treering_center)

    def __eq__(self, other):
        return (isinstance(other, SiliconSensor) and
                self.config == other.config and
                self.strength == other.strength and
                self.rng == other.rng and
                self.diffusion_factor == other.diffusion_factor and
                self.qdist == other.qdist and
                self.nrecalc == other.nrecalc and
                self.treering_func == other.treering_func and
                self.treering_center == other.treering_center)

    __hash__ = None

    def __getstate__(self):
        d = self.__dict__.copy()
        del d['_silicon']
        return d

    def __setstate__(self, d):
        self.__dict__ = d
        self._init_silicon()  # Build the _silicon object.

    def accumulate(self, photons, image, orig_center=galsim.PositionI(0,0)):
        """Accumulate the photons incident at the surface of the sensor into the appropriate
        pixels in the image.

        @param photons      A PhotonArray instance describing the incident photons
        @param image        The image into which the photons should be accumuated.
        @param orig_center  The position of the image center in the original image coordinates.
                            [default: (0,0)]
        """
        return self._silicon.accumulate(photons, self.rng, image._image.view(), orig_center)

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

    def _read_abs_length(self, filename):
        # This reads in a table of absorption
        # length vs wavelength in Si.
        # The ipython notebook that created the data
        # file from astropy is in the same directory
        # in share/sensors/absorption
        abs_data = np.loadtxt(filename, skiprows = 1)
        xarray = abs_data[:,0]
        farray = abs_data[:,1]
        table = galsim.LookupTable(x=xarray, f=farray, interpolant='linear')
        self.abs_length_table = table
        return

    def _calculate_diff_step(self):
        NumPhases = self.config['NumPhases']
        CollectingPhases = self.config['CollectingPhases']
        # I'm assuming square pixels for now.
        PixelSize = self.config['PixelSizeX']
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

        # Set up the diffusion step size at the operating temperature
        HiPhases = CollectingPhases
        LoPhases = NumPhases - CollectingPhases
        Vdiff = (LoPhases * Vparallel_lo + HiPhases * Vparallel_hi) / NumPhases - Vbb
        # 0.026 is kT/q at room temp (298 K)
        diff_step = np.sqrt(2 * 0.026 * CCDTemperature / 298.0 / Vdiff) * SensorThickness
        return diff_step

    @classmethod
    def simple_treerings(cls, amplitude=0.5, period=100., r_max=8000., dr=None):
        """Make a simple sinusoidal tree ring pattern that can be used as the treering_func
        parameter of SiliconSensor.

        The functional form is f(r) = amplitude * cos(2*pi*r/period)

        @param amplitude    The amplitude of the tree ring pattern distortion.  Typically
                            this is less than 0.01 pixels. [default: 0.5]
        @param period       The period of the tree ring distortion pattern, in pixels.
                            [default: 100.]
        @param r_max        The maximum value of r to store in the lookup table. [default: 8000]
        @param dr           The spacing to use for the r values. [default: period/100]
        """
        k = 2.*np.pi/float(period)
        func = lambda r: amplitude * np.cos(k * r)
        if dr is None:
            dr = period/100.
        npoints = int(r_max / dr) + 1
        return galsim.LookupTable.from_func(func, x_min=0., x_max=r_max, npoints=npoints)
