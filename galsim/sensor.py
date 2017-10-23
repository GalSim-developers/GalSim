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

    def accumulate(self, photons, image, orig_center):
        """Accumulate the photons incident at the surface of the sensor into the appropriate
        pixels in the image.

        Each photon has a position, which corresponds to the (x,y) position at the top of the
        sensor.  In general, they may also have incidence directions and wavelengths, although
        these are not used by the base class implementation.

        The base class implementation simply accumulates the photons above each pixel into that
        pixel.

        @param photons      A PhotonArray instance describing the incident photons.
        @param image        The image into which the photons should be accumuated.
        @param orig_center  The original center of the image, before the image was re-centered to (0,0)
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
    The following parameters characterize "tree rings", which add small distortions to the sensor
    pixel positions due to non-uniform background doping in the silicon sensor. The default pattern 
    is a cosine function with period specified by treeringperiod.  Alternatively, the user can
    specify an arbitrary f(r) function which characterizes the tree ring pattern.  This requires 
    the extra parameters listed below

    @param treeringcenterx  The x-coord of the center of the tree ring pattern, which may be 
                            outside the pixel region.  This is in pixels. [default = -1000.0]
    @param treeringcentery  The y-coord of the center of the tree ring pattern, which may be 
                            outside the pixel region.  This is in pixels. [default = -1000.0]
    @param treeringamplitude  The amplitude of the tree ring pattern distortion.  Typically
                              this is less than 0.01 pixels. [default = 0.0]
    @param treeringperiod   The period of the tree ring distortion pattern, in pixels.
                            [default = 22.5]

    In the case of the user-defined f(r) pattern, the parameters below must be specified.
    @param function     A callable function giving the tree ring pattern f(r), or a 
                        file containing the function as a 2-column ASCII table.  The
                        function should return values between zero and one, which is then
                        multiplied by the treeringamplitude parameter. [default: None]
    @param x_min        The minimum desired return value (required for non-LookupTable
                        callable functions; will raise an error if not passed in that case, or if
                        passed in any other case) [default: None]
    @param x_min        The maximum desired return value (required for non-LookupTable
                        callable functions; will raise an error if not passed in that case, or if
                        passed in any other case) [default: None]
    @param interpolant  Type of interpolation used for interpolating a file (causes an error if
                        passed alongside a callable function).  Options are given in the
                        documentation for LookupTable. [default: 'linear']
    @param npoints      Number of points we should create for the internal interpolation
                        tables. [default: 2048]



    """
    def __init__(self, dir='lsst_itl_8', strength=1.0, rng=None, diffusion_factor=1.0, qdist=3,
                 nrecalc=10000, treeringcenterx=-1000.0, treeringcentery=-1000.0,
                 treeringamplitude=0.0, treeringperiod=22.5, function=None, x_min=None, x_max=None,
                 interpolant='linear', npoints=2048):
        self.dir = dir
        self.strength = strength
        self.rng = galsim.UniformDeviate(rng)
        self.diffusion_factor = diffusion_factor
        self.qdist = qdist
        self.nrecalc = nrecalc
        self.treeringcenterx = treeringcenterx
        self.treeringcentery = treeringcentery
        self.treeringamplitude = treeringamplitude
        self.treeringperiod = treeringperiod        

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
        if function is None:
            # This is just a place holder until I figure out how to pass this to the C++ side.
            self.table = galsim.LookupTable(f=[0.0,1.0,2.0,3.0], x=[0.0,0.1,0.2,0.3], interpolant=interpolant)
        else:            
            self._create_lookup_table(self, function, x_min, x_max, interpolant, npoints)
        self._init_silicon()

    def _init_silicon(self):
        diff_step = self._calculate_diff_step() * self.diffusion_factor
        NumVertices = self.config['NumVertices']
        Nx = self.config['PixelBoundaryNx']
        Ny = self.config['PixelBoundaryNy']
        if 'PixelSize' in self.config:
            PixelSize = self.config['PixelSize']
        elif 'PixelSizeX' in self.config:
            PixelSize = self.config['PixelSizeX']
        else:
            PixelSize = 10.0
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
                                               diff_step, PixelSize, SensorThickness, vertex_data,
                                               self.treeringcenterx, self.treeringcentery,
                                               self.treeringamplitude, self.treeringperiod, self.table)

    def __str__(self):
        s = 'galsim.SiliconSensor(%r'%self.dir
        if self.strength != 1.: s += ', strength=%f'%self.strength
        if self.diffusion_factor != 1.: s += ', diffusion_factor=%f'%self.diffusion_factor
        s += ')'
        return s

    def __repr__(self):
        return ('galsim.SiliconSensor(dir=%r, strength=%f, rng=%r, diffusion_factor=%f, '
                'qdist=%d, nrecalc=%f, treeringcenterx = %f, treeringcentery=%f, '
                'treerincamplitude=%f, treeringperiod=%f'%(self.full_dir, self.strength, self.rng,
                                        self.diffusion_factor, self.qdist, self.nrecalc,
                                        self.treeringcenterx, self.treeringcentery,
                                        self.treeringamplitude, self.treeringperiod, self.table))

    def __eq__(self, other):
        return (isinstance(other, SiliconSensor) and
                self.config == other.config and
                self.strength == other.strength and
                self.rng == other.rng and
                self.diffusion_factor == other.diffusion_factor and
                self.qdist == other.qdist and
                self.nrecalc == other.nrecalc and
                self.treeringcenterx == other.treeringcenterx and 
                self.treeringcentery == other.treeringcentery and
                self.treeringamplitude == other.treeringamplitude and
                self.treeringperiod == other.treeringperiod and
                self.table == other.table)

    __hash__ = None

    def __getstate__(self):
        d = self.__dict__.copy()
        del d['_silicon']
        return d

    def __setstate__(self, d):
        self.__dict__ = d
        self._init_silicon()  # Build the _silicon object.

    def accumulate(self, photons, image, orig_center):
        """Accumulate the photons incident at the surface of the sensor into the appropriate
        pixels in the image.

        @param photons      A PhotonArray instance describing the incident photons
        @param image        The image into which the photons should be accumuated.
        @param orig_center  The original center of the image, before the image was re-centered to (0,0)
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

    def _calculate_diff_step(self):
        NumPhases = self.config['NumPhases']        
        CollectingPhases = self.config['CollectingPhases']
        # I'm assuming square pixels for now.
        if 'PixelSize' in self.config:
            PixelSize = self.config['PixelSize']
        elif 'PixelSizeX' in self.config:
            PixelSize = self.config['PixelSizeX']
        else:
            PixelSize = 10.0
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

    def _create_lookup_table(self, function, x_min, x_max, interpolant, npoints):

        # Figure out if a string is a filename or something we should be using in an eval call
        if isinstance(function, str):
            import os.path
            if os.path.isfile(function):
                if interpolant is None:
                    interpolant='linear'
                if x_min or x_max:
                    raise TypeError('Cannot pass x_min or x_max alongside a '
                                    'filename in arguments to SiliconSensor')
                table = galsim.LookupTable(file=function, interpolant=interpolant)
                x_min = function.x_min
                x_max = function.x_max
            else:
                try:
                    function = galsim.utilities.math_eval('lambda x : ' + function)
                    if x_min is not None: # is not None in case x_min=0.
                        function(x_min)
                    else:
                        # Somebody would be silly to pass a string for evaluation without x_min,
                        # but we'd like to throw reasonable errors in that case anyway
                        function(0.6) # A value unlikely to be a singular point of a function
                except Exception as e:
                    raise ValueError(
                        "String function must either be a valid filename or something that "+
                        "can eval to a function of x.\n"+
                        "Input provided: {0}\n".format(input_function)+
                        "Caught error: {0}".format(e))
        else:
            # Check that the function is actually a function
            if not (isinstance(function, galsim.LookupTable) or hasattr(function,'__call__')):
                raise TypeError('Keyword function must be a callable function or a string')
            if interpolant:
                raise TypeError('Cannot provide an interpolant with a callable function argument')
            if isinstance(function,galsim.LookupTable):
                if x_min or x_max:
                    raise TypeError('Cannot provide x_min or x_max with a LookupTable function '+
                                    'argument')
                x_min = function.x_min
                x_max = function.x_max
            else:
                if x_min is None or x_max is None:
                    raise TypeError('Must provide x_min and x_max when function argument is a '+
                                    'regular python callable function')

        xarray = x_min+(1.*x_max-x_min)/(npoints-1)*np.array(range(npoints),float)
        farray = [function(xarray[i]) for i in range(npoints)]
        table = galsim.LookupTable(farray, xarray, interpolant=interpolant)
        self.table = table
        return

    
