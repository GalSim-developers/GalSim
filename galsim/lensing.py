"""\file lensing.py The "lensing engine" for drawing shears from some power spectrum or a NFW halo.
"""
import galsim
import numpy as np

# A helper function for parsing the input position arguments for PowerSpectrum and NFWHalo:
def _convertPositions(pos, units, func):
    """Convert pos from the valid ways to input positions to two numpy arrays

       This is used by the functions getShear, getConvergence, and getMag for both 
       PowerSpectrum and NFWHalo (the former only has getShear currently).
    """
    try:
        # Check for PositionD or PositionI:
        if isinstance(pos,galsim.PositionD) or isinstance(pos,galsim.PositionI):
            pos = ( np.array([pos.x], dtype='float'),
                    np.array([pos.y], dtype='float') )

        # Check for list of PositionD or PositionI:
        # The only other options allow pos[0], so if this is invalid, an exception 
        # will be raised and appropriately dealt with:
        elif isinstance(pos[0],galsim.PositionD) or isinstance(pos[0],galsim.PositionI):
            pos = ( np.array([p.x for p in pos], dtype='float'),
                    np.array([p.y for p in pos], dtype='float') )

        # Now pos must be a tuple of length 2
        elif len(pos) != 2:
            raise TypeError() # This will be caught below and raised with a better error msg.

        # Check for (x,y):
        elif isinstance(pos[0],float):
            pos = ( np.array([pos[0]], dtype='float'),
                    np.array([pos[1]], dtype='float') )

        # Only other valid option is ( xlist , ylist )
        else:
            pos = ( np.array(pos[0], dtype='float'),
                    np.array(pos[1], dtype='float') )

        # Check validity of units
        if isinstance(units, basestring):
            # if the string is invalid, this raises a reasonable error message.
            units = galsim.angle.get_angle_unit(units)
        if not isinstance(units, galsim.AngleUnit):
            raise ValueError("units must be either an AngleUnit or a string")

        # Convert pos to arcsec
        if units != galsim.arcsec:
            scale = 1. * units / galsim.arcsec
            pos[0] *= scale
            pos[1] *= scale

        return pos

    except:
        raise TypeError("Unable to parse the input pos argument for %s."%func)
            

class PowerSpectrum(object):
    """Class to represent a lensing shear field according to some power spectrum P(k)

    A PowerSpectrum represents some (flat-sky) shear power spectrum, either for gridded points or at
    arbitary positions.  This class is originally initialized with a power spectrum from which we
    would like to generate g1 and g2 values.  It generate shears on a grid, and if necessary,
    when getShear is called, it wil interpolate to the requested positions. 

    When creating a PowerSpectrum instance, you need to specify at least one of the E or B mode 
    power spectrum, which is normally given as a function P(k).  The typical thing is to just 
    use a lambda function in Python (i.e., a function that is not associated with a name); 
    for example, to define P(k)=k^2, one would use `lambda k : k**2`.  But they can also be more 
    complicated user-defined functions that take a single argument k and return the power at that 
    k value, or they can be instances of the LookupTable class for power spectra that are known 
    at particular k values but for which there is not a simple analytic form.
    
    The power functions should return either power P(k), or Delta^2(k) = k^2 P(k) / 2pi.
    If the latter, then you should set `delta2 = True` in teh constructor.  We assume that P(k)
    goes to zero at k=0, as in any physically reasonable cosmological model.

    The power functions must return a list/array that is the same size as what it was given, e.g.,
    in the case of no power or constant power, a function that just returns a float would not be
    permitted; it would have to return an array of floats all with the same value.

    It is important to note that the power spectra used to initialize the PowerSpectrum object
    should be in the same units as any parameters to the getShear() method that define the locations
    at which we want to get shears.  When we actually draw images, there is a natural scale that
    defines the pitch of the image (dx), which is typically taken to be arcsec.  This definition of
    a specific length scale means that we should also use the same units (arcsec) for the positions
    at which we want our galaxies to be located when we draw shears from a power spectrum, and
    likewise the values of k (wavenumber) going into the power spectrum function should be inverse
    arcsec.  To give a specific example, if we want to draw Gaussians on an image with dx=0.2 arcsec
    (i.e., the argument dx to the draw method will be =0.2), and if we want a grid of galaxies
    spaced 40 pixels apart, then when we call the getShear method of the PowerSpectrum class, we
    should use grid_spacing=8 [arcsec, =(40 pixels)*(0.2 arcsec/pixel)].

    To use a different unit for k, you may specify such with the units kwarg in the constructor.
    This should be either a galsim.AngleUnit instance (e.g. galsim.radians) or a string 
    (e.g. 'radians')

    @param e_power_function A function or other callable that accepts a Numpy array of |k| values,
                            and returns the E-mode power spectrum P_E(|k|) in an array of the same
                            shape.  The function should return the power spectrum desired in the E
                            (gradient) mode of the image.  Set to None (default) for there to be no
                            E-mode power.
                            It may also be a string that can be converted to a function using
                            eval('lambda k : ' + e_power_function), or file_name from which to
                            read in a LookupTable.
    @param b_power_function A function or other callable that accepts a Numpy array of |k| values,
                            and returns the B-mode power spectrum P_B(|k|) in an array of the same
                            shape.  The function should return the power spectrum desired in the B
                            (curl) mode of the image.  Set to None (default) for there to be no
                            B-mode power.
                            It may also be a string that can be converted to a function using
                            eval('lambda k : ' + e_power_function), or file_name from which to
                            read in a LookupTable.
    @param delta2           Is the power actually given as Delta^2, which requires us to multiply 
                            by k^2 / (2pi) to get the shear power P(k)?  [default = False]
                            Note that if Delta^2 is provided, then it is critical that it have 
                            units consistent with the units for k.
    @param units            The angular units used for the power spectrum (i.e. the units of 
                            k^-1). [default = arcsec]
    """
    _req_params = {}
    _opt_params = { 'e_power_function' : str, 'b_power_function' : str,
                    'delta2' : bool, 'units' : str }
    _single_params = []
    def __init__(self, e_power_function=None, b_power_function=None,
                 delta2=False, units=galsim.arcsec):
        # Check that at least one power function is not None
        if e_power_function is None and b_power_function is None:
            raise AttributeError(
                "At least one of e_power_function or b_power_function must be provided.")
                
        self.e_power_function = e_power_function
        self.b_power_function = b_power_function
        self.delta2 = delta2

        # Try these conversions, but we don't actually keep the output.  This just 
        # provides a way to test if the arguments are sane.
        # Note: we redo this in buildGriddedShears for real rather than keeping the outputs
        # (e.g. in self.e_power_function, self.b_power_function) so that PowerSpectrum is 
        # picklable.  It turns out lambda functions are not picklable.
        self._convert_power_function(self.e_power_function,'e_power_function')
        self._convert_power_function(self.b_power_function,'b_power_function')

        # Check validity of units
        if isinstance(units, basestring):
            # if the string is invalid, this raises a reasonable error message.
            units = galsim.angle.get_angle_unit(units)
        if not isinstance(units, galsim.AngleUnit):
            raise ValueError("units must be either an AngleUnit or a string")

        if units == galsim.arcsec:
            self.scale = 1
        else:
            self.scale = 1. * units / galsim.arcsec


    def buildGriddedShears(self, grid_spacing=None, ngrid=None, rng=None,
                           interpolant=None, center=galsim.PositionD(0,0)):
        """Generate a realization of the current power spectrum an the specified grid.

        This function will generate a Gaussian random realization of the specified E and B mode 
        shear power spectra at a grid of positions, specified by the input parameters 
        `grid_spacing` (distance between grid points) and `ngrid` (number of grid points in each 
        direction.)

        The normalization of the shears from a given power spectrum is defined as follows: if 
        P_E(k)=P_B(k)=P [const],
        i.e., white noise in both shear components, then the shears g1 and g2 will be random
        Gaussian deviates with variance=P.  Note that if we really had power at all k, the variance
        would be infinite.  But we are getting shears on a grid, which has a limited k range, and
        hence the total power is finite.  For grid spacing dx and N grid points, the spacing between
        k values is dk = 2pi/(N dx) and k ranges from +/-(N/2) dk.  There are alternate definitions
        to consider, e.g., that the variance should be P*(dx)^2 for a grid spacing of dx (i.e., for
        fixed total grid extent, a smaller grid spacing requires smaller shear variances since the
        range of k values that are accessible is larger); those who input a continuum P(k) should,
        when predicting the behavior of shears on a grid, keep in mind our normalization convention
        and the fact that it's a discrete FFT.  If you strongly dislike our convention and would
        like support for an alternate one, please indicate this on our GitHub issues page.

        Also note that the convention for axis orientation differs from that for the GREAT10
        challenge, so when using codes that deal with GREAT10 challenge outputs, the sign of our g2
        shear component must be flipped.

        This function is called automatically from the constructor, but you can also call it
        manually to re-build the grid.  If you want to access the grid that is build during
        construction, you can use `getGriddedShears`.  On a rebuild, the grid is returned
        by buildGriddedShears.

        Note that the interpolation modifies the effective power spectrum somewhat.  The user is 
        responsible for choosing a grid size that is small enough not to significantly modify the
        power spectrum on the scales of interest. 

        Some examples:

        1. Get shears on a grid of points separated by 1 arcsec:

               my_ps = galsim.PowerSpectrum(lambda k : k**2, grid_spacing = 1., ngrid = 100)
               g1, g2 = my_ps.getGriddedShears()

           The returned g1,g2 are 2-d numpy arrays of values, corresponding to the values of 
           g1,g2 at the locations of the grid points.

           For a given value of grid_spacing and ngrid, we could get the x and y values on the
           grid using

               import numpy as np
               min = (-ngrid/2 + 0.5) * grid_spacing
               max = (ngrid/2 - 0.5) * grid_spacing
               x, y = np.meshgrid(np.arange(min,max,grid_spacing),
                                  np.arange(min,max,grid_spacing))

           where the center of the grid is taken to be (0,0).

        2. Rebuild the grid using a particular rng and set the location of the center of the grid
           to be something other than the default (0,0)

               g1, g2 = my_ps.buildGriddedShears(grid_spacing = 8., ngrid = 65,
                                                 rng = galsim.BaseDeviate(1413231),
                                                 center = (256.5, 256.5) )

        3. Make a PowerSpectrum from a tabulated P(k) that gets interpolated to find the power at
           all necessary values of k, then generate shears on a grid.  Assuming that k and P_k are
           either lists, tuples, or 1d Numpy arrays containing k and P(k):

               tab_pk = galsim.LookupTable(k, P_k)
               my_ps = galsim.PowerSpectrum(tab_pk, grid_spacing = 1., grid_nx = 100)

        @param grid_spacing     Spacing for an evenly spaced grid of points, in arcsec for
                                consistency with the natural length scale of images created using
                                the draw or drawShoot methods.
        @param ngrid            Number of grid points in each dimension.  If a number that is not
                                an int (e.g., a float) is supplied, then it gets converted to an int
                                automatically.
        @param rng              (Optional) A galsim.GaussianDeviate object for drawing the random
                                numbers.  (Alternatively, any BaseDeviate can be used.)
        @param interpolant      (Optional) Interpolant to use for interpolating the shears on a grid
                                to the requested positions.  [default = galsim.Linear()]
        @param center           (Optional) If setting up a new grid, define what position you
                                want to consider the center of that grid. [default = (0,0)]
        
        @return g1,g2           If given a single position: the two shear components g_1 and g_2.
                                If given a list of positions: each is a python list of values.
                                If pos=None, these are 2-d NumPy arrays.
        """
        # Check problem cases for regular grid of points
        if grid_spacing is None or ngrid is None:
            raise ValueError("Both a spacing and a size are required for buildGriddedShears.")
        # Check for non-integer ngrid
        if not isinstance(ngrid, int):
            try:
                ngrid = int(ngrid)
            except:
                raise ValueError("ngrid must be an int, or easily convertable to int!")

        # Check if center is a Position
        if isinstance(center,galsim.PositionD):
            pass  # This is what it should be
        elif isinstance(center,galsim.PositionI):
            # Convert to a PositionD
            center = galsim.PositionD(center.x, center.y)
        elif isinstance(center, tuple) and len(center) == 2:
            # Convert (x,y) tuple to PositionD
            center = galsim.PositionD(center[0], center[1])
        else:
            raise TypeError("Unable to parse the input center argument for getShear")

        # Make a GaussianDeviate if necessary
        if rng is None:
            gd = galsim.GaussianDeviate()
        elif isinstance(rng, galsim.GaussianDeviate):
            gd = rng
        elif isinstance(rng, galsim.BaseDeviate):
            gd = galsim.GaussianDeviate(rng)
        else:
            raise TypeError("The rng provided to getShear is not a BaseDeviate")

        # Check that the interpolant is valid
        self.interpolant = interpolant
        try :
            if self.interpolant is None:
                pass
            elif isinstance(interpolant, galsim.Interpolant):
                galsim.InterpolantXY(self.interpolant)
            elif isinstance(interpolant, galsim.InterpolantXY):
                pass
            else:
                raise TypeError()
        except:
            raise TypeError("Invalid interpolant provided to PowerSpectrum.buildGriddedShears")

        # Convert power_functions into callables:
        e_power_function = self._convert_power_function(self.e_power_function,'e_power_function')
        b_power_function = self._convert_power_function(self.b_power_function,'b_power_function')

        # If we actually have Delta^2, then we must convert to power, which is
        # dimensionless.  Also account for the possible unit scaling here.
        if e_power_function is None:
            p_E = None
        elif self.delta2:
            p_E = lambda k : (self.scale*k)**2 * e_power_function(self.scale*k)/(2.*np.pi)
        elif self.scale != 1:
            p_E = lambda k : e_power_function(self.scale*k)
        else: 
            p_E = e_power_function

        if b_power_function is None:
            p_B = None
        elif self.delta2:
            p_B = lambda k : (self.scale*k)**2 * b_power_function(self.scale*k)/(2.*np.pi)
        elif self.scale != 1:
            p_B = lambda k : b_power_function(self.scale*k)
        else:
            p_B = b_power_function

        # Build the grid 
        psr = PowerSpectrumRealizer(ngrid, ngrid, grid_spacing, p_E, p_B)
        self.grid_g1, self.grid_g2 = psr(gd)
            
        # Setup the images to be interpolated.
        # Note: We don't make the SBInterpolatedImages yet, since it's not picklable. 
        #       So just created them when we are actually going to use them.
        self.im_g1 = galsim.ImageViewD(self.grid_g1)
        self.im_g1.setScale(grid_spacing)

        self.im_g2 = galsim.ImageViewD(self.grid_g2)
        self.im_g2.setScale(grid_spacing)

        # Dealing with the center here is a bit confusing, especially if ngrid is even.
        # The InterpolatedImage will consider position (0,0) to correspond to 
        # self.im_g1.bounds.center() on the image.  We call this nominal_center.
        # However, if ngrid is even, this is slightly up and to the right of the 
        # true center. The true center x and y are at (1+ngrid)/2 * grid_spacing.
        # And finally, we may be passed a value to consider the center of the image.
        b = self.im_g1.bounds
        nominal_center = galsim.PositionD(b.center().x, b.center().y) * grid_spacing
        true_center = galsim.PositionD( (1.+ngrid)/2. , (1.+ngrid)/2. ) * grid_spacing
            
        # The offset to be added to any position is then such that if we are 
        # provided the target center position, the result will be the location of 
        # the true center with respect to the nominal center.  In other words:
        #   center + offset = true_center - nominal_center
        self.offset = true_center - nominal_center - center

        # Construct a bounds that we can use to check if a provided position will
        # end up falling on the interpolating image.
        self.bounds = galsim.BoundsD((b.xmin-0.5)*grid_spacing, (b.xmax+0.5)*grid_spacing,
                                     (b.ymin-0.5)*grid_spacing, (b.ymax+0.5)*grid_spacing)
        self.bounds.shift(-nominal_center - self.offset)

        return self.grid_g1, self.grid_g2

    def _convert_power_function(self, pf, pf_str):
        if pf is None: return None

        # Convert string inputs to either a lambda function or LookupTable
        if isinstance(pf,str):
            import os
            if os.path.isfile(pf):
                try:
                    #pf = galsim.LookupTable(file=pf, x_log=True, f_log=True)
                    pf = galsim.LookupTable(file=pf)
                except :
                    raise AttributeError(
                        "Unable to read %s = %s as a LookupTable"%(pf_str,pf))
            else:
                try : 
                    pf = eval('lambda k : ' + pf)
                except :
                    raise AttributeError(
                        "Unable to turn %s = %s into a valid function"%(pf_str,pf))

        # Check that the function is sane.
        # Note: Only try tests below if it's not a LookupTable.
        #       (If it's a LookupTable, then it could be a valid function that isn't 
        #        defined at k=1, and by definition it must return something that is the 
        #        same length as the input.)
        if not isinstance(pf, galsim.LookupTable):
            try:
                f1 = pf(1.)
            except:
                raise AttributeError("%s is not a valid function"%pf_str)
            fake_arr = np.zeros(2)
            fake_p = pf(fake_arr)
            if isinstance(fake_p, float):
                raise AttributeError(
                    "Power function MUST return a list/array same length as input")
        return pf


    def getShear(self, pos=None, units=galsim.arcsec):
        """
        This function can interpolate between the grid positions to find the shear values for a
        given list of input positions (or just a single position).  Before calling this function,
        you must call buildGriddedShears first to define the grid on which to interpolate.

        Some examples of how to use getShear:

        1. Get the shear for a particular point:

               g1, g2 = my_ps.getShear(pos = galsim.PositionD(12, 412))

           This time the returned values are just floats and correspond to the shear for the
           provided position.

        2. You can also provide a position as a tuple to save the explicit PositionD construction:

               g1, g2 = my_ps.getShear(pos = (12, 412))

        3. Get the shears for a bunch of points at once:
        
               xlist = [ 141, 313,  12, 241, 342 ]
               ylist = [  75, 199, 306, 225, 489 ]
               poslist = [ galsim.PositionD(xlist[i],ylist[i]) for i in range(len(xlist)) ]
               g1, g2 = my_ps.getShear( poslist )
               g1, g2 = my_ps.getShear( (xlist, ylist) )

           Both calls do the same thing.  The returned g1, g2 this time are lists of g1, g2 values.
           The lists are the same length as the number of input positions.

        @param pos              Position(s) of the source(s), assumed to be post-lensing!  (It is 
                                up to the user to check that the units are consistent with those in 
                                the P(k) function, just as for the grid_spacing keyword.)
                                Valid ways to input this:
                                  - Single galsim.PositionD (or PositionI) instance
                                  - tuple of floats: (x,y)
                                  - list of galsim.PositionD (or PositionI) instances
                                  - tuple of lists: ( xlist, ylist )
                                pos may also be None to just build a gridded array of (g1,g2).
        @param units            The angular units used for the positions.  [default = arcsec]
        
        @return g1,g2           If given a single position: the two shear components g_1 and g_2.
                                If given a list of positions: each is a python list of values.
        """

        if not hasattr(self, 'im_g1'):
            raise RuntimeError("PowerSpectrum.buildGriddedShears must be called before getShear")

        # Convert to numpy arrays for internal usage:
        pos_x, pos_y = _convertPositions(pos, units, 'getShear')

        # Set the interpolant:
        if self.interpolant is None:
            interpolantxy = galsim.InterpolantXY(galsim.Linear())
        elif isinstance(interpolant, galsim.Interpolant):
            interpolantxy = galsim.InterpolantXY(self.interpolant)
        elif isinstance(interpolant, galsim.InterpolantXY):
            interpolantxy = self.interpolant
        else:
            raise TypeError("Invalid interpolant provided to PowerSpectrum.buildGriddedShears")

        # Make an SBInterpolatedImage, which will do the heavy lifting for the 
        # interpolation.
        sbii_g1 = galsim.SBInterpolatedImage(self.im_g1, xInterp=interpolantxy)
        sbii_g2 = galsim.SBInterpolatedImage(self.im_g2, xInterp=interpolantxy)

        # interpolate if necessary
        g1,g2 = [], []
        for pos in [ galsim.PositionD(pos_x[i],pos_y[i]) for i in range(len(pos_x)) ]:
            # Check that the position is in the bounds of the interpolated image
            if not self.bounds.includes(pos):
                import warnings
                warnings.warn(
                    "Warning position (%f,%f) not within the bounds "%(pos.x,pos.y) +
                    "of the gridded shear values: " + str(self.bounds) + 
                    ".  Returning a shear of (0,0) for this point.")
                g1.append(0.)
                g2.append(0.)
            else:
                g1.append(sbii_g1.xValue(pos+self.offset))
                g2.append(sbii_g2.xValue(pos+self.offset))
        if len(pos_x) == 1:
            return g1[0], g2[0]
        else:
            return g1, g2

class PowerSpectrumRealizer(object):
    """Class for generating realizations of power spectra with any area and pixel size.
    
    This class is not one that end-users should expect to interact with.  It is designed to quickly
    generate many realizations of the same shear power spectra on a square grid.  The initializer
    sets up the grids in k-space and computes the power on them.  It also computes spin weighting
    terms.  You can alter any of the setup properties later.

    @param nx               The x-dimension of the desired image.
    @param ny               The y-dimension of the desired image.
    @param pixel_size       The size of the pixel sides, in units consistent with the units expected
                            by the power spectrum functions.
    @param e_power_function See description of this parameter in the documentation for the
                            PowerSpectrum class.
    @param b_power_function See description of this parameter in the documentation for the
                            PowerSpectrum class.
    """
    def __init__(self, nx, ny, pixel_size, p_E, p_B):
        # Set up the k grids in x and y, and the instance variables
        self.nx = nx
        self.ny = ny
        kx, ky=np.mgrid[0:nx/2+1,0:ny/2+1]
        self.kx = kx
        self.ky = ky
        pixel_size = float(pixel_size)

        # Set up the scalar |k| grid.
        self.k=((kx/(pixel_size*nx))**2+(ky/(pixel_size*ny))**2)**0.5
        
        #Compute the spin weightings
        self._cos, self._sin = self._generate_spin_weightings()
        
        self.p_E = p_E
        self.p_B = p_B
        if p_E is None:  self.amplitude_E = None
        else:            self.amplitude_E = np.sqrt(self._generate_power_array(p_E))
        if p_B is None:  self.amplitude_B = None
        else:            self.amplitude_B = np.sqrt(self._generate_power_array(p_B))


    def __call__(self, gd):
        """Generate a realization of the current power spectrum.
        
        @param gd               A gaussian deviate to use when generating the shear fields.
        @param new_power        If the power-spectrum functions that you specified are not
                                deterministic then you can set this value to True to call them again
                                to get new values.  For example, you could include a cosmic variance
                                term in your power spectrum and get a new spectrum realization each
                                time.
        
        @return g1,g2           Two image arrays for the two shear components g_1 and g_2
        """
        ISQRT2 = np.sqrt(1.0/2.0)

        if not isinstance(gd, galsim.GaussianDeviate):
            raise TypeError(
                "The gd provided to the PowerSpectrumRealizer is not a GaussianDeviate!")

        #Generate a random complex realization for the E-mode, if there is one
        if self.amplitude_E is not None:
            r1 = galsim.utilities.rand_arr(self.amplitude_E.shape, gd)
            r2 = galsim.utilities.rand_arr(self.amplitude_E.shape, gd)
            E_k = self.amplitude_E * (r1 + 1j*r2) * ISQRT2  
            #Do we need to multiply one of the rows by two to account for the reality?
        else: E_k = 0

        #Generate a random complex realization for the B-mode, if there is one
        if self.amplitude_B is not None:
            r1 = galsim.utilities.rand_arr(self.amplitude_B.shape, gd)
            r2 = galsim.utilities.rand_arr(self.amplitude_B.shape, gd)
            B_k = self.amplitude_B * (r1 + 1j*r2) * ISQRT2
        else:
            B_k = 0

        #Now convert from E,B to g1,g2  still in fourier space
        g1_k = self._cos*E_k - self._sin*B_k
        g2_k = self._sin*E_k + self._cos*B_k

        #And go to real space to get the images
        g1=g1_k.shape[0]*np.fft.irfft2(g1_k, s=(self.nx,self.ny))
        g2=g2_k.shape[0]*np.fft.irfft2(g2_k, s=(self.nx,self.ny))

        return g1, g2

    def _generate_power_array(self, power_function):
        #Internal function to generate the result of a power function evaluated on a grid,
        #taking into account the symmetries.
        power_array = np.zeros((self.nx, self.ny/2+1))

        # make a faked-up self.k array that fudges the value at k=0, so we don't have to evaluate
        # power there
        fake_k = self.k
        fake_k[0,0] = fake_k[1,0]
        # raise a clear exception for LookupTable that are not defined on the full k range!
        if isinstance(power_function, galsim.LookupTable):
            mink = np.min(fake_k)
            maxk = np.max(fake_k)
            if mink < power_function.x_min or maxk > power_function.x_max:
                raise ValueError(
                    "LookupTable P(k) is not defined for full k range on grid, %f<k<%f"%(mink,maxk))
        P_k = power_function(fake_k)
        
        # now fix the k=0 value of power to zero
        if type(P_k) is np.ndarray:
            P_k[0,0] = type(P_k[0,1])(0.)
        else:
            P_k = 0.
        power_array[ self.kx, self.ky] = P_k
        power_array[-self.kx, self.ky] = P_k
        if np.any(power_array < 0):
            raise ValueError("Negative power found for some values of k!")
        return power_array
    
    def _generate_spin_weightings(self):
        #Internal function to generate the cosine and sine spin weightings for the current 
        #array set up
        C=np.zeros((self.nx,self.ny/2+1))
        S=np.zeros((self.nx,self.ny/2+1))
        kx = self.kx
        ky = self.ky
        TwoPsi=2*np.arctan2(1.0*self.ky, 1.0*self.kx)
        C[kx,ky]=np.cos(TwoPsi)
        S[kx,ky]=np.sin(TwoPsi)
        C[-kx,ky]=-np.cos(TwoPsi)
        S[-kx,ky]=np.sin(TwoPsi)
        return C,S

class Cosmology(object):
    """Basic cosmology calculations.

    Cosmology calculates expansion function E(a) and angular diameter distances Da(z) for a
    LambdaCDM universe.  Radiation is assumed to be zero and Dark Energy constant with w = -1 (no
    quintessence), but curvature is arbitrary.

    Based on Matthias Bartelmann's libastro.

    @param omega_m    Present day energy density of matter relative to critical density.
    @param omega_lam  Present day density of Dark Energy relative to critical density.
    """
    def __init__(self, omega_m=0.3, omega_lam=0.7):
        # no quintessence, no radiation in this universe!
        self.omega_m = omega_m
        self.omega_lam = omega_lam
        self.omega_c = (1. - omega_m - omega_lam)
        self.omega_r = 0
    
    def a(self, z):
        """Compute scale factor

        @param z Redshift
        """
        return 1./(1+z)

    def E(self, a):
        """Evaluates expansion function

        @param a Scale factor
        """
        return (self.omega_r*a**(-4) + self.omega_m*a**(-3) + self.omega_c*a**(-2) + \
                self.omega_lam)**0.5

    def __angKernel(self, x):
        """Integration kernel for angular diameter distance computation
        """
        return self.E(x**-1)**-1

    def Da(self, z, z_ref=0):
        """Compute angular diameter distance between two redshifts in units of c/H0.

        In order to get the distance in Mpc/h, multiply by ~3000.

        @param z Redshift
        @param z_ref Reference redshift, with z_ref <= z.
        """
        if isinstance(z, np.ndarray):
            da = np.zeros_like(z)
            for i in range(len(da)):
                da[i] = self.Da(z[i], z_ref)
            return da
        else:
            if z < 0:
                raise ValueError("Redshift z must not be negative")
            if z < z_ref:
                raise ValueError("Redshift z must not be smaller than the reference redshift")

            d = galsim.integ.int1d(self.__angKernel, z_ref+1, z+1)
            # check for curvature
            rk = (abs(self.omega_c))**0.5
            if (rk*d > 0.01):
                if self.omega_c > 0:
                    d = sinh(rk*d)/rk
                if self.omega_c < 0:
                    d = sin(rk*d)/rk
            return d/(1+z)


class NFWHalo(object):
    """Class for NFW halos.

    Compute the lensing fields shear and convergence of a NFW halo of given mass, concentration, 
    redshift, assuming Cosmology. No mass-concentration relation is employed.

    Based on Matthias Bartelmann's libastro.

    The cosmology to use can be set either by providing a Cosmology instance as cosmo,
    or by providing omega_m and/or omega_lam.  
    If only one of the latter is provided, the other is taken to be one minus that.
    If no cosmology parameters are set, a default Cosmology() is constructed.

    @param mass       Mass defined using a spherical overdensity of 200 times the critical density
                      of the universe, in units of M_solar/h.
    @param conc       Concentration parameter, i.e., ratio of virial radius to NFW scale radius.
    @param redshift   Redshift of the halo.
    @param halo_pos   Position of halo center (in arcsec). [default=PositionD(0,0)]
    @param omega_m    Omega_matter to pass to Cosmology constructor [default=None]
    @param omega_lam  Omega_lambda to pass to Cosmology constructor [default=None]
    @param cosmo      A Cosmology instance [default=None]
    """
    _req_params = { 'mass' : float , 'conc' : float , 'redshift' : float }
    _opt_params = { 'halo_pos' : galsim.PositionD , 'omega_m' : float , 'omega_lam' : float }
    _single_params = []

    def __init__(self, mass, conc, redshift, halo_pos=galsim.PositionD(0,0), 
                 omega_m=None, omega_lam=None, cosmo=None):
        if omega_m or omega_lam:
            if cosmo:
                raise TypeError("NFWHalo constructor received both cosmo and omega parameters")
            if not omega_m: omega_m = 1.-omega_lam
            if not omega_lam: omega_lam = 1.-omega_m
            cosmo = Cosmology(omega_m=omega_m, omega_lam=omega_lam)
        elif not cosmo:
            cosmo = Cosmology()
        elif not isinstance(cosmo,Cosmology):
            raise TypeError("Invalid cosmo parameter in NFWHalo constructor")

        # Check if halo_pos is a Position
        if isinstance(halo_pos,galsim.PositionD):
            pass  # This is what it should be
        elif isinstance(halo_pos,galsim.PositionI):
            # Convert to a PositionD
            halo_pos = galsim.PositionD(halo_pos.x, halo_pos.y)
        elif isinstance(halo_pos, tuple) and len(halo_pos) == 2:
            # Convert (x,y) tuple to PositionD
            halo_pos = galsim.PositionD(halo_pos[0], halo_pos[1])
        else:
            raise TypeError("Unable to parse the input halo_pos argument for NFWHalo")

        self.M = float(mass)
        self.c = float(conc)
        self.z = float(redshift)
        self.halo_pos = halo_pos
        self.cosmo = cosmo

        # calculate scale radius
        a = self.cosmo.a(self.z)
        # First we get the virial radius, which is defined for some spherical overdensity as
        # 3 M / [4 pi (r_vir)^3] = overdensity
        # Here we have overdensity = 200 * rhocrit, to determine R200. The factor of 1.63e-5 comes
        # from the following set of prefactors: (3 / (4 pi * 200 * rhocrit))^(1/3)
        # where rhocrit = 2.8e11 h^2 M_solar / Mpc^3.  The mass in the equation below is in
        # M_solar/h, which is how the final units are Mpc/h.
        R200 = 1.63e-5/(1+self.z) * (self.M * self.__omega(a)/self.__omega(1))**0.3333 # in Mpc/h
        self.rs = R200/self.c

        # convert scale radius in arcsec
        dl = self.cosmo.Da(self.z)*3000.; # in Mpc/h
        scale = self.rs / dl
        arcsec2rad = 1./206265;
        self.rs_arcsec = scale/arcsec2rad;

    def __omega(self, a):
        """Matter density at scale factor a
        """
        return self.cosmo.omega_m/(self.cosmo.E(a)**2 * a**3)

    def __farcth (self, x, out=None):
        """Numerical implementation of integral functions of a spherical NFW profile.

        All expressions are a function of x, which is the radius r in units of the NFW scale radius,
        r_s.  For the derivation of these functions, see for example Wright & Brainerd (2000, ApJ,
        534, 34).
        """
        if out is None:
            out = np.zeros_like(x)

        # 3 cases: x > 1, x < 1, and |x-1| < 0.001
        mask = (x < 0.999)
        if mask.any():
            a = ((1.-x[mask])/(x[mask]+1.))**0.5
            out[mask] = 0.5*np.log((1.+a)/(1.-a))/(1-x[mask]**2)**0.5

        mask = (x > 1.001)
        if mask.any():
            a = ((x[mask]-1.)/(x[mask]+1.))**0.5
            out[mask] = np.arctan(a)/(x[mask]**2 - 1)**0.5

        # the approximation below has a maximum fractional error of 2.3e-7
        mask = (x >= 0.999) & (x <= 1.001)
        if mask.any():
            out[mask] = 5./6. - x[mask]/3.

        return out

    def __kappa(self, x, ks, out=None):
        """Calculate convergence of halo.

        @param x   Radial coordinate in units of rs (scale radius of halo), i.e., x=r/rs.
        @param ks  Lensing strength prefactor.
        @param out Numpy array into which results should be placed.
        """
        # convenience: call with single number
        if isinstance(x, np.ndarray) == False:
            return self.__kappa(np.array([x], dtype='float'), np.array([ks], dtype='float'))[0]

        if out is None:
            out = np.zeros_like(x)

        # 3 cases: x > 1, x < 1, and |x-1| < 0.001
        mask = (x < 0.999)
        if mask.any():
            a = ((1 - x[mask])/(x[mask] + 1))**0.5
            out[mask] = 2*ks[mask]/(x[mask]**2 - 1) * \
                (1 - np.log((1 + a)/(1 - a))/(1 - x[mask]**2)**0.5)

        mask = (x > 1.001)
        if mask.any():
            a = ((x[mask] - 1)/(x[mask] + 1))**0.5
            out[mask] = 2*ks[mask]/(x[mask]**2 - 1) * \
                (1 - 2*np.arctan(a)/(x[mask]**2 - 1)**0.5)

        # the approximation below has a maximum fractional error of 7.4e-7
        mask = (x >= 0.999) & (x <= 1.001)
        if mask.any():
            out[mask] = ks[mask]*(22./15. - 0.8*x[mask])

        return out

    def __gamma(self, x, ks, out=None):
        """Calculate tangential shear of halo.

        @param x   Radial coordinate in units of rs (scale radius of halo), i.e., x=r/rs.
        @param ks  Lensing strength prefactor.
        @param out Numpy array into which results should be placed
        """
        # convenience: call with single number
        if isinstance(x, np.ndarray) == False:
            return self.__gamma(np.array([x], dtype='float'), np.array([ks], dtype='float'))[0]
        if out is None:
            out = np.zeros_like(x)

        mask = (x > 0.01)
        if mask.any():
            out[mask] = 4*ks[mask]*(np.log(x[mask]/2) + 2*self.__farcth(x[mask])) * \
                x[mask]**(-2) - self.__kappa(x[mask], ks[mask])

        # the approximation below has a maximum fractional error of 1.1e-7
        mask = (x <= 0.01)
        if mask.any():
            out[mask] = 4*ks[mask]*(0.25 + 0.125 * x[mask]**2 * (3.25 + 3.0*np.log(x[mask]/2)))

        return out

    def __ks(self, z_s):
        """Lensing strength of halo as function of source redshift.
        """
        # critical density and surface density
        rho_c = 2.7722e11
        Sigma_c = 5.5444e14
        # density contrast of halo at redshift z
        a = self.cosmo.a(self.z)
        ez = self.cosmo.E(a)
        d0 = 200./3 * self.c**3/(np.log(1+self.c) - (1.*self.c)/(1+self.c))
        rho_s = rho_c * ez**2 *d0

        # lensing weights: the only thing that depends on z_s
        # this does takes some time...
        dl = self.cosmo.Da(z_s, self.z) * self.cosmo.Da(self.z) / self.cosmo.Da(z_s)
        k_s = dl * self.rs * rho_s / Sigma_c
        return k_s

    def getShear(self, pos, z_s, units=galsim.arcsec, reduced=True):
        """Calculate (reduced) shear of halo at specified positions.

        @param pos       Position(s) of the source(s), assumed to be post-lensing!
                         Valid ways to input this:
                           - Single galsim.PositionD (or PositionI) instance
                           - tuple of floats: (x,y)
                           - list of galsim.PositionD (or PositionI) instances
                           - tuple of lists: ( xlist, ylist )
        @param z_s       Source redshift(s).
        @param units     Angular units of coordinates [default = arcsec]
        @param reduced   Whether returned shear(s) should be reduced shears. [default=True]

        @return (g1,g2)   [g1 and g2 are each a list if input was a list]
        """
        # Convert to numpy arrays for internal usage:
        pos_x, pos_y = _convertPositions(pos, units, 'getShear')

        r = ((pos_x - self.halo_pos.x)**2 + (pos_y - self.halo_pos.y)**2)**0.5/self.rs_arcsec
        # compute strength of lensing fields
        ks = self.__ks(z_s)
        if isinstance(z_s, np.ndarray) == False:
            ks = ks*np.ones_like(r)
        g = self.__gamma(r, ks)

        # convert to observable = reduced shear
        if reduced:
            kappa = self.__kappa(r, ks)
            g /= 1 - kappa

        # pure tangential shear, no cross component
        phi = np.arctan2(pos_y - self.halo_pos.y, pos_x - self.halo_pos.x)
        g1 = -g / (np.cos(2*phi) + np.sin(2*phi)*np.tan(2*phi))
        g2 = g1 * np.tan(2*phi)

        # Convert to a tuple of floats or lists of floats
        if len(g) == 1:
            return g1[0], g2[0]
        else:
            return g1.tolist(), g2.tolist()


    def getConvergence(self, pos, z_s, units=galsim.arcsec):
        """Calculate convergence of halo at specified positions.

        @param pos     Position(s) of the source(s), assumed to be post-lensing!
                       Valid ways to input this:
                         - Single galsim.PositionD (or PositionI) instance
                         - tuple of floats: (x,y)
                         - list of galsim.PositionD (or PositionI) instances
                         - tuple of lists: ( xlist, ylist )
        @param z_s     Source redshift(s)
        @param units   Angular units of coordinates [default = arcsec]

        @return kappa or list of kappa values
        """

        # Convert to numpy arrays for internal usage:
        pos_x, pos_y = _convertPositions(pos, units, 'getKappa')

        r = ((pos_x - self.halo_pos.x)**2 + (pos_y - self.halo_pos.y)**2)**0.5/self.rs_arcsec
        # compute strength of lensing fields
        ks = self.__ks(z_s)
        if isinstance(z_s, np.ndarray) == False:
            ks = ks*np.ones_like(r)
        kappa = self.__kappa(r, ks)

        # Convert to a float or list of floats
        if len(kappa) == 1:
            return kappa[0]
        else:
            return [ k for k in kappa ]

    def getMag(self, pos, z_s, units=galsim.arcsec):
        """Calculate magnification of halo at specified positions.

        @param pos     Position(s) of the source(s), assumed to be post-lensing!
                       Valid ways to input this:
                         - Single galsim.PositionD (or PositionI) instance
                         - tuple of floats: (x,y)
                         - list of galsim.PositionD (or PositionI) instances
                         - tuple of lists: ( xlist, ylist )
        @param z_s     Source redshift(s)
        @param units   Angular units of coordinates (only arcsec implemented so far).
        @return mu     Numpy array containing the magnification at the specified position(s)
        """
        # Convert to numpy arrays for internal usage:
        pos_x, pos_y = _convertPositions(pos, units, 'getMag')

        r = ((pos_x - self.halo_pos.x)**2 + (pos_y - self.halo_pos.y)**2)**0.5/self.rs_arcsec
        # compute strength of lensing fields
        ks = self.__ks(z_s)
        if isinstance(z_s, np.ndarray) == False:
            ks = ks*np.ones_like(r)
        g = self.__gamma(r, ks)
        kappa = self.__kappa(r, ks)

        mu = 1. / ( (1.-kappa)**2 - g**2 )

        # Convert to a float or list of floats
        if len(mu) == 1:
            return mu[0]
        else:
            return [ m for m in mu ]
