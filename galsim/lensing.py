"""\file lensing.py The "lensing engine" for drawing shears from some power spectrum or a NFW halo.
"""
import galsim
import numpy as np

# A helper function for parsing the input position arguments for PowerSpectrum and NFWHalo:
def _convertPositions(pos, func):
    """Convert pos from the valid ways to input positions to two numpy arrays

       This is used by the functions getShear, getConvergence, and getMag for both 
       PowerSpectrum and NFWHalo (the former only has getShear currently).
    """
    try:
        # Check for PositionD or PositionI:
        if isinstance(pos,galsim.PositionD) or isinstance(pos,galsim.PositionI):
            return ( np.array([pos.x], dtype='float'),
                        np.array([pos.y], dtype='float') )

        # Check for list of PositionD or PositionI:
        # The only other options allow pos[0], so if this is invalid, an exception 
        # will be raised and appropriately dealt with:
        elif isinstance(pos[0],galsim.PositionD) or isinstance(pos[0],galsim.PositionI):
            return ( np.array([p.x for p in pos], dtype='float'),
                        np.array([p.y for p in pos], dtype='float') )

        # Now pos must be a tuple of length 2
        elif len(pos) != 2:
            raise TypeError() # This will be caught below and raised with a better error msg.

        # Check for (x,y):
        elif isinstance(pos[0],float):
            return ( np.array([pos[0]], dtype='float'),
                        np.array([pos[1]], dtype='float') )

        # Only other valid option is ( xlist , ylist )
        else:
            return ( np.array(pos[0], dtype='float'),
                        np.array(pos[1], dtype='float') )

    except:
        raise TypeError("Unable to parse the input pos argument for %s."%func)
            

class PowerSpectrum(object):
    """Class to represent a lensing shear field according to some power spectrum P(k)

    A PowerSpectrum represents some (flat-sky) shear power spectrum, either for gridded points or at
    arbitary positions.  This class is originally initialized with a power spectrum from which we
    would like to generate g1 and g2 values.  When the getShear() method is called, it uses a
    PowerSpectrumRealizer to generate shears on an appropriately-spaced grid, and if necessary,
    interpolates on that grid to the requested positions.  Finally, it carries around some
    information about the underlying shear power spectrum used to generate the field.

    When creating a PowerSpectrum instance, the E and B mode power spectra can optionally be set at
    initialization or later on with the method set_power_functions.  Note that the power spectra
    should be a function of k.  The typical thing is to just use a lambda function in Python (i.e.,
    a function that is not associated with a name); for example, to define P(k)=k^2, one would use
    `lambda k : k**2`.  But they can also be more complicated user-defined functions that take a
    single argument k and return the power at that k value, or they can be instances of the
    TabulatedPk class for power spectra that are known at particular k values but for which there is
    not a simple analytic form.  The power function should return power P(k), not Delta^2(k) = k^2
    P(k) / 2pi; the TabulatedPk class can be instructed to convert between the two automatically.
    We assume that P(k) goes to zero at k=0, as in any physically reasonable cosmological model.
    The power functions must return a list/array that is the same size as what it was given, e.g.,
    in the case of no power or constant power, a function that just returns a float would not be
    permitted; it would have to return an array of floats all with the same value.

    It is important to note that the power spectrum used to initialize the PowerSpectrum object
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

    If the power spectrum used for this calculation comes from a standard cosmology calculator that
    uses units of inverse radians for the wavenumber, then it is important to convert such that the
    units are consistent with our choice of inverse arcsec.  If reading in a tabulated P(k) from one
    of those calculators into a TabulatedPk object, it is possible to automatically have GalSim
    convert the units for you; this functionality is not currently available for other types of
    power functions.

    @param e_power_function A function or other callable that accepts a Numpy array of |k| values,
                            and returns the E-mode power spectrum P_E(|k|) in an array of the same
                            shape.  The function should return the power spectrum desired in the E
                            (gradient) mode of the image.  Set to None (default) for there to be no
                            E-mode power.
                            It may also be a string that can be converted to a function using
                            eval('lambda k : ' + e_power_function), or a tabulated P(k) represented
                            using a TabulatedPk object.
    @param b_power_function A function or other callable that accepts a Numpy array of |k| values,
                            and returns the B-mode power spectrum P_B(|k|) in an array of the same
                            shape.  The function should return the power spectrum desired in the B
                            (curl) mode of the image.  Set to None (default) for there to be no
                            B-mode power.
                            It may also be a string that can be converted to a function using
                            eval('lambda k : ' + b_power_function), or a tabulated P(k) represented
                            using a TabulatedPk object.
    @param units            The angular units used for the power spectrum (i.e. the units of 
                            k^-1).  Currently only arcsec is implemented.
    """
    _req_params = {}
    _opt_params = { 'e_power_function' : str, 'b_power_function' : str }
    _single_params = []
    def __init__(self, e_power_function=None, b_power_function=None, units=galsim.arcsec):
        # Check that the power functions are valid:
        for pf_str in [ 'e_power_function', 'b_power_function' ]:
            pf = eval(pf_str)
            if pf is not None:
                if isinstance(pf,str):
                    try : 
                        pf = eval('lambda k : ' + pf)
                    except :
                        raise AttributeError(
                            "Unable to turn %s = %s into a valid function"%(pf_str,pf))
                # Only try tests below if it's not a TabulatedPk.
                # (If it's a TabulatedPk, then it could be a valid function that isn't defined at
                # k=1, and by definition it must return something that is the same length as the
                # input.)
                if not isinstance(pf, TabulatedPk):
                    try:
                        f1 = pf(1.)
                    except:
                        raise AttributeError("%s is not a valid function"%pf_str)
                    fake_arr = np.zeros(2)
                    fake_p = pf(fake_arr)
                    if isinstance(fake_p, float):
                        raise AttributeError("Power function MUST return a list/array same length as input")

        # Check that at least one is not None
        if e_power_function is None and b_power_function is None:
            raise AttributeError(
                "At least one of e_power_function or b_power_function must be provided.")
                
        self.p_E = e_power_function
        self.p_B = b_power_function
        if units is not galsim.arcsec:
            raise ValueError("Currently we require units of arcsec for the inverse wavenumber!")

    def set_power_functions(self, e_power_function=None, b_power_function=None,
                            units=galsim.arcsec):
        """Set / change the functions that compute the E and B mode power spectra.

        @param e_power_function See description of this parameter in the documentation for the
                                PowerSpectrum class.
        @param b_power_function See description of this parameter in the documentation for the
                                PowerSpectrum class.
        @param units            See description of this parameter in the documentation for the
                                PowerSpectrum class.
        """
        # Check that the power functions are valid:
        for pf_str in [ 'e_power_function', 'b_power_function' ]:
            pf = eval(pf_str)
            if pf is not None:
                if isinstance(pf,str):
                    try : 
                        pf = eval('lambda k : ' + pf)
                    except :
                        raise AttributeError(
                            "Unable to turn %s = %s into a valid function"%(pf_str,pf))
                # Only try tests below if it's not a TabulatedPk.
                # (If it's a TabulatedPk, then it could be a valid function that isn't defined at
                # k=1, and by definition it must return something that is the same length as the
                # input.)
                if not isinstance(pf, TabulatedPk):
                    try:
                        f1 = pf(1.)
                    except:
                        raise AttributeError("%s is not a valid function"%pf_str)
                    fake_arr = np.zeros(2)
                    fake_p = pf(fake_arr)
                    if isinstance(fake_p, float):
                        raise AttributeError("Power function MUST return a list/array same length as input")

        self.p_E = e_power_function
        self.p_B = b_power_function
        if units is not galsim.arcsec:
            raise ValueError("Currently we require units of arcsec for the inverse wavenumber!")

    def getShear(self, pos=None, grid_spacing=None, ngrid=None, rng=None,
                 interpolant=None, center=galsim.PositionD(0,0)):
        """Generate a realization of the current power spectrum at the specified positions.

        This function currently does two relatively separate things.  The plan is to split it into
        two functions, but we haven't done so yet.
        
        First, it will generate a Gaussian random realization of the specified E and B mode shear
        power spectrum at a grid of positions, specified by the input parameters `grid_spacing` 
        (distance between grid points) and `ngrid` (number of grid points in each direction.)

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

        Second, this function can interpolate between the grid points to find the shear values for a
        given list of input positions (or just a single position).  This can be done in conjunction
        with the first functionality, in which case the grid will be computed using the `grid_*`
        parameters and then that new grid will be used to interpolate the shear values.  Or you can
        omit the `grid_*` parameters, in which case the function will use the most recently computed
        grid from a previous call.  Currently, if you try to interpolate a grid without having
        previously called `getShear` with the `grid_*` parameters, then an exception will be raised.
        A future version of the code will allow the estimation of shears on non-gridded points by
        first automatically choosing a grid spacing on which to estimate gridded shears before
        interpolating, but this functionality is not implemented yet.

        Some examples of how to use getShear:

        1. Create a grid of points separated by 1 arcsec:

               my_ps = galsim.PowerSpectrum(lambda k : k**2)
               g1, g2 = my_ps.getShear(grid_spacing = 1., ngrid = 100)

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

        2. Same thing, but use a particular rng and set the location of the center of the grid
           to be something other than the default (0,0)

               g1, g2 = my_ps.getShear(grid_spacing = 8., ngrid = 65,
                                       rng = galsim.BaseDeviate(1413231),
                                       center = (256.5, 256.5) )

        3. Use the previously created grid to get the shear for a particular point:

               g1, g2 = my_ps.getShear(pos = galsim.PositionD(12, 412))

           This time the returned values are just floats and correspond to the shear for the
           provided position.

        4. You can also provide a position as a tuple to save the explicit PositionD construction:

               g1, g2 = my_ps.getShear(pos = (12, 412))

        5. Get the shears for a bunch of points at once:
        
               xlist = [ 141, 313,  12, 241, 342 ]
               ylist = [  75, 199, 306, 225, 489 ]
               poslist = [ galsim.PositionD(xlist[i],ylist[i]) for i in range(len(xlist)) ]
               g1, g2 = my_ps.getShear( poslist )
               g1, g2 = my_ps.getShear( (xlist, ylist) )

           Both calls do the same thing.  The returned g1, g2 this time are lists of g1, g2 values.
           The lists are the same length as the number of input positions.

        6. Make a PowerSpectrum from a tabulated P(k) that gets interpolated to find the power at
        all necessary values of k, then generate shears on a grid.  Assuming that k and P_k are
        either lists, tuples, or 1d Numpy arrays containing k and P(k):

               tab_pk = galsim.lensing.TabulatedPk(k, P_k)
               my_ps = galsim.PowerSpectrum(tab_pk)
               g1, g2 = my_ps.getShear(grid_spacing = 1., grid_nx = 100)

        @param pos              Position(s) of the source(s), assumed to be post-lensing!  (It is 
                                up to the user to check that the units are consistent with those in 
                                the P(k) function, just as for the grid_spacing keyword.)
                                Valid ways to input this:
                                  - Single galsim.PositionD (or PositionI) instance
                                  - tuple of floats: (x,y)
                                  - list of galsim.PositionD (or PositionI) instances
                                  - tuple of lists: ( xlist, ylist )
                                pos may also be None to just build a gridded array of (g1,g2).
        @param grid_spacing     Spacing for an evenly spaced grid of points, in arcsec for
                                consistency with the natural length scale of images created using
                                the draw or drawShoot methods.
        @param ngrid            Number of grid points in each dimension.  If a number that is not
                                an int (e.g., a float) is supplied, then it gets converted to an int
                                automatically.
        @param rng              (Optional) A galsim.GaussianDeviate object for drawing the random
                                numbers.  (Alternatively, any BaseDeviate can be used.)
        @param interpolant      (Optional) Interpolant to use for interpolating the shears on a grid
                                to the requested positions.
                                This is highly recommended to be Linear (which will become
                                bi-linear, since it is in 2 dimensions).  Using other interpolants
                                is likely to be inaccurate, though on small enough scales even the
                                linear interpolant will be problematic.  A future version of the
                                code will quantify this inaccuracy due to the interpolant in greater
                                detail. [default = galsim.Linear()]
        @param center           (Optional) If setting up a new grid, define what position you
                                want to consider the center of that grid. [default = (0,0)]
        
        @return g1,g2           If given a single position: the two shear components g_1 and g_2.
                                If given a list of positions: each is a python list of values.
                                If pos=None, these are 2-d NumPy arrays.
        """
        # This used to be part of the doc string.  It was moved here for now, since this 
        # functionality isn't implemented yet.
        #
        # When using a non-gridded set of points, the code has to choose an appropriate spacing for
        # a grid and then interpolate the gridded shears to the specified set of points.  It does
        # this by requiring that the modification to the power spectrum due to a (bi)linear 
        # interpolant should not be significant at the minimum separation between points.  The user
        # should be aware that use of an interpolant that is not linear does not change how this 
        # calculation is done, and therefore it is necessary to test the fidelity of the recovered 
        # power spectrum for any errors due to the chosen non-linear interpolant.

        # Convert to numpy arrays for internal usage:
        if pos is not None:
            pos_x, pos_y = _convertPositions(pos, 'getShear')
            if grid_spacing is None and not hasattr(self,'im_g1'):
                raise AttributeError(
                    "Calling PowerSpectrum.getShear without grid parameters, and " +
                    "no grid previously set up.")


        # Check problem cases for regular grid of points
        if grid_spacing is not None or ngrid is not None:
            if grid_spacing is None or ngrid is None:
                raise ValueError("When specifying grid, we require both a spacing and a size!")
            # Check for non-integer ngrid
            if not isinstance(ngrid, int) and not isinstance(ngrid, long):
                if isinstance(ngrid, float):
                    ngrid = int(ngrid)
                else:
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

        # Make sure that we've specified some power spectrum
        if self.p_E is None and self.p_B is None:
            raise ValueError("Cannot generate shears when no E or B mode power spectrum are given!")

        # Make a GaussianDeviate if necessary
        if rng is None:
            gd = galsim.GaussianDeviate()
        elif isinstance(rng, galsim.GaussianDeviate):
            gd = rng
        elif isinstance(rng, galsim.BaseDeviate):
            gd = galsim.GaussianDeviate(rng)
        else:
            raise TypeError("The rng provided to getShear is not a BaseDeviate")

        # Set default interpolant if none given
        if interpolant is None:
            interpolantxy = galsim.InterpolantXY(galsim.Linear())
        elif isinstance(interpolant, galsim.Interpolant):
            interpolantxy = galsim.InterpolantXY(interpolant)
        elif isinstance(interpolant, galsim.InterpolantXY):
            interpolantxy = interpolant
        else:
            raise TypeError("Invalid interpolant provided to PowerSpectrum.getShear")

        # Build the grid if requested.
        if grid_spacing is not None:
            psr = PowerSpectrumRealizer(ngrid, ngrid, grid_spacing, self.p_E, self.p_B)
            self.grid_g1, self.grid_g2 = psr(gd)
            
            # Setup interpolated images
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

        if pos is None:
            return self.grid_g1, self.grid_g2
        else:
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
                    sbii_g1 = galsim.SBInterpolatedImage(self.im_g1, xInterp = interpolantxy)
                    sbii_g2 = galsim.SBInterpolatedImage(self.im_g2, xInterp = interpolantxy)
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
    def __init__(self, nx, ny, pixel_size, e_power_function, b_power_function):
        self.set_size(nx, ny, pixel_size, False)
        self.set_power(e_power_function, b_power_function)
        
    def set_size(self, nx, ny, pixel_size, remake_power=True):
        """Change the size of the array you want to simulate.
        
        @param nx           The x-dimension of the desired image
        @param ny           The y-dimension of the desired image
        @param pixel_size   The size of the pixel sides, in units consistent with the units
                            expected by the power spectrum functions.
        @param remake_power Whether to re-build the power spectra on the new grids.  Set this to
                            False if you are about to change the power spectrum functions too.
        
        """
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
        
        #Optionally (because this may be the first time we run this, or we may be about to change
        #these functions), re-build the power grids for the new sizes
        if remake_power: self.set_power(self.p_E, self.p_B)
        
    def set_power(self, p_E, p_B):
        """Change the functions that compute the E and B mode power spectra.
        
        This function re-generates the grids that the power spectrum is computed over.
        
        @param p_E See description of the e_power_function parameter in the documentation for the
                   PowerSpectrum class.
        @param p_B See description of the b_power_function parameter in the documentation for the
                   PowerSpectrum class.
        """
        # Convert from strings if necessary
        if p_E is not None and isinstance(p_E,str):
            p_E = eval('lambda k : ' + p_E)
        if p_B is not None and isinstance(p_B,str):
            p_B = eval('lambda k : ' + p_B)

        self.p_E = p_E
        self.p_B = p_B
        if p_E is None:  self.amplitude_E = None
        else:            self.amplitude_E = np.sqrt(self._generate_power_array(p_E))
        
        if p_B is None:  self.amplitude_B = None
        else:            self.amplitude_B = np.sqrt(self._generate_power_array(p_B))


    def __call__(self, gd, new_power=False):
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

        #If desired, recompute power spectra
        if new_power:
            self.set_power(self.p_E, self.p_B)
        
        if not isinstance(gd, galsim.GaussianDeviate):
            raise TypeError("The gd provided to the PowerSpectrumRealizer is not a GaussianDeviate!")

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
        # raise a clear exception for TabulatedPk that are not defined on the full k range!
        if isinstance(power_function, TabulatedPk):
            mink = np.min(fake_k)
            maxk = np.max(fake_k)
            if mink < power_function.k_min or maxk > power_function.k_max:
                print power_function.k_min, power_function.k_max
                raise ValueError("Tabulated P(k) is not defined for full k range on grid, %f<k<%f"%(mink,maxk))
        P_k = power_function(fake_k)
        # now fix the k=0 value of power to zero
        if type(P_k) is np.ndarray:
            P_k[0,0] = type(P_k[0,1])(0.)
        else:
            P_k = 0.
        power_array[ self.kx, self.ky] = P_k
        power_array[-self.kx, self.ky] = P_k
        if np.any(power_array < 0):
            raise ValueError("Negative power required for some values of k!")
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
        @param units     Angular units of coordinates (only arcsec implemented so far).
        @param reduced   Whether returned shear(s) should be reduced shears. (default=True)

        @return (g1,g2)   [g1 and g2 are each a list if input was a list]
        """
        if units != galsim.arcsec:
            raise NotImplementedError("Only arcsec units implemented!")

        # Convert to numpy arrays for internal usage:
        pos_x, pos_y = _convertPositions(pos, 'getShear')

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
        @param units   Angular units of coordinates (only arcsec implemented so far).

        @return kappa or list of kappa values
        """
        if units != galsim.arcsec:
            raise NotImplementedError("Only arcsec units implemented!")

        # Convert to numpy arrays for internal usage:
        pos_x, pos_y = _convertPositions(pos, 'getKappa')

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
        if units != galsim.arcsec:
            raise NotImplementedError("Only arcsec units implemented!")

        # Convert to numpy arrays for internal usage:
        pos_x, pos_y = _convertPositions(pos, 'getMag')

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

class TabulatedPk(object):
    """A class for storing a tabulated lensing power spectrum, for use by the lensing engine.

    The TabulatedPk class uses the galsim LookupTable functionality to take some input P(k) that is
    known at particular values of k, and interpolate it to other values of k.  The accuracy of the
    interpolation and the effective shear power spectrum will in part depend on the user ensuring
    that the P(k) is adequately sampled for the range of P(k) values of interest; very poor sampling
    could result in the interpolation being quite inaccurate.

    The user must supply the arrays for k and P, or the name of an ascii file containing columns for
    k and P.  The TabulatedPk object will store some basic information like the minimum and maximum
    k, and the number of k values used for interpolation.

    The user can opt to interpolate in log(k) and/or log(P), though this is not the default.  It may
    be a wise choice depending on the particular function, e.g., for a nearly power-law P(k) (or at
    least one that is locally power-law-ish for much of the k range) then it might be a good idea to
    interpolate in log(k) and log(P), using a linear interpolant.

    The k can in principle have any of the supported galsim AngleUnits (radians, degrees, hours,
    arcmin, arcsec), however since PowerSpectrum works only in terms of arcsec, at least for now,
    the input k will be automatically converted to 1/arcsec.  This means that when calling the
    object to get power at some k, the input units of k MUST be 1/arcsec.

    @param k             The list, tuple, or 1d Numpy array of k values (floats, doubles, or ints,
                         which get silently converted to floats for the purpose of interpolation).
    @param power         The list, tuple, or 1d Numpy array of P(k) values (floats, doubles, or ints,
                         which get silently converted to floats for the purpose of interpolation).
    @param delta2        Is the power actually given as Delta^2, which requires us to multiply by
                         k^2 / (2pi) to get the shear power P(k)?  [Default: False]
                         Note that if Delta^2 is provided, then it is critical that it have units
                         consistent with the units for k.
    @param interpolant   The interpolant to use, with the options being 'spline', 'linear', 'ceil',
                         and 'floor' [Default: 'spline'].
    @param k_log         Set to True if you wish to interpolate using log(k) rather than k.  Note
                         that all inputs / outputs will still be k, it's just a question of how the
                         interpolation is done. [Default: False]
    @param p_log         Set to True if you wish to interpolate using log(P) rather than P.  Note
                         that all inputs / outputs will still be P, it's just a question of how the
                         interpolation is done. [Default: False]
    @param units         The angular units used for the power spectrum (i.e. the units of k^-1).
                         Since the PowerSpectrum class currently requires units of arcsec, for now
                         we use the input value of `units` to convert k automatically to units of
                         inverse arcsec before tabulating and storing k and P(k).

    """
    def __init__(self, k = None, power = None, file = None, delta2 = False, interpolant = None,
                 k_log = False, p_log = False, units = galsim.arcsec):
        self.k_log = k_log
        self.p_log = p_log

        # read in from file if a filename was specified
        if file:
            data = np.loadtxt(file).transpose()
            k=data[0]
            power=data[1]

        # turn k and power into numpy arrays so that all subsequent math is possible (unlike for
        # lists, tuples).
        if not isinstance(k, np.ndarray):
            k = np.array(k)
        if not isinstance(power, np.ndarray):
            power = np.array(power)

        # first thing: if we actually have Delta^2, then we must convert to power, which is
        # dimensionless.
        if delta2:
            power = (k**2)*power/(2.*np.pi)

        # sanity check units
        if not isinstance(units, galsim.AngleUnit):
            raise ValueError("Input unit is not a galsim.AngleUnit!")
        if units is not galsim.arcsec:
            # Convert to galsim.arcsec for now; only k requires changing, since P is dimensionless.
            k = np.array(k)*(1.*galsim.arcsec)/units
        self.units = galsim.arcsec

        # check for proper interpolant
        if interpolant is None:
            interpolant = 'spline'
        else:
            if interpolant not in ['spline', 'linear', 'ceil', 'floor']:
                raise ValueError("Unknown interpolant: %s" % interpolant)

        # store some information that will be useful later
        self.k_min = min(k)
        self.k_max = max(k)
        self.n_k = len(k)

        # make and store table
        if k_log:
            if np.any(np.array(k) <= 0.):
                raise ValueError("Cannot interpolate in log(k) when table contains k<=0!")
            k = np.log(k)
        if p_log:
            if np.any(np.array(power) <= 0.):
                raise ValueError("Cannot interpolate in log(P) when table contains P<=0!")
            power = np.log(power)
        self.table = galsim.LookupTable(k, power, interpolant)

    def __call__(self, k):
        """Interpolate the TabulatedPk to get power at some k value(s).

        When the TabulatedPk object is called with a single argument, it returns the power at that
        argument.  An exception will be thrown automatically by the LookupTable class if the k value
        is outside the range of the original tabulated values.  The value that is returned is the
        same type as that provided as an argument, e.g., if a single value k is provided then a
        single value of P is returned; if a tuple of k values is provided then a tuple of P values
        is returned; and so on.  Even if interpolation was done using the `k_log` option, the user
        should still provide k rather than log(k).

        The input k should be in the same units as the TabulatedPk object; currently only
        1/galsim.arcsec are supported units.  (The TabulatedPk constructor can take other units and
        automatically convert to 1/arcsec.)

        @param k       The k value(s) for which the power should be calculated via interpolation on
                       the original (k, P) lookup table.  k can be a single float/double, or a
                       tuple, list, or arbitrarily shaped Numpy array.  It should have units of
                       inverse arcsec.
        @returns The power at the specified k value(s).
        """
        # first, keep track of whether interpolation was done in k or log(k)
        if self.k_log:
            k = np.log(k)

        # figure out what we received, and return the same thing
        # option 1: a Numpy array
        if isinstance(k, np.ndarray):
            dimen = len(k.shape)
            if dimen > 2:
                raise ValueError("Arrays with dimension larger than 2 not allowed!")
            elif dimen == 2:
                p = np.zeros_like(k)
                for i in xrange(k.shape[1]):
                    p[i,:] = np.fromiter((self.table(float(q)) for q in k[i,:]), dtype='float')
            else:
                p = np.fromiter((self.table(float(q)) for q in k), dtype='float')
            if self.p_log:
                p = np.exp(p)
            return p
        # option 2: a tuple
        elif isinstance(k, tuple):
            p = []
            for q in k:
                p.append(self.table(q))
            if self.p_log:
                p = np.exp(p)
            return tuple(p)
        # option 3: a list
        elif isinstance(k, list):
            p = []
            for q in k:
                p.append(self.table(q))
            if self.p_log:
                p = np.exp(p)
            return p
        # option 4: a single value
        else:
            # interpolate on table
            if self.p_log:
                return np.exp(self.table(k))
            else:
                return self.table(k)
