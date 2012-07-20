"""\file lensing.py The "lensing engine" for drawing random shears from some power spectrum.
"""
import galsim
import numpy as np

ISQRT2 = np.sqrt(1.0/2.0)

## TODO later: get convergences that are consistent with this shear field
class ShearField(object):
    """@brief Class to represent a lensing shear field

    A ShearField represents a random Gaussian realization of some (flat-sky) shear power spectrum,
    at arbitary positions (not just on a grid).  This class is originally initialized with a list of
    positions for which we would like to generate g1 and g2 values.   It uses a
    PowerSpectrumRealizer to generate shears on an appropriately-spaced grid, and then interpolates
    on that grid to the requested positions.  Finally, it carries around some information about the
    underlying shear power spectrum used to generate the field.
    """
    def __init__(self, ra, dec, E_power_function=None, B_power_function=None, units=None):
        """@brief Create a ShearField object for a list of positions

        We can optionally set the power spectra that will be used for E and B modes now, or this can
        be done later using set_p_E and set_p_B.

        @param[in] ra List of right ascensions (units should be consistent with P(k) function)
        @param[in] dec List of declinations (units should be consistent with P(k) function)
        @param[in] E_power_function A function or other callable that can take an array of k values
        and return a power.  It should cope happily with k=0.  The function should return the power
        spectrum desired in the E (gradient) mode of the image
        @param[in] B_power_function A function or other callable that can take an array of k values
        and return a power.  It should cope happily with k=0.  The function should return the power
        spectrum desired in the B (curl) mode of the image
        """
        self.ra = ra
        self.dec = dec
        if E_power_function is not None:
            self.p_E = E_power_function
        if B_power_function is not None:
            self.p_B = B_power_function
        if units is not None:
            self.units = units

    def set_power_functions(self, E_power_function=None, B_power_function=None):
        """@brief Set / change the functions that compute the E and B mode power spectra.

        @param[in] E_power_function A function or other callable that accepts a 2D numpy grid of |k| and returns
        the E-mode power spectrum of the same shape.  Set to None for there to be no E-mode power
        @param[in] B_power_function A function or other callable that accepts a 2D numpy grid of |k| and returns
        the B-mode power spectrum of the same shape.  Set to None for there to be no B-mode power
        """
        self.p_E = E_power_function
        self.p_B = B_power_function

    def __call__(self, E_power_function=None, B_power_function=None, gaussian_deviate=None,
                 seed=None, interpolantxy=None, psrealizer=None, new_realization=False):
        """@brief Generate a realization of the current power spectrum at the specified positions.

        Generate a Gaussian random realization of some specified shear power spectrum (E and B
        mode), given the (ra, dec) positions.  This code stores information about the quantities
        used to generate the random shear field, generates shears using a PowerSpectrumRealizer on a
        grid, and interpolates to get g1 and g2 at the specified positions.

        @param[in] E_power_function A function or other callable that can take an array of k values
        and return a power.  It should cope happily with k=0.  The function should return the power
        spectrum desired in the E (gradient) mode of the image
        @param[in] B_power_function A function or other callable that can take an array of k values
        and return a power.  It should cope happily with k=0.  The function should return the power
        spectrum desired in the B (curl) mode of the image
        @param[in] gaussian_deviate A galsim.GaussianDeviate object for drawing the random numbers
        @param[in] seed A seed to use when initializing a new galsim.GaussianDeviate for drawing the
        random numbers
        @param[in] interpolantxy (Optional) Interpolant to use for interpolating the shears on a
        grid to the requested positions [default =galsim.InterpolantXY(galsim.Linear())].
        @param[in] psrealizer (Optional) A PowerSpectrumRealizer object to use, rather than
        generating a new one given E_power_function/B_power_function
        @param[in] new_realization If True, then the ShearField should already contain a
        PowerSpectrumRealizer in psrealizer that can be used to generate another set of random
        shears
        """

        # check input values
        if gaussian_deviate is None:
            if seed is None:
                gd = galsim.GaussianDeviate()
            else:
                gd = galsim.GaussianDeviate(seed)
                self.seed = seed
        else:
            if seed:
                raise ValueError("Cannot provide both a Gaussian deviate and a random seed!")
            if isinstance(gaussian_deviate, galsim.GaussianDeviate) == False:
                raise TypeError("The requested gaussian_deviate is not a Gaussian deviate!")
        if interpolantxy is None:
            interpolantxy = galsim.InterpolantXY(galsim.Linear())
        if E_power_function is not None:
            if psrealizer is not None and new_realization is False:
                self.p_E = E_power_function
        if B_power_function is not None:
            if psrealizer is not None and new_realization is False:
                self.p_B = B_power_function

        # store some more information
        self.interpolantxy = interpolantxy

        # generate shears on a grid: choose set of input parameters for PowerSpectrumRealizer
        ## get total range in RA, dec
        tot_dra  = np.max(self.ra)  - np.min(self.ra)
        tot_ddec = np.max(self.dec) - np.min(self.dec)
        ## TODO: choose an appropriate delta(ra) and delta(dec) which results in setting pixel_size

        ## TODO: find grid size to cover the whole range at that resolution; or perhaps we should
        ## cover a wider range to allow for large-scale modes?

        if new_realization is False:
            if psrealizer is None:
                ### make the PowerSpectrumRealizer, and store
                psr = PowerSpectrumRealizer(nx, ny, pixel_size, self.p_E, self.p_B)
                self.psrealizer = psr
            else:
                self.psrealizer = psrealizer
            ### make a single realization
            g1_grid, g2_grid = self.psrealizer(gaussian_deviate=gd)
        else:
            g1_grid, g2_grid = self.psrealizer(gaussian_deviate=gd)

        # make the gridded shears from a numpy array into an Image
        g1_grid_img = galsim.ImageViewD(np.ascontiguousarray(g1_grid.astype(np.float64)))
        g2_grid_img = galsim.ImageViewD(np.ascontiguousarray(g2_grid.astype(np.float64)))

        # make the Image into an SBInterpolatedImage
        g1_sbimg = galsim.SBInterpolatedImage(g1_grid_img, xInterp = interpolantxy, dx = pixel_size)
        g2_sbimg = galsim.SBInterpolatedImage(g2_grid_img, xInterp = interpolantxy, dx = pixel_size)

        # interpolate from the grid points to the desired RA, dec values
        # TODO: watch out for constant shift between ra/dec values and image bounds
        # TODO: figure out how to do this for a vector all at once

class PowerSpectrumRealizer(object):
    """@brief Class for generating realizations of power spectra with any area and pixel size.
    
    Designed to quickly generate many realizations of the same shear power spectra on a grid.
    """
    def __init__(self, nx, ny, pixel_size, E_power_function, B_power_function):
        """@brief Create a Realizer object from image dimensions and power spectrum functions
        
        The initializer sets up the grids in k-space and computes the power on them.
        It also computes spin weighting terms.  You can alter any of the setup properties later.
        
        @param[in] nx The x-dimension of the desired image
        @param[in] ny The y-dimension of the desired image
        @param[in] pixel_size The size of the pixel sides, in units consistent with the units
        expected by the power spectrum functions
        @param[in] E_power_function A function or other callable that can take an array of k values
        and return a power.  It should cope happily with k=0.  The function should return the power
        spectrum desired in the E (gradient) mode of the image
        @param[in] B_power_function A function or other callable that can take an array of k values
        and return a power.  It should cope happily with k=0.  The function should return the power
        spectrum desired in the B (curl) mode of the image
        """
        self.set_size(nx, ny, pixel_size, False)
        self.set_power(E_power_function, B_power_function)
        
    def set_size(self, nx, ny, size, remake_power=True):
        """@brief Change the size of the array you want to simulate.
        
        @param[in] nx The x-dimension of the desired image
        @param[in] ny The y-dimension of the desired image
        @param[in] pixel_size The size of the pixel sides, in units consistent with the units
        expected by the power spectrum functions
        @param[in] remake_power Whether to re-build the power spectra on the new grids.  Set this to
        False if you are about to change the power spectrum functions too
        
        """
        # Set up the k grids in x and y, and the instance variables
        self.nx = nx
        self.ny = ny
        kx, ky=np.mgrid[0:nx/2+1,0:ny/2+1]
        self.kx = kx
        self.ky = ky
        size = float(size)

        # Set up the scalar |k| grid.
        self.k=((kx/(size*nx))**2+(ky/(size*ny))**2)**0.5
        
        #Compute the spin weightings
        self._cos, self._sin = self._generate_spin_weightings()
        
        #Optionally (because this may be the first time we run this, or we may be about to change
        #these functions), re-build the power grids for the new sizes
        if remake_power: self.set_power(self.p_E, self.p_B)
        
    def set_power(self, p_E, p_B):
        """@brief Change the functions that compute the E and B mode power spectra.
        
        This function re-generates the grids that the power spectrum is computed over.
        
        @param[in] p_E A function or other callable that accepts a 2D numpy grid of |k| and returns
        the E-mode power spectrum of the same shape.  Set to None for there to be no E-mode power
        @param[in] p_B A function or other callable that accepts a 2D numpy grid of |k| and returns
        the B-mode power spectrum of the same shape.  Set to None for there to be no B-mode power
        
        """
        self.p_E = p_E
        self.p_B = p_B
        if p_E is None:  self.amplitude_E = None
        else:            self.amplitude_E = np.sqrt(self._generate_power_array(p_E))
        
        if p_B is None:  self.amplitude_B = None
        else:            self.amplitude_B = np.sqrt(self._generate_power_array(p_B))


    def __call__(self, new_power=False, gaussian_deviate=None, seed=None):
        """@brief Generate a realization of the current power spectrum.
        
        @param[in] new_power If the power-spectrum functions that you specified are not
        deterministic then you can set this value to True to call them again to get new values.
        For example, you could include a cosmic variance term in your power spectrum and get a new
        spectrum realization each time.
        @param[in] gaussian_deviate (Optional) gaussian deviate to use when generating the shear
        fields
        @param[in] seed (Optional) random seed to use when generating the Gaussian random shear fields
        
        @return g1,g2 Two image arrays for the two shear components g_1 and g_2
        """
        #If desired, recompute power spectra
        if new_power:
            self.set_power(self.p_E, self.p_B)
        
        if gaussian_deviate is None:
            if seed is None:
                gd = galsim.GaussianDeviate()
            else:
                gd = galsim.GaussianDeviate(seed)
        else:
            if seed:
                raise ValueError("Cannot provide both a Gaussian deviate and a random seed!")

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
        g1=np.fft.irfft2(g1_k)
        g2=np.fft.irfft2(g2_k)
        
        return g1, g2

    def _generate_power_array(self, power_function):
        #Internal function to generate the result of a power function evaluated on a grid,
        #taking into account the symmetries.
        power_array = np.zeros((self.nx, self.ny/2+1))
        kx = self.kx
        ky = self.ky
        k  = self.k
        P_k = power_function(k)
        power_array[ kx, ky] = P_k
        power_array[-kx, ky] = P_k
        return power_array
    
    def _generate_spin_weightings(self):
        #Internal function to generate the cosine and sine spin weightings for the current array set up
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
