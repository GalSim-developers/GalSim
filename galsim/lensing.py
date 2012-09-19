"""\file lensing.py The "lensing engine" for drawing shears from some power spectrum or a NFW halo.
"""
import galsim
import numpy as np

class PowerSpectrum(object):
    """@brief Class to represent a lensing shear field according to some power spectrum P(k)

    A PowerSpectrum represents some (flat-sky) shear power spectrum, either for gridded points or at
    arbitary positions.  This class is originally initialized with a power spectrum from which we
    would like to generate g1 and g2 values.  When the getShear() method is called, it uses a
    PowerSpectrumRealizer to generate shears on an appropriately-spaced grid, and if necessary,
    interpolates on that grid to the requested positions.  Finally, it carries around some
    information about the underlying shear power spectrum used to generate the field.

    It is important to note that the power spectrum used to initialize the PowerSpectrum object
    should be in the same units as any parameters to the getShear() method that define the locations
    at which we want to get shears.  When we actually draw images, there is a natural scale that
    defines the pitch of the image (dx), which is typically taken to be arcsec.  This definition of
    a specific length scale means that we should also use the same units (arcsec) for the positions
    at which we want our galaxies to be located when we draw shears from a power spectrum, and
    likewise the values of k (wavenumber) going into the power spectrum function should be inverse
    arcsec.  To give a specific example, if we want to draw Gaussians on an image with dx=0.2"
    (i.e., the argument dx to the draw method will be =0.2), and if we want a grid of galaxies
    spaced 40 pixels apart, then when we call the getShear method of the PowerSpectrum class, we
    should use grid_spacing=8 [arcsec, =(40 pixels)*(0.2 arcsec/pixel)].

    If the power spectrum used for this calculation comes from a standard cosmology calculator that
    uses units of inverse radians for the wavenumber, then it is important to convert such that the
    units are consistent with our choice of inverse arcsec.  If there is sufficient interest from
    users for the code to have a "unit" class that handles conversions between the various units
    that one might use, then future versions of GalSim might be updated to include this
    functionality.
    """
    def __init__(self, E_power_function=None, B_power_function=None, units="arcsec"):
        """@brief Create a PowerSpectrum object corresponding to specific P(k) for E, B modes

        When creating a PowerSpectrum instance, the E and B mode power spectra can optionally be set
        at initialization or later on with the method set_power_functions.  Note that the power
        spectra can be ones provided in galsim.lensing (currently just a few simple power laws), or
        they can be user-provided functions that take a single argument k and return the power at
        that k value.  They should be power P(k), not Delta^2(k) = k^2 P(k) / 2pi.

        @param[in] E_power_function A function or other callable that accepts a 2D numpy grid of 
        |k| and returns the E-mode power spectrum of the same shape.  It should cope happily with 
        k=0.  The function should return the power spectrum desired in the E (gradient) mode of 
        the image
        @param[in] B_power_function A function or other callable that can take an array of k values
        and return a power.  It should cope happily with k=0.  The function should return the power
        spectrum desired in the B (curl) mode of the image
        @param[in] units A string specifying the units for the power spectrum.  This string is not
        used in any calculations, but is saved for later information.  Currently we require a value
        of "arcsec", so the user must do any necessary conversions to ensure that this is the case.
        """
        self.p_E = E_power_function
        self.p_B = B_power_function
        if units is not "arcsec":
            raise ValueError("Currently we require units of arcsec for the inverse wavenumber!")

    def set_power_functions(self, E_power_function=None, B_power_function=None, units="arcsec"):
        """@brief Set / change the functions that compute the E and B mode power spectra.

        @param[in] E_power_function A function or other callable that accepts a 2D numpy grid of |k|
        and returns the E-mode power spectrum of the same shape.  It should cope happily with k=0.
        Set to None for there to be no E-mode power.
        @param[in] B_power_function A function or other callable that accepts a 2D numpy grid of |k|
        and returns the B-mode power spectrum of the same shape.  It should cope happily with k=0.
        Set to None for there to be no B-mode power
        @param[in] units A string specifying the units for the power spectrum.  This string is not
        used in any calculations, but is saved for later information.  Currently we require a value
        of "arcsec", so the user must do any necessary conversions to ensure that this is the case.
        """
        self.p_E = E_power_function
        self.p_B = B_power_function
        if units is not "arcsec":
            raise ValueError("Currently we require units of arcsec for the inverse wavenumber!")

    def getShear(self, x=None, y=None, grid_spacing=None, grid_nx=None, gaussian_deviate=None,
                 interpolantxy=None):

        """@brief Generate a realization of the current power spectrum at the specified positions.

        Generate a Gaussian random realization of the specified E and B mode shear power spectra at
        some set of locations.  This can be done in two ways: first, given arbitrary (x, y)
        positions [NOT YET IMPLEMENTED]; or second, using grid_spacing and grid_nx to specify the
        grid spacing and grid size for a grid of positions.  This code stores information about the
        quantities used to generate the random shear field, generates shears using a
        PowerSpectrumRealizer on a grid, and if necessary, interpolates to get g1 and g2 at the
        specified positions.
        
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

        An example of how to use getShear is as follows, for the gridded case, where the choice of
        grid_spacing indicates that the grid points are spaced by 1":
        @code
        my_ps = galsim.lensing.PowerSpectrum(galsim.lensing.pk2)
        g1, g2 = my_ps.getShear(grid_spacing = 1., grid_nx = 100)
        @endcode

        To define some other P(k), the user can do the following:
        @code
        def mypk(k):
            return k**(0.1)

        my_ps = galsim.lensing.PowerSpectrum(mypk)
        g1, g2 = my_ps.getShear(grid_spacing = 1., grid_nx = 100)
        @endcode

        When using a non-gridded set of points, the code has to choose an appropriate spacing for a
        grid and then interpolate the gridded shears to the specified set of points.  It does this
        by requiring that the modification to the power spectrum due to a (bi)linear interpolant
        should not be significant at the minimum separation between points.  The user should be
        aware that use of an interpolant that is not linear does not change how this calculation is
        done, and therefore it is necessary to test the fidelity of the recovered power spectrum for
        any errors due to the chosen non-linear interpolant.

        For a given value of grid_spacing and grid_nx, we could get the x and y values on the
        grid using
        @code
        import numpy as np
        x, y = np.meshgrid(np.arange(0., grid_nx*grid_spacing, grid_spacing),
                           np.arange(0., grid_nx*grid_spacing, grid_spacing))
        @endcode
        where we assume a minimum x and y value of zero for the grid.

        @param[in] x List of x positions (it is up to the user to check that the units are
        consistent with those in the P(k) function, just as for the y and grid_spacing keywords)
        @param[in] y List of y positions
        @param[in] grid_spacing Spacing for an evenly spaced grid of points, in arcsec for
        consistency with the natural length scale of images created using the draw or drawShoot
        methods
        @param[in] grid_nx Number of grid points in the x dimension
        @param[in] gaussian_deviate (Optional) A galsim.GaussianDeviate object for drawing the
        random numbers.  (If this is a BaseDeviate class other than GaussianDeviate, that's
        fine too.)
        @param[in] interpolantxy (Optional) Interpolant to use for interpolating the shears on a
        grid to the requested positions [default = galsim.InterpolantXY(galsim.Linear())].
        @return g1,g2 Two Numpy arrays for the two shear components g_1 and g_2
        """

        # check input values for all keywords
        # (1) check problem cases for irregularly spaced set of points
        if x is not None or y is not None:
            if x is None or y is None:
                raise ValueError("When specifying points, must provide both x and y!")
            if grid_spacing is not None or grid_nx is not None:
                raise ValueError("When specifying points, do not also provide grid information!")
        # (2) check problem cases for regular grid of points
        if grid_spacing is not None or grid_nx is not None:
            if grid_spacing is None or grid_nx is None:
                raise ValueError("When specifying grid, we require both a spacing and a size!")
        # (3) make sure that we've specified some power spectrum
        if self.p_E is None and self.p_B is None:
            raise ValueError("Cannot generate shears when no E or B mode power spectrum are given!")
        # (4) make a GaussianDeviate if necessary
        if gaussian_deviate is None:
            gd = galsim.GaussianDeviate()
        elif isinstance(gaussian_deviate, galsim.GaussianDeviate):
            gd = gaussian_deviate
        elif isinstance(gaussian_deviate, galsim.BaseDeviate):
            gd = galsim.GaussianDeviate(gaussian_deviate)
        else:
            raise TypeError("The requested gaussian_deviate is not a BaseDeviate!")
        # (5) set default interpolant if none given
        if interpolantxy is None:
            interpolantxy = galsim.InterpolantXY(galsim.Linear())
        elif not isinstance(interpolantxy, galsim.InterpolantXY):
            raise TypeError("Any input interpolantxy must be a galsim.InterpolantXY instance.")

        # store some more information
        self.interpolantxy = interpolantxy

        if grid_spacing is not None:
            # do the calculation on a grid
            psr = PowerSpectrumRealizer(grid_nx, grid_nx, grid_spacing, self.p_E, self.p_B)
            g1, g2 = psr(gaussian_deviate=gd)
        else:
            # for now, we cannot do this
            raise NotImplementedError("Have not finished implementing the non-gridded case!")

        # after making either gridded shears or shears at specified x, y positions, return g1 and g2
        # arrays
        return g1, g2

class PowerSpectrumRealizer(object):
    """@brief Class for generating realizations of power spectra with any area and pixel size.
    
    Designed to quickly generate many realizations of the same shear power spectra on a square grid.
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
        
    def set_size(self, nx, ny, pixel_size, remake_power=True):
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
        pixel_size = float(pixel_size)

        # Set up the scalar |k| grid.
        self.k=((kx/(pixel_size*nx))**2+(ky/(pixel_size*ny))**2)**0.5
        
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


    def __call__(self, new_power=False, gaussian_deviate=None):
        """@brief Generate a realization of the current power spectrum.
        
        @param[in] new_power If the power-spectrum functions that you specified are not
        deterministic then you can set this value to True to call them again to get new values.
        For example, you could include a cosmic variance term in your power spectrum and get a new
        spectrum realization each time.
        @param[in] gaussian_deviate (Optional) gaussian deviate to use when generating the shear
        fields
        
        @return g1,g2 Two image arrays for the two shear components g_1 and g_2
        """
        ISQRT2 = np.sqrt(1.0/2.0)

        #If desired, recompute power spectra
        if new_power:
            self.set_power(self.p_E, self.p_B)
        
        if gaussian_deviate is None:
            gd = galsim.GaussianDeviate()
        elif isinstance(gaussian_deviate, galsim.GaussianDeviate):
            gd = gaussian_deviate
        elif isinstance(gaussian_deviate, galsim.BaseDeviate):
            gd = galsim.GaussianDeviate(gd)
        else:
            raise TypeError("The requested gaussian_deviate is not a BaseDeviate!")

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
        g1=g1_k.shape[0]*np.fft.irfft2(g1_k)
        g2=g2_k.shape[0]*np.fft.irfft2(g2_k)
        
        return g1, g2

    def _generate_power_array(self, power_function):
        #Internal function to generate the result of a power function evaluated on a grid,
        #taking into account the symmetries.
        power_array = np.zeros((self.nx, self.ny/2+1))
        P_k = power_function(self.k)
        power_array[ self.kx, self.ky] = P_k
        power_array[-self.kx, self.ky] = P_k
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

# for simple demonstration purposes, a few very simple power-law power spectra that don't crash and
# burn at k=0
def pk2(k):
    return k**(2.0)

def pk1(k):
    return k

def pkflat(k):
    # note: this gives random Gaussian shears with variance of 0.01
    return 0.01

class Cosmology(object):
    """@brief Basic cosmology calculations.

    Cosmology calculates expansion function E(a) and angular diameter distances Da(z) for a 
    LambdaCDM universe. 
    Radiation is assumed to be zero and Dark Energy constant with w = -1, but curvature
    is arbitrary.

    Based on Matthias Bartelmann's libastro.
    """
    def __init__(self, Omega_m=0.3, Omega_l=0.7):
        """@brief Create Cosmology with given energy densities for matter and for dark energy
        (specifically, a cosmological constant with w=-1); no quintessence, and no radiation.
        
        @param[in] Omega_m Present day energy density of matter relative to critical density
        @param[in] Omega_l Present day density of Dark Energy relative to critical density
        """
        # no quintessence, no radiation in this universe!
        self.omega_m = Omega_m
        self.omega_l = Omega_l
        self.omega_c = (1. - Omega_m - Omega_l)
        self.omega_r = 0
    
    def a(self, z):
        """@brief Compute scale factor

        @param[in] z Redshift
        """
        return 1./(1+z)

    def E(self, a):
        """@brief Evalutes expansion function

        @param[in] a Scale factor
        """
        return (self.omega_r*a**(-4) + self.omega_m*a**(-3) + self.omega_c*a**(-2) + \
                self.omega_l)**0.5

    def __angKernel(self, x):
        """@brief Integration kernel for angular diameter distance computation
        """
        return self.E(x**-1)**-1

    def Da(self, z, z_ref=0):
        """@brief Compute angular diameter distance between two redshifts in units of c/H0.

        In order to get the distance in Mpc/h, multiply by ~3000.

        @param[in] z Redshift
        @param[in] z_ref Reference redshift, with z_ref <= z.
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
            # TODO: We don't want to depend on scipy, so need to move this down to c++.
            from scipy.integrate import quad
            d = quad(self.__angKernel, z_ref+1, z+1)[0]
            # check for curvature
            rk = (abs(self.omega_c))**0.5
            if (rk*d > 0.01):
                if self.omega_c > 0:
                    d = sinh(rk*d)/rk
                if self.omega_c < 0:
                    d = sin(rk*d)/rk
            return d/(1+z)


class NFWHalo(object):
    """@brief Class for NFW halos.

    Compute the lensing fields shear and convergence of a NFW halo of given mass, concentration, 
    redshift, assuming Cosmology. No mass-concentration relation is employed.

    Based on Matthias Bartelmann's libastro.
    """
    def __init__(self, mass=1e15, conc=4, z=0.3, pos_x=0, pos_y=0, cosmo=Cosmology()):
        """@brief Create NFW halo.

        @param[in] mass Mass defined using a spherical overdensity of 200 times the critical density
                        of the universe, in units of M_solar/h.
        @param[in] conc Concentration parameter, i.e., ratio of virial radius to NFW scale radius
        @param[in] z Redshift
        @param[in] pos_x X-coordinate [arcsec]
        @param[in] pos_y Y-coordinate [arcsec]
        @param[in] cosmo A Cosmology instance
        """
        self.M = mass
        self.c = conc
        self.z = z
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.cosmo = cosmo

        # calculate scale radius
        a = self.cosmo.a(z)
        # First we get the virial radius, which is defined for some spherical overdensity as
        # 3 M / (4 pi r_vir)^3 = overdensity
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
        """@brief Matter density at scale factor a
        """
        return self.cosmo.omega_m/(self.cosmo.E(a)**2 * a**3)

    def __farcth (self, x, out=None):
        """@brief Numerical implementation of integral functions of NFW profile
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
        """@brief Calculate convergence of halo.

        @param[in] x Radial coordinate in units of r/rs (normalized to scale radius of halo)
        @param[in] ks Lensing strength prefactor
        @param[in] out Numpy array into which results should be placed
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
        """@brief Calculate tangential shear of halo.

        @param[in] x Radial coordinate in units of r/rs (normalized to scale radius of halo)
        @param[in] ks Lensing strength prefactor
        @param[in] out Numpy array into which results should be placed
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
        """@brief Lensing strength of halo as function of source redshift
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

    def getShear(self, pos_x, pos_y, z_s, units='arcsec', reduced=True):
        """@brief Calculate (reduced) shear of halo at specified positions

        @param[in] pos_x X-coordinate(s) of the source, input as a numpy array. This is assumed to
                         be post-lensing!
        @param[in] pos_y Y-coordinate(s) of the source, input as a numpy array. This is assumed to
                         be post-lensing!
        @param[in] z_s Source redshift(s)
        @param[in] units Units of coordinates (only arcsec implemented so far).
        @param[in] reduced Whether reduced shears are returned
        @return g1,g2 Numpy arrays containing the two shear components g_1 and g_2 at the specified
                      position(s)
        """
        if units != 'arcsec':
            raise NotImplementedError("Only arcsec units implemented!")

        x = ((pos_x - self.pos_x)**2 + (pos_y - self.pos_y)**2)**0.5/self.rs_arcsec
        # compute strength of lensing fields
        ks = self.__ks(z_s)
        if isinstance(z_s, np.ndarray) == False:
            ks = ks*np.ones_like(pos_x)
        g = self.__gamma(x, ks)

        # convert to observable = reduced shear
        if reduced:
            kappa = self.__kappa(x, ks)
            g /= 1 - kappa

        # split into g1 and g2 component:
        # pure tangential shear, no cross component
        phi = np.arctan2(pos_y - self.pos_y, pos_x - self.pos_x)
        g1 = -g / (np.cos(2*phi) + np.sin(2*phi)*np.tan(2*phi))
        g2 = g1 * np.tan(2*phi)
        return g1, g2

    def getConvergence(self, pos_x, pos_y, z_s, units='arcsec'):
        """@brief Calculate convergence of halo at specified positions

        @param[in] pos_x X-coordinate(s) of the source, input as a Numpy array. This is assumed to
                         be post-lensing!
        @param[in] pos_y Y-coordinate(s) of the source, input as a Numpy array. This is assumed to 
                         be post-lensing!
        @param[in] z_s Source redshift(s)
        @param[in] units Units of coordinates (only arcsec implemented so far).
        @return kappa Numpy array containing the convergence at the specified position(s)
        """
        if units != 'arcsec':
            raise NotImplementedError("Only arcsec units implemented!")

        x = ((pos_x - self.pos_x)**2 + (pos_y - self.pos_y)**2)**0.5/self.rs_arcsec
        # compute strength of lensing fields
        ks = self.__ks(z_s)
        if isinstance(z_s, np.ndarray) == False:
            ks = ks*np.ones_like(pos_x)
        return self.__kappa(x, ks)

