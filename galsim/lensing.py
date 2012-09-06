"""\file lensing.py The "lensing engine" for drawing shears from some power spectrum or a NFW halo.
"""
import galsim
import numpy as np
from math import log
import warnings

ISQRT2 = np.sqrt(1.0/2.0)

## TODO later: get convergences that are consistent with this shear field
class PowerSpectrum(object):
    """@brief Class to represent a lensing shear field according to some specified power spectrum

    A PowerSpectrum represents some (flat-sky) shear power spectrum, either for gridded points or at
    arbitary positions.  This class is originally initialized with a power spectrum from which we
    would like to generate g1 and g2 values.  When the getShear() method is called, it uses a
    PowerSpectrumRealizer to generate shears on an appropriately-spaced grid, and if necessary,
    interpolates on that grid to the requested positions.  Finally, it carries around some
    information about the underlying shear power spectrum used to generate the field.
    """
    def __init__(self, E_power_function=None, B_power_function=None, units=None):
        """@brief Create a PowerSpectrum object corresponding to specific P(k) for E, B modes

        When creating a PowerSpectrum instance, the E and B mode power spectra can optionally be set
        at initialization or later on with the method set_power_functions.

        @param[in] E_power_function A function or other callable that accepts a 2D numpy grid of |k| and returns
        the E-mode power spectrum of the same shape.  It should cope happily with k=0.  The function should return the power
        spectrum desired in the E (gradient) mode of the image
        @param[in] B_power_function A function or other callable that can take an array of k values
        and return a power.  It should cope happily with k=0.  The function should return the power
        spectrum desired in the B (curl) mode of the image
        @param[in] units A string specifying the units for the power spectrum.  This string is not
        used in any calculations, but is saved for later information.
        """
        self.p_E = E_power_function
        self.p_B = B_power_function
        if units is not None:
            self.units = units

    def set_power_functions(self, E_power_function=None, B_power_function=None, units=None):
        """@brief Set / change the functions that compute the E and B mode power spectra.

        @param[in] E_power_function A function or other callable that accepts a 2D numpy grid of |k|
        and returns the E-mode power spectrum of the same shape.  It should cope happily with k=0.
        Set to None for there to be no E-mode power.
        @param[in] B_power_function A function or other callable that accepts a 2D numpy grid of |k|
        and returns the B-mode power spectrum of the same shape.  It should cope happily with k=0.
        Set to None for there to be no B-mode power
        @param[in] units A string specifying the units for the power spectrum.  This string is not
        used in any calculations, but is saved for later information.
        """
        self.p_E = E_power_function
        self.p_B = B_power_function
        if units is not None:
            self.units = units

    def getShear(self, x=None, y=None, grid_spacing=None, grid_nx=None, grid_ny=None,
                 gaussian_deviate=None, seed=None, interpolantxy=None):

        """@brief Generate a realization of the current power spectrum at the specified positions.

        Generate a Gaussian random realization of the specified E and B mode shear power spectra at
        some set of locations.  This can be done in two ways: first, given arbitrary (x, y)
        positions [NOT YET IMPLEMENTED]; or second, using grid_spacing, grid_nx, and (optionally)
        grid_ny to specify the grid spacing and grid size for a grid of positions.  This code stores
        information about the quantities used to generate the random shear field, generates shears
        using a PowerSpectrumRealizer on a grid, and if necessary, interpolates to get g1 and g2 at
        the specified positions.  An example of how to use getShear is as follows, for the gridded
        case:
        @code
        my_ps = galsim.lensing.PowerSpectrum(galsim.lensing.pk2) g1, g2 =
        my_ps.getShear(grid_spacing = 1., grid_nx = 100)
        @endcode

        When using a non-gridded set of points, the code has to choose an appropriate spacing for a
        grid and then interpolate the gridded shears to the specified set of points.  It does this
        by requiring that the modification to the power spectrum due to a (bi)linear interpolant
        should not be significant at the minimum separation between points.  The user should be
        aware that use of an interpolant that is not linear does not change how this calculation is
        done, and therefore it is necessary to test the fidelity of the recovered power spectrum for
        any errors due to the chosen non-linear interpolant.

        For a given value of grid_spacing, grid_nx, grid_ny, we could get the x and y values on the
        grid using
        @code
        import numpy as np
        x, y = np.meshgrid(np.arange(0., grid_nx*grid_spacing, grid_spacing),
                           np.arange(0., grid_ny*grid_spacing, grid_spacing))
        @endcode
        where we assume a minimum x and y value of zero for the grid.

        @param[in] x List of x positions (units should be consistent with P(k) function)
        @param[in] y List of y positions (units should be consistent with P(k) function)
        @param[in] grid_spacing Spacing for an evenly spaced grid of points (units should be
        consistent with P(k) function)
        @param[in] grid_nx Number of grid points in the x dimension
        @param[in] grid_ny Number of grid points in the y dimension
        @param[in] gaussian_deviate (Optional) A galsim.GaussianDeviate object for drawing the
        random numbers
        @param[in] seed (Optional) A seed to use when initializing a new galsim.GaussianDeviate for
        drawing the random numbers
        @param[in] interpolantxy (Optional) Interpolant to use for interpolating the shears on a
        grid to the requested positions [default = galsim.InterpolantXY(galsim.Linear())].
        @return g1,g2 Two Numpy arrays for the two shear components g_1 and g_2
        """

        # check input values for all keywords
        # (1) check problem cases for irregularly spaced set of points
        if x is not None or y is not None:
            if x is None or y is None:
                raise ValueError("When specifying points, must provide both x and y!")
            if grid_spacing is not None or grid_nx is not None or grid_ny is not None:
                raise ValueError("When specifying points, do not also provide grid information!")
        # (2) check problem cases for regular grid of points
        if grid_spacing is not None or grid_nx is not None or grid_ny is not None:
            if grid_spacing is None or grid_nx is None:
                raise ValueError("When specifying grid, we require at least a spacing and x size!")
            if grid_ny is None:
                grid_ny = grid_nx
        # (3) make sure that we've specified some power spectrum
        if self.p_E is None and self.p_B is None:
            raise ValueError("Cannot generate shears when no E or B mode power spectrum are given!")
        # (4) handle any specification of the Gaussian deviate and/or seed
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
            gd = gaussian_deviate
        # (5) set default interpolant if none given
        if interpolantxy is None:
            interpolantxy = galsim.InterpolantXY(galsim.Linear())

        # store some more information
        self.interpolantxy = interpolantxy

        if grid_spacing is not None:
            # do the calculation on a grid
            psr = PowerSpectrumRealizer(grid_nx, grid_ny, grid_spacing, self.p_E, self.p_B)
            g1, g2 = psr(gaussian_deviate=gd)
        else:
            # for now, we cannot do this
            raise NotImplementedError("Have not finished implementing the non-gridded case!")

        # after making either gridded shears or shears at specified x, y positions, return g1 and g2
        # arrays
        return g1, g2

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
            gd = gaussian_deviate

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

# for simple demonstration purposes, a very simple power-law power spectrum that doesn't crash and
# burn at k=0
def pk2(k):
    return k**(2.0)



class Cosmology(object):
    """@brief Basic cosmology calculations.

    Cosmology calculates expansion function E(a) and angular diameter distances Da(z) for a 
    LambdaCDM universe. 
    Radiation is assumed to be zero and Dark Energy constant with w = -1, but curvature
    is arbitrary.

    Based on Matthias Bartelmann's libastro.
    """
    def __init__(self, Omega_m=0.3, Omega_l=0.7):
        """@brief Create Cosmology with given energy densities.
        
        @param[in] Omega_m Energy density of matter
        @param[in] Omega_l Density of Dark Energy
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
        return (self.omega_r*a**(-4) + self.omega_m*a**(-3) + self.omega_c*a**(-2) + self.omega_l)**0.5

    def __angKernel(self, x):
        """@brief Integration kernel for angular diameter distance computation
        """
        return self.E(x**-1)**-1

    def Da(self, z, z_ref=0):
        """@brief Compute angular diameter distance between two redshifts in units of c/H0.

        In order to get the distance in Mpc, multiply by ~3000.

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
            try:
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
            except ImportError:
                warnings.warn("scipy not found! Integrator required for angular diameter distances")
                return z


class NFWHalo(object):
    """@brief Class for NFW halos.

    Compute the lensing fields shear and convergence of a NFW halo of given mass, concentration, 
    redshift, assuming Cosmology. No mass-concentration relation is employed.

    Based on Matthias Bartelmann's libastro.
    """
    def __init__(self, mass=1e15, conc=4, z=0.3, pos_x=0, pos_y=0, cosmo=Cosmology()):
        """@brief Create NFW halo.

        @param[in] mass Mass
        @param[in] conc Concentration parameter
        @param[in] z Redshift
        @param[in] pos_x X-coordinate [arcsec]
        @param[in] pos_y Y-coordinate [arcsec]
        @param[in] cosmo A Comology instance
        """
        self.M = mass
        self.c = conc
        self.z = z
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.cosmo = cosmo

        # calculate scale radius
        a = self.cosmo.a(z)
        R200 = 1.63e-5/(1+self.z) * (self.M * self.__omega(a)/self.__omega(1))**0.3333 # in Mpc
        self.rs = R200/self.c

        # convert scale radius in arcsec
        dl = self.cosmo.Da(self.z)*3000.; # in Mpc
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

        # 3 cases: x > 1, x < 1, and |x-1| < 0.01
        mask = (x < 0.99)
        if mask.any():
            a = ((1.-x[mask])/(x[mask]+1.))**0.5
            out[mask] = 0.5*np.log((1.+a)/(1.-a))/(1-x[mask]**2)**0.5

        mask = (x > 1.01)
        if mask.any():
            a = ((x[mask]-1.)/(x[mask]+1.))**0.5
            out[mask] = np.arctan(a)/(x[mask]**2 - 1)**0.5

        mask = (x >= 0.99) & (x <= 1.01)
        if mask.any():
            out[mask] = 5./6. - x[mask]/3.

        return out

    def __kappa(self, x, ks, out=None):
        """@brief Calculate convergence of halo.

        @param[in] x Radial coordinate in units of r/rs (normalized to scale radius of halo)
        @param[in] ks Lensing strength prefactor
        """
        # convenience: call with single number
        if isinstance(x, np.ndarray) == False:
            return kappa(np.array([x], dtype='float'), np.array([ks], dtype='float'))[0]

        if out is None:
            out = np.zeros_like(x)

        # 3 cases: x > 1, x < 1, and |x-1| < 0.01
        mask = (x < 0.99)
        if mask.any():
            a = ((1 - x[mask])/(x[mask] + 1))**0.5
            out[mask] = 2*ks[mask]/(x[mask]**2 - 1)*(1 - np.log((1 + a)/(1 - a))/(1 - x[mask]**2)**0.5)

        mask = (x > 1.01)
        if mask.any():
            a = ((x[mask] - 1)/(x[mask] + 1))**0.5
            out[mask] = 2*ks[mask]/(x[mask]**2 - 1)*(1 - 2*np.arctan(a)/(x[mask]**2 - 1)**0.5)

        mask = (x >= 0.99) & (x <= 1.01)
        if mask.any():
            out[mask] = ks[mask]*(22./15. - 0.8*x[mask])

        return out

    def __gamma(self, x, ks, out=None):
        """@brief Calculate tangential shear of halo.

        @param[in] x Radial coordinate in units of r/rs (normalized to scale radius of halo)
        @param[in] ks Lensing strength prefactor
        """
        # convenience: call with single number
        if isinstance(x, np.ndarray) == False:
            return gamma(np.array([x], dtype='float'), np.array([ks], dtype='float'))[0]
        if out is None:
            out = np.zeros_like(x)

        mask = (x < 0.05)
        if mask.any():
            out[mask] = 4*ks[mask]*(0.25 + 0.125 * x[mask]**2 * (3.25 + 3.0*np.log(x[mask]/2)))

        mask = (mask == False)
        if mask.any():
            out[mask] = 4*ks[mask]*(np.log(x[mask]/2) + 2*self.__farcth(x[mask])) * x[mask]**(-2) - self.__kappa(x[mask], ks[mask])
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
        d0 = 200./3 * self.c**3/(log(1+self.c) - (1.*self.c)/(1+self.c))
        rho_s = rho_c * ez**2 *d0

        # lensing weights: the only thing that depends on z_s
        # this does takes some time...
        dl = self.cosmo.Da(z_s, self.z) * self.cosmo.Da(self.z) / self.cosmo.Da(z_s)
        k_s = dl * self.rs * rho_s / Sigma_c
        return k_s

    def getShear(self, pos_x, pos_y, z_s, units='arcsec', reduced=True):
        """@brief Calculate (reduced) shear of halo at specified positions

        @param[in] pos_x X-coordinate(s) of the source. This is assumed to be post-lensing!
        @param[in] pos_y Y-coordinate(s) of the source. This is assumed to be post-lensing!
        @param[in] z_s Source redshift(s)
        @param[in] units Units of coordinates (only arcsec implemented so far).
        @param[in] reduced Whether reduced shears are returned
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

        @param[in] pos_x X-coordinate(s) of the source. This is assumed to be post-lensing!
        @param[in] pos_y Y-coordinate(s) of the source. This is assumed to be post-lensing!
        @param[in] z_s Source redshift(s)
        @param[in] units Units of coordinates (only arcsec implemented so far).
        """
        if units != 'arcsec':
            raise NotImplementedError("Only arcsec units implemented!")

        x = ((pos_x - self.pos_x)**2 + (pos_y - self.pos_y)**2)**0.5/self.rs_arcsec
        # compute strength of lensing fields
        ks = self.__ks(z_s)
        if isinstance(z_s, np.ndarray) == False:
            ks = ks*np.ones_like(pos_x)
        return self.__kappa(x, ks)

