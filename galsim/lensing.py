import galsim
import numpy as np

ISQRT2 = np.sqrt(1.0/2.0)

class PowerSpectrumRealizer(object):
    """@brief Class for generating realizations of power spectra with any area and pixel size.
    
    Designed to quickly generate many realizations of the same spectra.
    
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
        
        
        



def example():
    from numpy import isfinite, min, max, zeros_like
    from pylab import figure, subplot, imshow, title, show, colorbar, axes
    nx=1024
    ny=1024
    size=0.01

    @power_spectrum_function
    def p_E(k):
        return k**-2

    p_B = None

    generator = PowerSpectrumRealizer(nx, ny, size, p_E, p_B)
    g1,g2 = generator()

    pmin=min([g1.min(), g2.min()])
    pmax=max([g2.max(), g2.max()])
    figure(figsize=(12,12))
    subplot(121)
    imshow(g1,interpolation='nearest',vmin=pmin, vmax=pmax,origin='lower')
    title("$g_1$")
    subplot(122)
    imshow(g2,interpolation='nearest',vmin=pmin, vmax=pmax,origin='lower')
    title("$g_2$")
    #subplots_adjust(bottom=0.1)
    ax=axes((0.1,0.25,0.8,0.03),frameon=False)
    colorbar(cax=ax,orientation='horizontal')
    figure()
    P = (g1**2+g2**2)**0.5
    imshow(np.log10(P), interpolation='nearest', vmin=-5)
    colorbar()
    show()



def power_spectrum_function(P):
    """A decorator that takes a simple function P(k) and returns a well-behaved power spectrum"""
    def P_out(k):
        p = P(k)
        p[~np.isfinite(p)]=0
        return p
    return P_out
        

if __name__=="__main__":
    example()

#Some tests:
# Should always generate desired size images (test odd and even nx, ny)
# Check result for everything zero
# Check monopole term
# Check results are different when we re-realize
