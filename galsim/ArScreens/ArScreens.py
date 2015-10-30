import numpy as np
import numpy.random as ra
import scipy.fftpack as sf
import pyfits
from create_multilayer_arbase import create_multilayer_arbase

class ArScreens(object):
    """
    Class to generate atmosphere phase screens using an autoregressive
    process to add stochastic noise to an otherwise frozen flow.
    @param n          Number of subapertures across the screen
    @param m          Number of pixels per subaperature
    @param pscale     Pixel scale
    @param rate       A0 system rate (Hz)
    @param paramcube  Parameter array describing each layer of the atmosphere
                      to be modeled.  Each row contains a tuple of 
                      (r0 (m), velocity (m/s), direction (deg), altitude (m))
                      describing the corresponding layer.
    @param alpha_mag  magnitude of autoregressive parameter.  (1-alpha_mag)
                      is the fraction of the phase from the prior time step
                      that is "forgotten" and replaced by Gaussian noise.
    """
    def __init__(self, n, m, pscale, rate, paramcube, alpha_mag, 
                 ranseed=None):
        self.pl, self.alpha = create_multilayer_arbase(n, m, pscale, rate,
                                                       paramcube, alpha_mag)
        self._phaseFT = None
        self.screens = [[] for x in paramcube]
        ra.seed(ranseed)
    def get_ar_atmos(self):
        shape = self.alpha.shape
        newphFT = []
        newphase = []
        for i, powerlaw, alpha in zip(range(shape[0]), self.pl, self.alpha):
            noise = ra.normal(size=shape[1:3])
            noisescalefac = np.sqrt(1. - np.abs(alpha**2))
            noiseFT = sf.fft2(noise)*powerlaw
            if self._phaseFT is None:
                newphFT.append(noiseFT)
            else:
                newphFT.append(alpha*self._phaseFT[i] + noiseFT*noisescalefac)
            newphase.append(sf.ifft2(newphFT[i]).real)
        return np.array(newphFT), np.array(newphase)
    def run(self, nframes, verbose=False):
        for j in range(nframes):
            if verbose:
                print "time step", j
            self._phaseFT, screens = self.get_ar_atmos()
            for i, item in enumerate(screens):
                self.screens[i].append(item)
    def write(self, outfile, clobber=True):
        output = pyfits.HDUList()
        output.append(pyfits.PrimaryHDU())
        for i, screen in enumerate(self.screens):
            output.append(pyfits.ImageHDU(np.array(screen)))
            output[-1].name = "Layer %i" % i
        output.writeto(outfile, clobber=clobber)

if __name__ == '__main__':
    n = 48
    m = 8
    bigD = 8.4
    pscale = bigD/(n*m)
    rate = 1000.
    alpha_mag = 0.99
    paramcube = np.array([(0.85, 23.2, 59, 7600),
                          (1.08, 5.7, 320, 16000)])

    my_screens = ArScreens(n, m, pscale, rate, paramcube, alpha_mag)
    my_screens.run(100, verbose=True)
    my_screens.write('my_screens_0.999.fits')
    
