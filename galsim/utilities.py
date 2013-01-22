"""@file utilities.py
Module containing general utilities for the GalSim software.
"""

import numpy as np
import galsim

def roll2d(image, (iroll, jroll)):
    """Perform a 2D roll (circular shift) on a supplied 2D numpy array, conveniently.

    @param image            the numpy array to be circular shifted.
    @param (iroll, jroll)   the roll in the i and j dimensions, respectively.

    @returns the rolled image.
    """
    return np.roll(np.roll(image, jroll, axis=1), iroll, axis=0)

def kxky(array_shape=(256, 256)):
    """Return the tuple kx, ky corresponding to the DFT of a unit integer-sampled array of input
    shape.
    
    Uses the SBProfile conventions for Fourier space, so k varies in approximate range (-pi, pi].
    Uses the most common DFT element ordering conventions (and those of FFTW), so that `(0, 0)`
    array element corresponds to `(kx, ky) = (0, 0)`.

    See also the docstring for np.fftfreq, which uses the same DFT convention, and is called here,
    but misses a factor of pi.
    
    Adopts Numpy array index ordering so that the trailing axis corresponds to kx, rather than the
    leading axis as would be expected in IDL/Fortran.  See docstring for numpy.meshgrid which also
    uses this convention.

    @param array_shape   the Numpy array shape desired for `kx, ky`. 
    """
    # Note: numpy shape is y,x
    k_xaxis = np.fft.fftfreq(array_shape[1]) * 2. * np.pi
    k_yaxis = np.fft.fftfreq(array_shape[0]) * 2. * np.pi
    return np.meshgrid(k_xaxis, k_yaxis)

def g1g2_to_e1e2(g1, g2):
    """Convenience function for going from (g1, g2) -> (e1, e2).

    Here g1 and g2 are reduced shears, and e1 and e2 are distortions - see shear.py for definitions
    of reduced shear and distortion in terms of axis ratios or other ways of specifying ellipses.
    @param g1  First reduced shear component (along pixel axes)
    @param g2  Second reduced shear component (at 45 degrees with respect to image axes)
    @returns The corresponding distortions, e1 and e2.
    """
    # Conversion:
    # e = (a^2-b^2) / (a^2+b^2)
    # g = (a-b) / (a+b)
    # b/a = (1-g)/(1+g)
    # e = (1-(b/a)^2) / (1+(b/a)^2)
    gsq = g1*g1 + g2*g2
    if gsq > 0.:
        g = np.sqrt(gsq)
        boa = (1-g) / (1+g)
        e = (1 - boa*boa) / (1 + boa*boa)
        e1 = g1 * (e/g)
        e2 = g2 * (e/g)
        return e1, e2
    elif gsq == 0.:
        return 0., 0.
    else:
        raise ValueError("Input |g|^2 < 0, cannot convert.")

class AttributeDict(object):
    """Dictionary class that allows for easy initialization and refs to key values via attributes.

    NOTE: Modified a little from Jim's bot.git AttributeDict class so that tab completion now works
    in ipython since attributes are actually added to __dict__.
    
    HOWEVER this means the __dict__ attribute has been redefined to be a collections.defaultdict()
    so that Jim's previous default attribute behaviour is also replicated.
    """
    def __init__(self):
        import collections
        object.__setattr__(self, "__dict__", collections.defaultdict(AttributeDict))

    def __getattr__(self, name):
        return self.__dict__[name]

    def __setattr__(self, name, value):
        self.__dict__[name] = value

    def merge(self, other):
        self.__dict__.update(other.__dict__)

    def _write(self, output, prefix=""):
        for k, v in self.__dict__.iteritems():
            if isinstance(v, AttributeDict):
                v._write(output, prefix="{0}{1}.".format(prefix, k))
            else:
                output.append("{0}{1} = {2}".format(prefix, k, repr(v)))

    def __nonzero__(self):
        return not not self.__dict__

    def __repr__(self):
        output = []
        self._write(output, "")
        return "\n".join(output)

    __str__ = __repr__

    def __len__(self):
        return len(self.__dict__)

def rand_arr(shape, deviate):
    """Function to make a 2d array of random deviates (of any sort).

    @param shape A list of length 2, indicating the desired 2d array dimensions
    @param deviate Any GalSim deviate (see random.py) such as UniformDeviate, GaussianDeviate,
    etc. to be used to generate random numbers
    @returns A Numpy array of the desired dimensions with random numbers generated using the
    supplied deviate.
    """
    if len(shape) is not 2:
        raise ValueError("Can only make a 2d array from this function!")
    # note reversed indices due to Numpy vs. Image array indexing conventions!
    tmp_img = galsim.ImageD(shape[1], shape[0])
    deviate.applyTo(tmp_img.view())
    return tmp_img.array

def eval_sbinterpolatedimage(sbi, x_list, y_list):
    """Function to get the value of some SBInterpolatedImage at a list of positions.

    @param sbi An SBInterpolatedImage on which we want to carry out the interpolation.
    @param x_list A list of the x values at which we want to do the interpolation.
    @param y_list A list of the y values at which we awnt to do the interpolation.
    @returns A list of the image values at the desired (x, y) positions.
    """
    if len(x_list) != len(y_list):
        raise RuntimeError("x and y list lengths must match!")
    vals = []
    for x_ind in range(len(x_list)):
        vals.append(sbi.xValue(galsim.PositionD(x_list[x_ind], y_list[x_ind])))
    return vals

def convert_interpolant_to_2d(interpolant):
    """Convert a given interpolant to an Interpolant2d if it is given as a string or 1-d.
    """
    if interpolant == None:
        return None  # caller is responsible for setting a default if desired.
    elif isinstance(interpolant, galsim.Interpolant2d):
        return interpolant
    elif isinstance(interpolant, galsim.Interpolant):
        return galsim.InterpolantXY(interpolant)
    else:
        try:
            return galsim.Interpolant2d(interpolant)
        except:
            raise RuntimeError('Specified interpolant is not valid!')


