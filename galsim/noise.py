import numpy as np
import galsim

def addGaussian(image, uniform, mean=0., sigma=1.):
    """@brief Add Gaussian noise to an input Image using a user-supplied UniformDeviate instance.

    DEPRECATED: Please use the CCDNoise class and addNoise() function / Image method.

    Parameters
    ----------
    @param[in,out] image    input Image instance
    @param[in,out] uniform  a galsim UniformDeviate instance to supply the random numbers
    @param[in]     mean     optional mean for the Gaussian noise (default mean = 0.)
    @param[in]     sigma    optional sigma for the Gaussian noise (default sigma = 1.)
    
    On output, the supplied Image will now have noise added with the desired properties; this Image
    is also passed as the function return value.

    Note that noise will be integer-rounded for integer type input Images.

    TODO: Speed this up my moving repeated calls to C++/use the Ccd noise class.
    """
    g = galsim.GaussianDeviate(uniform, mean=mean, sigma=sigma)
    imtype = image.array.dtype.type
    image += galsim.Image[imtype](np.array([g() for i in xrange(np.product(image.array.shape))],
                                           dtype=imtype).reshape(image.array.shape), 
                                  xMin=image.getXMin(), yMin=image.getYMin())
    return image

def addPoisson(image, uniform, gain=1.):
    """@brief Add Poisson noise to an input Image using a user-supplied UniformDeviate instance.

    DEPRECATED: Please use the CCDNoise class and addNoise() function / Image method.

    Parameters
    ----------
    @param[in,out] image    input Image
    @param[in,out] uniform  a galsim UniformDeviate instance to supply the random numbers
    @param[in]     gain     optional gain e-/ADU for calculating the noise (default gain = 1.)
    
    On output, the supplied Image will now have noise added with the desired properties; this Image
    is also passed as the function return value.
    
    Note integer Poisson noise will be converted to floats when added to float-type input Images.

    TODO: Speed this up my moving repeated calls to C++/use the Ccd noise class.
    """
    p = galsim.PoissonDeviate(uniform)
    imtype = image.array.dtype.type
    elist = list(image.array.flatten('C') * gain)
    image += galsim.Image[imtype](np.array([poissonwithmean(p, elecs) / gain for elecs in elist],
                                  dtype=imtype).reshape(image.array.shape, order='C'),
                                  xMin=image.getXMin(), yMin=image.getYMin()) \
           - image  # Barney note: need to understand image operations better, couldn't get
                    # other assignment to work on output after leaving function scope
    return image

def poissonwithmean(p, mean):
    """Output a Poisson deviate with distribution updated to the specified mean in a single call.
    """
    if mean <= 0.:
        return 0.
    else:
        p.setMean(float(mean))
        return p()

def addNoise(image, noise):
    """@brief Add noise according to a supplied noise model.

    Parameters
    ----------
    @param[in,out]  noise  instantiated noise model (currently only CCDNoise instances supported).

    If the supplied noise object does not have an applyTo() method, then this will raise an
    AttributeError exception.

    Currently only CCDNoise instances have this method for quickly applying noise to images, see
    documentation for galsim.CCDNoise objects.
    """
    noise.applyTo(image)

# inject addNoise as a method of Image classes
for Class in galsim.Image.itervalues():
    Class.addNoise = addNoise
del Class
