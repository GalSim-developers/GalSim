import numpy as np
import galsim

def addGaussian(Image, uniform, mean=0., sigma=1.):
    """@brief Add Gaussian noise to an input Image using a user-supplied UniformDeviate instance.

    Parameters
    ----------
    @param[in,out] Image    input Image
    @param[in,out] uniform  a galsim UniformDeviate instance to supply the random numbers
    @param[in]     mean     optional mean for the Gaussian noise (default mean = 0.)
    @param[in]     sigma    optional sigma for the Gaussian noise (default sigma = 1.)
    
    On output, the supplied Image will now have noise added with the desired properties; this Image
    is also passed as the function return value.

    Note that noise will be integer-rounded for integer type input Images.

    TODO: Speed this up my moving repeated calls to C++/use the Ccd noise class.
    """
    g = galsim.GaussianDeviate(uniform, mean=mean, sigma=sigma)
    imtype = Image.array.dtype.type
    Image += galsim.Image[imtype]( np.array([g() for i in xrange(np.product(Image.array.shape))],
                                            dtype=imtype).reshape(Image.array.shape) )
    return Image

def addPoisson(Image, uniform, gain=1.):
    """@brief Add Poisson noise to an input Image using a user-supplied UniformDeviate instance.

    Parameters
    ----------
    @param[in,out] Image    input Image
    @param[in,out] uniform  a galsim UniformDeviate instance to supply the random numbers
    @param[in]     gain     optional gain e-/ADU for calculating the noise (default gain = 1.)
    
    On output, the supplied Image will now have noise added with the desired properties; this Image
    is also passed as the function return value.
    
    Note integer Poisson noise will be converted to floats when added to float-type input Images.

    TODO: Speed this up my moving repeated calls to C++/use the Ccd noise class.
    """
    p = galsim.PoissonDeviate(uniform)
    imtype = Image.array.dtype.type
    elist = list(Image.array.flatten('C') * gain)
    Image += galsim.Image[imtype](np.array([poissonwithmean(p, elecs) / gain for elecs in elist],
                                  dtype=imtype).reshape(Image.array.shape, order='C')) \
           - Image  # Barney note: need to understand image operations better, couldn't get
                    # other assignment to work on output after leaving function scope
    return Image

def poissonwithmean(p, mean):
    """Output a Poisson deviate with distribution updated to the specified mean in a single call.
    """
    if mean <= 0.:
        return 0.
    else:
        p.setMean(float(mean))
        return p()

