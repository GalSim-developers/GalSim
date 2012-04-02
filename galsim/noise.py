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

def addPoisson(Image, uniform, mean=1.):
    """@brief Add Poisson noise to an input Image using a user-supplied UniformDeviate instance.

    Parameters
    ----------
    @param[in,out] Image    input Image
    @param[in,out] uniform  a galsim UniformDeviate instance to supply the random numbers
    @param[in]     mean     optional mean for the Poisson noise (default mean = 0.)
    
    On output, the supplied Image will now have noise added with the desired properties; this Image
    is also passed as the function return value.
    
    Note integer Poisson noise will be converted to floats when added to float-type input Images.

    TODO: Speed this up my moving repeated calls to C++/use the Ccd noise class.
    """
    p = galsim.PoissonDeviate(uniform, mean=mean)
    imtype = Image.array.dtype.type
    Image += galsim.Image[imtype]( np.array([p() for i in xrange(np.product(Image.array.shape))],
                                             dtype=imtype).reshape(Image.array.shape) )
    return Image
