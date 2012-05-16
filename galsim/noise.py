from . import _galsim

def addNoise(image, noise):
    """@brief Noise addition Image method, adding noise according to a supplied noise model.

    >>> Image.addNoise(noise)

    Noise following supplied model will be added to the image.

    Parameters
    ----------
    @param[in,out]  image  The image on which to add the noise.
    @param[in,out]  noise  Instantiated noise model (currently CCDNoise, UniformDeviate,
                           BinomialDeviate, GaussianDeviate and PoissonDeviate are supported).

    If the supplied noise model object does not have an applyTo() method, then this will raise an
    AttributeError exception.
    """
    im_view = image.view()
    noise.applyTo(im_view)

# inject addNoise as a method of Image classes
for Class in _galsim.Image.itervalues():
    Class.addNoise = addNoise

for Class in _galsim.ImageView.itervalues():
    Class.addNoise = addNoise

del Class # cleanup public namespace
