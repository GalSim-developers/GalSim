from . import _galsim

"""\file psfcorr.py Routines for adaptive moment estimation and PSF correction
"""

def EstimateShearHSM(gal_image, PSF_image, sky_var = 0.0, shear_est = "REGAUSS", flags = 0xe,
                     guess_sig_gal = 5.0, guess_sig_PSF = 3.0, precision = 1.0e-6,
                     guess_x_centroid = -1000.0, guess_y_centroid = -1000.0, strict = True):
    """
    @brief PSF correction method from HSM.

    Carry out PSF correction using one of the methods of the HSM package to estimate shears.
    Example usage:

    @code
    galaxy = galsim.Gaussian(flux = 1.0, sigma = 1.0)
    galaxy.applyShear(g1=0.05, g2=0.0)  # shear it by (0.05, 0) using the g=(a-b)/(a+b) definition
    psf = galsim.atmosphere.DoubleGaussian(flux1 = 0.7, sigma1 = 0.7, flux2 = 0.3, sigma2 = 1.5)
    pixel = galsim.Pixel(xw = 0.2, yw = 0.2)
    final = galsim.Convolve([galaxy, psf, pixel])
    final_epsf = galsim.Convolve([psf, pixel])
    final_image = final.draw(dx = 0.2)
    final_epsf_image = final_epsf.draw(dx = 0.2)
    result = galsim.EstimateShearHSM(final_image, final_epsf_image)
    @endcode

    After running the above code, result.observed_shape ["shape" = distortion, (a^2-b^2)/(a^2+b^2)] is
    (0.0595676, 0), and result.corrected_shape is (0.0981158, 0), compared with the expected
    (0.09975, 0) for a perfect PSF correction method.  Note that the method will fail if the PSF or
    galaxy are too point-like to easily fit an elliptical Gaussian; when running on batches of many
    galaxies, it may be preferable to set strict=False and catch errors explicitly, i.e.

    @code
    n_image = 100
    n_fail = 0
    for i=0, range(n_image):
        ...some code defining this_image, this_final_epsf_image...
        result = galsim.EstimateShearHSM(this_image, this_final_epsf_image, strict = False)
        if result.error_message != "":
            n_fail += 1
    print "Number of failures: ", n_fail
    @endcode

    Parameters
    ----------
    @param[in] gal_image The Image or ImageView of the galaxy being measured.
    @param[in] PSF_image The Image or ImageView for the PSF
    @param[in] sky_var The variance of the sky level, used for estimating uncertainty on the
               measured shape; default 0.
    @param[in] *shear_est A string indicating the desired method of PSF correction: REGAUSS,
               LINEAR, BJ, or KSB; default REGAUSS.
    @param[in] flags A flag determining various aspects of the shape measurement process (only
               necessary for REGAUSS); default 0xe.
    @param[in] guess_sig_gal Optional argument with an initial guess for the Gaussian sigma of
               the galaxy, default 5.0 (pixels).
    @param[in] guess_sig_PSF Optional argument with an initial guess for the Gaussian sigma of
               the PSF, default 3.0 (pixels).
    @param[in] precision The convergence criterion for the moments; default 1e-6.
    @param[in] guess_x_centroid An initial guess for the x component of the object centroid (useful
               in case it is not located at the center, which is the default assumption).
    @param[in] guess_y_centroid An initial guess for the y component of the object centroid (useful
               in case it is not located at the center, which is the default assumption).
    @param[in] strict If True (default), then there will be a run-time exception if shear
               estimation fails.  If set to False, then information about failures will be silently
               stored in the output HSMShapeData object.
    @return A HSMShapeData object containing the results of shape measurement.
    """
    gal_image_view = gal_image.view()
    PSF_image_view = PSF_image.view()
    try:
        result = _galsim._EstimateShearHSMView(gal_image_view, PSF_image_view, sky_var = sky_var,
                                               shear_est = shear_est, flags = flags,
                                               guess_sig_gal = guess_sig_gal,
                                               guess_sig_PSF = guess_sig_PSF,
                                               precision = precision, guess_x_centroid =
                                               guess_x_centroid, guess_y_centroid =
                                               guess_y_centroid)
    except RuntimeError as err:
        if (strict == True):
            raise err
        else:
            result = _galsim.HSMShapeData()
            result.error_message = err.message
    return result

def FindAdaptiveMom(object_image, guess_sig = 5.0, precision = 1.0e-6, guess_x_centroid = -1000.0,
                    guess_y_centroid = -1000.0, strict = True):
    """
    @brief Measure adaptive moments of an object.

    The key result is the best-fit elliptical Gaussian to the object, which is computed iteratively
    by initially guessing a circular Gaussian that is used as a weight function, computing the
    weighted moments, recomputing the moments using the result of the previous step as the weight
    function, and so on until the moments that are measured are the same as those used for the
    weight function.  FindAdaptiveMom can be used either as a free function, or as a method of the
    ImageView class.  Example usage:

    @code
    my_gaussian = galsim.Gaussian(flux = 1.0, sigma = 1.0)
    my_gaussian_image = my_gaussian.draw(dx = 0.2)
    my_moments = galsim.FindAdaptiveMom(my_gaussian_image)
    @endcode

    OR
    
    @code
    my_moments = my_gaussian_image.FindAdaptiveMom()
    @endcode

    Assuming a successful measurement, the most relevant pieces of information are
    my_moments.moments_sigma, which is |det(M)|^(1/4) [=sigma for a circular Gaussian] and
    my_moments.observed_shape, which is a C++ Shear.  Methods of the Shear class can be used to
    get the distortion (e1, e2) = (a^2-b^2)/(a^2+b^2), i.e. my_moments.observed_shape.getE1() -- or,
    to get the shear g, the conformal shear eta, and so on.

    Note that the method will fail if the object is too point-like to easily fit an elliptical
    Gaussian; when running on batches of many galaxies, it may be preferable to set strict=False and
    catch errors explicitly, i.e.

    @code
    n_image = 100
    n_fail = 0
    for i=0, range(n_image):
        ...some code defining this_image for index i...
        result = this_image.FindAdaptiveMom(strict = False)
        if result.error_message != "":
            n_fail += 1
    print "Number of failures: ", n_fail
    @endcode

    @param[in] object_image The Image or ImageView for the object being measured.
    @param[in] guess_sig Optional argument with an initial guess for the Gaussian sigma of
               the object, default 5.0 (pixels).
    @param[in] precision The convergence criterion for the moments; default 1e-6.
    @param[in] guess_x_centroid An initial guess for the x component of the object centroid (useful
               in case it is not located at the center, which is the default assumption).
    @param[in] guess_y_centroid An initial guess for the y component of the object centroid (useful
               in case it is not located at the center, which is the default assumption).
    @param[in] strict If True (default), then there will be a run-time exception when moment
               measurement fails.  If set to False, then information about failures will be silently
               stored in the output HSMShapeData object.
    @return A HSMShapeData object containing the results of moment measurement.
    """
    object_image_view = object_image.view()
    try:
        result = _galsim._FindAdaptiveMomView(object_image_view, guess_sig = guess_sig, precision =
                                              precision, guess_x_centroid = guess_x_centroid,
                                              guess_y_centroid = guess_y_centroid)
    except RuntimeError as err:
        if (strict == True):
            raise err
        else:
            result = _galsim.HSMShapeData()
            result.error_message = err.message
    return result

# make FindAdaptiveMom a method of Image and ImageView classes
for Class in _galsim.ImageView.itervalues():
    Class.FindAdaptiveMom = FindAdaptiveMom

for Class in _galsim.Image.itervalues():
    Class.FindAdaptiveMom = FindAdaptiveMom

del Class # cleanup public namespace
