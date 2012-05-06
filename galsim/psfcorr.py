from . import _galsim

def EstimateShearHSM(gal_image, PSF_image, sky_var = 0.0, shear_est = "REGAUSS", flags = 0xe,
                     guess_sig_gal = 5.0, guess_sig_PSF = 3.0, precision = 1.0e-6, strict = True):
    """@brief PSF correction method from HSM.

    Carry out PSF correction using one of the methods of the HSM package to estimate shears.

    Parameters
    ----------
    @param[in] gal_image  The Image of the galaxy being measured.
    @param[in] PSF_image The ImageView for the PSF
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
    @param[in] strict If True (default), then there will be a run-time exception when moment
               measurement fails.  If set to False, then information about failures will be silently
               stored in the output HSMShapeData object.
    @return A HSMShapeData object containing the results of shape measurement. 
    """
    gal_image_view = gal_image.view()
    PSF_image_view = PSF_image.view()
    result = _galsim._EstimateShearHSMView(gal_image_view, PSF_image_view, sky_var, shear_est, flags,
                                           guess_sig_gal, guess_sig_PSF, precision)
    if (strict == True and len(result.error_message) > 0):
        raise RuntimeError(result.error_message)
    return result

def FindAdaptiveMom(object_image, guess_sig = 5.0, precision = 1.0e-6, strict = True):
    """@brief Measure adaptive moments of an object.

    The key result is the best-fit elliptical Gaussian to the object, which is computed by initially
    guessing a circular Gaussian that is used as a weight function, computing the weighted moments,
    recomputing the moments using the result of the previous step as the weight function, and so on
    until the moments that are measured are the same as those used for the weight function.
    
    @param[in] object_image The image for the object being measured.
    @param[in] guess_sig Optional argument with an initial guess for the Gaussian sigma of
               the object, default 5.0 (pixels).
    @param[in] precision The convergence criterion for the moments; default 1e-6.
    @param[in] strict If True (default), then there will be a run-time exception when moment
               measurement fails.  If set to False, then information about failures will be silently
               stored in the output HSMShapeData object.
    @return A HSMShapeData object containing the results of moment measurement.
    """
    object_image_view = object_image.view()
    result = _galsim._FindAdaptiveMomView(object_image_view, guess_sig, precision)
    if (strict == True and len(result.error_message) > 0):
        raise RuntimeError(result.error_message)
    return result

# make FindAdaptiveMom a method of Image classes
for Class in _galsim.ImageView.itervalues():
    Class.FindAdaptiveMom = FindAdaptiveMom

del Class # cleanup public namespace
