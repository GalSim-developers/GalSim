/**
 * @file PSFCorr.h 
 *
 * @brief Contains functions for the hsm shape measurement code, which has functions to carry out
 *        PSF correction and measurement of adaptive moments. 
 *
 * All functions are in the hsm namespace.
 */

/****************************************************************
  Copyright 2003, 2004 Christopher Hirata: original code
  2007, 2009 Rachel Mandelbaum: minor modifications

  For a copy of the license, see COPYING; for more information,
  including contact information, see README.

  This file is part of the meas_shape distribution.

  Meas_shape is free software: you can redistribute it and/or modify it
  under the terms of the GNU General Public License as published by the
  Free Software Foundation, either version 3 of the License, or (at your
  option) any later version.

  Meas_shape is distributed in the hope that it will be useful, but
  WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with meas_shape.  If not, see <http://www.gnu.org/licenses/>.
 *******************************************************************/

/* structures and declarations for shape measurement */

/* object data type */

#include "../Shear.h"
#include "../Image.h"
#include "../Bounds.h"

namespace galsim {
namespace hsm {

    /**
     * @brief Characterizes the shape and other parameters of objects.
     *
     * Describe the hsm shape-related parameters of some object (usually galaxy) before and after
     * PSF correction.  All ellipticities are defined as (1-q^2)/(1+q^2), with the 1st component
     * aligned with the pixel grid and the 2nd aligned at 45 degrees with respect to it.  There are
     * two choices for measurement type: 'e' = Bernstein & Jarvis (2002) ellipticity, 'g' = shear
     * estimator = shear*responsivity 
     */
    struct ObjectData 
    {
        double x0; ///< x centroid position within the postage stamp
        double y0; ///< y centroid position within the postage stamp
        double sigma; ///< size parameter
        double flux; ///< total flux
        double e1; ///< first ellipticity component
        double e2; ///< second ellipticity component
        double responsivity; ///< responsivity of ellipticity estimator 
        char meas_type; ///< type of measurement (see function description) 
        double resolution; ///< resolution factor (0=unresolved, 1=resolved) 
    };
  
    /* rectangular image type */
  
    /**
     * @brief Represents an image in the way that the hsm code expects to see.
     *
     * The hsm representation of an image of some object, with arbitrary pixel indexing, the image
     * itself, and a mask image. The mask image indicates which pixels to use (1) and which to
     * ignore (0).  All values must be 0 or 1.  While the mask functionality underwent basic testing
     * to weed out obvious mistakes, it was not used for the science that came out of the hsm code,
     * so it could conceivably have some subtle bugs. 
     */
    struct RectImage 
    {
        long xmin; ///< Lower x boundary for image
        long xmax; ///< Upper x boundary for image
        long ymin; ///< Lower y boundary for image
        long ymax; ///< Upper y boundary for image
        double **image; ///< The actual image
        int **mask; ///< The mask image 
    };

    /**
     * @brief Struct containing information about the shape of an object.
     *
     * This hsm representation of an object shape contains two Shear objects, one for the observed
     * shape and one after PSF correction.  It also contains information about what PSF correction
     * was used; if no PSF correction was carried out and only the observed moments were measured,
     * the PSF correction method will be 'None'.
     */
    struct HSMShapeData
    {
        /// @brief galsim::Bounds object describing the image of the object
        Bounds<int> image_bounds;

        // Now put information about moments measurement

        /// @brief Status after measuring adaptive moments; -1 indicates no attempt to measure them
        int moments_status;

        /// @brief galsim::Shear object representing the observed shape
        Shear observed_shape;

        /// @brief Size sigma = (det M)^(1/4) from the adaptive moments, in units of pixels; -1 if
        /// not measured
        float moments_sigma;

        /// @brief Total image intensity for best-fit elliptical Gaussian from adaptive moments;
        /// note that this is not flux, since flux = (total image intensity)*(pixel scale)^2
        float moments_amp;

        /// @brief Centroid of best-fit elliptical Gaussian
        Position<double> moments_centroid;

        /// @brief The weighted radial fourth moment of the image
        double moments_rho4;

        /// @brief Number of iterations needed to get adaptive moments; 0 if not measured
        int moments_n_iter;

        // Then put information about PSF correction

        /// @brief Status after carrying out PSF correction; -1 indicates no attempt to do so
        int correction_status;

        /// @brief galsim::Shear object representing the PSF-corrected shape
        Shear corrected_shape;

        /// @brief Shape measurement uncertainty sigma_gamma (not sigma_e) per component
        float corrected_shape_err;

        /// @brief String indicating PSF-correction method; "None" if PSF correction was not done
        std::string correction_method;

        /// @brief Resolution factor R_2; 0 indicates object is consistent with a PSF, 1 indicates
        /// perfect resolution; default -1
        float resolution_factor;

        /// @brief Constructor, setting defaults
    HSMShapeData() : image_bounds(galsim::Bounds<int>()), moments_status(-1),
            observed_shape(galsim::Shear()), moments_sigma(-1.), moments_amp(-1.),
            moments_centroid(galsim::Position<double>(0.,0.)), moments_rho4(-1.), moments_n_iter(0),
            correction_status(-1), corrected_shape(galsim::Shear()), corrected_shape_err(-1.),
            correction_method("None"), resolution_factor(-1.)
        {}

        /// @brief get observed Mxx from e1, e2, sigma
        double getMxx() {
            double A = (1.0 + observed_shape.getE1()) / (1.0 - observed_shape.getE1());
            return (A*moments_sigma*moments_sigma) / 
                std::sqrt(A - 0.25*(A+1)*(A+1)*observed_shape.getE2()*observed_shape.getE2());
        }

        /// @brief get observed Myy from e1, e2, sigma
        double getMyy() {
            double A = (1.0 + observed_shape.getE1()) / (1.0 - observed_shape.getE1());
            return (moments_sigma*moments_sigma) / 
                std::sqrt(A - 0.25*(A+1)*(A+1)*observed_shape.getE2()*observed_shape.getE2());
        }

        /// @brief get observed Mxy from e1, e2, sigma
        double getMxy() {
            return 0.5*observed_shape.getE2()*(getMxx() + getMyy());
        }
    };

    /* functions that the user will want to call from outside */

    /**
     * @brief Carry out PSF correction directly using ImageViews.
     *
     * A template function to carry out one of the multiple possible methods of PSF correction using
     * the HSM package, directly accessing the input ImageViews.  The input arguments get repackaged
     * into RectImage and ObjectData structs before calling general_shear_estimator.  Results for
     * the shape measurement are returned as HSMShapeData.  There are two arguments that have
     * default values, namely shear_est (the type of shear estimator) and flags (for the REGAUSS
     * method only).
     *
     * @param[in] gal_image The ImageView for the galaxy being measured
     * @param[in] PSF_image The ImageView for the PSF
     * @param[in] sky_var The variance of the sky level, used for estimating uncertainty on the
     *            measured shape; default 0.
     * @param[in] *shear_est A string indicating the desired method of PSF correction: REGAUSS,
     *            LINEAR, BJ, or KSB; default REGAUSS.
     * @param[in] flags A flag determining various aspects of the shape measurement process (only
     *            necessary for REGAUSS); default 0xe.
     * @param[in] guess_sig_gal Optional argument with an initial guess for the Gaussian sigma of
     *            the galaxy, default 5.0 (pixels).
     * @param[in] guess_sig_PSF Optional argument with an initial guess for the Gaussian sigma of
     *            the PSF, default 3.0 (pixels).
     * @param[in] precision The convergence criterion for the moments; default 1e-6.
     * @return A HSMShapeData object containing the results of shape measurement. 
     */
    template <typename T>
        HSMShapeData EstimateShearHSMView(const ImageView<T> &gal_image, const ImageView<T> &PSF_image,
                                          float sky_var = 0.0, const char *shear_est = "REGAUSS",
                                          unsigned long flags = 0xe, double guess_sig_gal = 5.0,
                                          double guess_sig_PSF = 3.0, double precision = 1.0e-6);

    /**
     * @brief Measure the adaptive moments of an object directly using ImageViews.
     *
     * This function repackages the input ImageView in a format that find_ellipmom_2 accepts, in order
     * to iteratively compute the adaptive moments of an object.  The key result is the best-fit
     * elliptical Gaussian to the object, which is computed by initially guessing a circular
     * Gaussian that is used as a weight function, computing the weighted moments, recomputing the
     * moments using the result of the previous step as the weight function, and so on until the
     * moments that are measured are the same as those used for the weight function.  
     *
     * @param[in] object_image The ImageView for the object being measured.
     * @param[in] guess_sig Optional argument with an initial guess for the Gaussian sigma of
     *            the object, default 5.0 (pixels).
     * @param[in] precision The convergence criterion for the moments; default 1e-6.
     * @return A HSMShapeData object containing the results of moment measurement.
     */
    template <typename T>
        HSMShapeData FindAdaptiveMomView(const ImageView<T> &object_image, double guess_sig = 5.0,
                                         double precision = 1.0e-6);

    /**
     * @brief Allocate memory for a RectImage representing the image of some object
     *
     * @param[in] *A The pointer to the RectImage
     * @param[in] xmin The lower x boundary for the image
     * @param[in] xmax The upper x boundary for the imgae
     * @param[in] ymin The lower y boundary for the image
     * @param[in] ymax The upper y boundary for the image
     */
    void allocate_rect_image(RectImage *A, long xmin, long xmax, long ymin, long ymax);
  
    /**
     * @brief De-allocate memory for a RectImage
     * @param[in] *A The pointer to the RectImage
     */
    void deallocate_rect_image(RectImage *A);
  
    /**
     * @brief Carry out PSF correction.
     *
     * Carry out one of the multiple possible methods of PSF correction using the HSM package.
     * Results for the shape measurement are returned by modifying the galaxy data directly.  The
     * flags parameter is only used for the REGAUSS shape measurement method, and is defined as
     * follows: 0x1=recompute galaxy flux by summing unmasked pixels, 0x2=recompute galaxy flux from
     * Gaussian-quartic fit, 0x4=cut off Gaussian approximator at NSIG_RG sigma to save time,
     * 0x8=cut off PSF residual at NSIG_RG2 to save time.    
     * @param[in] *gal_image The RectImage object for the galaxy
     * @param[in] *PSF The RectImage object for the PSF
     * @param[in] *gal_data The ObjectData object for the galaxy
     * @param[in] *PSF_data The ObjectData object for the PSF
     * @param[in] *shear_est A string indicating the desired method of PSF correction: REGAUSS,
     *            LINEAR, BJ, or KSB
     * @param[in] flags A parameter for REGAUSS (typical usage is 0xe).
     * @return A status flag that should be zero if the measurement was successful.
     */
    unsigned int general_shear_estimator(
        RectImage *gal_image, RectImage *PSF, ObjectData *gal_data, ObjectData *PSF_data, 
        char *shear_est, unsigned long flags);

    /**
     * @brief Measure the adaptive moments of an object.
     *
     * This function iteratively computes the adaptive moments of an image, and tells the user the
     * results plus other diagnostic information.  The key result is the best-fit elliptical
     * Gaussian to the object, which is computed by initially guessing a circular Gaussian that is
     * used as a weight function, computing the weighted moments, recomputing the moments using the
     * result of the previous step as the weight function, and so on until the moments that are
     * measured are the same as those used for the weight function.
     * @param[in] *data The RectImage for the object being measured.
     * @param[out] *A The amplitude of the best-fit elliptical Gaussian (total image intensity for
     *             the Gaussian is 2A). 
     * @param[out] *x0 The x centroid of the best-fit elliptical Gaussian
     * @param[out] *y0 The y centroid of the best-fit elliptical Gaussian
     * @param[out] *Mxx The xx component of the moment matrix
     * @param[out] *Mxy The xy component of the moment matrix
     * @param[out] *Myy The yy component of the moment matrix
     * @param[out] *rho4 The weighted radial fourth moment
     * @param[in] epsilon The required level of accuracy
     * @param[out] *num_iter The number of iterations needed to converge
     */
    void find_ellipmom_2(
        RectImage *data, double *A, double *x0, double *y0,
        double *Mxx, double *Mxy, double *Myy, double *rho4, double epsilon, int *num_iter);
  
}}
