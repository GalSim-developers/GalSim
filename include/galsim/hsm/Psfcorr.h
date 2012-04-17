/**
 * @file psfcorr.h 
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
    struct OBJECT_DATA 
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
    struct RECT_IMAGE 
    {
        long xmin; ///< Lower x boundary for image
        long xmax; ///< Upper x boundary for image
        long ymin; ///< Lower y boundary for image
        long ymax; ///< Upper y boundary for image
        double **image; ///< The actual image
        int **mask; ///< The mask image 
    };

    /* functions that the user will want to call from outside */

    /**
     * @brief Allocate memory for a RECT_IMAGE representing the image of some object
     *
     * @param[in] *A The pointer to the RECT_IMAGE
     * @param[in] xmin The lower x boundary for the image
     * @param[in] xmax The upper x boundary for the imgae
     * @param[in] ymin The lower y boundary for the image
     * @param[in] ymax The upper y boundary for the image
     */
    void allocate_rect_image(RECT_IMAGE *A, long xmin, long xmax, long ymin, long ymax);
  
    /**
     * @brief De-allocate memory for a RECT_IMAGE
     * @param[in] *A The pointer to the RECT_IMAGE
     */
    void deallocate_rect_image(RECT_IMAGE *A);
  
    /**
     * @brief Carry out PSF correction.
     *
     * Carry out one of the multiple possible methods of PSF correction using the HSM package.
     * Results for the shape measurement are returned by modifying the galaxy data directly.  The
     * flags parameter is only used for the REGAUSS shape measurement method, and is defined as
     * follows: 0x1=recompute galaxy flux by summing unmasked pixels, 0x2=recompute galaxy flux from
     * Gaussian-quartic fit, 0x4=cut off Gaussian approximator at NSIG_RG sigma to save time,
     * 0x8=cut off PSF residual at NSIG_RG2 to save time.    
     * @param[in] *gal_image The RECT_IMAGE object for the galaxy
     * @param[in] *PSF The RECT_IMAGE object for the PSF
     * @param[in] *gal_data The OBJECT_DATA object for the galaxy
     * @param[in] *PSF_data The OBJECT_DATA object for the PSF
     * @param[in] *shear_est A string indicating the desired method of PSF correction: REGAUSS,
     *            LINEAR, BJ, or KSB
     * @param[in] flags A parameter for REGAUSS, which is hardcoded in meas_shape.cpp to 0xe. 
     * @return A status flag that should be zero if the measurement was successful.
     */
    unsigned int general_shear_estimator(
        RECT_IMAGE *gal_image, RECT_IMAGE *PSF, OBJECT_DATA *gal_data, OBJECT_DATA *PSF_data, 
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
     * @param[in] *data The RECT_IMAGE for the object being measured.
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
        RECT_IMAGE *data, double *A, double *x0, double *y0,
        double *Mxx, double *Mxy, double *Myy, double *rho4, double epsilon, int *num_iter);
  
}}
