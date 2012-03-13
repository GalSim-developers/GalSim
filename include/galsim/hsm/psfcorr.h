/// \file psfcorr.h Contains functions in the hsm namespace, which are
/// required to run the hsm shape measurement and moment estimation code. 

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

  /// Describe the hsm shape-related parameters of some object (usually
  /// galaxy) before and after PSF correction.
  struct OBJECT_DATA 
  {
    double x0; ///< x centroid position within the postage stamp
    double y0; ///< y centroid position within the postage stamp
    double sigma; ///< size parameter
    double flux; ///< total flux
    double e1; ///< ellipticity component aligned with pixel grid
    double e2; ///< x ellipticity component
    double responsivity; ///< responsivity of ellipticity estimator 
    char meas_type; ///< type of measurement: 'e' = Bernstein & Jarvis (2002) ellipticity, 'g' = shear estimator = shear*responsivity
    double resolution; ///< resolution factor (0=unresolved, 1=resolved) 
  };
  
  /* rectangular image type */
  
  /// The hsm representation of an image of some object, with
  /// arbitrary pixel indexing, the image itself, and a mask image.
  struct RECT_IMAGE 
  {
    long xmin; ///< Lower x boundary for image
    long xmax; ///< Upper x boundary for image
    long ymin; ///< Lower y boundary for image
    long ymax; ///< Upper y boundary for image
    double **image; ///< The actual image
    int **mask; ///< A mask image indicating which pixels to use
  };

    /* functions that the user will want to call from outside */

  /// Allocate memory for a RECT_IMAGE representing the image of some
  /// object
  /// \param *A The pointer to the RECT_IMAGE
  /// \param xmin The lower x boundary for the image
  /// \param xmax The upper x boundary for the imgae
  /// \param ymin The lower y boundary for the image
  /// \param ymax The upper y boundary for the image
  void allocate_rect_image(RECT_IMAGE *A, long xmin, long xmax, long ymin, long ymax);
  
  /// De-allocate memory for a RECT_IMAGE
  /// \param *A The pointer to the RECT_IMAGE
  void deallocate_rect_image(RECT_IMAGE *A);
  
  /// Carry out PSF correction.
  //
  /// Carry out one of the multiple possible methods of PSF correction
  /// using the HSM package.  Results for the shape measurement are
  /// returned by modifying the galaxy data directly.
  /// \param *gal_image Input: the RECT_IMAGE object for the galaxy
  /// \param *PSF Input: the RECT_IMAGE object for the PSF
  /// \param *gal_data Input: the OBJECT_DATA object for the galaxy
  /// \param *PSF_data Input: the OBJECT_DATA object for the PSF
  /// \param *shear_est A string indicating the desired method of PSF correction: REGAUSS, LINEAR, BJ, or KSB
  /// \param flags A parameter that is only needed for REGAUSS.  0x1=recompute galaxy flux by summing unmasked pixels, 0x2=recompute galaxy flux from Gaussian-quartic fit, 0x4=cut off Gaussian approximator at NSIG_RG sigma to save time, 0x8=cut off PSF residual at NSIG_RG2 to save time.  meas_shape.cpp now has hardcoded flags==0xe.
  /// \return A status flag that should be zero if the measurement was successful.
  unsigned int general_shear_estimator(
				       RECT_IMAGE *gal_image, RECT_IMAGE *PSF, OBJECT_DATA *gal_data, OBJECT_DATA *PSF_data,
				       char *shear_est, unsigned long flags);
  
  /// Measure the adaptive moments of an object.
  //
  /// This function iteratively computes the adaptive moments of an
  /// image, and tells the user the results plus other diagnostic
  /// information.  The key result is the best-fit elliptical Gaussian
  /// to the object, which is computed by initially guessing a
  /// circular Gaussian that is used as a weight function, computing
  /// the weighted moments, recomputing the moments using the result
  /// of the previous step as the weight function, and so on until the
  /// moments that are measured are the same as those used for the
  /// weight function.
  /// \param *data Input: the RECT_IMAGE for the object being measured.
  /// \param *A Output: the amplitude of the best-fit elliptical Gaussian (defined such that total image intensity for the Gaussian is 2A)
  /// \param *x0 Output: the x centroid of the best-fit elliptical Gaussian
  /// \param *y0 Output: the y centroid of the best-fit elliptical Gaussian
  /// \param *Mxx Output: the xx component of the moment matrix
  /// \param *Mxy Output: the xy component of the moment matrix
  /// \param *Myy Output: the yy component of the moment matrix
  /// \param *rho4 Output: the weighted radial fourth moment
  /// \param epsilon Input: the required level of accuracy
  /// \param *num_iter Output: the number of iterations needed to converge
  void find_ellipmom_2(
		       RECT_IMAGE *data, double *A, double *x0, double *y0,
		       double *Mxx, double *Mxy, double *Myy, double *rho4, double epsilon, int *num_iter);
  
}}
