/****************************************************************
  Copyright 2003, 2004 Christopher Hirata: original code
  2007, 2009, 2010 Rachel Mandelbaum: minor modifications

  For a copy of the license, see LICENSE; for more information,
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

#include <cstring>
#include <string>
#define TMV_DEBUG
#include "TMV.h"
#include "hsm/PSFCorr.h"

//#define DEBUGLOGGING
#ifdef DEBUGLOGGING
#include <fstream>
std::ostream* dbgout = new std::ofstream("debug.out");
int verbose_level = 2;
// There are three levels of verbosity which can be helpful when debugging,
// which are written as dbg, xdbg, xxdbg (all defined in Std.h).
// It's Mike's way to have debug statements in the code that are really easy to turn 
// on and off.
//
// If DEBUGLOGGING is #defined, then these write out to *dbgout, according to the value
// of verbose_level.
// dbg requires verbose_level >= 1
// xdbg requires verbose_level >= 2
// xxdbg requires verbose_level >= 3
//
// If DEBUGLOGGING is not defined, the all three becomes just `if (false) std::cerr`,
// so the compiler parses the statement fine, but trivially optimizes the code away,
// so there is no efficiency hit from leaving them in the code.
#endif


namespace galsim {
namespace hsm {

    template <typename T, typename U>
    unsigned int general_shear_estimator(
        ConstImageView<T> gal_image, ConstImageView<int> gal_mask,
        ConstImageView<U> PSF_image, ConstImageView<int> PSF_mask,
        ObjectData& gal_data, ObjectData& PSF_data, const std::string& shear_est,
        unsigned long flags, boost::shared_ptr<HSMParams> hsmparams);

    template <typename T>
    void find_ellipmom_2(
        ConstImageView<T> data, ConstImageView<int> mask, double& A, double& x0, double& y0,
        double& Mxx, double& Mxy, double& Myy, double& rho4, double epsilon, int& num_iter,
        boost::shared_ptr<HSMParams> hsmparams);

    // Carry out PSF correction directly using ImageViews, repackaging for general_shear_estimator.
    template <typename T, typename U>
    CppHSMShapeData EstimateShearHSMView(
        const ImageView<T>& gal_image, const ImageView<U>& PSF_image,
        const ImageView<int> &gal_mask_image,
        float sky_var, const char* shear_est, const std::string& recompute_flux,
        double guess_sig_gal,
        double guess_sig_PSF, double precision,
        double guess_x_centroid, double guess_y_centroid,
        boost::shared_ptr<HSMParams> hsmparams) 
    {
        // define variables, create output CppHSMShapeData struct, etc.
        CppHSMShapeData results;
        ObjectData gal_data, PSF_data;
        double amp, m_xx, m_xy, m_yy;
        unsigned long flags=0;

        if (!hsmparams.get()) hsmparams = hsm::default_hsmparams;

        dbg<<"Start EstimateShearHSMView"<<std::endl;
        dbg<<"Setting defaults and so on before calling general_shear_estimator"<<std::endl;
        // Set defaults etc. and pass to general_shear_estimator
        if (guess_x_centroid != -1000.0) {
            gal_data.x0 = guess_x_centroid;
        } else {
            gal_data.x0 = 0.5*(gal_image.getXMin() + gal_image.getXMax());
        }
        if (guess_y_centroid != -1000.0) {
            gal_data.y0 = guess_y_centroid;
        } else {
            gal_data.y0 = 0.5*(gal_image.getYMin() + gal_image.getYMax());
        }
        gal_data.sigma = guess_sig_gal;

        PSF_data.x0 = 0.5*(PSF_image.getXMin() + PSF_image.getXMax());
        PSF_data.y0 = 0.5*(PSF_image.getYMin() + PSF_image.getYMax());
        PSF_data.sigma = guess_sig_PSF;

        m_xx = guess_sig_gal*guess_sig_gal;
        m_yy = m_xx;
        m_xy = 0.0;

        // Need to set flag values for general_shear_estimator
        if (hsmparams->nsig_rg > 0) flags |= 0x4;
        if (hsmparams->nsig_rg2 > 0) flags |= 0x8;
        if (recompute_flux == "FIT") flags |= 0x2;
        else if (recompute_flux == "SUM") flags |= 0x1;
        else if (recompute_flux != "NONE") {
            throw HSMError("Unknown value for recompute_flux parameter!");
        }

        // call general_shear_estimator
        results.image_bounds = gal_image.getBounds();
        results.correction_method = shear_est;
        Image<int> PSF_mask(PSF_image.getBounds());
        PSF_mask.fill(1);
        ConstImageView<T> gal_image_cview = gal_image;
        ConstImageView<U> PSF_image_cview = PSF_image;
        ConstImageView<int> gal_mask_view = gal_mask_image;
        ConstImageView<int> PSF_mask_view = PSF_mask.view();

        dbg<<"About to get moments using find_ellipmom_2"<<std::endl;
        find_ellipmom_2(gal_image_cview, gal_mask_view, amp, gal_data.x0,
                        gal_data.y0, m_xx, m_xy, m_yy, results.moments_rho4,
                        precision, results.moments_n_iter, hsmparams);
        // repackage outputs to the output CppHSMShapeData struct
        dbg<<"Repackaging find_ellipmom_2 results"<<std::endl;
        results.moments_amp = 2.0*amp;
        results.moments_sigma = std::pow(m_xx*m_yy-m_xy*m_xy, 0.25);
        results.observed_shape.setE1E2((m_xx-m_yy)/(m_xx+m_yy), 2.*m_xy/(m_xx+m_yy));
        results.moments_status = 0;

        // and if that worked, try doing PSF correction
        gal_data.sigma = results.moments_sigma;
        dbg<<"About to get shear using general_shear_estimator"<<std::endl;
        results.correction_status = general_shear_estimator(
            gal_image_cview, gal_mask_view, PSF_image_cview, PSF_mask_view,
            gal_data, PSF_data, shear_est, flags, hsmparams);
        dbg<<"Repackaging general_shear_estimator results"<<std::endl;

        results.meas_type = gal_data.meas_type;
        if (gal_data.meas_type == 'e') {
            results.corrected_e1 = gal_data.e1;
            results.corrected_e2 = gal_data.e2;
        } else if (gal_data.meas_type == 'g') {
            results.corrected_g1 = gal_data.e1;
            results.corrected_g2 = gal_data.e2;
        } else {
            throw HSMError("Unknown shape measurement type!\n");
        }

        if (results.correction_status != 0) {
            throw HSMError("PSF correction status indicates failure!\n");
        }

        results.corrected_shape_err = std::sqrt(4. * M_PI * sky_var) * gal_data.sigma /
            (gal_data.resolution * gal_data.flux);
        results.moments_sigma = gal_data.sigma;
        results.moments_amp = gal_data.flux;
        results.resolution_factor = gal_data.resolution;

        if (results.resolution_factor <= 0.) {
            throw HSMError("Unphysical situation: galaxy convolved with PSF is smaller than PSF!\n");
        }

        dbg<<"Exiting EstimateShearHSMView"<<std::endl;
        return results;
    }

    // Measure the adaptive moments of an object directly using ImageViews, repackaging for 
    // find_ellipmom_2.
    template <typename T>
    CppHSMShapeData FindAdaptiveMomView(
        const ImageView<T>& object_image, const ImageView<int> &object_mask_image, 
        double guess_sig, double precision, double guess_x_centroid,
        double guess_y_centroid, boost::shared_ptr<HSMParams> hsmparams) 
    {
        dbg<<"Start FindAdaptiveMomView"<<std::endl;
        dbg<<"Setting defaults and so on before calling find_ellipmom_2"<<std::endl;
        // define variables, create output CppHSMShapeData struct, etc.
        CppHSMShapeData results;
        double amp, m_xx, m_xy, m_yy;

        if (!hsmparams.get()) hsmparams = hsm::default_hsmparams;

        // set some values for initial guesses
        if (guess_x_centroid != -1000.0) {
            results.moments_centroid.x = guess_x_centroid;
        } else {
            results.moments_centroid.x = 0.5*(object_image.getXMin() + object_image.getXMax());
        }
        if (guess_y_centroid != -1000.0) {
            results.moments_centroid.y = guess_y_centroid;
        } else {
            results.moments_centroid.y = 0.5*(object_image.getYMin() + object_image.getYMax());
        }
        m_xx = guess_sig*guess_sig;
        m_yy = m_xx;
        m_xy = 0.0;

        // call find_ellipmom_2
        results.image_bounds = object_image.getBounds();
        ConstImageView<T> object_image_cview = object_image;
        ConstImageView<int> object_mask_view = object_mask_image;
        try {
            dbg<<"About to get moments using find_ellipmom_2"<<std::endl;
            find_ellipmom_2(object_image_cview, object_mask_view, amp, results.moments_centroid.x,
                            results.moments_centroid.y, m_xx, m_xy, m_yy, results.moments_rho4,
                            precision, results.moments_n_iter, hsmparams);
            dbg<<"Repackaging find_ellipmom_2 results"<<std::endl;

            // repackage outputs from find_ellipmom_2 to the output CppHSMShapeData struct
            results.moments_amp = 2.0*amp;
            results.moments_sigma = std::pow(m_xx*m_yy-m_xy*m_xy, 0.25);
            results.observed_shape.setE1E2((m_xx-m_yy)/(m_xx+m_yy), 2.*m_xy/(m_xx+m_yy));
            results.moments_status = 0;
        }
        catch (char *err_msg) {
            results.error_message = err_msg;
            results.moments_status = 1;
            results.moments_centroid.x = 0.0;
            results.moments_centroid.y = 0.0;
            results.moments_rho4 = -1.0;
            results.moments_n_iter = 0;
            dbg<<"Caught an error: "<<err_msg<<std::endl;
            throw HSMError(err_msg);
        }

        dbg<<"Exiting FindAdaptiveMomView"<<std::endl;
        return results;
    }

    /* fourier_trans_1
     * *** FOURIER TRANSFORMS A DATA SET WITH LENGTH A POWER OF 2 ***
     *
     * This is a Fourier transform routine.  It has the same calling
     * interface as Numerical Recipes four1 and uses the same algorithm
     * as that routine.  This function is slightly faster than NR four1
     * because we have minimized the use of expensive array look-ups.
     *
     * Replaces data[1..2*nn] by its discrete Fourier transform, if
     * isign is input as 1; or replaces data[1..2*nn] by nn times its
     * inverse discrete Fourier transform, if isign is input as -1.
     * data is a complex array of length nn.
     *
     */

    void fourier_trans_1(double *data, long nn, int isign) 
    {

        double *data_i, *data_i1;
        double *data_j, *data_j1;
        double temp, theta, sintheta, oneminuscostheta;
        double wr, wi, wtemp;
        double tempr1, tempr2, tempr, tempi;

        unsigned long ndata; /* = 2*nn */
        unsigned long lcurrent; /* length of current FFT; will range from 2..n */
        unsigned long i,j,k,m,istep;

        ndata = (unsigned long)nn << 1;

        /* Bit reversal */
        data_i = data;
        for(i=0;i<(unsigned long)nn;i++) {

            /* Here we set data_j equal to data_null plus twice the bit-reverse of i */
            j=0;
            k=i;
            for(m=ndata>>2;m>=1;m>>=1) {
                if (k & 1) j+=m;
                k >>= 1;
            }

            /* If i<j, swap the i and j complex elements of data
             * Notice that these are the (2i,2i+1) and (2j,2j+1)
             * real elements.
             */
            if (i<j) {
                data_j = data + (j<<1);
                temp = *data_i; *data_i = *data_j; *data_j = temp;
                data_i++; data_j++;
                temp = *data_i; *data_i = *data_j; *data_j = temp;
            } else {
                data_i++;
            }

            /* Now increment data_i so it points to data[2i+1]; this is
             * important when we start the next iteration.
             */
            data_i++;
        }

        /* Now do successive FFTs */
        for(lcurrent=2;lcurrent<ndata;lcurrent<<=1) {

            /* Find the angle between successive points and its trig
             * functions, the sine and 1-cos. (Use 1-cos for stability.)
             */
            theta = 2.*M_PI/lcurrent * isign;
            sintheta = std::sin(theta);
            oneminuscostheta = std::sin(0.5*theta);
            oneminuscostheta = 2.0*oneminuscostheta*oneminuscostheta;

            /* FFT the individual length-lcurrent segments */
            wr = 1.0;
            wi = 0.0;
            istep = lcurrent<<1;
            for(m=0;m<lcurrent;m+=2) {
                for(i=m;i<ndata;i+=istep) {
                    /* Set the data pointers so we don't need to do
                     * time-consuming array lookups.
                     */
                    data_j1=data_j = (data_i1=data_i = data + i) + lcurrent;
                    data_i1++;
                    data_j1++;

                    /* Now apply Danielson-Lanczos formula */
                    tempr1 = wr*(*data_j);
                    tempr2 = wi*(*data_j1);
                    tempr = tempr1 - tempr2;
                    tempi = (wr+wi)*((*data_j)+(*data_j1)) - tempr1 - tempr2;
                    /*
                     * at this point, tempr + i*tempi is equal to the product of
                     * the jth complex array element and w.
                     */
                    *data_j = (*data_i) - tempr;
                    *data_j1 = (*data_i1) - tempi;
                    *data_i += tempr;
                    *data_i1 += tempi;

                }

                /* Now increment trig recurrence */
                wr -= (wtemp=wr)*oneminuscostheta + wi*sintheta;
                wi += wtemp*sintheta - wi*oneminuscostheta;
            }
        }
    }

    /* qho1d_wf_1
     * *** COMPUTES 1D QHO WAVE FUNCTIONS ***
     *
     * The QHO wavefunctions psi_0 ... psi_Nmax are computed
     * at points x=xmin ... xmin+xstep*(nx-1).  The ground
     * state is a Gaussian centered at x=0 of width sigma,
     * i.e. ~ exp(-x^2/(2*sigma^2)).  [NOT 4*sigma^2 as is
     * usual in QM, since we are using these wave functions
     * for classical image measurement.]
     *
     * Arguments:
     *   nx: number of x-coordinates at which to compute wavefunction
     *   xmin: minimum x at which to compute wavefunction
     *   xstep: change in x at each successive point
     *   Nmax: maximum-order wavefunction to calculate
     *   sigma: width of ground state
     * > psi: result; psi[n][j] = n th order wavefunction evaluated
     *      at x = xmin+xstep*j.
     */

    void qho1d_wf_1(long nx, double xmin, double xstep, long Nmax, double sigma, 
                    tmv::Matrix<double>& psi) 
    {

        double beta, beta2__2, norm0;
        double coef1, coef2;
        double x;
        long j,n;

#ifdef N_CHECKVAL
        if (nx<=0) {
            throw HSMError("Error: nx<=0 in qho1d_wf_1\n");
        }
        if (Nmax<0) {
            throw HSMError("Error: Nmax<0 in qho1d_wf_1\n");
        }
#endif

        /* Set up constants */
        beta = 1./sigma;
        beta2__2 = 0.5*beta*beta;

        /* Get ground state */
        norm0 = 0.75112554446494248285870300477623 * std::sqrt(beta);
        x=xmin;
        for(j=0;j<nx;j++) {
            psi[0][j] = norm0 * std::exp( -beta2__2 * x*x );
            if (Nmax>=1) psi[1][j] = std::sqrt(2.) * psi[0][j] * beta * x;
            x += xstep;
        }

        /* Return if we don't need 2nd order or higher wavefunctions */
        if (Nmax<2) return;

        /* Use recursion relation for QHO wavefunctions to generate
         * the higher-order functions
         */
        for(n=1;n<Nmax;n++) {

            /* Recursion relation coefficients */
            coef1 = beta * std::sqrt( 2. / (n+1.) );
            coef2 = -std::sqrt( (double)n / (n+1.) );

            x=xmin;
            for(j=0;j<nx;j++) {

                /* The recurrance */
                psi[n+1][j] = coef1 * x * psi[n][j] + coef2 * psi[n-1][j];         

                x += xstep; /* Increment x */
            } /* End j loop */
        } /* End n loop */

    }

    /* find_mom_1
     * *** FINDS MOMENTS OF AN IMAGE ***
     *
     * Computes the shapelet moments of an image by integration of
     * int f(x,y) psi_m(x) psi_n(y) for the relevant weight
     *
     * Arguments:
     *   data: ImageView structure containing the image to be measured
     * > moments: moments(m,n) is the m:n coefficient
     *   max_order: maximum order of moments to compute
     *   x0: center around which to compute moments (x-coordinate)
     *   y0: " (y-coordinate)
     *   sigma: width of Gaussian to measure image
     */
    template <typename T>
    void find_mom_1(
        ConstImageView<T> data, ConstImageView<int> mask,
        tmv::Matrix<double>& moments, int max_order, 
        double x0, double y0, double sigma)
    {

        /* Setup */
        int xmin = data.getXMin();
        int xmax = data.getXMax();
        int ymin = data.getYMin();
        int ymax = data.getYMax();
        int nx = xmax-xmin+1;
        int ny = ymax-ymin+1;
        tmv::Matrix<double> psi_x(max_order+1,nx);
        tmv::Matrix<double> psi_y(max_order+1,ny);

        /* Compute wavefunctions */
        qho1d_wf_1(nx, (double)xmin - x0, 1., max_order, sigma, psi_x);
        qho1d_wf_1(ny, (double)ymin - y0, 1., max_order, sigma, psi_y);

        /* Now let's compute moments -- outer loop is over (m,n) */
        for(int m=0;m<=max_order;m++) for(int n=0;n<=max_order-m;n++) {

            /* Initialize moments(m,n), then loop over map */
            moments(m,n) = 0;
            for(int y=ymin;y<=ymax;y++) for(int x=xmin;x<=xmax;x++) {
                if (mask(x,y)) {

                    /* Moment "integral" (here simply a finite sum) */
                    moments(m,n) += data(x,y) * psi_x(m,x-xmin) * psi_y(n,y-ymin);

                } /* End mask condition */
            } /* End (x,y) loop */
        } /* End (m,n) loop */
    }

    /* find_mom_2
     * *** FINDS ADAPTIVE CIRCULAR MOMENTS OF AN IMAGE ***
     *
     * Computes the center, 1sigma radius, and moments of an image.  "Guesses"
     * must be given for x0, y0, and sigma.
     *
     * Arguments:
     *   data: ImageView structure containing the image to be measured
     * > moments: moments(m,n) is the m:n coefficient
     *   max_order: maximum order of moments to compute
     * > x0: Gaussian-weighted centroid (x-coordinate)
     * > y0: " (y-coordinate)
     * > sigma: width of Gaussian to measure image (best fit 1sigma)
     *   epsilon: accuracy (in x0, y0, and sigma as a fraction of sigma.
     *      The value of sigma used for the convergence criterion is the
     *      minimum of the "guessed" value and the "current" value.)
     * > num_iter: number of iterations required for convergence
     */

    template <typename T>
    void find_mom_2(
        ConstImageView<T> data, ConstImageView<int> mask,
        tmv::Matrix<double>& moments, int max_order,
        double& x0, double& y0, double& sigma, double epsilon, int& num_iter,
        boost::shared_ptr<HSMParams> hsmparams) 
    {

        double sigma0 = sigma;
        double convergence_factor = 1; /* Ensure at least one iteration. */

        num_iter = 0;
        tmv::Matrix<double> iter_moments(hsmparams->adapt_order+1,hsmparams->adapt_order+1);

#ifdef N_CHECKVAL
        if (epsilon <= 0) {
            throw HSMError("Error: epsilon out of range in find_mom_2.\n");
        }
#endif

        /* Iterate until we converge */
        while(convergence_factor > epsilon) {

            /* Get moments */
            find_mom_1(data,mask,iter_moments,hsmparams->adapt_order,x0,y0,sigma);

            /* Get updates to weight function */
            double dx     = 1.414213562373 * iter_moments(1,0) / iter_moments(0,0);
            double dy     = 1.414213562373 * iter_moments(0,1) / iter_moments(0,0);
            double dsigma = 0.7071067811865
                * (iter_moments(2,0)+iter_moments(0,2)) / iter_moments(0,0);

            if (dx     >  hsmparams->bound_correct_wt) dx     =  hsmparams->bound_correct_wt;
            if (dx     < -hsmparams->bound_correct_wt) dx     = -hsmparams->bound_correct_wt;
            if (dy     >  hsmparams->bound_correct_wt) dy     =  hsmparams->bound_correct_wt;
            if (dy     < -hsmparams->bound_correct_wt) dy     = -hsmparams->bound_correct_wt;
            if (dsigma >  hsmparams->bound_correct_wt) dsigma =  hsmparams->bound_correct_wt;
            if (dsigma < -hsmparams->bound_correct_wt) dsigma = -hsmparams->bound_correct_wt;

            /* Convergence */
            convergence_factor = std::abs(dx)>std::abs(dy)? std::abs(dx): std::abs(dy);
            if (std::abs(dsigma)>convergence_factor) convergence_factor = std::abs(dsigma);
            if (sigma<sigma0) convergence_factor *= sigma0/sigma;

            /* Update numbers */
            x0    += dx     * sigma;
            y0    += dy     * sigma;
            sigma += dsigma * sigma;

            if (++num_iter > hsmparams->max_mom2_iter) {
                convergence_factor = 0.;
                num_iter = hsmparams->num_iter_default;
                throw HSMError("Warning: too many iterations in find_mom_2.\n");
            }
        }

        /* Now compute all of the moments that we want to return */
        find_mom_1(data,mask,moments,max_order,x0,y0,sigma);
    }

    /* find_ellipmom_1
     * *** FINDS ELLIPTICAL GAUSSIAN MOMENTS OF AN IMAGE ***
     *
     * Returns the parameters:
     * A = int f(x,y) w(x,y)
     * B_i = int (r_i-r0_i) f(r) w(r)
     * C_ij = int (r_i-r0_i) (r_j-r0_j) f(r) w(r)
     * rho4 = int rho^4 f(r) w(r)
     *
     * where w(r) = exp(-rho^2/2), rho^2 = (x-x0) * M^{-1} * (y-y0),
     * M = adaptive covariance matrix, and note that the weight may be set to zero for rho^2 >
     * hsmparams->max_moment_nsig2 if that parameter is defined.
     *
     * Arguments:
     *   data: the input image (ImageView format)
     *   x0: weight centroid (x coordinate)
     *   y0: weight centroid (y coordinate)
     *   Mxx: xx element of adaptive covariance
     *   Mxy: xy element of adaptive covariance
     *   Myy: yy element of adaptive covariance
     * > A: amplitude
     * > Bx: weighted centroid displacement (x)
     * > By: weighted centroid displacement (y)
     * > Cxx: weighted covariance (xx)
     * > Cxy: weighted covariance (xy)
     * > Cyy: weighted covariance (yy)
     * > rho4w: weighted radial fourth moment
     */

    template <typename T>
    void find_ellipmom_1(
        ConstImageView<T> data, ConstImageView<int> mask, double x0, double y0, double Mxx,
        double Mxy, double Myy, double& A, double& Bx, double& By, double& Cxx,
        double& Cxy, double& Cyy, double& rho4w, boost::shared_ptr<HSMParams> hsmparams) 
    {
        //long npix=0;
        long xmin = data.getXMin();
        long xmax = data.getXMax();
        long ymin = data.getYMin();
        long ymax = data.getYMax();
        dbg<<"Entering find_ellipmom_1 with Mxx, Myy, Mxy: "<<Mxx<<" "<<Myy<<" "<<Mxy<<std::endl;
        dbg<<"x0, y0: "<<x0<<" "<<y0<<std::endl;
        dbg<<"xmin, xmax: "<<xmin<<" "<<xmax<<std::endl;

        /* Compute M^{-1} for use in computing weights */
        double detM = Mxx * Myy - Mxy * Mxy;
        if (detM<=0 || Mxx<=0 || Myy<=0) {
            throw HSMError("Error: non positive definite adaptive moments!\n");
        }
        double Minv_xx    =  Myy/detM;
        double TwoMinv_xy = -Mxy/detM * 2.0;
        double Minv_yy    =  Mxx/detM;

        /* Generate Minv_xx__x_x0__x_x0 array */
        tmv::Vector<double> Minv_xx__x_x0__x_x0(xmax-xmin+1);
        for(long x=xmin;x<=xmax;x++) Minv_xx__x_x0__x_x0[x-xmin] = Minv_xx*(x-x0)*(x-x0);

        /* Now let's initialize the outputs and then sum
         * over all the unmasked pixels
         */
        A = Bx = By = Cxx = Cxy = Cyy = rho4w = 0.;
        /* Use these pointers to speed up referencing arrays */
        const int* maskptr = mask.getData();
        const T* imageptr = data.getData();
        const int maskstride = mask.getStride() - (xmax - xmin + 1);
        const int datastride = data.getStride() - (xmax - xmin + 1);
        for(long y=ymin;y<=ymax;y++, maskptr+=maskstride, imageptr+=datastride) {
            double y_y0 = y-y0;
            double x_x0 = xmin - 1 - x0;
            double TwoMinv_xy__y_y0 = TwoMinv_xy * y_y0;
            double Minv_yy__y_y0__y_y0 = Minv_yy * y_y0 * y_y0;
            const double* mxxptr = Minv_xx__x_x0__x_x0.cptr();
            for(long x=xmin;x<=xmax;x++) {
                x_x0 += 1.; // do this increment to x_x0 out here, before the following if
                            // statement, because it has to happen whether we actually use the pixel
                            // or not if we want to properly track where we are in the image
                if (*(maskptr++)) {
                    //npix++;
                    /* Compute displacement from weight centroid, then
                     * get elliptical radius and weight.
                     */
                    double rho2 = Minv_yy__y_y0__y_y0 + TwoMinv_xy__y_y0*x_x0 + *(mxxptr++);
                    dbg<<"Using pixel: "<<x<<" "<<y<<" with value "<<*(imageptr)<<" rho2 "<<rho2<<" x_x0 "<<x_x0<<" y_y0 "<<y_y0<<std::endl;
                    if (rho2 < hsmparams->max_moment_nsig2) {
                        double intensity = std::exp(-0.5 * rho2) * *(imageptr++);

                        /* Now do the addition */
                        double intensity__x_x0 = intensity * x_x0;
                        double intensity__y_y0 = intensity * y_y0;
                        A    += intensity;
                        Bx   += intensity__x_x0;
                        By   += intensity__y_y0;
                        Cxx  += intensity__x_x0 * x_x0;
                        Cxy  += intensity__x_x0 * y_y0;
                        Cyy  += intensity__y_y0 * y_y0;
                        rho4w+= intensity * rho2 * rho2;
                    } else {
                        // if we are skipping this pixel because it's too far from center of
                        // Gaussian, then just increment the pointer to the next pixel in the image,
                        // don't waste time doing any math for this pixel
                        ++imageptr;
                    }
                } else {
                    // we still have to increment pointers when jumping over masked pixels,
                    // otherwise serious badness will happen.
                    ++imageptr;
                    ++mxxptr;
                }
            }
        }
        //dbg<<"Number of pixels used: "<<npix<<std::endl;
        dbg<<"Exiting find_ellipmom_1 with results: "<<A<<" "<<Bx<<" "<<By<<" "<<Cxx<<" "<<Cyy<<" "<<Cxy<<" "<<rho4w<<std::endl;
    }

    /* find_ellipmom_2
     * *** COMPUTES ADAPTIVE ELLIPTICAL MOMENTS OF AN IMAGE ***
     *
     * Finds the best-fit Gaussian:
     *
     * f ~ A / (pi*sqrt det M) * exp( - (r-r0) * M^-1 * (r-r0) )
     *
     * The fourth moment rho4 is also returned.
     * Note that the total image intensity for the Gaussian is 2A.
     *
     * Arguments:
     *   data: ImageView structure containing the image
     * > A: adaptive amplitude
     * > x0: adaptive centroid (x)
     * > y0: adaptive centroid (y)
     * > Mxx: adaptive covariance (xx)
     * > Mxy: adaptive covariance (xy)
     * > Myy: adaptive covariance (yy)
     * > rho4: rho4 moment
     *   epsilon: required accuracy
     * > num_iter: number of iterations required to converge
     */

    template <typename T>
    void find_ellipmom_2(
        ConstImageView<T> data, ConstImageView<int> mask, double& A, double& x0, double& y0,
        double& Mxx, double& Mxy, double& Myy, double& rho4, double epsilon, int& num_iter,
        boost::shared_ptr<HSMParams> hsmparams) 
    {

        double convergence_factor = 1.0;
        double Amp,Bx,By,Cxx,Cxy,Cyy;
        double semi_a2, semi_b2, two_psi;
        double dx, dy, dxx, dxy, dyy;
        double shiftscale, shiftscale0=0.;
        double x00 = x0;
        double y00 = y0;

        num_iter = 0;

#ifdef N_CHECKVAL
        if (epsilon <= 0 || epsilon >= convergence_factor) {
            throw HSMError("Error: epsilon out of range in find_ellipmom_2.\n");
        }
#endif

        /*
         * Set Amp = -1000 as initial value just in case the while() block below is never triggered;
         * in this case we have at least *something* defined to divide by, and for which the output
         * will fairly clearly be junk.
         */
        Amp = -1000.;

        /* Iterate until we converge */
        while(convergence_factor > epsilon) {

            /* Get moments */
            find_ellipmom_1(data, mask, x0, y0, Mxx, Mxy, Myy, Amp, Bx, By, Cxx, Cxy, Cyy, rho4, hsmparams);

            /* Compute configuration of the weight function */
            two_psi = std::atan2( 2* Mxy, Mxx-Myy );
            semi_a2 = 0.5 * ((Mxx+Myy) + (Mxx-Myy)*std::cos(two_psi)) + Mxy*std::sin(two_psi);
            semi_b2 = Mxx + Myy - semi_a2;

            if (semi_b2 <= 0) {
                throw HSMError("Error: non positive-definite weight in find_ellipmom_2.\n");
            }

            shiftscale = std::sqrt(semi_b2);
            if (num_iter == 0) shiftscale0 = shiftscale;

            /* Now compute changes to x0, etc. */
            dx = 2. * Bx / (Amp * shiftscale);
            dy = 2. * By / (Amp * shiftscale);
            dxx = 4. * (Cxx/Amp - 0.5*Mxx) / semi_b2;
            dxy = 4. * (Cxy/Amp - 0.5*Mxy) / semi_b2;
            dyy = 4. * (Cyy/Amp - 0.5*Myy) / semi_b2;

            if (dx     >  hsmparams->bound_correct_wt) dx     =  hsmparams->bound_correct_wt;
            if (dx     < -hsmparams->bound_correct_wt) dx     = -hsmparams->bound_correct_wt;
            if (dy     >  hsmparams->bound_correct_wt) dy     =  hsmparams->bound_correct_wt;
            if (dy     < -hsmparams->bound_correct_wt) dy     = -hsmparams->bound_correct_wt;
            if (dxx    >  hsmparams->bound_correct_wt) dxx    =  hsmparams->bound_correct_wt;
            if (dxx    < -hsmparams->bound_correct_wt) dxx    = -hsmparams->bound_correct_wt;
            if (dxy    >  hsmparams->bound_correct_wt) dxy    =  hsmparams->bound_correct_wt;
            if (dxy    < -hsmparams->bound_correct_wt) dxy    = -hsmparams->bound_correct_wt;
            if (dyy    >  hsmparams->bound_correct_wt) dyy    =  hsmparams->bound_correct_wt;
            if (dyy    < -hsmparams->bound_correct_wt) dyy    = -hsmparams->bound_correct_wt;

            /* Convergence tests */
            convergence_factor = std::abs(dx)>std::abs(dy)? std::abs(dx): std::abs(dy);
            convergence_factor *= convergence_factor;
            if (std::abs(dxx)>convergence_factor) convergence_factor = std::abs(dxx);
            if (std::abs(dxy)>convergence_factor) convergence_factor = std::abs(dxy);
            if (std::abs(dyy)>convergence_factor) convergence_factor = std::abs(dyy);
            convergence_factor = std::sqrt(convergence_factor);
            if (shiftscale<shiftscale0) convergence_factor *= shiftscale0/shiftscale;

            /* Now update moments */
            x0 += dx * shiftscale;
            y0 += dy * shiftscale;
            Mxx += dxx * semi_b2;
            Mxy += dxy * semi_b2;
            Myy += dyy * semi_b2;

            /* If the moments have gotten too large, or the centroid is out of range,
             * report a failure */
            if (std::abs(Mxx)>hsmparams->max_amoment || std::abs(Mxy)>hsmparams->max_amoment
                || std::abs(Myy)>hsmparams->max_amoment
                || std::abs(x0-x00)>hsmparams->max_ashift 
                || std::abs(y0-y00)>hsmparams->max_ashift) {
                throw HSMError("Error: adaptive moment failed\n");
            }

            if (++num_iter > hsmparams->max_mom2_iter) {
                throw HSMError("Error: too many iterations in adaptive moments\n");
            }

            if (std::isnan(convergence_factor) || std::isnan(Mxx) || std::isnan(Myy)
                || std::isnan(Mxy) || std::isnan(x0) || std::isnan(y0)) {
                throw HSMError("Error: NaN in calculation of adaptive moments\n");
            }
        }

        /* Re-normalize rho4 */
        A = Amp;
        rho4 /= Amp;
    }

    /* fast_convolve_image_1
     *
     * *** CONVOLVES TWO IMAGES *** 
     *
     * The bounding boxes and masks from image1 and image2 are taken
     * into account; only unmasked pixels are set in the output.  Note
     * that this routine ADDS the convolution to the pre-existing image.
     *
     * Arguments:
     *   image1: 1st image to be convolved, ImageView format
     *   image2: 2nd image to be convolved, ImageView format
     * > image_out: output (convolved) image, ImageView format
     */

    template <typename T, typename U>
    void fast_convolve_image_1(
        ConstImageView<T> image1, ConstImageView<int> mask1, 
        ConstImageView<U> image2, ConstImageView<int> mask2, 
        ImageView<T> image_out, ConstImageView<int> mask_out)
    {
        long dim1x, dim1y, dim1o, dim1, dim2, dim3, dim4;
        double xr,xi,yr,yi;
        long i,i_conj,j,k,ii,ii_conj;
        long out_xmin, out_xmax, out_ymin, out_ymax, out_xref, out_yref;

        /* Determine array sizes:
         * dim1 = (linear) size of pixel grid used for FFT
         * dim2 = 2*dim1
         * dim3 = dim2*dim2
         * dim4 = 2*dim3
         */
        dim1x = image1.getXMax() - image1.getXMin() + image2.getXMax() - image2.getXMin() + 2;
        dim1y = image1.getYMax() - image1.getYMin() + image2.getYMax() - image2.getYMin() + 2;
        dim1o = (dim1x>dim1y)? dim1x: dim1y;
        dim1 = 1; while(dim1<dim1o) dim1 <<= 1; /* dim1 must be a power of two */
        dim2 = dim1 << 1;
        dim3 = dim2 * dim2;
        dim4 = dim3 << 1;

        /* Allocate & initialize memory */
        tmv::Matrix<double> m1(dim1,dim1,0.);
        tmv::Matrix<double> m2(dim1,dim1,0.);
        tmv::Matrix<double> mout(dim1,dim1,0.);
        tmv::Vector<double> Ax(dim4,0.);
        tmv::Vector<double> Bx(dim4,0.);

        /* Build input maps */
        for(int x=image1.getXMin();x<=image1.getXMax();x++)
            for(int y=image1.getYMin();y<=image1.getYMax();y++) 
                if (mask1(x,y)) 
                    m1(x-image1.getXMin(),y-image1.getYMin()) = image1(x,y);
        for(i=image2.getXMin();i<=image2.getXMax();i++)
            for(j=image2.getYMin();j<=image2.getYMax();j++)
                if (mask2(i,j)) 
                    m2(i-image2.getXMin(),j-image2.getYMin()) = image2(i,j);

        /* Build the arrays for FFT -
         * - put m1 and m2 into the real and imaginary parts of Bx, respectively. */
        for(i=0;i<dim1;i++) for(j=0;j<dim1;j++) {
            k=2*(dim2*i+j);
            Bx[k  ] = m1[i][j];
            Bx[k+1] = m2[i][j];
        }

        /* We've filled only part of Bx, the other locations are for
         * zero padding.  First we separate the real (m1) and imaginary (m2) parts of the FFT,
         * then multiply to get the convolution.
         */
        fourier_trans_1(Bx.ptr(),dim3,1);
        for(i=0;i<dim3;i++) {
            i_conj = i==0? 0: dim3-i;      /* part of FFT of B holding complex conjugate mode */
            ii      = 2*i;
            ii_conj = 2*i_conj;
            xr = 0.5 * (  Bx[ii  ] + Bx[ii_conj  ] );
            xi = 0.5 * (  Bx[ii+1] - Bx[ii_conj+1] );
            yr = 0.5 * (  Bx[ii+1] + Bx[ii_conj+1] );
            yi = 0.5 * ( -Bx[ii  ] + Bx[ii_conj  ] );
            Ax[ii  ] = xr*yr-xi*yi;      /* complex multiplication */
            Ax[ii+1] = xr*yi+xi*yr;
        }
        fourier_trans_1(Ax.ptr(),dim3,-1);   /* Reverse FFT Ax to get convolved image */
        for(i=0;i<dim1;i++)
            for(j=0;j<dim1;j++)
                mout[i][j] = Ax[2*(dim2*i+j)] / (double)dim3;

        /* Calculate the effective bounding box for the output image,
         * [out_xmin..out_xmax][out_ymin..out_ymax], and the offset between mout and
         * image_out, namely (out_xref,out_yref)
         */
        out_xmin = out_xref = image1.getXMin() + image2.getXMin();
        out_xmax =            image1.getXMax() + image2.getXMax();
        out_ymin = out_yref = image1.getYMin() + image2.getYMin();
        out_ymax =            image1.getYMax() + image2.getYMax();
        if (out_xmin<image_out.getXMin()) out_xmin = image_out.getXMin();
        if (out_xmax>image_out.getXMax()) out_xmax = image_out.getXMax();
        if (out_ymin<image_out.getYMin()) out_ymin = image_out.getYMin();
        if (out_ymax>image_out.getYMax()) out_ymax = image_out.getYMax();

        /* And now do the writing */
        for(i=out_xmin;i<=out_xmax;i++)
            for(j=out_ymin;j<=out_ymax;j++)
                if(mask_out(i,j))
                    image_out(i,j) += mout(i-out_xref,j-out_yref);

    }

    void matrix22_invert(double& a, double& b, double& c, double& d) 
    {

        double det,temp;

        det = a*d-b*c;
        b = -b; c = -c;
        temp = a; a = d; d = temp;
        a /= det; b /= det; c /= det; d /= det;
    }

    /* shearmult
     * *** COMPOSES TWO SHEARS ***
     *
     * Takes two shears and finds the effective shear on an initially circular object from
     * applying ea and then eb.  The "e"'s are in the ellipticity format of Bernstein &
     * Jarvis.
     *
     * Arguments:
     *   e1a: + component of 1st shear
     *   e1b: x component of 1st shear
     *   e2a: + component of 2nd shear
     *   e2b: x component of 2nd shear
     * > e1out: + component of total shear
     * > e2out: x component of total shear
     */

    void shearmult(double e1a, double e2a, double e1b, double e2b,
                   double& e1out, double& e2out) 
    {

        /* This is eq. 2-13 of Bernstein & Jarvis */
        /* Shear ea is applied, then eb -- it matters! */
        double dotp, factor;

        dotp = e1a*e1b + e2a*e2b;
        factor = (1.-std::sqrt(1-e1b*e1b-e2b*e2b)) / (e1b*e1b + e2b*e2b);
        e1out = (e1a + e1b + e2b*factor*(e2a*e1b - e1a*e2b))/(1+dotp);
        e2out = (e2a + e2b + e1b*factor*(e1a*e2b - e2a*e1b))/(1+dotp);
    }

    /* psf_corr_bj
     * *** CARRIES OUT BERNSTEIN & JARVIS A4 PSF CORRECTION ***
     *
     * This routine cleans up the PSF by shearing to the circular-PSF frame,
     * then applying the resolution factor, then un-shearing.
     *
     * Arguments:
     *   Tratio: trace ratio, (Mxx+Myy)(psf)/(Mxx+Myy)(measured)
     *   e1p: + ellipticity of PSF
     *   e2p: x ellipticity of PSF
     *   a4p: radial 4th moment of PSF
     *   e1o: + ellipticity of galaxy (measured)
     *   e2o: x ellipticity of galaxy (measured)
     *   a4o: radial 4th moment of galaxy (measured)
     * > e1: output ellipticity
     * > e2: output ellipticity
     */

    void psf_corr_bj(
        double Tratio, double e1p, double e2p, double a4p, double e1o,
        double e2o, double a4o, double& e1, double& e2) 
    {

        double e1red, e2red; /* ellipticities reduced to circular PSF */
        double sig2ratio;
        double coshetap, coshetao;
        double R;

        /* Take us to sig2ratio = sigma2(P)/sigma2(O) since this is shear-invariant */
        coshetap = 1./std::sqrt(1-e1p*e1p-e2p*e2p);
        coshetao = 1./std::sqrt(1-e1o*e1o-e2o*e2o);
        sig2ratio = Tratio * coshetao/coshetap; /* since sigma2 = T / cosh eta */

        shearmult(e1o,e2o,-e1p,-e2p,e1red,e2red);

        /* compute resolution factor and un-dilute */
        coshetao = 1./std::sqrt(1-e1red*e1red-e2red*e2red);
        R = 1. - sig2ratio * (1-a4p)/(1+a4p) * (1+a4o)/(1-a4o) / coshetao;

        e1red /= R;
        e2red /= R;

        shearmult(e1red,e2red,e1p,e2p,e1,e2);
    }

    /* psf_corr_linear
     * *** CARRIES OUT HIRATA & SELJAK LINEAR PSF CORRECTION ***
     *
     * This routine cleans up the PSF by shearing to the circular-PSF frame,
     * then applying the resolution factor, then un-shearing.
     *
     * Arguments:
     *   Tratio: trace ratio, (Mxx+Myy)(psf)/(Mxx+Myy)(measured)
     *   e1p: + ellipticity of PSF
     *   e2p: x ellipticity of PSF
     *   a4p: radial 4th moment of PSF
     *   e1o: + ellipticity of galaxy (measured)
     *   e2o: x ellipticity of galaxy (measured)
     *   a4o: radial 4th moment of galaxy (measured)
     * > e1: output ellipticity
     * > e2: output ellipticity
     */

    void psf_corr_linear(
        double Tratio, double e1p, double e2p, double a4p, double e1o,
        double e2o, double a4o, double& e1, double& e2) 
    {

        double e1red, e2red; /* ellipticities reduced to circular PSF */
        double sig2ratio;
        double coshetap, coshetao;
        double e,eta,a2,b2,A,B;
        double R;
        double a4i;
        double ca4i,ca4p;
        double deltaeta,deltamu;
        //double etai;
        double Ti,Tp;
        double EI;

        /* Take us to sig2ratio = sigma2(P)/sigma2(O) since this is shear-invariant */
        coshetap = 1./std::sqrt(1-e1p*e1p-e2p*e2p);
        coshetao = 1./std::sqrt(1-e1o*e1o-e2o*e2o);
        sig2ratio = Tratio * coshetao/coshetap; /* since sigma2 = T / cosh eta */

        shearmult(e1o,e2o,-e1p,-e2p,e1red,e2red);

        /* compute resolution factor and un-dilute */
        e = std::sqrt(e1red*e1red+e2red*e2red);
        eta = atanh(e);
        a2 = std::exp(-eta)*sig2ratio; /* fraction of major axis variance from PSF */
        b2 = std::exp(eta)*sig2ratio; /* fraction of minor axis variance from PSF */
        A = 1-a2; B = 1-b2; /* fractions from intrinsic image */
        ca4p = 0.375*(a2*a2+b2*b2)+0.25*a2*b2;
        ca4i = 0.375*(A*A+B*B)+0.25*A*B;
        a4i = (a4o - ca4p*a4p) / ca4i;
        Ti = (A-B) * (-2+1.5*(A+B));
        Tp = (a2-b2) * (-2+1.5*(a2+b2));
        deltaeta = Ti * a4i + Tp * a4p;

        /* 4th moment correction for R: must find etai */
        EI = std::sqrt(e1red*e1red + e2red*e2red);
        // TODO: etai was set, but not used.
        // Is this a bug?  Or just a legacy of an old calculation?
        //etai = 0.5 * log( (1./a2-1) / (1./b2-1) ); 
        coshetao = 1./std::sqrt(1-e1red*e1red-e2red*e2red);
        deltamu = (-1.5*A*A - A*B - 1.5*B*B +2*(A+B)) * a4i
            + (-1.5*a2*a2 - a2*b2 - 1.5*b2*b2 + 2*(a2+b2))*a4p;
        deltamu *= 0.5;
        deltaeta *= -1.0;
        R = ( 1 - 2*deltamu - deltaeta*EI - sig2ratio/coshetao ) /
            ( -deltaeta/EI + 1-2*deltamu ) ;

        e1red /= R;
        e2red /= R;

        shearmult(e1red,e2red,e1p,e2p,e1,e2);
    }

    /* psf_corr_ksb_1
     * *** COMPUTES KSB PSF CORRECTION ***
     *
     * Uses the galaxy and PSF images to compute an estimator for the shear (e1,e2)
     * using the method of Kaiser, Squires, and Broadhurst (1995), updated
     * to include the anisotropic PSF correction of Luppino and Kaiser (1997).
     *
     * Arguments:
     *   gal_image: image of measured galaxy (ImageView format)
     *   PSF: PSF map (ImageView format)
     * > e1: + shear estimator, PSF-corrected
     * > e2: x shear estimator, PSF-corrected
     * > responsivity: shear responsivity of estimator
     * > R: resolution factor
     *   flags: processing flags (NOT IMPLEMENTED)
     * > x0_gal: galaxy center (x coordinate) -- input initial guess
     * > y0_gal: galaxy center (y coordinate) -- input initial guess
     * > sig_gal: galaxy radius (pixels) -- input initial guess
     * > flux_gal: galaxy flux (counts)
     * > x0_psf: PSF center (x coordinate) -- input initial guess
     * > y0_psf: PSF center (y coordinate) -- input initial guess
     * > sig_psf: PSF radius (pixels) -- input initial guess
     */

    template <typename T, typename U>
    unsigned int psf_corr_ksb_1(
        ConstImageView<T> gal_image, ConstImageView<int> gal_mask,
        ConstImageView<U> PSF_image, ConstImageView<int> PSF_mask,
        double& e1, double& e2,
        double& responsivity, double& R, unsigned long flags, double& x0_gal, double& y0_gal,
        double& sig_gal, double& flux_gal, double& x0_psf, double& y0_psf, double& sig_psf,
        boost::shared_ptr<HSMParams> hsmparams) 
    {

        unsigned int status = 0;
        int num_iter;
        double oT,oeQ,oeU,oegQ,oegU,opgQQ,opgQU,opgUQ,opgUU,oesQ,oesU,opsQQ,opsQU,opsUQ,opsUU;
        double pT,peQ,peU,pegQ,pegU,ppgQQ,ppgQU,ppgUQ,ppgUU,pesQ,pesU,ppsQQ,ppsQU,ppsUQ,ppsUU;
        double gQ,gU;
        double eQ,eU;
        double PQQ,PQU,PUQ,PUU;
        double x0, y0, sigma0;
        double I00,I20r,I20i,I11,I40r,I40i,I31r,I31i,I22;
        double P00,P20r,P20i,P11,P40r,P40i,P31r,P31i,P22;

        /* Initialize -- if we don't set the outputs, they will be reported
         * as failures.
         */
        e1 = e2 = R = hsmparams->failed_moments;

        tmv::Matrix<double> moments(hsmparams->ksb_moments_max+1,hsmparams->ksb_moments_max+1);
        tmv::Matrix<double> psfmoms(hsmparams->ksb_moments_max+1,hsmparams->ksb_moments_max+1);

        /* Determine the adaptive variance of the measured galaxy */
        x0 = x0_gal;
        y0 = y0_gal;
        sigma0 = sig_gal;
        find_mom_2(gal_image, gal_mask, moments, hsmparams->ksb_moments_max, x0_gal, y0_gal, sig_gal,
                   1.0e-6, num_iter, hsmparams);
        if (num_iter == hsmparams->num_iter_default) {
            status |= 0x0002; /* Report convergence failure */
            x0_gal = x0;
            y0_gal = y0;
            sig_gal = sigma0;
            find_mom_1(gal_image, gal_mask, moments, hsmparams->ksb_moments_max, x0, y0, sigma0);
        }
        flux_gal = 3.544907701811 * sig_gal * moments(0,0);

        /* Determine the centroid of the PSF */
        x0 = x0_psf;
        y0 = y0_psf;
        sigma0 = sig_psf;
        find_mom_2(PSF_image, PSF_mask, psfmoms, hsmparams->ksb_moments_max, x0_psf, y0_psf, sig_psf,
                   1.0e-6, num_iter, hsmparams);
        if (num_iter == hsmparams->num_iter_default) {
            status |= 0x0001; /* Report convergence failure */
            x0_psf = x0;
            y0_psf = y0;
            sig_psf = sigma0;
        }

        /* ... but we want the moments with the galaxy weight fcn */
        find_mom_1(PSF_image, PSF_mask, psfmoms, hsmparams->ksb_moments_max, x0_psf, y0_psf, sig_gal);

        /* Get resolution factor */
        R = 1. - (sig_psf*sig_psf)/(sig_gal*sig_gal);

        /* Now we convert from the rectangular |nx,ny> basis into the polar
         * (nl,nr) basis.  The conversion is:
         *
         * zeroeth order
         * (0,0) = |0,0>
         *
         * second order
         * (2,0) = 1/2 * [ |2,0> - |0,2> ] + i/sqrt2 * |1,1>
         * (1,1) = 1/sqrt2 * [ |2,0> + |0,2> ]
         *
         * fourth order
         * (4,0) = 1/4 * [|4,0>+|0,4>] - sqrt(3/8) * |2,2> + i/2 * [|3,1>-|1,3>]
         * (3,1) = 1/2 * [|4,0>-|0,4>] + i/2 * [|3,1>+|1,3>]
         * (2,2) = sqrt(3/8) [|4,0>+|0,4>] + 1/2 * |2,2>
         */
        P00  = psfmoms(0,0);
        P20r = 0.5 * (psfmoms(2,0) - psfmoms(0,2));
        P20i = 0.7071067811865 * psfmoms(1,1);
        P11  = 0.7071067811865 * (psfmoms(2,0) + psfmoms(0,2));
        P40r = 0.25 * (psfmoms(4,0) + psfmoms(0,4)) - 0.6123724356958 * psfmoms(2,2);
        P40i = 0.5 * (psfmoms(3,1)-psfmoms(1,3));
        P31r = 0.5 * (psfmoms(4,0)-psfmoms(0,4));
        P31i = 0.5 * (psfmoms(3,1)+psfmoms(1,3));
        P22  = 0.6123724356958 * (psfmoms(4,0)+psfmoms(0,4)) + 0.5 * psfmoms(2,2);

        I00  = moments(0,0);
        I20r = 0.5 * (moments(2,0) - moments(0,2));
        I20i = 0.7071067811865 * moments(1,1);
        I11  = 0.7071067811865 * (moments(2,0) + moments(0,2));
        I40r = 0.25 * (moments(4,0) + moments(0,4)) - 0.6123724356958 * moments(2,2);
        I40i = 0.5 * (moments(3,1)-moments(1,3));
        I31r = 0.5 * (moments(4,0)-moments(0,4));
        I31i = 0.5 * (moments(3,1)+moments(1,3));
        I22  = 0.6123724356958 * (moments(4,0)+moments(0,4)) + 0.5 * moments(2,2);

        /* and from this we get all of KSB's quantities.  Their Greek letters have index values
         * here of Q or U.  The "shear" and "smear" tensors are denoted with "g" and "s"
         * respectively.  We'll do the object first.  WARNING: Hirata&Seljak (2003) Appendix C
         * has a complex-conjugation error in all the formulas.
         */
        oT = I00 + I11;

        oeQ = 1.414213562373 * I20r / oT;
        oeU = 1.414213562373 * I20i / oT;

        oegQ = (-1.414213562373*I20r - 2.449489742783*I31r)/oT;
        oegU = (-1.414213562373*I20i - 2.449489742783*I31i)/oT;

        opgQQ = -oeQ*oegQ - 2.449489742783*I40r/oT + 2-(I00 + 2*I11 + I22)/oT;
        opgQU = -oeQ*oegU - 2.449489742783*I40i/oT;
        opgUQ = -oeU*oegQ - 2.449489742783*I40i/oT;
        opgUU = -oeU*oegU + 2.449489742783*I40r/oT + 2-(I00 + 2*I11 + I22)/oT;

        oesQ = (-1.414213562373*I20r + 2.449489742783*I31r ) / (4*oT);
        oesU = (-1.414213562373*I20i + 2.449489742783*I31i ) / (4*oT);

        opsQQ = -oeQ*oesQ + 2.449489742783*I40r / (4*oT) + (I00 - 2*I11 + I22) / (2*oT);
        opsQU = -oeQ*oesU + 2.449489742783*I40i / (4*oT);
        opsUQ = -oeU*oesQ + 2.449489742783*I40i / (4*oT);
        opsUU = -oeU*oesU - 2.449489742783*I40r / (4*oT) + (I00 - 2*I11 + I22) / (2*oT);

        /* Now the PSF */
        pT = P00 + P11;

        peQ = 1.414213562373 * P20r / pT;
        peU = 1.414213562373 * P20i / pT;

        pegQ = (-1.414213562373*P20r - 2.449489742783*P31r)/pT;
        pegU = (-1.414213562373*P20i - 2.449489742783*P31i)/pT;

        ppgQQ = -peQ*pegQ - 2.449489742783*P40r/pT + 2-(P00 + 2*P11 + P22)/pT;
        ppgQU = -peQ*pegU - 2.449489742783*P40i/pT;
        ppgUQ = -peU*pegQ - 2.449489742783*P40i/pT;
        ppgUU = -peU*pegU + 2.449489742783*P40r/pT + 2-(P00 + 2*P11 + P22)/pT;

        pesQ = (-1.414213562373*P20r + 2.449489742783*P31r ) / (4*pT);
        pesU = (-1.414213562373*P20i + 2.449489742783*P31i ) / (4*pT);

        ppsQQ = -peQ*pesQ + 2.449489742783*P40r / (4*pT) + (P00 - 2*P11 + P22) / (2*pT);
        ppsQU = -peQ*pesU + 2.449489742783*P40i / (4*pT);
        ppsUQ = -peU*pesQ + 2.449489742783*P40i / (4*pT);
        ppsUU = -peU*pesU - 2.449489742783*P40r / (4*pT) + (P00 - 2*P11 + P22) / (2*pT);

        /* Let's invert the PSF smear responsivity matrix */
        matrix22_invert(ppsQQ,ppsQU,ppsUQ,ppsUU);

        /* We've got these, let's find g (KSB's "p") and do the smear correction */
        gQ = (ppsQQ*peQ+ppsQU*peU);
        gU = (ppsUQ*peQ+ppsUU*peU);
        eQ = oeQ - opsQQ*gQ - opsQU*gU;
        eU = oeU - opsUQ*gQ - opsUU*gU;

        /* Now compute and invert P = opg - ops*ppg*pps^-1 */
        PQQ = opgQQ - opsQQ*ppgQQ*ppsQQ - opsQQ*ppgQU*ppsUQ -
            opsQU*ppgUQ*ppsQQ - opsQU*ppgUU*ppsUQ;
        PQU = opgQU - opsQQ*ppgQQ*ppsQU - opsQQ*ppgQU*ppsUU -
            opsQU*ppgUQ*ppsQU - opsQU*ppgUU*ppsUU;
        PUQ = opgUQ - opsUQ*ppgQQ*ppsQQ - opsUQ*ppgQU*ppsUQ -
            opsUU*ppgUQ*ppsQQ - opsUU*ppgUU*ppsUQ;
        PUU = opgUU - opsUQ*ppgQQ*ppsQU - opsUQ*ppgQU*ppsUU -
            opsUU*ppgUQ*ppsQU - opsUU*ppgUU*ppsUU;

        matrix22_invert(PQQ,PQU,PUQ,PUU);

        /* This finally gives us a shear. */
        e1 = PQQ*eQ + PQU*eU;
        e2 = PUQ*eQ + PUU*eU;
        responsivity = 1.;

        return status;
    }

    /* psf_corr_regauss
     * *** COMPUTES RE-GAUSSIANIZATION PSF CORRECTION ***
     *
     * Takes in galaxy and PSF images and computes a PSF-corrected ellipticity (e1,e2)
     * using the re-Gaussianization method of Hirata & Seljak (2003).  The
     * ellipticity computed corresponds to Bernstein & Jarvis (2002) definition.
     *
     * flags:
     *   0x00000001: recompute galaxy flux by summing unmasked pixels
     *   0x00000002: recompute galaxy flux from Gaussian-quartic fit (overrides 0x00000001)
     *   0x00000004: cut off Gaussian approximator at hsmparams->nsig_rg sigma (saves computation time in
     *               the convolution step)
     *   0x00000008: cut off PSF residual at hsmparams->nsig_rg2 sigma (saves computation time in
     *               the convolution step)
     *
     * Arguments:
     *   gal_image: image of the galaxy as measured (i.e. not deconvolved)
     *   PSF: image of point spread function
     * > e1: + ellipticity
     * > e2: x ellipticity
     * > R: effective resolution factor (0 = unresolved, 1 = well resolved)
     *   flags: controls options for shear measurement
     * > x0_gal: guess for galaxy centroid (x) [replaced with best value]
     * > y0_gal: guess for galaxy centroid (y) [replaced with best value]
     * > sig_gal: guess for galaxy sigma [replaced with best value]
     * > x0_psf: guess for PSF centroid (x) [replaced with best value]
     * > y0_psf: guess for PSF centroid (y) [replaced with best value]
     * > sig_psf: guess for PSF sigma [replaced with best value]
     * > flux_gal: total flux of galaxy
     *
     * Returns: status of shear measurement. (0 = completely successful)
     *   The status integer is bit-encoded:
     *   0x0001 = PSF adaptive moment failure to converge
     *   0x0002 = galaxy adaptive moment failure to converge
     *   0x0004 = galaxy smaller than PSF
     *   0x0008 = adaptive measurement of re-Gaussianized image failed to converge
     */

    template <typename T, typename U>
    unsigned int psf_corr_regauss(
        ConstImageView<T> gal_image, ConstImageView<int> gal_mask,
        ConstImageView<U> PSF_image, ConstImageView<int> PSF_mask,
        double& e1, double& e2, double& R, unsigned long flags, double& x0_gal,
        double& y0_gal, double& sig_gal, double& x0_psf, double& y0_psf,
        double& sig_psf, double& flux_gal, boost::shared_ptr<HSMParams> hsmparams) 
    {
        int num_iter;
        unsigned int status = 0;
        double A_g, Mxxpsf, Mxypsf, Myypsf, rho4psf, flux_psf, sum;
        double A_I, Mxxgal, Mxygal, Myygal, rho4gal;
        double Minvpsf_xx, Minvpsf_xy, Minvpsf_yy, detM, center_amp_psf;
        double dx, dy;
        double a2, b2, two_phi;
        double x0_old=0., y0_old=0., Mfxx, Mfxy, Mfyy, detMf, Minvf_xx, Minvf_xy, Minvf_yy;
        double Tpsf, e1psf, e2psf;
        double Tgal, e1gal, e2gal;
        long fgauss_xmin, fgauss_xmax, fgauss_ymin, fgauss_ymax;
        double fgauss_xctr, fgauss_yctr, fgauss_xsig, fgauss_ysig;

        /* Initialize -- if we don't set the outputs, they will be reported
         * as failures.
         */
        e1 = e2 = R = hsmparams->failed_moments;

        /* Get the PSF flux */
        flux_psf = 0;
        for(int y=PSF_image.getYMin();y<=PSF_image.getYMax();y++)
            for(int x=PSF_image.getXMin();x<=PSF_image.getXMax();x++)
                if (PSF_mask(x,y))
                    flux_psf += PSF_image(x,y);

        /* Recompute the galaxy flux only if the relevant flag is set */
        if (flags & 0x00000001) {
            flux_gal = 0;
            for(int y=gal_image.getYMin();y<=gal_image.getYMax();y++)
                for(int x=gal_image.getXMin();x<=gal_image.getXMax();x++)
                    if (gal_mask(x,y)) 
                        flux_gal += gal_image(x,y);
        }

        /* Get the elliptical adaptive moments of PSF */
        Mxxpsf = Myypsf = sig_psf * sig_psf;
        Mxypsf = 0.;
        find_ellipmom_2(PSF_image, PSF_mask, A_g, x0_psf, y0_psf, Mxxpsf, Mxypsf, Myypsf, rho4psf,
                        1.0e-6, num_iter, hsmparams);

        if (num_iter == hsmparams->num_iter_default) {
            x0_psf = x0_old;
            y0_psf = y0_old;
            status |= 0x0001;
        }

        /* Get the elliptical adaptive moments of galaxy */
        Mxxgal = Myygal = sig_gal * sig_gal;
        Mxygal = 0.;
        find_ellipmom_2(gal_image, gal_mask, A_I, x0_gal, y0_gal, Mxxgal, Mxygal, Myygal, rho4gal,
                        1.0e-6, num_iter, hsmparams);

        if (num_iter == hsmparams->num_iter_default) {
            x0_gal = x0_old;
            y0_gal = y0_old;
            status |= 0x0002;
        }

        /* If the flags tell us to, we reset the galaxy flux estimate */
        if (flags & 0x00000002) {
            flux_gal = rho4gal * A_I;
        }

        /* Compute approximate deconvolved moments (i.e. without non-Gaussianity correction).
         * We also test this matrix for positive definiteness.
         */
        Mfxx = Mxxgal - Mxxpsf;
        Mfxy = Mxygal - Mxypsf;
        Mfyy = Myygal - Myypsf;
        detMf = Mfxx * Mfyy - Mfxy * Mfxy;
        if (hsmparams->regauss_too_small == 0) {
            if (Mfxx<=0 || Mfyy<=0 || detMf<=0) status |= 0x0004;
        } else {

            /* Compute the semimajor and semiminor axes of Mf and the position angle */
            two_phi = std::atan2(2*Mfxy, Mfxx-Mfyy);
            a2 = 0.5 * ( Mfxx+Mfyy + (Mfxx-Mfyy)*std::cos(two_phi) ) + Mfxy*std::sin(two_phi);
            b2 = Mfxx + Mfyy - a2;

            /* Now impose restrictions to ensure this doesn't blow up */
            if (a2<=0.25) a2=0.25;
            if (b2<=0.25) b2=0.25;

            /* Convert back to Mf matrix */
            Mfxx = 0.5 * ( a2+b2 + (a2-b2)*std::cos(two_phi) );
            Mfyy = 0.5 * ( a2+b2 - (a2-b2)*std::cos(two_phi) );
            Mfxy = 0.5 * (a2-b2) * std::sin(two_phi);
            detMf = Mfxx*Mfyy - Mfxy*Mfxy;
        }

        /* Test to see if anything has gone wrong -- if so, complain! */
        if (status) return (status);

        /* We also need the Gaussian de-convolved fit.  First get bounding box */
        fgauss_xmin = gal_image.getXMin() - PSF_image.getXMax();
        fgauss_xmax = gal_image.getXMax() - PSF_image.getXMin();
        fgauss_ymin = gal_image.getYMin() - PSF_image.getYMax();
        fgauss_ymax = gal_image.getYMax() - PSF_image.getYMin();
        fgauss_xctr = x0_gal - x0_psf;
        fgauss_yctr = y0_gal - y0_psf;
        fgauss_xsig = std::sqrt(Mfxx>1? Mfxx: 1);
        fgauss_ysig = std::sqrt(Mfyy>1? Mfyy: 1);

        /* Shrink if the box extends beyond hsmparams->nsig_rg sigma range */
        if (flags & 0x00000004) {
            if (fgauss_xmin < fgauss_xctr - hsmparams->nsig_rg*fgauss_xsig)
                fgauss_xmin = (long) std::floor(fgauss_xctr - hsmparams->nsig_rg*fgauss_xsig);
            if (fgauss_xmax > fgauss_xctr + hsmparams->nsig_rg*fgauss_xsig)
                fgauss_xmax = (long) std::ceil (fgauss_xctr + hsmparams->nsig_rg*fgauss_xsig);
            if (fgauss_ymin < fgauss_yctr - hsmparams->nsig_rg*fgauss_ysig)
                fgauss_ymin = (long) std::floor(fgauss_yctr - hsmparams->nsig_rg*fgauss_ysig);
            if (fgauss_ymax > fgauss_yctr + hsmparams->nsig_rg*fgauss_ysig)
                fgauss_ymax = (long) std::ceil (fgauss_yctr + hsmparams->nsig_rg*fgauss_ysig);
        }
        Minvf_xx =  Mfyy/detMf;
        Minvf_xy = -Mfxy/detMf;
        Minvf_yy =  Mfxx/detMf;
        sum = 0.;
        Bounds<int> fgauss_bounds(fgauss_xmin, fgauss_xmax, fgauss_ymin, fgauss_ymax);
        Image<double> fgauss(fgauss_bounds);
        for(int y=fgauss.getYMin();y<=fgauss.getYMax();y++) {
            for(int x=fgauss.getXMin();x<=fgauss.getXMax();x++) {
                dx = x - x0_gal + x0_psf;
                dy = y - y0_gal + y0_psf;
                sum += fgauss(x,y) =
                    std::exp (-0.5 * ( Minvf_xx*dx*dx + Minvf_yy*dy*dy ) - Minvf_xy*dx*dy);
            }
        }

        /* Properly normalize fgauss */
        fgauss *= flux_gal/(sum*flux_psf);
        Image<int> fgauss_mask(fgauss_bounds);
        fgauss_mask.fill(1);
        ConstImageView<int> fgauss_mask_view = fgauss_mask.view();

        /* Figure out the size of the bounding box for the PSF residual.
         * We don't necessarily need the whole PSF,
         * just the part that will affect regions inside the hsmparams->nsig_rg2 sigma ellipse 
         * of the Intensity.
         */
        Bounds<int> pbounds = PSF_image.getBounds();
        if (flags & 0x00000008) {
            int pxmin = (int) std::floor(
                x0_psf - hsmparams->nsig_rg2*std::sqrt(Mxxgal) - hsmparams->nsig_rg*fgauss_xsig );
            int pxmax = (int) std::ceil (
                x0_psf + hsmparams->nsig_rg2*std::sqrt(Mxxgal) + hsmparams->nsig_rg*fgauss_xsig );
            int pymin = (int) std::floor(
                y0_psf - hsmparams->nsig_rg2*std::sqrt(Myygal) - hsmparams->nsig_rg*fgauss_ysig );
            int pymax = (int) std::ceil (
                y0_psf + hsmparams->nsig_rg2*std::sqrt(Myygal) + hsmparams->nsig_rg*fgauss_ysig );
            if (PSF_image.getXMin() >= pxmin) pxmin = PSF_image.getXMin();
            if (PSF_image.getXMax() <= pxmax) pxmax = PSF_image.getXMax();
            if (PSF_image.getYMin() >= pymin) pymin = PSF_image.getYMin();
            if (PSF_image.getYMax() <= pymax) pymax = PSF_image.getYMax();
            pbounds = Bounds<int>(pxmin,pxmax,pymin,pymax);
        }

        /* Now let's compute the residual from the PSF fit.  This is called
         * - epsilon in Hirata & Seljak.
         */
        Image<double> PSF_resid(pbounds);
        detM = Mxxpsf * Myypsf - Mxypsf * Mxypsf;
        Minvpsf_xx =  Myypsf/detM;
        Minvpsf_xy = -Mxypsf/detM;
        Minvpsf_yy =  Mxxpsf/detM;
        center_amp_psf = flux_psf / (2.*M_PI * std::sqrt(detM));
        for(int y=pbounds.getYMin();y<=pbounds.getYMax();y++) {
            for(int x=pbounds.getXMin();x<=pbounds.getXMax();x++) {
                dx = x - x0_psf;
                dy = y - y0_psf;

                if (PSF_mask(x,y))
                    PSF_resid(x,y) = -PSF_image(x,y) + center_amp_psf * 
                        std::exp (-0.5 * ( Minvpsf_xx*dx*dx + Minvpsf_yy*dy*dy ) - Minvpsf_xy*dx*dy);
            }
        }

        /* Now compute the re-Gaussianized galaxy image */
        Image<double> Iprime = gal_image;
        ConstImageView<double> fgauss_view = fgauss.view();
        ConstImageView<double> PSF_resid_view = PSF_resid.view();
        ImageView<double> Iprime_view = Iprime.view();
        ConstImageView<double> Iprime_cview = Iprime_view;
        fast_convolve_image_1(fgauss_view, fgauss_mask_view, PSF_resid_view, PSF_mask,
                              Iprime_view, gal_mask);

        /* Now that Iprime is constructed, we measure it */
        find_ellipmom_2(Iprime_cview, gal_mask, A_I, x0_gal, y0_gal, Mxxgal, Mxygal, Myygal,
                        rho4gal, 1.0e-6, num_iter, hsmparams);
        if (num_iter == hsmparams->num_iter_default) {
            x0_gal = x0_old;
            y0_gal = y0_old;
            status |= 0x0008;
        }
        if (Mxxgal<=0 || Myygal<=0 || Mxxgal*Myygal<=Mxygal*Mxygal ) {
            throw HSMError("Error: non positive definite adaptive moments.\n");
        }
        sig_gal = std::pow( Mxxgal*Myygal - Mxygal*Mxygal, 0.25);

        /* And do the PSF correction */
        Tgal  = Mxxgal + Myygal;
        e1gal = (Mxxgal - Myygal) / Tgal;
        e2gal = 2 *Mxygal / Tgal;
        Tpsf  = Mxxpsf + Myypsf;
        e1psf = (Mxxpsf - Myypsf) / Tpsf;
        e2psf = 2 * Mxypsf / Tpsf;

        psf_corr_bj(Tpsf/Tgal, e1psf, e2psf, 0., e1gal, e2gal, 0.5*rho4gal-1., e1, e2); 
        /* Use 0 for radial 4th moment of PSF because it's been * re-Gaussianized.  */

        R = 1. - Tpsf/Tgal;

        return status;
    }

    /* general_shear_estimator
     * *** WRAPPER FOR SHEAR ESTIMATION ROUTINES ***
     *
     * Arguments:
     *   gal_image: measured image of galaxy
     *   PSF: estimated PSF map
     *   gal_data: galaxy data
     *   PSF_data: PSF data
     *   shear_est: which shear estimator to use
     *   flags: any flags for the shear estimator
     *
     * Returns: status from shear measurement:
     *   0      = completely successful
     *   0x8000 = couldn't figure out which estimator you wanted to use
     *
     * For BJ and LINEAR methods, returns 1 if measurement fails.
     */

    template <typename T, typename U>
    unsigned int general_shear_estimator(
        ConstImageView<T> gal_image, ConstImageView<int> gal_mask,
        ConstImageView<U> PSF_image, ConstImageView<int> PSF_mask,
        ObjectData& gal_data, ObjectData& PSF_data, const std::string& shear_est,
        unsigned long flags, boost::shared_ptr<HSMParams> hsmparams) 
    {
        unsigned int status = 0;
        int num_iter;
        double x0, y0, R;
        double A_gal, Mxx_gal, Mxy_gal, Myy_gal, rho4_gal;
        double A_psf, Mxx_psf, Mxy_psf, Myy_psf, rho4_psf;

        if (shear_est == "BJ" || shear_est == "LINEAR") {
            /* Bernstein & Jarvis and linear estimator */

            /* Measure the PSF */
            x0 = PSF_data.x0;
            y0 = PSF_data.y0;
            Mxx_psf = Myy_psf = PSF_data.sigma * PSF_data.sigma; Mxy_psf = 0.;
            find_ellipmom_2(PSF_image, PSF_mask, A_psf, x0, y0, Mxx_psf, Mxy_psf, Myy_psf,
                            rho4_psf, 1.0e-6, num_iter, hsmparams);
            if (num_iter == hsmparams->num_iter_default) {
                return 1;
            } else {
                PSF_data.x0 = x0;
                PSF_data.y0 = y0;
                PSF_data.sigma = std::pow( Mxx_psf * Myy_psf - Mxy_psf * Mxy_psf, 0.25);
            }

            /* Measure the galaxy */
            x0 = gal_data.x0;
            y0 = gal_data.y0;
            Mxx_gal = Myy_gal = gal_data.sigma * gal_data.sigma; Mxy_gal = 0.;
            find_ellipmom_2(gal_image, gal_mask, A_gal, x0, y0, Mxx_gal, Mxy_gal,
                            Myy_gal, rho4_gal, 1.0e-6, num_iter, hsmparams);
            if (num_iter == hsmparams->num_iter_default) {
                return 1;
            } else {
                gal_data.x0 = x0;
                gal_data.y0 = y0;
                gal_data.sigma = std::pow( Mxx_gal * Myy_gal - Mxy_gal * Mxy_gal, 0.25);
                gal_data.flux = 2.0 * A_gal;
            }

            /* Perform PSF correction */
            R = 1. - (Mxx_psf+Myy_psf)/(Mxx_gal+Myy_gal);
            if (shear_est == "BJ") {
                psf_corr_bj( 1.-R, (Mxx_psf-Myy_psf)/(Mxx_psf+Myy_psf),
                             2*Mxy_psf/(Mxx_psf+Myy_psf), 0.5*rho4_psf-1.,
                             (Mxx_gal-Myy_gal)/(Mxx_gal+Myy_gal),
                             2*Mxy_gal/(Mxx_gal+Myy_gal), 0.5*rho4_gal-1.,
                             gal_data.e1, gal_data.e2 );
            } else {
                psf_corr_linear( 1.-R, (Mxx_psf-Myy_psf)/(Mxx_psf+Myy_psf),
                                 2*Mxy_psf/(Mxx_psf+Myy_psf), 0.5*rho4_psf-1.,
                                 (Mxx_gal-Myy_gal)/(Mxx_gal+Myy_gal),
                                 2*Mxy_gal/(Mxx_gal+Myy_gal), 0.5*rho4_gal-1.,
                                 gal_data.e1, gal_data.e2 );
            }
            gal_data.meas_type = 'e';
            gal_data.responsivity = 1.;
        } else if (shear_est == "KSB") {

            status = psf_corr_ksb_1(
                gal_image, gal_mask, PSF_image, PSF_mask, gal_data.e1, gal_data.e2,
                gal_data.responsivity, R, flags, gal_data.x0, gal_data.y0,
                gal_data.sigma, gal_data.flux, PSF_data.x0, PSF_data.y0,
                PSF_data.sigma, hsmparams );
            gal_data.meas_type = 'g';

        } else if (shear_est == "REGAUSS") {

            status = psf_corr_regauss(
                gal_image, gal_mask, PSF_image, PSF_mask, gal_data.e1, gal_data.e2, R,
                flags, gal_data.x0, gal_data.y0, gal_data.sigma, PSF_data.x0,
                PSF_data.y0, PSF_data.sigma, gal_data.flux, hsmparams );
            gal_data.meas_type = 'e';
            gal_data.responsivity = 1.;

        } else {
            return 0x4000;
        }

        /* Report resolution factor and return */
        gal_data.resolution = R;
        return status;
    }

    // instantiate template classes for expected types
    template CppHSMShapeData EstimateShearHSMView(
        const ImageView<float>& gal_image, const ImageView<float>& PSF_image,
        const ImageView<int>& gal_mask_image,
        float sky_var, const char* shear_est, const std::string& recompute_flux, double guess_sig_gal,
        double guess_sig_PSF, double precision, double guess_x_centroid, double guess_y_centroid,
        boost::shared_ptr<HSMParams> hsmparams);
    template CppHSMShapeData EstimateShearHSMView(
        const ImageView<double>& gal_image, const ImageView<double>& PSF_image,
        const ImageView<int>& gal_mask_image,
        float sky_var, const char* shear_est, const std::string& recompute_flux, double guess_sig_gal,
        double guess_sig_PSF, double precision, double guess_x_centroid, double guess_y_centroid,
        boost::shared_ptr<HSMParams> hsmparams);
    template CppHSMShapeData EstimateShearHSMView(
        const ImageView<float>& gal_image, const ImageView<double>& PSF_image,
        const ImageView<int>& gal_mask_image,
        float sky_var, const char* shear_est, const std::string& recompute_flux, double guess_sig_gal,
        double guess_sig_PSF, double precision, double guess_x_centroid, double guess_y_centroid,
        boost::shared_ptr<HSMParams> hsmparams);
    template CppHSMShapeData EstimateShearHSMView(
        const ImageView<double>& gal_image, const ImageView<float>& PSF_image,
        const ImageView<int>& gal_mask_image,
        float sky_var, const char* shear_est, const std::string& recompute_flux, double guess_sig_gal,
        double guess_sig_PSF, double precision, double guess_x_centroid, double guess_y_centroid,
        boost::shared_ptr<HSMParams> hsmparams);
    template CppHSMShapeData EstimateShearHSMView(
        const ImageView<int>& gal_image, const ImageView<int>& PSF_image,
        const ImageView<int>& gal_mask_image,
        float sky_var, const char* shear_est, const std::string& recompute_flux, double guess_sig_gal,
        double guess_sig_PSF, double precision, double guess_x_centroid, double guess_y_centroid,
        boost::shared_ptr<HSMParams> hsmparams);

    template CppHSMShapeData FindAdaptiveMomView(
        const ImageView<float>& object_image, const ImageView<int> &object_mask_image,
        double guess_sig, double precision, double guess_x_centroid, double guess_y_centroid,
        boost::shared_ptr<HSMParams> hsmparams);
    template CppHSMShapeData FindAdaptiveMomView(
        const ImageView<double>& object_image, const ImageView<int> &object_mask_image,
        double guess_sig, double precision, double guess_x_centroid, double guess_y_centroid,
        boost::shared_ptr<HSMParams> hsmparams);
    template CppHSMShapeData FindAdaptiveMomView(
        const ImageView<int>& object_image, const ImageView<int> &object_mask_image,
        double guess_sig, double precision, double guess_x_centroid, double guess_y_centroid,
        boost::shared_ptr<HSMParams> hsmparams);

    template unsigned int general_shear_estimator(
        ConstImageView<float> gal_image, ConstImageView<int> gal_mask, 
        ConstImageView<float> PSF_image, ConstImageView<int> PSF_mask, 
        ObjectData& gal_data, ObjectData& PSF_data, const std::string& shear_est, 
        unsigned long flags, boost::shared_ptr<HSMParams> hsmparams);
    template unsigned int general_shear_estimator(
        ConstImageView<float> gal_image, ConstImageView<int> gal_mask, 
        ConstImageView<double> PSF_image, ConstImageView<int> PSF_mask, 
        ObjectData& gal_data, ObjectData& PSF_data, const std::string& shear_est, 
        unsigned long flags, boost::shared_ptr<HSMParams> hsmparams);
    template unsigned int general_shear_estimator(
        ConstImageView<double> gal_image, ConstImageView<int> gal_mask, 
        ConstImageView<float> PSF_image, ConstImageView<int> PSF_mask, 
        ObjectData& gal_data, ObjectData& PSF_data, const std::string& shear_est, 
        unsigned long flags, boost::shared_ptr<HSMParams> hsmparams);
    template unsigned int general_shear_estimator(
        ConstImageView<double> gal_image, ConstImageView<int> gal_mask, 
        ConstImageView<double> PSF_image, ConstImageView<int> PSF_mask, 
        ObjectData& gal_data, ObjectData& PSF_data, const std::string& shear_est, 
        unsigned long flags, boost::shared_ptr<HSMParams> hsmparams);
    template unsigned int general_shear_estimator(
        ConstImageView<int> gal_image, ConstImageView<int> gal_mask, 
        ConstImageView<int> PSF_image, ConstImageView<int> PSF_mask, 
        ObjectData& gal_data, ObjectData& PSF_data, const std::string& shear_est, 
        unsigned long flags, boost::shared_ptr<HSMParams> hsmparams);

    template void find_ellipmom_2(
        ConstImageView<double> data, ConstImageView<int> mask, double& A, double& x0, double& y0,
        double& Mxx, double& Mxy, double& Myy, double& rho4, double epsilon, int& num_iter,
        boost::shared_ptr<HSMParams> hsmparams);
    template void find_ellipmom_2(
        ConstImageView<float> data, ConstImageView<int> mask, double& A, double& x0, double& y0,
        double& Mxx, double& Mxy, double& Myy, double& rho4, double epsilon, int& num_iter,
        boost::shared_ptr<HSMParams> hsmparams);
    template void find_ellipmom_2(
        ConstImageView<int> data, ConstImageView<int> mask, double& A, double& x0, double& y0,
        double& Mxx, double& Mxy, double& Myy, double& rho4, double epsilon, int& num_iter,
        boost::shared_ptr<HSMParams> hsmparams);

}
}
