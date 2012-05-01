/****************************************************************
  Copyright 2003, 2004 Christopher Hirata: original code
  2007, 2009, 2010 Rachel Mandelbaum: minor modifications

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

#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <cstring>
#include "hsm/PSFCorr.h"

namespace galsim {
namespace hsm {

#define Pi    3.1415926535897932384626433832795
#define TwoPi 6.283185307179586476925286766559

#define NSIG_RG   3.0
#define NSIG_RG2  3.6
#define REGAUSS_TOO_SMALL
    // REGAUSS_TOO_SMALL: this prevents the re-Gaussianization PSF correction from completely
    // failing at the beginning, before trying to do PSF correction, due to the crudest possible PSF
    // correction (Gaussian approximation) suggesting that the galaxy is very small.  This could
    // happen for some usable galaxies particularly when they have very non-Gaussian surface
    // brightness profiles -- for example, if there's a prominent bulge that the adaptive moments
    // attempt to fit, ignoring the more extended disk.  Setting REGAUSS_TOO_SMALL is useful for
    // keeping galaxies that would have failed for that reason.  If they later turn out to be too
    // small to really use, this will be reflected in the final estimate of the resolution factor,
    // and they can be rejected after the fact.

    /* NUMERICAL RECIPES ROUTINES SECTION */
    /* Note: all NR routines included here are from nrutil.c, which is in
       the public domain.  Some of them have been modified from their
       original form. */

    /* nrerror
     * *** NUMERICAL RECIPES ERROR HANDLER ***
     * This code is in the public domain.
     */
    void nrerror(const char* error_text)
        /* Numerical Recipes standard error handler */
    {
        printf("Numerical Recipes run-time error...\n");
        printf("%s\n",error_text);
        printf("...now exiting to system...\n");
        std::exit(1);
    }

    /* dmatrix
     * *** ALLOCATES DOUBLE PRECISION MATRICES ***
     * the matrix has range m[nrl..nrh][ncl..nch]
     * This code is in the public domain.
     */

    double **dmatrix(long nrl, long nrh, long ncl, long nch)
        /* allocate a double matrix with subscript range               */
        /* m[nrl..nrh][ncl..nch]                                       */
        /* NR_END has been replaced with its value, 1.                 */
    {
        long i,j, nrow=nrh-nrl+1,ncol=nch-ncl+1;
        double **m;

        /* allocate pointers to rows */
        m=(double **) malloc((size_t)((nrow+1)*sizeof(double*)));
        if (!m) nrerror("allocation failure 1 in matrix()");
        m += 1;
        m -= nrl;

        /* allocate rows and set pointers to them */
        m[nrl]=(double *)malloc((size_t)((nrow*ncol+1)*sizeof(double)));
        if (!m[nrl]) nrerror("allocation failure 2 in matrix()");
        m[nrl] += 1;
        m[nrl] -= ncl;

        for(i=nrl+1;i<=nrh;i++) m[i]=m[i-1]+ncol;

        /* Sets the newly created matrix to zero */
        for(i=nrl;i<=nrh;i++) for(j=ncl;j<=nch;j++) m[i][j] = 0.;

        /* return pointer to array of pointers to rows */
        return m;
    }

    /* free_dmatrix
     * *** DE-ALLOCATES DOUBLE PRECISION MATRICES ***
     * the matrix has range m[nrl..nrh][ncl..nch]
     * This code is in the public domain.
     */
    void free_dmatrix(double **m, long nrl, long nrh, long ncl, long nch)
        /* free an double matrix allocated by dmatrix() */
        /* replaced NR_END => 1, FREE_ARG => (char *)   */
    {
        free((char *) (m[nrl]+ncl-1));
        free((char *) (m+nrl-1));
    }

    /* imatrix
     * *** ALLOCATES INTEGER MATRICES ***
     * the matrix has range m[nrl..nrh][ncl..nch]
     * This code is in the public domain.
     */
    int **imatrix(long nrl, long nrh, long ncl, long nch)
        /* allocate an integer matrix with subscript range             */
        /* m[nrl..nrh][ncl..nch]                                       */
        /* NR_END has been replaced with its value, 1.                 */
    {
        long i,j, nrow=nrh-nrl+1,ncol=nch-ncl+1;
        int **m;

        /* allocate pointers to rows */
        m=(int **) malloc((size_t)((nrow+1)*sizeof(int*)));
        if (!m) nrerror("allocation failure 1 in matrix()");
        m += 1;
        m -= nrl;

        /* allocate rows and set pointers to them */
        m[nrl]=(int *)malloc((size_t)((nrow*ncol+1)*sizeof(int)));
        if (!m[nrl]) nrerror("allocation failure 2 in matrix()");
        m[nrl] += 1;
        m[nrl] -= ncl;

        for(i=nrl+1;i<=nrh;i++) m[i]=m[i-1]+ncol;

        /* Sets the newly created matrix to zero */
        for(i=nrl;i<=nrh;i++) for(j=ncl;j<=nch;j++) m[i][j] = 0;

        /* return pointer to array of pointers to rows */
        return m;
    }

    /* free_imatrix
     * *** DE-ALLOCATES INTEGER MATRICES ***
     * the matrix has range m[nrl..nrh][ncl..nch]
     * This code is in the public domain.
     */
    void free_imatrix(int **m, long nrl, long nrh, long ncl, long nch)
        /* free an integer matrix allocated by imatrix() */
        /* replaced NR_END => 1, FREE_ARG => (char *)   */
    {
        free((char *) (m[nrl]+ncl-1));
        free((char *) (m+nrl-1));
    }
    /* End free_imatrix */

    /* dvector
     * *** ALLOCATES DOUBLE PRECISION VECTORS ***
     * the vector has range m[nl..nh]
     * This code is in the public domain.
     */

    double *dvector(long nl, long nh)
        /* allocate a double vector with subscript range v[nl..nh] */
        /* replaced macros, as with dmatrix etc.                   */
    {
        double *v;
        long i;

        v=(double *)malloc((size_t) ((nh-nl+2)*sizeof(double)));
        if (!v) nrerror("allocation failure in dvector()");

        /* Sets the newly created vector to zero */
        for(i=0;i<nh-nl+2;i++) v[i] = 0.;

        return(v-nl+1);
    }
    /* End dvector */

    /* free_dvector
     * *** DE-ALLOCATES DOUBLE PRECISION VECTORS ***
     * the vector has range m[nl..nh]
     * This code is in the public domain.
     */
    void free_dvector(double *v, long nl, long nh)
        /* free a double vector allocated with dvector() */
    {
        free((char*) (v+nl-1));
    }
    /* End free_dvector */

    /* lvector
     * *** ALLOCATES LONG INTEGER VECTORS ***
     * the vector has range m[nl..nh]
     * This code is in the public domain.
     */
    long *lvector(long nl, long nh)
        /* allocate a long vector with subscript range v[nl..nh] */
    {
        long *v;
        long i;

        v=(long *)malloc((size_t) ((nh-nl+2)*sizeof(long)));
        if (!v) nrerror("allocation failure in lvector()");

        /* Sets the newly created vector to zero */
        for(i=0;i<=nh-nl;i++) v[i] = 0;

        return(v-nl+1);
    }
    /* End lvector */

    /* free_lvector
     * *** DE-ALLOCATES LONG INTEGER VECTORS ***
     * the vector has range m[nl..nh]
     * This code is in the public domain.
     */
    void free_lvector(long *v, long nl, long nh)
        /* free a long vector allocated with lvector() */
    {
        free((char*) (v+nl-1));
    }
    /* End free_lvector */

    /* ivector
     * *** ALLOCATES LONG INTEGER VECTORS ***
     * the vector has range m[nl..nh]
     * This code is in the public domain.
     */
    int *ivector(long nl, long nh)
        /* allocate an integer vector with subscript range v[nl..nh] */
    {
        int *v;

        v=(int *)malloc((size_t) ((nh-nl+2)*sizeof(int)));
        if (!v) nrerror("allocation failure in ivector()");
        return(v-nl+1);
    }
    /* End ivector */

    /* free_ivector
     * *** DE-ALLOCATES INTEGER VECTORS ***
     * the vector has range m[nl..nh]
     * This code is in the public domain.
     */
    void free_ivector(int *v, long nl, long nh)
        /* free an integer vector allocated with ivector() */
    {
        free((char*) (v+nl-1));
    }
    /* End free_ivector */

    /* BEGIN OUR CODE HERE */

    /* Carry out PSF correction directly using Images, repackaging for general_shear_estimator.*/
    template <typename T>
    HSMShapeData EstimateShearHSM(Image<T> const &gal_image, Image<T> const &PSF_image, 
                                  const char *shear_est = "REGAUSS", unsigned long flags = 0xe) {
        // define variables, create output HSMShapeData struct, etc.
        HSMShapeData results;

        // repackage Images --> RectImage

        // allocate ObjectData for setting defaults etc. and passing to general_shear_estimator

        // call general_shear_estimator [generally, go through MeasMoments.cpp to make sure that
        // I've done everything needed]

        // repackage outputs from the ObjectData to an HSMShapeData struct
        return results;
    }

    /** Measure the adaptive moments of an object directly using Images, repackaging for find_ellipmom_2.*/
    template <typename T>
    HSMShapeData FindAdaptiveMom(Image<T> const &object_image, double precision = 1.0e-6) {
        // define variables, create output HSMShapeData struct, etc.
        HSMShapeData results;
        RectImage object_rect_image;
        double amp, m_xx, m_xy, m_yy;

        // repackage input Image --> RectImage

        // call find_ellipmom_2
        find_ellipmom_2(&object_rect_image, &amp, &(results.moments_centroid.x),
                        &(results.moments_centroid.y), &m_xx, &m_xy, &m_yy, &(results.moments_rho4),
                        precision, &(results.moments_n_iter));

        // repackage outputs from find_ellipmom_2 to the output HSMShapeData struct
        results.moments_amp = 2.0*amp;
        results.moments_sigma = std::pow(m_xx*m_yy-m_xy*m_xy, 0.25);
        results.image_bounds = object_image.getBounds();
        results.observed_shape.setE1E2((m_xx-m_yy)/(m_xx+m_yy), 2.*m_xy/(m_xx+m_xy));
        // results.moments_status

        return results;
    }


    /* allocate_rect_image
     * *** ALLOCATES A RectImage STRUCTURE ***
     *
     * Allocates and initializes a RectImage.  Note that the
     * map is initialized to zero and the mask is initialized
     * to the unmasked state.
     *
     * Arguments:
     * > A: RectImage to be allocated
     *   xmin: bounding box xmin
     *   xmax: bounding box xmax
     *   ymin: bounding box ymin
     *   ymax: bounding box ymax
     */

    void allocate_rect_image(RectImage *A, long xmin, long xmax, long ymin, long ymax) 
    {

        long x,y;

        /* Set bounding box */
        A->xmin = xmin;
        A->xmax = xmax;
        A->ymin = ymin;
        A->ymax = ymax;

        /* Allocate the image and mask */
        A->image = dmatrix(xmin,xmax,ymin,ymax);
        A->mask  = imatrix(xmin,xmax,ymin,ymax);

        for(x=xmin;x<=xmax;x++) for(y=ymin;y<=ymax;y++) {
            A->image[x][y] = 0.;
            A->mask[x][y] = 1;
        }

    }

    /* deallocate_rect_image
     * *** DEALLOCATES A RectImage STRUCTURE ***
     *
     * Arguemnts:
     * > A: RectImage to be deallocated
     */

    void deallocate_rect_image(RectImage *A) 
    {

        free_dmatrix(A->image,A->xmin,A->xmax,A->ymin,A->ymax);
        free_imatrix(A->mask ,A->xmin,A->xmax,A->ymin,A->ymax);
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

        double *data_null;
        double *data_i, *data_i1;
        double *data_j, *data_j1;
        double temp, theta, sintheta, oneminuscostheta;
        double wr, wi, wtemp;
        double tempr1, tempr2, tempr, tempi;

        unsigned long ndata; /* = 2*nn */
        unsigned long lcurrent; /* length of current FFT; will range from 2..n */
        unsigned long i,j,k,m,istep;

        /* Convert to null-offset vector, find number of elements */
        data_null = data + 1;
        ndata = (unsigned long)nn << 1;

        /* Bit reversal */
        data_i = data_null;
        for(i=0;i<(unsigned long)nn;i++) {

            /* Here we set data_j equal to data_null plus twice the bit-reverse of i */
            j=0;
            k=i;
            for(m=ndata>>2;m>=1;m>>=1) {
                if (k & 1) j+=m;
                k >>= 1;
            }

            /* If i<j, swap the i and j complex elements of data_null
             * Notice that these are the (2i,2i+1) and (2j,2j+1)
             * real elements.
             */
            if (i<j) {
                data_j = data_null + (j<<1);
                temp = *data_i; *data_i = *data_j; *data_j = temp;
                data_i++; data_j++;
                temp = *data_i; *data_i = *data_j; *data_j = temp;
            } else {
                data_i++;
            }

            /* Now increment data_i so it points to data_null[2i+2]; this is
             * important when we start the next iteration.
             */
            data_i++;
        }

        /* Now do successive FFTs */
        for(lcurrent=2;lcurrent<ndata;lcurrent<<=1) {

            /* Find the angle between successive points and its trig
             * functions, the sine and 1-cos. (Use 1-cos for stability.)
             */
            theta = TwoPi/lcurrent * isign;
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
                    data_j1=data_j = (data_i1=data_i = data_null + i) + lcurrent;
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

    void qho1d_wf_1(long nx, double xmin, double xstep, long Nmax, double sigma, double **psi) 
    {

        double beta, beta2__2, norm0;
        double coef1, coef2;
        double x;
        long j,n;

#ifdef N_CHECKVAL
        if (nx<=0) {
            fprintf(stderr,"Error: nx<=0 in qho1d_wf_1\n");
            std::exit(1);
        }
        if (Nmax<0) {
            fprintf(stderr,"Error: Nmax<0 in qho1d_wf_1\n");
            std::exit(1);
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
     *   data: RectImage structure containing the image to be measured
     * > moments: moments[m][n] is the m:n coefficient
     *   max_order: maximum order of moments to compute
     *   x0: center around which to compute moments (x-coordinate)
     *   y0: " (y-coordinate)
     *   sigma: width of Gaussian to measure image
     */

    void find_mom_1(
		    RectImage *data, double **moments, int max_order, double x0, double y0, double sigma) 
    {

        int m,n;
        long x, y, xmin, xmax, ymin, ymax, nx, ny;
        double **psi_x, **psi_y;

        /* Setup */
        xmin = data->xmin;
        xmax = data->xmax;
        ymin = data->ymin;
        ymax = data->ymax;
        nx = xmax-xmin+1;
        ny = ymax-ymin+1;
        psi_x = dmatrix(0,max_order,0,nx-1);
        psi_y = dmatrix(0,max_order,0,ny-1);

        /* Compute wavefunctions */
        qho1d_wf_1(nx, (double)xmin - x0, 1., max_order, sigma, psi_x);
        qho1d_wf_1(ny, (double)ymin - y0, 1., max_order, sigma, psi_y);

        /* Now let's compute moments -- outer loop is over (m,n) */
        for(m=0;m<=max_order;m++) for(n=0;n<=max_order-m;n++) {

            /* Initialize moments[m][n], then loop over map */
            moments[m][n] = 0;
            for(x=xmin;x<=xmax;x++) for(y=ymin;y<=ymax;y++) {
                if (data->mask[x][y]) {

                    /* Moment "integral" (here simply a finite sum) */
                    moments[m][n] += data->image[x][y] * psi_x[m][x-xmin]
                        * psi_y[n][y-ymin];

                } /* End mask condition */
            } /* End (x,y) loop */
        } /* End (m,n) loop */

        /* Cleanup memory */
        free_dmatrix(psi_x,0,max_order,0,nx-1);
        free_dmatrix(psi_y,0,max_order,0,ny-1);
    }

    /* find_mom_2
     * *** FINDS ADAPTIVE CIRCULAR MOMENTS OF AN IMAGE ***
     *
     * Computes the center, 1sigma radius, and moments of an image.  "Guesses"
     * must be given for x0, y0, and sigma.
     *
     * Arguments:
     *   data: RectImage structure containing the image to be measured
     * > moments: moments[m][n] is the m:n coefficient
     *   max_order: maximum order of moments to compute
     * > x0: Gaussian-weighted centroid (x-coordinate)
     * > y0: " (y-coordinate)
     * > sigma: width of Gaussian to measure image (best fit 1sigma)
     *   epsilon: accuracy (in x0, y0, and sigma as a fraction of sigma.
     *      The value of sigma used for the convergence criterion is the
     *      minimum of the "guessed" value and the "current" value.)
     * > num_iter: number of iterations required for convergence
     */

    void find_mom_2(
        RectImage *data, double **moments, int max_order,
        double *x0, double *y0, double *sigma, double epsilon, int *num_iter) 
    {

#define ADAPT_ORDER 2
#define MAX_MOM2_ITER 400
#define NUM_ITER_DEFAULT -1

        double sigma0 = *sigma;
        double convergence_factor = 1; /* Ensure at least one iteration. */
        double dx, dy, dsigma;
        double **iter_moments;

        *num_iter = 0;
        iter_moments = dmatrix(0,ADAPT_ORDER,0,ADAPT_ORDER);

#ifdef N_CHECKVAL
        if (epsilon <= 0) {
            fprintf(stderr,"Error: epsilon out of range in find_mom_2.\n");
            std::exit(1);
        }
#endif

        /* Iterate until we converge */
        while(convergence_factor > epsilon) {

            /* Get moments */
            find_mom_1(data,iter_moments,ADAPT_ORDER,*x0,*y0,*sigma);

            /* Get updates to weight function */
            dx     = 1.414213562373 * iter_moments[1][0] / iter_moments[0][0];
            dy     = 1.414213562373 * iter_moments[0][1] / iter_moments[0][0];
            dsigma = 0.7071067811865
                * (iter_moments[2][0]+iter_moments[0][2]) / iter_moments[0][0];

#define BOUND_CORRECT_WEIGHT 0.25
            if (dx     >  BOUND_CORRECT_WEIGHT) dx     =  BOUND_CORRECT_WEIGHT;
            if (dx     < -BOUND_CORRECT_WEIGHT) dx     = -BOUND_CORRECT_WEIGHT;
            if (dy     >  BOUND_CORRECT_WEIGHT) dy     =  BOUND_CORRECT_WEIGHT;
            if (dy     < -BOUND_CORRECT_WEIGHT) dy     = -BOUND_CORRECT_WEIGHT;
            if (dsigma >  BOUND_CORRECT_WEIGHT) dsigma =  BOUND_CORRECT_WEIGHT;
            if (dsigma < -BOUND_CORRECT_WEIGHT) dsigma = -BOUND_CORRECT_WEIGHT;

            /* Convergence */
            convergence_factor = std::abs(dx)>std::abs(dy)? std::abs(dx): std::abs(dy);
            if (std::abs(dsigma)>convergence_factor) convergence_factor = std::abs(dsigma);
            if (*sigma<sigma0) convergence_factor *= sigma0/(*sigma);

            /* Update numbers */
            *x0    += dx     * (*sigma);
            *y0    += dy     * (*sigma);
            *sigma += dsigma * (*sigma);

            if (++ *num_iter > MAX_MOM2_ITER) {
                fprintf(stderr,"Warning: too many iterations in find_mom_2.\n");
                convergence_factor = 0.;
                *num_iter = NUM_ITER_DEFAULT;
            }
        }

        /* Clean up memory */
        free_dmatrix(iter_moments,0,ADAPT_ORDER,0,ADAPT_ORDER);

        /* Now compute all of the moments that we want to return */
        find_mom_1(data,moments,max_order,*x0,*y0,*sigma);
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
     * M = adaptive covariance matrix.
     *
     * Arguments:
     *   data: the input image (RectImage format)
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

    void find_ellipmom_1(
        RectImage *data, double x0, double y0, double Mxx,
        double Mxy, double Myy, double *A, double *Bx, double *By, double *Cxx,
        double *Cxy, double *Cyy, double *rho4w) 
    {

        long x,y;
        long xmin = data->xmin;
        long xmax = data->xmax;
        long ymin = data->ymin;
        long ymax = data->ymax;
        double Minv_xx, TwoMinv_xy, Minv_yy, detM;
        double rho2, intensity;
        double x_x0, y_y0;
        int *maskptr;
        double *imageptr;
        double intensity__x_x0, intensity__y_y0, TwoMinv_xy__x_x0, Minv_xx__x_x0__x_x0;
        double *Minv_yy__y_y0__y_y0, *myyptr;

        /* Compute M^{-1} for use in computing weights */
        detM = Mxx * Myy - Mxy * Mxy;
        if (detM<=0 || Mxx<=0 || Myy<=0) {
            fprintf(stderr, "Error: non positive definite adaptive moments!\n");
            std::exit(1);
        }
        Minv_xx    =  Myy/detM;
        TwoMinv_xy = -Mxy/detM * 2.0;
        Minv_yy    =  Mxx/detM;

        /* Generate Minv_yy__y_y0__y_y0 array */
        Minv_yy__y_y0__y_y0 = dvector(ymin,ymax);
        for(y=ymin;y<=ymax;y++) Minv_yy__y_y0__y_y0[y] = Minv_yy*(y-y0)*(y-y0);

        /* Now let's initialize the outputs and then sum
         * over all the unmasked pixels
         */
        *A = *Bx = *By = *Cxx = *Cxy = *Cyy = *rho4w = 0.;
        for(x=xmin;x<=xmax;x++) {
            /* Use these pointers to speed up referencing arrays */
            maskptr  = data->mask[x] + ymin;   
            imageptr = data->image[x] + ymin;
            x_x0 = x-x0;
            TwoMinv_xy__x_x0 = TwoMinv_xy * x_x0;
            Minv_xx__x_x0__x_x0 = Minv_xx * x_x0 * x_x0;
            myyptr = Minv_yy__y_y0__y_y0 + ymin;
            y_y0 = ymin - 1 - y0;
            for(y=ymin;y<=ymax;y++) {
                if (*(maskptr++)) {
                    /* Compute displacement from weight centroid, then
                     * get elliptical radius and weight.
                     */
                    y_y0 += 1.;
                    rho2 = Minv_xx__x_x0__x_x0 + TwoMinv_xy__x_x0*y_y0 + *(myyptr++);
                    intensity = std::exp(-0.5 * rho2) * *(imageptr++);

                    /* Now do the addition */
                    *A    += intensity;
                    *Bx   += intensity__x_x0 = intensity * x_x0;
                    *By   += intensity__y_y0 = intensity * y_y0;
                    *Cxx  += intensity__x_x0 * x_x0;
                    *Cxy  += intensity__x_x0 * y_y0;
                    *Cyy  += intensity__y_y0 * y_y0;
                    *rho4w+= intensity * rho2 * rho2;
                }
            }
        }

        /* Clean up memory */
        free_dvector(Minv_yy__y_y0__y_y0, ymin,ymax);
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
     *   data: RectImage structure containing the image
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

    void find_ellipmom_2(
        RectImage *data, double *A, double *x0, double *y0,
        double *Mxx, double *Mxy, double *Myy, double *rho4, double epsilon, int *num_iter) 
    {

        double convergence_factor = 1.0;
        double Amp,Bx,By,Cxx,Cxy,Cyy;
        double semi_a2, semi_b2, two_psi;
        double dx, dy, dxx, dxy, dyy;
        double shiftscale, shiftscale0=0.;
        double x00 = *x0;
        double y00 = *y0;

        *num_iter = 0;

#ifdef N_CHECKVAL
        if (epsilon <= 0) {
            fprintf(stderr,"Error: epsilon out of range in find_mom_2.\n");
            std::exit(1);
        }
#endif

        /* Iterate until we converge */
        while(convergence_factor > epsilon) {
	  
            /* Get moments */
            find_ellipmom_1(data, *x0, *y0, *Mxx, *Mxy, *Myy, &Amp, &Bx, &By,
                            &Cxx, &Cxy, &Cyy, rho4);

            /* Compute configuration of the weight function */
            two_psi = std::atan2( 2* *Mxy, *Mxx- *Myy );
            semi_a2 = 0.5 * ((*Mxx+*Myy) + (*Mxx-*Myy)*std::cos(two_psi)) + *Mxy*std::sin(two_psi);
            semi_b2 = *Mxx + *Myy - semi_a2;

            if (semi_b2 <= 0) {
                fprintf(stderr,"Error: non positive-definite weight in find_ellipmom_2.\n");
                std::exit(1);
            }

            shiftscale = std::sqrt(semi_b2);
            if (*num_iter == 0) shiftscale0 = shiftscale;

            /* Now compute changes to x0, etc. */
            dx = 2. * Bx / (Amp * shiftscale);
            dy = 2. * By / (Amp * shiftscale);
            dxx = 4. * (Cxx/Amp - 0.5* *Mxx) / semi_b2;
            dxy = 4. * (Cxy/Amp - 0.5* *Mxy) / semi_b2;
            dyy = 4. * (Cyy/Amp - 0.5* *Myy) / semi_b2;

            if (dx     >  BOUND_CORRECT_WEIGHT) dx     =  BOUND_CORRECT_WEIGHT;
            if (dx     < -BOUND_CORRECT_WEIGHT) dx     = -BOUND_CORRECT_WEIGHT;
            if (dy     >  BOUND_CORRECT_WEIGHT) dy     =  BOUND_CORRECT_WEIGHT;
            if (dy     < -BOUND_CORRECT_WEIGHT) dy     = -BOUND_CORRECT_WEIGHT;
            if (dxx    >  BOUND_CORRECT_WEIGHT) dxx    =  BOUND_CORRECT_WEIGHT;
            if (dxx    < -BOUND_CORRECT_WEIGHT) dxx    = -BOUND_CORRECT_WEIGHT;
            if (dxy    >  BOUND_CORRECT_WEIGHT) dxy    =  BOUND_CORRECT_WEIGHT;
            if (dxy    < -BOUND_CORRECT_WEIGHT) dxy    = -BOUND_CORRECT_WEIGHT;
            if (dyy    >  BOUND_CORRECT_WEIGHT) dyy    =  BOUND_CORRECT_WEIGHT;
            if (dyy    < -BOUND_CORRECT_WEIGHT) dyy    = -BOUND_CORRECT_WEIGHT;

            /* Convergence tests */
            convergence_factor = std::abs(dx)>std::abs(dy)? std::abs(dx): std::abs(dy);
            convergence_factor *= convergence_factor;
            if (std::abs(dxx)>convergence_factor) convergence_factor = std::abs(dxx);
            if (std::abs(dxy)>convergence_factor) convergence_factor = std::abs(dxy);
            if (std::abs(dyy)>convergence_factor) convergence_factor = std::abs(dyy);
            convergence_factor = std::sqrt(convergence_factor);
            if (shiftscale<shiftscale0) convergence_factor *= shiftscale0/shiftscale;

            /* Now update moments */
            *x0 += dx * shiftscale;
            *y0 += dy * shiftscale;
            *Mxx += dxx * semi_b2;
            *Mxy += dxy * semi_b2;
            *Myy += dyy * semi_b2;

            /* If the moments have gotten too large, or the centroid is out of range,
             * report a failure */
#define MAX_AMOMENT 8000.0
#define MAX_ASHIFT 15.0
            if (std::abs(*Mxx)>MAX_AMOMENT || std::abs(*Mxy)>MAX_AMOMENT || std::abs(*Myy)>MAX_AMOMENT
                || std::abs(*x0-x00)>MAX_ASHIFT || std::abs(*y0-y00)>MAX_ASHIFT) {
	      fprintf(stderr, "Error: adaptive moment failed: %lf %lf %lf %lf %lf %d\n",std::abs(*Mxx),std::abs(*Mxy),std::abs(*Myy),std::abs(*x0-x00),std::abs(*y0-y00),*num_iter);
                std::exit(1);
            }

            if (++ *num_iter > MAX_MOM2_ITER) {
                fprintf(stderr,"Warning: too many iterations in find_mom_2.\n");
                convergence_factor = 0.;
                *num_iter = NUM_ITER_DEFAULT;
            }
        }

        /* Re-normalize rho4 */
        *A = Amp;
        *rho4 /= Amp;
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
     *   image1: 1st image to be convolved, RectImage format
     *   image2: 2nd image to be convolved, RectImage format
     * > image_out: output (convolved) image, RectImage format
     */

    void fast_convolve_image_1(
        RectImage *image1, RectImage *image2, RectImage *image_out) 
    {

        long dim1x, dim1y, dim1o, dim1, dim2, dim3, dim4;
        double *Ax, *Bx;
        double **m1, **m2, **mout;
        double xr,xi,yr,yi;
        long i,i_conj,j,k,ii,ii_conj;
        long out_xmin, out_xmax, out_ymin, out_ymax, out_xref, out_yref;

        /* Determine array sizes:
         * dim1 = (linear) size of pixel grid used for FFT
         * dim2 = 2*dim1
         * dim3 = dim2*dim2
         * dim4 = 2*dim3
         */
        dim1x = image1->xmax - image1->xmin + image2->xmax - image2->xmin + 2;
        dim1y = image1->ymax - image1->ymin + image2->ymax - image2->ymin + 2;
        dim1o = (dim1x>dim1y)? dim1x: dim1y;
        dim1 = 1; while(dim1<dim1o) dim1 <<= 1; /* dim1 must be a power of two */
        dim2 = dim1 << 1;
        dim3 = dim2 * dim2;
        dim4 = dim3 << 1;

        /* Allocate & initialize memory */
        m1 = dmatrix(0,dim1-1,0,dim1-1);
        m2 = dmatrix(0,dim1-1,0,dim1-1);
        mout = dmatrix(0,dim1-1,0,dim1-1);
        for(i=0;i<dim1;i++) for(j=0;j<dim1;j++) m1[i][j] = m2[i][j] = mout[i][j] = 0.;
        Ax = dvector(1,dim4);
        Bx = dvector(1,dim4);
        for(i=1;i<=dim4;i++) Ax[i]=Bx[i]=0;

        /* Build input maps */
        for(i=image1->xmin;i<=image1->xmax;i++)
            for(j=image1->ymin;j<=image1->ymax;j++)
                if (image1->mask[i][j])
                    m1[i-image1->xmin][j-image1->ymin] = image1->image[i][j];
        for(i=image2->xmin;i<=image2->xmax;i++)
            for(j=image2->ymin;j<=image2->ymax;j++)
                if (image2->mask[i][j])
                    m2[i-image2->xmin][j-image2->ymin] = image2->image[i][j];

        /* Build the arrays for FFT -
         * - put m1 and m2 into the real and imaginary parts of Bx, respectively. */
        for(i=0;i<dim1;i++) for(j=0;j<dim1;j++) {
            k=2*(dim2*i+j)+1;
            Bx[k  ] = m1[i][j];
            Bx[k+1] = m2[i][j];
        }

        /* We've filled only part of Bx, the other locations are for
         * zero padding.  First we separate the real (m1) and imaginary (m2) parts of the FFT,
         * then multiply to get the convolution.
         */
        fourier_trans_1(Bx,dim3,1);
        for(i=0;i<dim3;i++) {
            i_conj = i==0? 0: dim3-i;      /* part of FFT of B holding complex conjugate mode */
            ii      = 2*i     +1;
            ii_conj = 2*i_conj+1;
            xr = 0.5 * (  Bx[ii  ] + Bx[ii_conj  ] );
            xi = 0.5 * (  Bx[ii+1] - Bx[ii_conj+1] );
            yr = 0.5 * (  Bx[ii+1] + Bx[ii_conj+1] );
            yi = 0.5 * ( -Bx[ii  ] + Bx[ii_conj  ] );
            Ax[ii  ] = xr*yr-xi*yi;      /* complex multiplication */
            Ax[ii+1] = xr*yi+xi*yr;
        }
        fourier_trans_1(Ax,dim3,-1);   /* Reverse FFT Ax to get convolved image */
        for(i=0;i<dim1;i++)
            for(j=0;j<dim1;j++)
                mout[i][j] = Ax[2*(dim2*i+j)+1] / (double)dim3;

        /* Calculate the effective bounding box for the output image,
         * [out_xmin..out_xmax][out_ymin..out_ymax], and the offset between mout and
         * image_out, namely (out_xref,out_yref)
         */
        out_xmin = out_xref = image1->xmin + image2->xmin;
        out_xmax =            image1->xmax + image2->xmax;
        out_ymin = out_yref = image1->ymin + image2->ymin;
        out_ymax =            image1->ymax + image2->ymax;
        if (out_xmin<image_out->xmin) out_xmin = image_out->xmin;
        if (out_xmax>image_out->xmax) out_xmax = image_out->xmax;
        if (out_ymin<image_out->ymin) out_ymin = image_out->ymin;
        if (out_ymax>image_out->ymax) out_ymax = image_out->ymax;

        /* And now do the writing */
        for(i=out_xmin;i<=out_xmax;i++)
            for(j=out_ymin;j<=out_ymax;j++)
                if(image_out->mask[i][j])
                    image_out->image[i][j] += mout[i-out_xref][j-out_yref];

        /* Clean up memory */
        free_dmatrix(m1,0,dim1-1,0,dim1-1);
        free_dmatrix(m2,0,dim1-1,0,dim1-1);
        free_dmatrix(mout,0,dim1-1,0,dim1-1);
        free_dvector(Ax,1,dim4);
        free_dvector(Bx,1,dim4);
    }

    void matrix22_invert(double *a, double *b, double *c, double *d) 
    {

        double det,temp;

        det = (*a)*(*d)-(*b)*(*c);
        *b = -(*b); *c = -(*c);
        temp = *a; *a = *d; *d = temp;
        *a /= det; *b /= det; *c /= det; *d /= det;
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
                   double *e1out, double *e2out) 
    {

        /* This is eq. 2-13 of Bernstein & Jarvis */
        /* Shear ea is applied, then eb -- it matters! */
        double dotp, factor;

        dotp = e1a*e1b + e2a*e2b;
        factor = (1.-std::sqrt(1-e1b*e1b-e2b*e2b)) / (e1b*e1b + e2b*e2b);
        *e1out = (e1a + e1b + e2b*factor*(e2a*e1b - e1a*e2b))/(1+dotp);
        *e2out = (e2a + e2b + e1b*factor*(e1a*e2b - e2a*e1b))/(1+dotp);
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
        double e2o, double a4o, double *e1, double *e2) 
    {

        double e1red, e2red; /* ellipticities reduced to circular PSF */
        double sig2ratio;
        double coshetap, coshetao;
        double R;

        /* Take us to sig2ratio = sigma2(P)/sigma2(O) since this is shear-invariant */
        coshetap = 1./std::sqrt(1-e1p*e1p-e2p*e2p);
        coshetao = 1./std::sqrt(1-e1o*e1o-e2o*e2o);
        sig2ratio = Tratio * coshetao/coshetap; /* since sigma2 = T / cosh eta */

        shearmult(e1o,e2o,-e1p,-e2p,&e1red,&e2red);

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
        double e2o, double a4o, double *e1, double *e2) 
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

        shearmult(e1o,e2o,-e1p,-e2p,&e1red,&e2red);

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
     *   gal_image: image of measured galaxy (RectImage format)
     *   PSF: PSF map (RectImage format)
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

    unsigned int psf_corr_ksb_1(
        RectImage *gal_image, RectImage *PSF, double *e1, double *e2,
        double *responsivity, double *R, unsigned long flags, double *x0_gal, double *y0_gal,
        double *sig_gal, double *flux_gal, double *x0_psf, double *y0_psf, double *sig_psf) 
    {

#define KSB_MOMENTS_MAX 4

        unsigned int status = 0;
        int num_iter;
        double **moments;
        double **psfmoms;
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
#define FAILED_MOMENTS (-1000.0)
        *e1 = *e2 = *R = FAILED_MOMENTS;

        moments = dmatrix(0,KSB_MOMENTS_MAX,0,KSB_MOMENTS_MAX);
        psfmoms = dmatrix(0,KSB_MOMENTS_MAX,0,KSB_MOMENTS_MAX);

        /* Determine the adaptive variance of the measured galaxy */
        x0 = *x0_gal;
        y0 = *y0_gal;
        sigma0 = *sig_gal;
        find_mom_2(gal_image, moments, KSB_MOMENTS_MAX, x0_gal, y0_gal, sig_gal,
                   1.0e-6, &num_iter);
        if (num_iter == NUM_ITER_DEFAULT) {
            status |= 0x0002; /* Report convergence failure */
            *x0_gal = x0;
            *y0_gal = y0;
            *sig_gal = sigma0;
            find_mom_1(gal_image, moments, KSB_MOMENTS_MAX, x0, y0, sigma0);
        }
        *flux_gal = 3.544907701811 * *sig_gal * moments[0][0];

        /* Determine the centroid of the PSF */
        x0 = *x0_psf;
        y0 = *y0_psf;
        sigma0 = *sig_psf;
        find_mom_2(PSF, psfmoms, KSB_MOMENTS_MAX, x0_psf, y0_psf, sig_psf, 1.0e-6, &num_iter);
        if (num_iter == NUM_ITER_DEFAULT) {
            status |= 0x0001; /* Report convergence failure */
            *x0_psf = x0;
            *y0_psf = y0;
            *sig_psf = sigma0;
        }

        /* ... but we want the moments with the galaxy weight fcn */
        find_mom_1(PSF, psfmoms, KSB_MOMENTS_MAX, *x0_psf, *y0_psf, *sig_gal);

        /* Get resolution factor */
        *R = 1. - (*sig_psf**sig_psf)/(*sig_gal**sig_gal);

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
        P00  = psfmoms[0][0];
        P20r = 0.5 * (psfmoms[2][0] - psfmoms[0][2]);
        P20i = 0.7071067811865 * psfmoms[1][1];
        P11  = 0.7071067811865 * (psfmoms[2][0] + psfmoms[0][2]);
        P40r = 0.25 * (psfmoms[4][0] + psfmoms[0][4]) - 0.6123724356958 * psfmoms[2][2];
        P40i = 0.5 * (psfmoms[3][1]-psfmoms[1][3]);
        P31r = 0.5 * (psfmoms[4][0]-psfmoms[0][4]);
        P31i = 0.5 * (psfmoms[3][1]+psfmoms[1][3]);
        P22  = 0.6123724356958 * (psfmoms[4][0]+psfmoms[0][4]) + 0.5 * psfmoms[2][2];

        I00  = moments[0][0];
        I20r = 0.5 * (moments[2][0] - moments[0][2]);
        I20i = 0.7071067811865 * moments[1][1];
        I11  = 0.7071067811865 * (moments[2][0] + moments[0][2]);
        I40r = 0.25 * (moments[4][0] + moments[0][4]) - 0.6123724356958 * moments[2][2];
        I40i = 0.5 * (moments[3][1]-moments[1][3]);
        I31r = 0.5 * (moments[4][0]-moments[0][4]);
        I31i = 0.5 * (moments[3][1]+moments[1][3]);
        I22  = 0.6123724356958 * (moments[4][0]+moments[0][4]) + 0.5 * moments[2][2];

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
        matrix22_invert(&ppsQQ,&ppsQU,&ppsUQ,&ppsUU);

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

        matrix22_invert(&PQQ,&PQU,&PUQ,&PUU);

        /* This finally gives us a shear. */
        *e1 = PQQ*eQ + PQU*eU;
        *e2 = PUQ*eQ + PUU*eU;
        *responsivity = 1.;

        /* And now clean up the memory and exit */
        free_dmatrix(moments,0,KSB_MOMENTS_MAX,0,KSB_MOMENTS_MAX);
        free_dmatrix(psfmoms,0,KSB_MOMENTS_MAX,0,KSB_MOMENTS_MAX);
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
     *   0x00000004: cut off Gaussian approximator at NSIG_RG sigma (saves computation time in
     *      the convolution step)
     *   0x00000008: cut off PSF residual at NSIG_RG2 sigma (saves computation time in
     *      the convolution step)
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

    unsigned int psf_corr_regauss(
        RectImage *gal_image, RectImage *PSF, double *e1,
        double *e2, double *R, unsigned long flags, double *x0_gal,
        double *y0_gal, double *sig_gal, double *x0_psf, double *y0_psf,
        double *sig_psf, double *flux_gal) 
    {

        /* WARNING: we have some returns buried in the ellipticity section, so we
         * can't allocate memory until after these (unless we want memory bugs or
         * we want tho de-allocate before any of the returns) !!!
         */

        long x,y;
        int num_iter;
        unsigned int status = 0;
        double A_g, Mxxpsf, Mxypsf, Myypsf, rho4psf, flux_psf, sum;
        double A_I, Mxxgal, Mxygal, Myygal, rho4gal;
        double Minvpsf_xx, Minvpsf_xy, Minvpsf_yy, detM, center_amp_psf;
        double dx, dy;
#ifdef REGAUSS_TOO_SMALL
        double a2, b2, two_phi;
#endif
        double x0_old=0., y0_old=0., Mfxx, Mfxy, Mfyy, detMf, Minvf_xx, Minvf_xy, Minvf_yy;
        double Tpsf, e1psf, e2psf;
        double Tgal, e1gal, e2gal;
        RectImage PSF_resid, fgauss, Iprime;
        long fgauss_xmin, fgauss_xmax, fgauss_ymin, fgauss_ymax;
        double fgauss_xctr, fgauss_yctr, fgauss_xsig, fgauss_ysig;
        long pxmin, pxmax, pymin, pymax;

        /* Initialize -- if we don't set the outputs, they will be reported
         * as failures.
         */
        *e1 = *e2 = *R = FAILED_MOMENTS;

        /* Get the PSF flux */
        flux_psf = 0;
        for(x=PSF->xmin;x<=PSF->xmax;x++)
            for(y=PSF->ymin;y<=PSF->ymax;y++)
                if (PSF->mask[x][y])
                    flux_psf += PSF->image[x][y];

        /* Recompute the galaxy flux only if the relevant flag is set */
        if (flags & 0x00000001) {
            *flux_gal = 0;
            for(x=gal_image->xmin;x<=gal_image->xmax;x++)
                for(y=gal_image->ymin;y<=gal_image->ymax;y++)
                    if (gal_image->mask[x][y])
                        *flux_gal += gal_image->image[x][y];
        }

        /* Get the elliptical adaptive moments of PSF */
        Mxxpsf = Myypsf = *sig_psf * *sig_psf;
        Mxypsf = 0.;
        find_ellipmom_2(PSF, &A_g, x0_psf, y0_psf, &Mxxpsf, &Mxypsf, &Myypsf, &rho4psf,
                        1.0e-6, &num_iter);
        if (num_iter == NUM_ITER_DEFAULT) {
            *x0_psf = x0_old;
            *y0_psf = y0_old;
            status |= 0x0001;
        }

        /* Get the elliptical adaptive moments of galaxy */
        Mxxgal = Myygal = *sig_gal * *sig_gal;
        Mxygal = 0.;
        find_ellipmom_2(gal_image, &A_I, x0_gal, y0_gal, &Mxxgal, &Mxygal, &Myygal, &rho4gal,
                        1.0e-6, &num_iter);
        if (num_iter == NUM_ITER_DEFAULT) {
            *x0_gal = x0_old;
            *y0_gal = y0_old;
            status |= 0x0002;
        }

        /* If the flags tell us to, we reset the galaxy flux estimate */
        if (flags & 0x00000002) {
            *flux_gal = rho4gal * A_I;
        }

        /* Compute approximate deconvolved moments (i.e. without non-Gaussianity correction).
         * We also test this matrix for positive definiteness.
         */
        Mfxx = Mxxgal - Mxxpsf;
        Mfxy = Mxygal - Mxypsf;
        Mfyy = Myygal - Myypsf;
        detMf = Mfxx * Mfyy - Mfxy * Mfxy;
#ifndef REGAUSS_TOO_SMALL
        if (Mfxx<=0 || Mfyy<=0 || detMf<=0) status |= 0x0004;
#endif
#ifdef REGAUSS_TOO_SMALL

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

#endif

        /* Test to see if anything has gone wrong -- if so, complain! */
        if (status) return (status);

        /* <--- OK to allocate memory after here ---> */

        /* We also need the Gaussian de-convolved fit.  First get bounding box */
        fgauss_xmin = gal_image->xmin - PSF->xmax;
        fgauss_xmax = gal_image->xmax - PSF->xmin;
        fgauss_ymin = gal_image->ymin - PSF->ymax;
        fgauss_ymax = gal_image->ymax - PSF->ymin;
        fgauss_xctr = *x0_gal - *x0_psf;
        fgauss_yctr = *y0_gal - *y0_psf;
        fgauss_xsig = std::sqrt(Mfxx>1? Mfxx: 1);
        fgauss_ysig = std::sqrt(Mfyy>1? Mfyy: 1);

        /* Shrink if the box extends beyond NSIG_RG sigma range */
        if (flags & 0x00000004) {
            if (fgauss_xmin < fgauss_xctr - NSIG_RG*fgauss_xsig)
                fgauss_xmin = (long) std::floor(fgauss_xctr - NSIG_RG*fgauss_xsig);
            if (fgauss_xmax > fgauss_xctr + NSIG_RG*fgauss_xsig)
                fgauss_xmax = (long) std::ceil (fgauss_xctr + NSIG_RG*fgauss_xsig);
            if (fgauss_ymin < fgauss_yctr - NSIG_RG*fgauss_ysig)
                fgauss_ymin = (long) std::floor(fgauss_yctr - NSIG_RG*fgauss_ysig);
            if (fgauss_ymax > fgauss_yctr + NSIG_RG*fgauss_ysig)
                fgauss_ymax = (long) std::ceil (fgauss_yctr + NSIG_RG*fgauss_ysig);
        }

        allocate_rect_image(&fgauss, fgauss_xmin, fgauss_xmax, fgauss_ymin, fgauss_ymax);

        Minvf_xx =  Mfyy/detMf;
        Minvf_xy = -Mfxy/detMf;
        Minvf_yy =  Mfxx/detMf;
        sum = 0.;
        for(x=fgauss.xmin;x<=fgauss.xmax;x++) {
            for(y=fgauss.ymin;y<=fgauss.ymax;y++) {
                dx = x - *x0_gal + *x0_psf;
                dy = y - *y0_gal + *x0_psf;
                sum += fgauss.image[x][y] = 
                    exp (-0.5 * ( Minvf_xx*dx*dx + Minvf_yy*dy*dy ) - Minvf_xy*dx*dy);
            }
        }

        /* Properly normalize fgauss */
        for(x=fgauss.xmin;x<=fgauss.xmax;x++)
            for(y=fgauss.ymin;y<=fgauss.ymax;y++)
                fgauss.image[x][y] *= (*flux_gal)/(sum*flux_psf); 

        /* Figure out the size of the bounding box for the PSF residual.
         * We don't necessarily need the whole PSF,
         * just the part that will affect regions inside the NSIG_RG2 sigma ellipse 
         * of the Intensity.
         */
        pxmin = PSF->xmin;
        pxmax = PSF->xmax;
        pymin = PSF->ymin;
        pymax = PSF->ymax;
        if (flags & 0x00000008) {
            pxmin = (long) std::floor(*x0_psf - NSIG_RG2*std::sqrt(Mxxgal) - NSIG_RG*fgauss_xsig );
            pxmax = (long) std::ceil (*x0_psf + NSIG_RG2*std::sqrt(Mxxgal) + NSIG_RG*fgauss_xsig );
            pymin = (long) std::floor(*y0_psf - NSIG_RG2*std::sqrt(Myygal) - NSIG_RG*fgauss_ysig );
            pymax = (long) std::ceil (*y0_psf + NSIG_RG2*std::sqrt(Myygal) + NSIG_RG*fgauss_ysig );
            if (PSF->xmin >= pxmin) pxmin = PSF->xmin;
            if (PSF->xmax <= pxmax) pxmax = PSF->xmax;
            if (PSF->ymin >= pymin) pymin = PSF->ymin;
            if (PSF->ymax <= pymax) pymax = PSF->ymax;
        }

        /* Now let's compute the residual from the PSF fit.  This is called
         * - epsilon in Hirata & Seljak.
         */
        allocate_rect_image(&PSF_resid, pxmin, pxmax, pymin, pymax);
        detM = Mxxpsf * Myypsf - Mxypsf * Mxypsf;
        Minvpsf_xx =  Myypsf/detM;
        Minvpsf_xy = -Mxypsf/detM;
        Minvpsf_yy =  Mxxpsf/detM;
        center_amp_psf = flux_psf / (TwoPi * std::sqrt(detM));
        for(x=PSF_resid.xmin;x<=PSF_resid.xmax;x++)
            for(y=PSF_resid.ymin;y<=PSF_resid.ymax;y++) {
                dx = x - *x0_psf;
                dy = y - *y0_psf;

                /* Set the PSF_resid mask to be the same as PSF mask.  Note that we
                 * only set the image for unmasked pixels.  The "=" in the condition
                 * really is an "=", not "==".
                 */
                if ((PSF_resid.mask[x][y] = PSF->mask[x][y]))
                    PSF_resid.image[x][y] = -PSF->image[x][y] + center_amp_psf * 
                        exp (-0.5 * ( Minvpsf_xx*dx*dx + Minvpsf_yy*dy*dy ) - Minvpsf_xy*dx*dy);
            }

        /* Now compute the re-Gaussianized galaxy image */
        allocate_rect_image(
            &Iprime, gal_image->xmin,gal_image->xmax,gal_image->ymin,gal_image->ymax);
        for(x=gal_image->xmin;x<=gal_image->xmax;x++)
            for(y=gal_image->ymin;y<=gal_image->ymax;y++)
                /* Iprime mask will be set equal to the gal_image mask */
                if ((Iprime.mask[x][y] = gal_image->mask[x][y]))
                    Iprime.image[x][y] = gal_image->image[x][y];
        fast_convolve_image_1(&fgauss, &PSF_resid, &Iprime);

        /* Now that Iprime is constructed, we measure it */
        find_ellipmom_2(&Iprime, &A_I, x0_gal, y0_gal, &Mxxgal, &Mxygal, &Myygal,
                        &rho4gal, 1.0e-6, &num_iter);
        if (num_iter == NUM_ITER_DEFAULT) {
            *x0_gal = x0_old;
            *y0_gal = y0_old;
            status |= 0x0008;
        }
        if (Mxxgal<=0 || Myygal<=0 || Mxxgal*Myygal<=Mxygal*Mxygal ) {
            fprintf(stderr, "Error: non positive definite adaptive moments.\n");
        }
        *sig_gal = std::pow( Mxxgal*Myygal - Mxygal*Mxygal, 0.25);

        /* And do the PSF correction */
        Tgal  = Mxxgal + Myygal;
        e1gal = (Mxxgal - Myygal) / Tgal;
        e2gal = 2 *Mxygal / Tgal;
        Tpsf  = Mxxpsf + Myypsf;
        e1psf = (Mxxpsf - Myypsf) / Tpsf;
        e2psf = 2 * Mxypsf / Tpsf;

        psf_corr_bj(Tpsf/Tgal, e1psf, e2psf, 0., e1gal, e2gal, 0.5*rho4gal-1., e1, e2); 
        /* Use 0 for radial 4th moment of PSF because it's been * re-Gaussianized.  */

        *R = 1. - Tpsf/Tgal;

        /* Clean up memory; exit */
        deallocate_rect_image(&PSF_resid);
        deallocate_rect_image(&fgauss);
        deallocate_rect_image(&Iprime);
        return(status);
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

    unsigned int general_shear_estimator(
        RectImage *gal_image, RectImage *PSF,
        ObjectData *gal_data, ObjectData *PSF_data, char *shear_est, unsigned long flags) 
    {

        //int max_order_psf, max_order_gal, num_args;
        int length_shear_est;
        unsigned short int est_id = 0; /* which estimator do we want? */
        unsigned int status = 0;
        int num_iter;
        double x0, y0, R;
        double A_gal, Mxx_gal, Mxy_gal, Myy_gal, rho4_gal;
        double A_psf, Mxx_psf, Mxy_psf, Myy_psf, rho4_psf;

        /* Estimator macros */
#define EST_BJ 1
#define EST_LINEAR 2
#define EST_KSB 3
#define EST_REGAUSS 4
#define EST_SHAPELET 5

        /* Figure out which estimator is wanted */
        length_shear_est = (int) strlen(shear_est);
        if (length_shear_est==0) return 0x4000;
        if (strcmp(shear_est,"BJ"     )==0)   est_id = EST_BJ;
        if (strcmp(shear_est,"LINEAR" )==0)   est_id = EST_LINEAR;
        if (strcmp(shear_est,"KSB"    )==0)   est_id = EST_KSB;
        if (strcmp(shear_est,"REGAUSS")==0)   est_id = EST_REGAUSS;

        if (!est_id) return 0x4000;

        /* Now we've identified which estimator to use.  Brach to appropriate one. */
        switch(est_id) {

            /* Bernstein & Jarvis and linear estimator */
          case EST_BJ:
          case EST_LINEAR:

               /* Measure the PSF */
               x0 = PSF_data->x0;
               y0 = PSF_data->y0;
               Mxx_psf = Myy_psf = PSF_data->sigma * PSF_data->sigma; Mxy_psf = 0.;
               find_ellipmom_2(PSF, &A_psf, &x0, &y0, &Mxx_psf, &Mxy_psf, &Myy_psf,
                               &rho4_psf, 1.0e-6, &num_iter);
               if (num_iter == NUM_ITER_DEFAULT) {
                   return 1;
               } else {
                   PSF_data->x0 = x0;
                   PSF_data->y0 = y0;
                   PSF_data->sigma = std::pow( Mxx_psf * Myy_psf - Mxy_psf * Mxy_psf, 0.25);
               }

               /* Measure the galaxy */
               x0 = gal_data->x0;
               y0 = gal_data->y0;
               Mxx_gal = Myy_gal = gal_data->sigma * gal_data->sigma; Mxy_gal = 0.;
               find_ellipmom_2(gal_image, &A_gal, &x0, &y0, &Mxx_gal, &Mxy_gal,
                               &Myy_gal, &rho4_gal, 1.0e-6, &num_iter);
               if (num_iter == NUM_ITER_DEFAULT) {
                   return 1;
               } else {
                   gal_data->x0 = x0;
                   gal_data->y0 = y0;
                   gal_data->sigma = std::pow( Mxx_gal * Myy_gal - Mxy_gal * Mxy_gal, 0.25);
                   gal_data->flux = 2.0 * A_gal;
               }

               /* Perform PSF correction */
               R = 1. - (Mxx_psf+Myy_psf)/(Mxx_gal+Myy_gal);
               if (est_id == EST_BJ)
                   psf_corr_bj( 1.-R, (Mxx_psf-Myy_psf)/(Mxx_psf+Myy_psf),
                                2*Mxy_psf/(Mxx_psf+Myy_psf), 0.5*rho4_psf-1.,
                                (Mxx_gal-Myy_gal)/(Mxx_gal+Myy_gal),
                                2*Mxy_gal/(Mxx_gal+Myy_gal), 0.5*rho4_gal-1.,
                                &(gal_data->e1), &(gal_data->e2) );
               if (est_id == EST_LINEAR)
                   psf_corr_linear( 1.-R, (Mxx_psf-Myy_psf)/(Mxx_psf+Myy_psf),
                                    2*Mxy_psf/(Mxx_psf+Myy_psf), 0.5*rho4_psf-1.,
                                    (Mxx_gal-Myy_gal)/(Mxx_gal+Myy_gal),
                                    2*Mxy_gal/(Mxx_gal+Myy_gal), 0.5*rho4_gal-1.,
                                    &(gal_data->e1), &(gal_data->e2) );
               gal_data->meas_type = 'e';
               gal_data->responsivity = 1.;
               break;

          case EST_KSB:

               status = psf_corr_ksb_1(
                   gal_image, PSF, &(gal_data->e1), &(gal_data->e2),
                   &(gal_data->responsivity), &R, flags, &(gal_data->x0), &(gal_data->y0),
                   &(gal_data->sigma), &(gal_data->flux), &(PSF_data->x0), &(PSF_data->y0),
                   &(PSF_data->sigma) );
               gal_data->meas_type = 'g';
               break;

          case EST_REGAUSS:

               status = psf_corr_regauss(
                   gal_image, PSF, &(gal_data->e1), &(gal_data->e2), &R,
                   flags, &(gal_data->x0), &(gal_data->y0), &(gal_data->sigma), &(PSF_data->x0),
                   &(PSF_data->y0), &(PSF_data->sigma), &(gal_data->flux));
               gal_data->meas_type = 'e';
               gal_data->responsivity = 1.;
               break;

          default:
               return 0x4000;
        }

        /* Report resolution factor and return */
        gal_data->resolution = R;
        return status;
    }

    // instantiate template classes for expected types
    template HSMShapeData EstimateShearHSM(Image<float> const &gal_image, Image<float> const &PSF_Image, const char *shear_est, unsigned long flags);
    template HSMShapeData EstimateShearHSM(Image<double> const &gal_image, Image<double> const &PSF_Image, const char *shear_est, unsigned long flags);
    template HSMShapeData EstimateShearHSM(Image<int> const &gal_image, Image<int> const &PSF_Image, const char *shear_est, unsigned long flags);
    template HSMShapeData EstimateShearHSM(Image<const float> const &gal_image, Image<const float> const &PSF_Image, const char *shear_est, unsigned long flags);
    template HSMShapeData EstimateShearHSM(Image<const double> const &gal_image, Image<const double> const &PSF_Image, const char *shear_est, unsigned long flags);
    template HSMShapeData EstimateShearHSM(Image<const int> const &gal_image, Image<const int> const &PSF_Image, const char *shear_est, unsigned long flags);

    template HSMShapeData FindAdaptiveMom(Image<float> const &object_image, double precision);
    template HSMShapeData FindAdaptiveMom(Image<double> const &object_image, double precision);
    template HSMShapeData FindAdaptiveMom(Image<int> const &object_image, double precision);
    template HSMShapeData FindAdaptiveMom(Image<const float> const &object_image, double precision);
    template HSMShapeData FindAdaptiveMom(Image<const double> const &object_image, double precision);
    template HSMShapeData FindAdaptiveMom(Image<const int> const &object_image, double precision);

}
}
