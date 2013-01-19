// -*- c++ -*-

//#define DEBUGLOGGING

#include "CorrelatedNoise.h"

#ifdef DEBUGLOGGING
#include <fstream>
std::ostream* dbgout = new std::ofstream("debug.out");
int verbose_level = 2;
/*
 * There are three levels of verbosity which can be helpful when debugging, which are written as
 * dbg, xdbg, xxdbg (all defined in Std.h).
 * It's Mike's way to have debug statements in the code that are really easy to turn on and off.
 *
 * If DEBUGLOGGING is #defined, then these write out to *dbgout, according to the value of 
 * verbose_level.
 * dbg requires verbose_level >= 1
 * xdbg requires verbose_level >= 2
 * xxdbg requires verbose_level >= 3
 * If DEBUGLOGGING is not defined, the all three becomes just `if (false) std::cerr`,
 * so the compiler parses the statement fine, but trivially optimizes the code away, so there is no
 * efficiency hit from leaving them in the code.
 */
#endif

namespace galsim {

    /*
     * Covariance matrix calculation using the an input SBProfile, the dimensions of the image for
     * which a covariance matrix is desired (in the form of a Bounds), and a scale dx
     */
    Image<double> calculateCovarianceMatrix(
        const SBProfile& sbp, const Bounds<int>& bounds, double dx)
    {
        // Calculate the required dimensions of the image for which a covariance matrix is needed
        int idim = 1 + bounds.getXMax() - bounds.getXMin();
        int jdim = 1 + bounds.getYMax() - bounds.getYMin();
        int covdim = idim * jdim;
        tmv::SymMatrix<double, 
	    tmv::FortranStyle|tmv::Upper> symcov = calculateCovarianceSymMatrix(sbp, bounds, dx);
        Image<double> cov = Image<double>(covdim, covdim, 0.);

        for (int i=1; i<=covdim; i++){ // note that the Image indices use the FITS convention and 
                                       // start from 1!!
            for (int j=i; j<=covdim; j++){
                cov.setValue(i, j, symcov(i, j)); // fill in the upper triangle with the
                                                  // correct CorrFunc value
            }
        }
        return cov;
    }

    tmv::SymMatrix<double, tmv::FortranStyle|tmv::Upper> calculateCovarianceSymMatrix(
        const SBProfile& sbp, const Bounds<int>& bounds, double dx)
    {
         // Calculate the required dimensions
        int idim = 1 + bounds.getXMax() - bounds.getXMin();
        int jdim = 1 + bounds.getYMax() - bounds.getYMin();
        int covdim = idim * jdim;

        int k, ell; // k and l are indices that refer to image pixel separation vectors in the 
                    // correlation func.
        double x_k, y_ell; // physical vector separations in the correlation func, dx * k etc.

        tmv::SymMatrix<double, tmv::FortranStyle|tmv::Upper> cov = tmv::SymMatrix<
            double, tmv::FortranStyle|tmv::Upper>(covdim);

        for (int i=1; i<=covdim; i++){ // note that the Image indices use the FITS convention and 
                                       // start from 1!!
            for (int j=i; j<=covdim; j++){

                k = (j / jdim) - (i / idim);  // using integer division rules here
                ell = (j % jdim) - (i % idim);
                x_k = double(k) * dx;
                y_ell = double(ell) * dx;
                Position<double> p = Position<double>(x_k, y_ell);
                cov(i, j) = sbp.xValue(p); // fill in the upper triangle with the correct value

            }

        }
        return cov;
    }

}
