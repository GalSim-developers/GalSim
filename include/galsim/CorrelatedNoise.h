// -*- c++ -*-
#ifndef CORRELATEDNOISE_H
#define CORRELATEDNOISE_H

//#define DEBUGLOGGING

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

/**
 * @file CorrelatedNoise.h @brief Contains a class definition for handling the correlation
 * properties of noise in Images.
 */

#include <complex>
#include "TMV_Sym.h"
#include "Image.h"
#include "SBProfile.h"

namespace galsim {

        /**
         * @brief Return, as a square Image, a noise covariance matrix between every element in an 
         * input Image with pixel scale dx.
         *
         * The matrix is symmetric, and therefore only the upper triangular elements are actually
         * written into.  The rest are initialized and remain as zero.
         *
         * Currently, this actually copies elements from an internal calculation of the covariance
         * matrix (using Mike Jarvis' TMV library).  It could, therefore, be calculated more 
         * efficiently by direct assignment.  However, as this public member function is foreseen
         * as being mainly for visualization/checking purposes, we go via the TMV intermediary to 
         * avoid code duplication.  If, in future, it becomes critical to speed up this function
         * this can be revisited.
         */
        Image<double> calculateCovarianceMatrix(
            const SBProfile& sbp, const Bounds<int>& bounds, double dx) const;

         /**
         * @brief Return, as a TMV SymMatrix, a noise covariance matrix between every element in 
         * an input Image with pixel scale dx.
         *
         * The TMV SymMatrix uses FortranStyle indexing (to match the FITS-compliant usage in Image)
         * along with ColumnMajor ordering (the default), and Upper triangle storage.
         */
        tmv::SymMatrix<double, tmv::FortranStyle|tmv::Upper> calculateCovarianceSymMatrix(
            const SBProfile& sbp, const Bounds<int>& bounds, double dx) const;

    };

}
#endif
