/* -*- c++ -*-
 * Copyright (c) 2012-2016 by the GalSim developers team on GitHub
 * https://github.com/GalSim-developers
 *
 * This file is part of GalSim: The modular galaxy image simulation toolkit.
 * https://github.com/GalSim-developers/GalSim
 *
 * GalSim is free software: redistribution and use in source and binary forms,
 * with or without modification, are permitted provided that the following
 * conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions, and the disclaimer given in the accompanying LICENSE
 *    file.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions, and the disclaimer given in the documentation
 *    and/or other materials provided with the distribution.
 */

#ifndef GalSim_IncompleteGamma_H
#define GalSim_IncompleteGamma_H
/**
 * @file math/IncompleteGamma.h
 * @brief Contains an implementation of the incomplete Gamma function ported from netlib
 */

#include <cmath>

//#define TEST // Uncomment this to turn on testing of this code against boost code.
               // It does some additional testing beyond just what we get from the SBSersic
               // code being accurate.
#ifdef TEST
#include <boost/math/special_functions/gamma.hpp>
#include <iostream>
#endif

namespace galsim {
namespace math {

    double dgamit(double a, double x);
    double d9lgit(double a, double x);
    double d9gmit(double a, double x, double algap1, double sgngam);
    double d9lgic(double a, double x);

    inline double gamma_p(double a, double x)
    {
        // This specific function is what boost calls gamma_p:
        //
        //     P(a,x) = gamma(a, x) / Gamma(a).
        //
        // Wolfram calls it the Regularized Gamma Function:
        // cf. http://mathworld.wolfram.com/RegularizedGammaFunction.html
        //
        // The implementation here is based on the netlib fortran code found here:
        // http://www.netlib.org/slatec/fnlib/gamit.f
        // but reimplemented in C++.
        // The relevant netlib code is called dgamit(a,x), which actually returns what they
        // call Tricomi's incomplete Gamma function, which is P(a,x) * x^-a
        // So we just take that value and multiply by x^a.
        double gp = dgamit(a,x) * std::pow(x,a);
#ifdef TEST
        double gp2 = boost::math::gamma_p(a,x);
        if (std::abs(gp-gp2) > 1.e-6) {
            std::cerr<<"gammap("<<a<<","<<x<<") = "<<gp2<<"  =? "<<gp<<std::endl;
            throw std::runtime_error("gamma_p doesn't agree with boost gamma_p");
        }
        // We don't normally use a < 1 or x < 1, so test those too.
        if (a>1) gamma_p(1./a,x);
        if (x>1) gamma_p(a,1./x);
#endif
        return gp;
    }


    // The below functions are manual conversions from the public domain fortran code here:
    //   http://www.netlib.org/slatec/fnlib/
    // to C++ (guided by f2c, but then manually edited).
    // I left the original PROLOGUEs from the fortran code intact, but added a line to their
    // revision histories that I converted them to C++.
    inline double dgamit(double a, double x)
    {
        // ***BEGIN PROLOGUE  DGAMIT
        // ***PURPOSE  Calculate Tricomi's form of the incomplete Gamma function.
        // ***LIBRARY   SLATEC (FNLIB)
        // ***CATEGORY  C7E
        // ***TYPE      DOUBLE PRECISION (GAMIT-S, DGAMIT-D)
        // ***KEYWORDS  COMPLEMENTARY INCOMPLETE GAMMA FUNCTION, FNLIB,
        //              SPECIAL FUNCTIONS, TRICOMI
        // ***AUTHOR  Fullerton, W., (LANL)
        // ***DESCRIPTION
        //
        // Evaluate Tricomi's incomplete Gamma function defined by
        //
        // DGAMIT = X**(-A)/GAMMA(A) * integral from 0 to X of EXP(-T)
        // T**(A-1.)
        //
        // for A .GT. 0.0 and by analytic continuation for A .LE. 0.0.
        // GAMMA(X) is the complete gamma function of X.
        //
        // DGAMIT is evaluated for arbitrary real values of A and for non-
        // negative values of X (even though DGAMIT is defined for X .LT.
        // 0.0), except that for X = 0 and A .LE. 0.0, DGAMIT is infinite,
        // which is a fatal error.
        //
        // The function and both arguments are DOUBLE PRECISION.
        //
        // A slight deterioration of 2 or 3 digits accuracy will occur when
        // DGAMIT is very large or very small in absolute value, because log-
        // arithmic variables are used.  Also, if the parameter  A  is very
        // close to a negative integer (but not a negative integer), there is
        // a loss of accuracy, which is reported if the result is less than
        // half machine precision.
        //
        // ***REFERENCES  W. Gautschi, A computational procedure for incomplete
        //    gamma functions, ACM Transactions on Mathematical
        //    Software 5, 4 (December 1979), pp. 466-481.
        //    W. Gautschi, Incomplete gamma functions, Algorithm 542,
        //    ACM Transactions on Mathematical Software 5, 4
        //    (December 1979), pp. 482-489.
        //
        // ***REVISION HISTORY  (YYMMDD)
        //    770701  DATE WRITTEN
        //    890531  Changed all specific intrinsics to generic.  (WRB)
        //    890531  REVISION DATE from Version 3.2
        //    891214  Prologue converted to Version 4.0 format.  (BAB)
        //    900315  CALLs to XERROR changed to CALLs to XERMSG.  (THJ)
        //    920528  DESCRIPTION and REFERENCES sections revised.  (WRB)
        //    170203  Converted to C++ (MJ)
        // ***END PROLOGUE  DGAMIT

        const double eps = std::numeric_limits<double>::epsilon();
        const double alneps = -std::log(eps);
        const double sqeps = std::sqrt(eps);

        assert(x >= 0.);

        double sga = a == 0. ? 1.0 : std::copysign(1.0, a);
        double ainta = std::floor(a + sga * 0.5);
        double aeps = a - ainta;

        if (x == 0.)
            return (ainta > 0. || aeps != 0.) ? 1. / std::tgamma(a+1.) : 0.;

        if (x <= 1.) {
            double algap1 = (a >= -0.5 || aeps != 0.) ? std::lgamma(a+1.) : 0.;
            double sgngam = (a < 0. && int(std::floor(a)) % 2 == 1) ? -1 : 1.;
            return d9gmit(a, x, algap1, sgngam);
        }

        if (a >= x)
            return std::exp(d9lgit(a, x));

        // EVALUATE DGAMIT IN TERMS OF LOG (DGAMIC (A, X))

        if (aeps == 0. && ainta <= 0.)
            return std::pow(x,-a);

        double alng = d9lgic(a, x);
        double algap1 = std::lgamma(a+1.);
        double sgngam = (a < 0. && int(std::floor(a)) % 2 == 1) ? -1 : 1.;
        double t = std::log((std::abs(a))) + alng - algap1;
        if (t > alneps) {
            t -= a * std::log(x);
            return -sga * sgngam * std::exp(t);
        }

        double h = 1.;
        if (t > -alneps) {
            h = 1. - sga * sgngam * std::exp(t);
        }
        if (std::abs(h) > sqeps) {
            t = -a * std::log(x) + std::log((std::abs(h)));
            return std::copysign(std::exp(t), h);
        }

        throw std::runtime_error("DGAMIT RESULT LESS THAN HALF PRECISION");
    }

    inline double d9lgit(double a, double x)
    {
        // ***BEGIN PROLOGUE  D9LGIT
        // ***SUBSIDIARY
        // ***PURPOSE  Compute the logarithm of Tricomi's incomplete Gamma
        //            function with Perron's continued fraction for large X and
        //            A .GE. X.
        // ***LIBRARY   SLATEC (FNLIB)
        // ***CATEGORY  C7E
        // ***TYPE      DOUBLE PRECISION (R9LGIT-S, D9LGIT-D)
        // ***KEYWORDS  FNLIB, INCOMPLETE GAMMA FUNCTION, LOGARITHM,
        //             PERRON'S CONTINUED FRACTION, SPECIAL FUNCTIONS, TRICOMI
        // ***AUTHOR  Fullerton, W., (LANL)
        // ***DESCRIPTION
        //
        // Compute the log of Tricomi's incomplete gamma function with Perron's
        // continued fraction for large X and for A .GE. X.
        //
        // ***REFERENCES  (NONE)
        // ***ROUTINES CALLED  D1MACH, XERMSG
        // ***REVISION HISTORY  (YYMMDD)
        //    770701  DATE WRITTEN
        //    890531  Changed all specific intrinsics to generic.  (WRB)
        //    890531  REVISION DATE from Version 3.2
        //    891214  Prologue converted to Version 4.0 format.  (BAB)
        //    900315  CALLs to XERROR changed to CALLs to XERMSG.  (THJ)
        //    900720  Routine changed from user-callable to subsidiary.  (WRB)
        //    170203  Converted to C++ (MJ)
        // ***END PROLOGUE  D9LGIT

        const double eps = std::numeric_limits<double>::epsilon() * 0.5;
        const double sqeps = std::sqrt(std::numeric_limits<double>::epsilon() * 2.);

        assert(x > 0.);
        assert(a >= x);

        double algap1 = std::lgamma(a+1.);
        double ax = a + x;
        double a1x = ax + 1.;
        double r = 0.;
        double p = 1.;
        double s = p;
        for (int k = 1; k <= 200; ++k) {
            double fk(k);
            double t = (a + fk) * x * (r + 1.);
            r = t / ((ax + fk) * (a1x + fk) - t);
            p = r * p;
            s += p;
            if (std::abs(p) < eps * s) {
                double hstar = 1. - x * s / a1x;
                if (hstar < sqeps)
                    throw std::runtime_error("D9LGIT RESULT LESS THAN HALF PRECISION");
                return -x - algap1 - std::log(hstar);
            }
        }
        throw std::runtime_error("D9LGIT NO CONVERGENCE IN 200 TERMS OF CONTINUED FRACTION");
    }

    inline double d9gmit(double a, double x, double algap1, double sgngam)
    {
        // ***BEGIN PROLOGUE  D9GMIT
        // ***SUBSIDIARY
        // ***PURPOSE  Compute Tricomi's incomplete Gamma function for small
        //            arguments.
        // ***LIBRARY   SLATEC (FNLIB)
        // ***CATEGORY  C7E
        // ***TYPE      DOUBLE PRECISION (R9GMIT-S, D9GMIT-D)
        // ***KEYWORDS  COMPLEMENTARY INCOMPLETE GAMMA FUNCTION, FNLIB, SMALL X,
        //             SPECIAL FUNCTIONS, TRICOMI
        // ***AUTHOR  Fullerton, W., (LANL)
        // ***DESCRIPTION
        //
        // Compute Tricomi's incomplete gamma function for small X.
        //
        // ***REFERENCES  (NONE)
        // ***ROUTINES CALLED  D1MACH, DLNGAM, XERMSG
        // ***REVISION HISTORY  (YYMMDD)
        //    770701  DATE WRITTEN
        //    890531  Changed all specific intrinsics to generic.  (WRB)
        //    890911  Removed unnecessary intrinsics.  (WRB)
        //    890911  REVISION DATE from Version 3.2
        //    891214  Prologue converted to Version 4.0 format.  (BAB)
        //    900315  CALLs to XERROR changed to CALLs to XERMSG.  (THJ)
        //    900720  Routine changed from user-callable to subsidiary.  (WRB)
        //    170203  Converted to C++ (MJ)
        // ***END PROLOGUE  D9GMIT

        const double eps = std::numeric_limits<double>::epsilon() * 0.5;
        const double bot = std::log(std::numeric_limits<double>::min());

        assert(x > 0.);

        int ma = int(std::floor(a + 0.5));
        double aeps = a - ma;
        double ae = a < 0.5 ? aeps : a;
        double t = 1.;
        double te = ae;
        double s = t;
        bool converged = false;
        for (int k = 1; k <= 200; ++k) {
            double fk(k);
            te = -x * te / fk;
            t = te / (ae + fk);
            s += t;
            if (std::abs(t) < eps * std::abs(s)) {
                converged = true;
                break;
            }
        }
        if (!converged)
            throw std::runtime_error("D9GMIT NO CONVERGENCE IN 200 TERMS OF TAYLOR-S SERIES");

        if (a >= -0.5)
            return std::exp(-algap1 + std::log(s));

        double algs = -std::lgamma(aeps + 1.) + std::log(s);
        s = 1.;
        int m = -ma - 1;
        if (m != 0) {
            t = 1.;
            int i = m;
            for (int k = 1; k <= i; ++k) {
                t = x * t / (aeps - (m + 1 - k));
                s += t;
                if (std::abs(t) < eps * std::abs(s))
                    break;
            }
        }

        if (s == 0. || aeps == 0.)
            return std::exp(-ma * std::log(x) + algs);

        double sgng2 = sgngam * std::copysign(1., s);
        double alg2 = -x - algap1 + std::log((std::abs(s)));

        double ret_val = 0.;
        if (alg2 > bot) ret_val = sgng2 * std::exp(alg2);
        if (algs > bot) ret_val += std::exp(algs);
        return ret_val;
    }

    inline double d9lgic(double a, double x)
    {
        // ***BEGIN PROLOGUE  D9LGIC
        // ***SUBSIDIARY
        // ***PURPOSE  Compute the log complementary incomplete Gamma function
        //            for large X and for A .LE. X.
        // ***LIBRARY   SLATEC (FNLIB)
        // ***CATEGORY  C7E
        // ***TYPE      DOUBLE PRECISION (R9LGIC-S, D9LGIC-D)
        // ***KEYWORDS  COMPLEMENTARY INCOMPLETE GAMMA FUNCTION, FNLIB, LARGE X,
        //             LOGARITHM, SPECIAL FUNCTIONS
        // ***AUTHOR  Fullerton, W., (LANL)
        // ***DESCRIPTION
        //
        // Compute the log complementary incomplete gamma function for large X
        // and for A .LE. X.
        //
        // ***REFERENCES  (NONE)
        // ***ROUTINES CALLED  D1MACH, XERMSG
        // ***REVISION HISTORY  (YYMMDD)
        //    770701  DATE WRITTEN
        //    890531  Changed all specific intrinsics to generic.  (WRB)
        //    890531  REVISION DATE from Version 3.2
        //    891214  Prologue converted to Version 4.0 format.  (BAB)
        //    900315  CALLs to XERROR changed to CALLs to XERMSG.  (THJ)
        //    900720  Routine changed from user-callable to subsidiary.  (WRB)
        //    170203  Converted to C++ (MJ)
        // ***END PROLOGUE  D9LGIC

        const double eps = std::numeric_limits<double>::epsilon() * 0.5;

        double xpa = x + 1. - a;
        double xma = x - 1. - a;

        double r = 0.;
        double p = 1.;
        double s = p;
        for (int k = 1; k <= 300; ++k) {
            double fk(k);
            double t = fk * (a - fk) * (r + 1.);
            r = -t / ((xma + fk * 2.) * (xpa + fk * 2.) + t);
            p = r * p;
            s += p;
            if (std::abs(p) < eps * s)
                return a * std::log(x) - x + std::log(s / xpa);
        }
        throw std::runtime_error("D9LGIC NO CONVERGENCE IN 300 TERMS OF CONTINUED FRACTION");
    }

} }

#endif

