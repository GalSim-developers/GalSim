/* -*- c++ -*-
 * Copyright (c) 2012-2021 by the GalSim developers team on GitHub
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

#include <cmath>
#include <limits>

#include "math/Gamma.h"
#include "Std.h"

//#define TEST // Uncomment this to turn on testing of this code against boost code.
               // It does some additional testing beyond just what we get from the SBSersic
               // code being accurate.
#ifdef TEST
#include <boost/math/special_functions/gamma.hpp>
#include <iostream>
#endif

namespace galsim {
namespace math {

    // Defined below.
    double dgamma(double x);
    double dlngam(double x);
    double d9lgmc(double x);
    double dgamit(double a, double x);
    double d9lgit(double a, double x);
    double d9gmit(double a, double x, double algap1, double sgngam);
    double d9lgic(double a, double x);

    // Defined in BesselJ.cpp
    double dcsevl(double x, const double* cs, int n);

#if not (__cplusplus >= 201103L)
    double tgamma(double x)
    {
        double g = dgamma(x);
#ifdef TEST
        double g2 = boost::math::tgamma(x);
        if (std::abs(g-g2) > 1.e-6) {
            std::cerr<<"tgamma("<<x<<") = "<<g2<<"  =? "<<g<<std::endl;
            throw std::runtime_error("tgamma doesn't agree with boost tgamma");
        }
        // We don't normally use x < 1, so test those too.
        if (x>1) tgamma(1./x);
#endif
        return g;
    }

    double lgamma(double x)
    {
        double g = dlngam(x);
#ifdef TEST
        double g2 = boost::math::lgamma(x);
        if (std::abs(g-g2) > 1.e-6) {
            std::cerr<<"lgamma("<<x<<") = "<<g2<<"  =? "<<g<<std::endl;
            throw std::runtime_error("lgamma doesn't agree with boost lgamma");
        }
        if (x>1) lgamma(1./x);
#endif
        return g;
    }
#endif

    double gamma_p(double a, double x)
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
    double dgamma(double x)
    {
        // ***BEGIN PROLOGUE  DGAMMA
        // ***PURPOSE  Compute the complete Gamma function.
        // ***LIBRARY   SLATEC (FNLIB)
        // ***CATEGORY  C7A
        // ***TYPE      DOUBLE PRECISION (GAMMA-S, DGAMMA-D, CGAMMA-C)
        // ***KEYWORDS  COMPLETE GAMMA FUNCTION, FNLIB, SPECIAL FUNCTIONS
        // ***AUTHOR  Fullerton, W., (LANL)
        // ***DESCRIPTION

        // DGAMMA(X) calculates the double precision complete Gamma function
        // for double precision argument X.

        // Series for GAM        on the interval  0.          to  1.00000E+00
        //                                        with weighted error   5.79E-32
        //                                         log weighted error  31.24
        //                               significant figures required  30.00
        //                                    decimal places required  32.05

        // ***REFERENCES  (NONE)
        // ***ROUTINES CALLED  D1MACH, D9LGMC, DCSEVL, DGAMLM, INITDS, XERMSG
        // ***REVISION HISTORY  (YYMMDD)
        //   770601  DATE WRITTEN
        //   890531  Changed all specific intrinsics to generic.  (WRB)
        //   890911  Removed unnecessary intrinsics.  (WRB)
        //   890911  REVISION DATE from Version 3.2
        //   891214  Prologue converted to Version 4.0 format.  (BAB)
        //   900315  CALLs to XERROR changed to CALLs to XERMSG.  (THJ)
        //   920618  Removed space from variable name.  (RWC, WRB)
        //   170817  Converted to C++. (MJ)
        // ***END PROLOGUE  DGAMMA

        const double gamcs[42] = {
            0.008571195590989331421920062399942,
            0.004415381324841006757191315771652,
            0.05685043681599363378632664588789,
            -0.004219835396418560501012500186624,
            0.001326808181212460220584006796352,
            -1.893024529798880432523947023886e-4,
            3.606925327441245256578082217225e-5,
            -6.056761904460864218485548290365e-6,
            1.055829546302283344731823509093e-6,
            -1.811967365542384048291855891166e-7,
            3.117724964715322277790254593169e-8,
            -5.354219639019687140874081024347e-9,
            9.19327551985958894688778682594e-10,
            -1.577941280288339761767423273953e-10,
            2.707980622934954543266540433089e-11,
            -4.646818653825730144081661058933e-12,
            7.973350192007419656460767175359e-13,
            -1.368078209830916025799499172309e-13,
            2.347319486563800657233471771688e-14,
            -4.027432614949066932766570534699e-15,
            6.910051747372100912138336975257e-16,
            -1.185584500221992907052387126192e-16,
            2.034148542496373955201026051932e-17,
            -3.490054341717405849274012949108e-18,
            5.987993856485305567135051066026e-19,
            -1.027378057872228074490069778431e-19,
            1.762702816060529824942759660748e-20,
            -3.024320653735306260958772112042e-21,
            5.188914660218397839717833550506e-22,
            -8.902770842456576692449251601066e-23,
            1.527474068493342602274596891306e-23,
            -2.620731256187362900257328332799e-24,
            4.496464047830538670331046570666e-25,
            -7.714712731336877911703901525333e-26,
            1.323635453126044036486572714666e-26,
            -2.270999412942928816702313813333e-27,
            3.896418998003991449320816639999e-28,
            -6.685198115125953327792127999999e-29,
            1.146998663140024384347613866666e-29,
            -1.967938586345134677295103999999e-30,
            3.376448816585338090334890666666e-31,
            -5.793070335782135784625493333333e-32 };
        const double pi = 3.1415926535897932384626433832795;
        const double sq2pil = .91893853320467274178032973640562;
        const int ngam = 23;

        if (x == 0.)
            throw std::runtime_error("Argument of dgamma is 0.");

        double y = std::abs(x);
        if (y > 10.) {

            // GAMMA(X) FOR ABS(X) .GT. 10.0.  RECALL Y = ABS(X).
            double ret_val = std::exp((y - 0.5) * std::log(y) - y + sq2pil + d9lgmc(y));
            if (x > 0.) return ret_val;
            else {
                double sinpiy = std::sin(pi * y);
                if (sinpiy == 0.)
                    throw std::runtime_error("Argument of dgamma is a negative integer");
                return -pi / (y * sinpiy * ret_val);
            }

        } else {

            // COMPUTE GAMMA(X) FOR -XBND .LE. X .LE. XBND.  REDUCE INTERVAL AND FIND
            // GAMMA(1+Y) FOR 0.0 .LE. Y .LT. 1.0 FIRST OF ALL.

            int n = int(x);
            if (x < 0.) --n;
            y = x - n;
            --n;
            double z = 2.*y - 1.;
            double ret_val = dcsevl(z, gamcs, ngam) + .9375;
            if (n == 0) return ret_val;
            else if (n > 0) {

                // GAMMA(X) FOR X .GE. 2.0 AND X .LE. 10.0
                for (int i=1; i <= n; ++i)
                    ret_val = (y + i) * ret_val;
                return ret_val;

            } else {

                // COMPUTE GAMMA(X) FOR X .LT. 1.0
                n = -n;
                if (x < 0. && x+n-2 == 0.) 
                    throw std::runtime_error("argument of dgamma is a negative integer");

                for (int i=1; i<=n; ++i) 
                    ret_val /= x + i - 1;
                return ret_val;
            }
        }
    }

    double dlngam(double x)
    {
        // ***BEGIN PROLOGUE  DLNGAM
        // ***PURPOSE  Compute the logarithm of the absolute value of the Gamma
        //            function.
        // ***LIBRARY   SLATEC (FNLIB)
        // ***CATEGORY  C7A
        // ***TYPE      DOUBLE PRECISION (ALNGAM-S, DLNGAM-D, CLNGAM-C)
        // ***KEYWORDS  ABSOLUTE VALUE, COMPLETE GAMMA FUNCTION, FNLIB, LOGARITHM,
        //             SPECIAL FUNCTIONS
        // ***AUTHOR  Fullerton, W., (LANL)
        // ***DESCRIPTION

        // DLNGAM(X) calculates the double precision logarithm of the
        // absolute value of the Gamma function for double precision
        // argument X.

        // ***REFERENCES  (NONE)
        // ***ROUTINES CALLED  D1MACH, D9LGMC, DGAMMA, XERMSG
        // ***REVISION HISTORY  (YYMMDD)
        //   770601  DATE WRITTEN
        //   890531  Changed all specific intrinsics to generic.  (WRB)
        //   890531  REVISION DATE from Version 3.2
        //   891214  Prologue converted to Version 4.0 format.  (BAB)
        //   900315  CALLs to XERROR changed to CALLs to XERMSG.  (THJ)
        //   900727  Added EXTERNAL statement.  (WRB)
        //   170817  Converted to C++. (MJ)
        // ***END PROLOGUE  DLNGAM

        static double sq2pil = 0.91893853320467274178032973640562;
        static double sqpi2l = 0.225791352644727432363097614947441;
        static double pi = 3.1415926535897932384626433832795;

        if (x == 0.)
            throw std::runtime_error("Argument of dlngam is 0.");

        double y = std::abs(x);
        if (y > 10.) {
            // LOG ( ABS (DGAMMA(X)) ) FOR ABS(X) .GT. 10.0

            if (x > 0.) {
                return sq2pil + (x - 0.5) * std::log(x) - x + d9lgmc(y);
            } else {
                double sinpiy = std::abs(std::sin(pi * y));
                if (sinpiy == 0.)
                    throw std::runtime_error("Argument of lgamma is a negative integer");
                return sqpi2l + (x - 0.5) * std::log(y) - x - std::log(sinpiy) - d9lgmc(y);
            }
        } else {
            // LOG (ABS (DGAMMA(X)) ) FOR ABS(X) .LE. 10.0
            return std::log(std::abs(dgamma(x)));
        }
    }

    double d9lgmc(double x)
    {
        // ***BEGIN PROLOGUE  D9LGMC
        // ***SUBSIDIARY
        // ***PURPOSE  Compute the log Gamma correction factor so that
        //            LOG(DGAMMA(X)) = LOG(SQRT(2*PI)) + (X-5.)*LOG(X) - X
        //            + D9LGMC(X).
        // ***LIBRARY   SLATEC (FNLIB)
        // ***CATEGORY  C7E
        // ***TYPE      DOUBLE PRECISION (R9LGMC-S, D9LGMC-D, C9LGMC-C)
        // ***KEYWORDS  COMPLETE GAMMA FUNCTION, CORRECTION TERM, FNLIB,
        //             LOG GAMMA, LOGARITHM, SPECIAL FUNCTIONS
        // ***AUTHOR  Fullerton, W., (LANL)
        // ***DESCRIPTION

        // Compute the log gamma correction factor for X .GE. 10. so that
        // LOG (DGAMMA(X)) = LOG(SQRT(2*PI)) + (X-.5)*LOG(X) - X + D9lGMC(X)

        // Series for ALGM       on the interval  0.          to  1.00000E-02
        //                                        with weighted error   1.28E-31
        //                                         log weighted error  30.89
        //                               significant figures required  29.81
        //                                    decimal places required  31.48

        // ***REFERENCES  (NONE)
        // ***ROUTINES CALLED  D1MACH, DCSEVL, INITDS, XERMSG
        // ***REVISION HISTORY  (YYMMDD)
        //   770601  DATE WRITTEN
        //   890531  Changed all specific intrinsics to generic.  (WRB)
        //   890531  REVISION DATE from Version 3.2
        //   891214  Prologue converted to Version 4.0 format.  (BAB)
        //   900315  CALLs to XERROR changed to CALLs to XERMSG.  (THJ)
        //   900720  Routine changed from user-callable to subsidiary.  (WRB)
        //   170817  Converted to C++. (MJ)
        // ***END PROLOGUE  D9LGMC

        const double algmcs[15] = {
            0.1666389480451863247205729650822,
            -1.384948176067563840732986059135e-5,
            9.810825646924729426157171547487e-9,
            -1.809129475572494194263306266719e-11,
            6.221098041892605227126015543416e-14,
            -3.399615005417721944303330599666e-16,
            2.683181998482698748957538846666e-18,
            -2.868042435334643284144622399999e-20,
            3.962837061046434803679306666666e-22,
            -6.831888753985766870111999999999e-24,
            1.429227355942498147573333333333e-25,
            -3.547598158101070547199999999999e-27,1.025680058010470912e-28,
            -3.401102254316748799999999999999e-30,
            1.276642195630062933333333333333e-31
        };
        const double xbig = 1. / std::numeric_limits<double>::epsilon();
        const int nalgm = 7;

        double ret_val = 1. / (x * 12.);
        if (x < xbig) {
            // Computing 2nd power
            double temp = 10. / x;
            temp = temp * temp* 2. - 1.;
            ret_val = dcsevl(temp, algmcs, nalgm) / x;
        }
        return ret_val;
    }

    double dgamit(double a, double x)
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

        assert(x >= 0.);

        double sga = a >= 0. ? 1.0 : -1.0;
        double ainta = std::floor(a + sga * 0.5);
        double aeps = a - ainta;

        if (x == 0.)
            return (ainta > 0. || aeps != 0.) ? 1. / math::tgamma(a+1.) : 0.;

        if (x <= 1.) {
            double algap1 = (a >= -0.5 || aeps != 0.) ? math::lgamma(a+1.) : 0.;
            double sgngam = (a < 0. && int(std::floor(a)) % 2 == 1) ? -1 : 1.;
            return d9gmit(a, x, algap1, sgngam);
        }

        if (a >= x)
            return std::exp(d9lgit(a, x));

        // EVALUATE DGAMIT IN TERMS OF LOG (DGAMIC (A, X))

        if (aeps == 0. && ainta <= 0.)
            return std::pow(x,-a);

        double alng = d9lgic(a, x);
        double algap1 = math::lgamma(a+1.);
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
        t = -a * std::log(x) + std::log((std::abs(h)));
        return h >= 0. ? std::exp(t) : -std::exp(t);
    }

    double d9lgit(double a, double x)
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

        assert(x > 0.);
        assert(a >= x);

        double algap1 = math::lgamma(a+1.);
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
                return -x - algap1 - std::log(hstar);
            }
        }
        throw std::runtime_error("D9LGIT NO CONVERGENCE IN 200 TERMS OF CONTINUED FRACTION");
    }

    double d9gmit(double a, double x, double algap1, double sgngam)
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

        double algs = -math::lgamma(aeps + 1.) + std::log(s);
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

        double sgng2 = s >= 0. ? sgngam : -sgngam;
        double alg2 = -x - algap1 + std::log((std::abs(s)));

        double ret_val = 0.;
        if (alg2 > bot) ret_val = sgng2 * std::exp(alg2);
        if (algs > bot) ret_val += std::exp(algs);
        return ret_val;
    }

    double d9lgic(double a, double x)
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

