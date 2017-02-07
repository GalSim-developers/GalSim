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

#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <assert.h>
#include <limits>
#include <algorithm>
#include "math/Bessel.h"

//#define TEST // Uncomment this to turn on testing of this code against boost code.
#ifdef TEST
#include <boost/math/special_functions/bessel.hpp>
#include <iostream>
#endif

namespace galsim {
namespace math {

    // Routines ported from netlib, defined below.
    double dbesj(double x, double fnu);
    double dbesj0(double x);
    double dbesj1(double x);
    double dasyjy(double x, double fnu, bool is_j, double *wk, int* iflw);
    void djairy(double x, double rx, double c, double *ai, double *dai);

    double dbesy(double x, double fnu);
    double dbesy0(double x);
    double dbesy1(double x);
    void dbsynu(double x, double fnu, int n, double *y);
    void dyairy(double x, double rx, double c, double *ai, double *dai);

    double dbesi(double x, double fnu);
    double dbesi0(double x);
    double dbesi1(double x);
    double dbsi0e(double x);
    double dbsi1e(double x);
    double dasyik(double x, double fnu, bool is_i);

    double dbesk(double x, double fnu);
    double dbesk0(double x);
    double dbsk0e(double x);
    double dbesk1(double x);
    double dbsk1e(double x);
    void dbsknu(double x, double fnu, int n, double *y);

    double dcsevl(double x, const double* cs, int n);


    double cyl_bessel_j(double nu, double x)
    {
        // Negative arguments yield complex values, so don't do that.
        if (x < 0)
            throw std::runtime_error("cyl_bessel_j x must be >= 0");

        // Identity: J_-nu(x) = cos(pi nu) J_nu(x) - sin(pi nu) Y_nu(x)
        if (nu < 0) {
            nu = -nu;
            if (int(nu) == nu) {
                if (int(nu) % 2 == 0) return cyl_bessel_j(nu, x);
                else return -cyl_bessel_j(nu, x);
            } else {
                double c = std::cos(M_PI * nu);
                double s = std::sin(M_PI * nu);
                return c * cyl_bessel_j(nu, x) - s * cyl_bessel_y(nu, x);
            }
        }

        double jnu = dbesj(x, nu);
#ifdef TEST
        double jnu2 = boost::math::cyl_bessel_j(nu, x);
        if (std::abs(jnu-jnu2)/std::abs(jnu2) > 1.e-10) {
            std::cerr.precision(16);
            std::cerr<<"J("<<nu<<","<<x<<") = "<<jnu2<<"  =? "<<jnu<<std::endl;
            std::cerr<<"diff = "<<jnu-jnu2<<std::endl;
            std::cerr<<"rel diff = "<<(jnu-jnu2)/std::abs(jnu2)<<std::endl;
            throw std::runtime_error("cyl_bessel_j doesn't agree with boost::cyl_bessel_j");
        }
#endif
        return jnu;
    }

    double cyl_bessel_y(double nu, double x)
    {
        // Negative arguments yield complex values, so don't do that.
        // And Y_nu(0) = inf.
        if (x <= 0)
            throw std::runtime_error("cyl_bessel_y x must be > 0");

        // Identity: Y_-nu(x) = cos(pi nu) Y_nu(x) + sin(pi nu) J_nu(x)
        if (nu < 0) {
            nu = -nu;
            if (int(nu) == nu) {
                if (int(nu) % 2 == 1) return -cyl_bessel_y(nu, x);
                else return cyl_bessel_y(nu, x);
            } else {
                double c = std::cos(M_PI * nu);
                double s = std::sin(M_PI * nu);
                return c * cyl_bessel_y(nu, x) + s * cyl_bessel_j(nu, x);
            }
        }

        double ynu = dbesy(x, nu);
#ifdef TEST
        double ynu2 = boost::math::cyl_neumann(nu, x);
        if (std::abs(ynu-ynu2)/std::abs(ynu2) > 1.e-10) {
            std::cerr.precision(16);
            std::cerr<<"Y("<<nu<<","<<x<<") = "<<ynu2<<"  =? "<<ynu<<std::endl;
            std::cerr<<"diff = "<<ynu-ynu2<<std::endl;
            std::cerr<<"rel diff = "<<(ynu-ynu2)/std::abs(ynu2)<<std::endl;
            throw std::runtime_error("cyl_bessel_y doesn't agree with boost::cyl_bessel_y");
        }
#endif
        return ynu;
    }

    double cyl_bessel_i(double nu, double x)
    {
        // Negative arguments yield complex values, so don't do that.
        if (x < 0)
            throw std::runtime_error("cyl_bessel_i x must be >= 0");

        // Identity: I_−nu(x) = I_nu(z) + 2/pi sin(pi nu) K_nu(x)
        if (nu < 0)
            return cyl_bessel_i(-nu, x) + 2./M_PI * std::sin(-M_PI*nu) * cyl_bessel_k(-nu,x);

        double inu = dbesi(x, nu);
#ifdef TEST
        double inu2 = boost::math::cyl_bessel_i(nu, x);
        if (std::abs(inu-inu2)/std::abs(inu2) > 1.e-10) {
            std::cerr.precision(16);
            std::cerr<<"K("<<nu<<","<<x<<") = "<<inu2<<"  =? "<<inu<<std::endl;
            std::cerr<<"diff = "<<inu-inu2<<std::endl;
            std::cerr<<"rel diff = "<<(inu-inu2)/std::abs(inu2)<<std::endl;
            throw std::runtime_error("cyl_bessel_i doesn't agree with boost::cyl_bessel_i");
        }
#endif
        return inu;
    }

    double cyl_bessel_k(double nu, double x)
    {
        const double sqrteps = std::sqrt(std::numeric_limits<double>::epsilon());

        // Identity: K_-nu(x) = K_nu(x)
        nu = std::abs(nu);

        // Negative arguments yield complex values, so don't do that.
        // And K_nu(0) = inf.
        if (x <= 0)
            throw std::runtime_error("cyl_bessel_k x must be > 0");

        double knu = dbesk(x, nu);
#ifdef TEST
        double knu2 = boost::math::cyl_bessel_k(nu, x);
        if (std::abs(knu-knu2)/std::abs(knu2) > 1.e-10) {
            std::cerr.precision(16);
            std::cerr<<"K("<<nu<<","<<x<<") = "<<knu2<<"  =? "<<knu<<std::endl;
            std::cerr<<"diff = "<<knu-knu2<<std::endl;
            std::cerr<<"rel diff = "<<(knu-knu2)/std::abs(knu2)<<std::endl;
            throw std::runtime_error("cyl_bessel_k doesn't agree with boost::cyl_bessel_k");
        }
#endif
        return knu;
    }


    // The below functions are manual conversions from the public domain fortran code here:
    //   http://www.netlib.org/slatec/fnlib/
    // to C++ (guided by f2c, but then manually edited).
    // I left the original PROLOGUEs from the fortran code intact, but added a line to their
    // revision histories that I converted them to C++.

    //
    // J_nu(x)
    //
    double dbesj(double x, double fnu)
    {
        // ***BEGIN PROLOGUE  DBESJ
        // ***PURPOSE  Compute an N member sequence of J Bessel functions
        //            J/SUB(ALPHA+K-1)/(X), K=1,...,N for non-negative ALPHA
        //            and X.
        // ***LIBRARY   SLATEC
        // ***CATEGORY  C10A3
        // ***TYPE      DOUBLE PRECISION (BESJ-S, DBESJ-D)
        // ***KEYWORDS  J BESSEL FUNCTION, SPECIAL FUNCTIONS
        // ***AUTHOR  Amos, D. E., (SNLA)
        //           Daniel, S. L., (SNLA)
        //           Weston, M. K., (SNLA)
        // ***DESCRIPTION
        //
        //     Abstract  **** a double precision routine ****
        //         DBESJ computes an N member sequence of J Bessel functions
        //         J/sub(ALPHA+K-1)/(X), K=1,...,N for non-negative ALPHA and X.
        //         A combination of the power series, the asymptotic expansion
        //         for X to infinity and the uniform asymptotic expansion for
        //         NU to infinity are applied over subdivisions of the (NU,X)
        //         plane.  For values of (NU,X) not covered by one of these
        //         formulae, the order is incremented or decremented by integer
        //         values into a region where one of the formulae apply. Backward
        //         recursion is applied to reduce orders by integer values except
        //         where the entire sequence lies in the oscillatory region.  In
        //         this case forward recursion is stable and values from the
        //         asymptotic expansion for X to infinity start the recursion
        //         when it is efficient to do so. Leading terms of the series and
        //         uniform expansion are tested for underflow.  If a sequence is
        //         requested and the last member would underflow, the result is
        //         set to zero and the next lower order tried, etc., until a
        //         member comes on scale or all members are set to zero.
        //         Overflow cannot occur.
        //
        //         The maximum number of significant digits obtainable
        //         is the smaller of 14 and the number of digits carried in
        //         double precision arithmetic.
        //
        //     Description of Arguments
        //
        //         Input      X,ALPHA are double precision
        //           X      - X .GE. 0.0D0
        //           ALPHA  - order of first member of the sequence,
        //                    ALPHA .GE. 0.0D0
        //           N      - number of members in the sequence, N .GE. 1
        //
        //         Output     Y is double precision
        //           Y      - a vector whose first N components contain
        //                    values for J/sub(ALPHA+K-1)/(X), K=1,...,N
        //           NZ     - number of components of Y set to zero due to
        //                    underflow,
        //                    NZ=0   , normal return, computation completed
        //                    NZ .NE. 0, last NZ components of Y set to zero,
        //                             Y(K)=0.0D0, K=N-NZ+1,...,N.
        //
        //     Error Conditions
        //         Improper input arguments - a fatal error
        //         Underflow  - a non-fatal error (NZ .NE. 0)
        //
        // ***REFERENCES  D. E. Amos, S. L. Daniel and M. K. Weston, CDC 6600
        //                 subroutines IBESS and JBESS for Bessel functions
        //                 I(NU,X) and J(NU,X), X .GE. 0, NU .GE. 0, ACM
        //                 Transactions on Mathematical Software 3, (1977),
        //                 pp. 76-92.
        //               F. W. J. Olver, Tables of Bessel Functions of Moderate
        //                 or Large Orders, NPL Mathematical Tables 6, Her
        //                 Majesty's Stationery Office, London, 1962.
        // ***ROUTINES CALLED  D1MACH, DASYJY, DJAIRY, DLNGAM, I1MACH, XERMSG
        // ***REVISION HISTORY  (YYMMDD)
        //    750101  DATE WRITTEN
        //    890531  Changed all specific intrinsics to generic.  (WRB)
        //    890911  Removed unnecessary intrinsics.  (WRB)
        //    890911  REVISION DATE from Version 3.2
        //    891214  Prologue converted to Version 4.0 format.  (BAB)
        //    900315  CALLs to XERROR changed to CALLs to XERMSG.  (THJ)
        //    900326  Removed duplicate information from DESCRIPTION section.
        //            (WRB)
        //    920501  Reformatted the REFERENCES section.  (WRB)
        //    170203  Converted to C++, and modified to only cover n==1 option. (MJ)
        // ***END PROLOGUE  DBESJ

        const double rtwo = 1.34839972492648;
        const double pdf = 0.785398163397448;
        const double rttp = 0.797884560802865;
        const double pidt = 1.5707963267949;
        const double pp[4] = {
            8.72909153935547, 0.26569393226503, 0.124578576865586, 7.70133747430388e-4 };
        const int inlim = 150;
        const double fnulim[2] = {100., 60.};

        const double tol = std::max(std::numeric_limits<double>::epsilon(), 1e-15);
        const double elim1 = -std::log(std::numeric_limits<double>::min() * 1.e3);
        const double rtol = 1. / tol;
        const double slim = std::numeric_limits<double>::min() * rtol * 1e3;
        const double tolln = -std::log(tol);

        assert(fnu >= 0.);
        assert(x >= 0.);

        if (fnu == 0.) return dbesj0(x);
        else if (fnu == 1.) return dbesj1(x);
        else if (x == 0.) return 0.;

        const double fni = std::floor(fnu);
        const double fnf = fnu - fni;
        const double xo2 = x * 0.5;
        const double sxo2 = xo2 * xo2;

        //     DECISION TREE FOR REGION WHERE SERIES, ASYMPTOTIC EXPANSION FOR X
        //     TO INFINITY AND ASYMPTOTIC EXPANSION FOR NU TO INFINITY ARE
        //     APPLIED.

        bool series;
        int ns;
        if (sxo2 <= fnu + 1.) {
            series = true;
            ns = 0;
        } else if (x <= 12.) {
            series = true;
            ns = int(sxo2 - fnu) + 1;
        } else {
            series = false;
            if (x > std::max(20., fnu)) {
                double rtx = std::sqrt(x);
                if (fnu <= rtwo * rtx + fnulim[1]) {
                    //     ASYMPTOTIC EXPANSION FOR X TO INFINITY WITH FORWARD RECURSION IN
                    //     OSCILLATORY REGION X.GT.MAX(20, NU), PROVIDED THE LAST MEMBER
                    //     OF THE SEQUENCE IS ALSO IN THE REGION.
                    double arg = x - pidt * fnu - pdf;
                    double sa = std::sin(arg);
                    double sb = std::cos(arg);
                    double coef = rttp / rtx;
                    double etx = x * 8.;
                    double dtm = 4. * fni * fni;
                    double tm = fnf * 4. * (fni + fni + fnf);
                    double trx = dtm - 1.;
                    double t2 = (trx + tm) / etx;
                    double s2 = t2;
                    double relb = tol * std::abs(t2);
                    double t1 = etx;
                    double s1 = 1.;
                    double fn = 1.;
                    double ak = 8.;
                    for (int k = 1; k <= 13; ++k) {
                        t1 += etx;
                        fn += ak;
                        trx = dtm - fn;
                        double ap = trx + tm;
                        t2 = -t2 * ap / t1;
                        s1 += t2;
                        t1 += etx;
                        ak += 8.;
                        fn += ak;
                        trx = dtm - fn;
                        ap = trx + tm;
                        t2 = t2 * ap / t1;
                        s2 += t2;
                        if (std::abs(t2) <= relb) break;
                        ak += 8.;
                    }
                    return coef * (s1 * sb - s2 * sa);
                }
                ns = 0;
            } else {
                double ans = std::max(36.-fnu, 0.);
                ns = int(ans);
            }
        }

        double fn = fnu + ns;
        int in;
        double jnu;
        if (series) {
            //     SERIES FOR (X/2)**2.LE.NU+1
            double gln = std::lgamma(fn+1);
            double xo2l = std::log(xo2);
            double arg = fn * xo2l - gln;
            if (arg < -elim1) return 0.;
            double earg = std::exp(arg);

            double s = 1.;
            if (x >= tol) {
                double ak = 3.;
                double t2 = 1.;
                double t = 1.;
                double s1 = fn;
                for (int k = 1; k <= 17; ++k) {
                    double s2 = t2 + s1;
                    t = -t * sxo2 / s2;
                    s += t;
                    if (std::abs(t) < tol) break;
                    t2 += ak;
                    ak += 2.;
                    s1 += fn;
                }
            }
            jnu = s * earg;
            if (ns == 0) return jnu;

            //     BACKWARD RECURSION WITH NORMALIZATION BY
            //     ASYMPTOTIC EXPANSION FOR NU TO INFINITY OR POWER SERIES.

            //     COMPUTATION OF LAST ORDER FOR SERIES NORMALIZATION
            double akm = std::max(3.-fn, 0.);
            int km = int(akm);
            double tfn = fn + km;
            double ta = (gln + tfn - 0.9189385332 - 0.0833333333 / tfn) / (tfn + 0.5);
            ta = xo2l - ta;
            double tb = -(1. - 1.5 / tfn) / tfn;
            akm = tolln / (-ta + std::sqrt(ta * ta - tolln * tb)) + 1.5;
            in = km + int(akm);
        } else {
            //     UNIFORM ASYMPTOTIC EXPANSION FOR NU TO INFINITY
            double wk[7];
            int iflw;
            double jnu = dasyjy(x, fn, true, wk, &iflw);
            if (iflw != 0) return 0.;
            if (ns == 0) return jnu;

            //     COMPUTATION OF LAST ORDER FOR ASYMPTOTIC EXPANSION NORMALIZATION
            double ta,tb;
            if (wk[5] > 30.) {
                ta = tolln * 0.5 / wk[3];
                ta = ((ta * 0.049382716 - 0.1111111111) * ta + 0.6666666667) * ta * wk[5];
            } else {
                double rden = (pp[3] * wk[5] + pp[2]) * wk[5] + 1.;
                double rzden = pp[0] + pp[1] * wk[5];
                ta = rzden / rden;
            }
            if (wk[0] < 0.1) {
                tb = ((wk[0] * 0.0887944358 + 0.167989473) * wk[0] + 1.259921049) / wk[6];
            } else {
                tb = (wk[2] + wk[1]) / wk[4];
            }
            in = int(ta / tb + 1.5);
        }

        double trx = 2. / x;
        double tm = (fn + in) * trx;
        double ta = 0.;
        double tb = tol;
        double ak = 1.;

        //     BACKWARD RECUR UNINDEXED
        for (int kk=1; kk<=2; ++kk) {
            for (int i=1; i<=in; ++i) {
                double temp = tb;
                tb = tm * tb - ta;
                ta = temp;
                tm -= trx;
            }
            if (kk == 2) break;

            double sa = ta / tb;
            ta = jnu;
            tb = jnu;
            if (std::abs(jnu) <= slim) {
                ta *= rtol;
                tb *= rtol;
                ak = tol;
            }
            ta *= sa;
            in = ns;
        }
        return tb*ak;
    }

    double dbesj0(double x)
    {
        // ***BEGIN PROLOGUE  DBESJ0
        // ***PURPOSE  Compute the Bessel function of the first kind of order
        //            zero.
        // ***LIBRARY   SLATEC (FNLIB)
        // ***CATEGORY  C10A1
        // ***TYPE      DOUBLE PRECISION (BESJ0-S, DBESJ0-D)
        // ***KEYWORDS  BESSEL FUNCTION, FIRST KIND, FNLIB, ORDER ZERO,
        //             SPECIAL FUNCTIONS
        // ***AUTHOR  Fullerton, W., (LANL)
        // ***DESCRIPTION

        // DBESJ0(X) calculates the double precision Bessel function of
        // the first kind of order zero for double precision argument X.

        // Series for BJ0        on the interval  0.          to  1.60000E+01
        //                                        with weighted error   4.39E-32
        //                                         log weighted error  31.36
        //                               significant figures required  31.21
        //                                    decimal places required  32.00

        // ***REFERENCES  (NONE)
        // ***ROUTINES CALLED  D1MACH, D9B0MP, DCSEVL, INITDS
        // ***REVISION HISTORY  (YYMMDD)
        //   770701  DATE WRITTEN
        //   890531  Changed all specific intrinsics to generic.  (WRB)
        //   890531  REVISION DATE from Version 3.2
        //   891214  Prologue converted to Version 4.0 format.  (BAB)
        // ***END PROLOGUE  DBESJ0

        const double bj0cs[19] = {
            0.10025416196893913701073127264074,
            -0.66522300776440513177678757831124,
            0.2489837034982813137046046872668,
            -0.033252723170035769653884341503854,
            0.0023114179304694015462904924117729,
            -9.9112774199508092339048519336549e-5,
            2.8916708643998808884733903747078e-6,
            -6.1210858663032635057818407481516e-8,
            9.8386507938567841324768748636415e-10,
            -1.2423551597301765145515897006836e-11,
            1.2654336302559045797915827210363e-13,
            -1.0619456495287244546914817512959e-15,
            7.4706210758024567437098915584e-18,
            -4.4697032274412780547627007999999e-20,
            2.3024281584337436200523093333333e-22,
            -1.0319144794166698148522666666666e-24,
            4.06081782748733227008e-27,
            -1.4143836005240913919999999999999e-29,
            4.391090549669888e-32
        };
        const double bm0cs[37] = {
            0.09211656246827742712573767730182,
            -0.001050590997271905102480716371755,
            1.470159840768759754056392850952e-5,
            -5.058557606038554223347929327702e-7,
            2.787254538632444176630356137881e-8,
            -2.062363611780914802618841018973e-9,
            1.870214313138879675138172596261e-10,
            -1.969330971135636200241730777825e-11,
            2.325973793999275444012508818052e-12,
            -3.009520344938250272851224734482e-13,
            4.194521333850669181471206768646e-14,
            -6.219449312188445825973267429564e-15,
            9.718260411336068469601765885269e-16,
            -1.588478585701075207366635966937e-16,
            2.700072193671308890086217324458e-17,
            -4.750092365234008992477504786773e-18,
            8.61512816260437087319170374656e-19,
            -1.605608686956144815745602703359e-19,
            3.066513987314482975188539801599e-20,
            -5.987764223193956430696505617066e-21,
            1.192971253748248306489069841066e-21,
            -2.420969142044805489484682581333e-22,
            4.996751760510616453371002879999e-23,
            -1.047493639351158510095040511999e-23,
            2.227786843797468101048183466666e-24,
            -4.801813239398162862370542933333e-25,
            1.047962723470959956476996266666e-25,
            -2.3138581656786153251012608e-26,
            5.164823088462674211635199999999e-27,
            -1.164691191850065389525401599999e-27,
            2.651788486043319282958336e-28,
            -6.092559503825728497691306666666e-29,
            1.411804686144259308038826666666e-29,
            -3.298094961231737245750613333333e-30,
            7.763931143074065031714133333333e-31,
            -1.841031343661458478421333333333e-31,
            4.395880138594310737100799999999e-32
        };
        const double bth0cs[44] = {
            -0.24901780862128936717709793789967,
            4.8550299609623749241048615535485e-4,
            -5.4511837345017204950656273563505e-6,
            1.3558673059405964054377445929903e-7,
            -5.569139890222762622758321841492e-9,
            3.2609031824994335304004205719468e-10,
            -2.4918807862461341125237903877993e-11,
            2.3449377420882520554352413564891e-12,
            -2.6096534444310387762177574766136e-13,
            3.3353140420097395105869955014923e-14,
            -4.7890000440572684646750770557409e-15,
            7.5956178436192215972642568545248e-16,
            -1.3131556016891440382773397487633e-16,
            2.4483618345240857495426820738355e-17,
            -4.8805729810618777683256761918331e-18,
            1.0327285029786316149223756361204e-18,
            -2.3057633815057217157004744527025e-19,
            5.4044443001892693993017108483765e-20,
            -1.3240695194366572724155032882385e-20,
            3.3780795621371970203424792124722e-21,
            -8.9457629157111779003026926292299e-22,
            2.4519906889219317090899908651405e-22,
            -6.9388422876866318680139933157657e-23,
            2.0228278714890138392946303337791e-23,
            -6.0628500002335483105794195371764e-24,
            1.864974896403763538182378839627e-24,
            -5.8783732384849894560245036530867e-25,
            1.8958591447999563485531179503513e-25,
            -6.2481979372258858959291620728565e-26,
            2.1017901684551024686638633529074e-26,
            -7.2084300935209253690813933992446e-27,
            2.5181363892474240867156405976746e-27,
            -8.9518042258785778806143945953643e-28,
            3.2357237479762298533256235868587e-28,
            -1.1883010519855353657047144113796e-28,
            4.4306286907358104820579231941731e-29,
            -1.6761009648834829495792010135681e-29,
            6.4292946921207466972532393966088e-30,
            -2.4992261166978652421207213682763e-30,
            9.8399794299521955672828260355318e-31,
            -3.9220375242408016397989131626158e-31,
            1.5818107030056522138590618845692e-31,
            -6.4525506144890715944344098365426e-32,
            2.6611111369199356137177018346367e-32
        };
        const double bm02cs[40] = {
            0.0950041514522838136933086133556,
            -3.801864682365670991748081566851e-4,
            2.258339301031481192951829927224e-6,
            -3.895725802372228764730621412605e-8,
            1.246886416512081697930990529725e-9,
            -6.065949022102503779803835058387e-11,
            4.008461651421746991015275971045e-12,
            -3.350998183398094218467298794574e-13,
            3.377119716517417367063264341996e-14,
            -3.964585901635012700569356295823e-15,
            5.286111503883857217387939744735e-16,
            -7.852519083450852313654640243493e-17,
            1.280300573386682201011634073449e-17,
            -2.263996296391429776287099244884e-18,
            4.300496929656790388646410290477e-19,
            -8.705749805132587079747535451455e-20,
            1.86586271396209514118144277205e-20,
            -4.210482486093065457345086972301e-21,
            9.956676964228400991581627417842e-22,
            -2.457357442805313359605921478547e-22,
            6.307692160762031568087353707059e-23,
            -1.678773691440740142693331172388e-23,
            4.620259064673904433770878136087e-24,
            -1.311782266860308732237693402496e-24,
            3.834087564116302827747922440276e-25,
            -1.151459324077741271072613293576e-25,
            3.547210007523338523076971345213e-26,
            -1.119218385815004646264355942176e-26,
            3.611879427629837831698404994257e-27,
            -1.190687765913333150092641762463e-27,
            4.005094059403968131802476449536e-28,
            -1.373169422452212390595193916017e-28,
            4.794199088742531585996491526437e-29,
            -1.702965627624109584006994476452e-29,
            6.149512428936330071503575161324e-30,
            -2.255766896581828349944300237242e-30,
            8.3997075092942994860616583532e-31,
            -3.172997595562602355567423936152e-31,
            1.215205298881298554583333026514e-31,
            -4.715852749754438693013210568045e-32
        };
        const double bt02cs[39] = {
            -0.24548295213424597462050467249324,
            0.0012544121039084615780785331778299,
            -3.1253950414871522854973446709571e-5,
            1.4709778249940831164453426969314e-6,
            -9.9543488937950033643468850351158e-8,
            8.5493166733203041247578711397751e-9,
            -8.6989759526554334557985512179192e-10,
            1.0052099533559791084540101082153e-10,
            -1.2828230601708892903483623685544e-11,
            1.7731700781805131705655750451023e-12,
            -2.6174574569485577488636284180925e-13,
            4.0828351389972059621966481221103e-14,
            -6.6751668239742720054606749554261e-15,
            1.1365761393071629448392469549951e-15,
            -2.0051189620647160250559266412117e-16,
            3.6497978794766269635720591464106e-17,
            -6.83096375645823031693558437888e-18,
            1.3107583145670756620057104267946e-18,
            -2.5723363101850607778757130649599e-19,
            5.1521657441863959925267780949333e-20,
            -1.0513017563758802637940741461333e-20,
            2.1820381991194813847301084501333e-21,
            -4.6004701210362160577225905493333e-22,
            9.8407006925466818520953651199999e-23,
            -2.1334038035728375844735986346666e-23,
            4.6831036423973365296066286933333e-24,
            -1.0400213691985747236513382399999e-24,
            2.33491056773015100517777408e-25,
            -5.2956825323318615788049749333333e-26,
            1.2126341952959756829196287999999e-26,
            -2.8018897082289428760275626666666e-27,
            6.5292678987012873342593706666666e-28,
            -1.5337980061873346427835733333333e-28,
            3.6305884306364536682359466666666e-29,
            -8.6560755713629122479172266666666e-30,
            2.0779909972536284571238399999999e-30,
            -5.0211170221417221674325333333333e-31,
            1.2208360279441714184191999999999e-31,
            -2.9860056267039913454250666666666e-32
        };
        const int ntj0 = 12;
        const int nbm0 = 15;
        const int nbt02 = 16;
        const int nbm02 = 13;
        const int nbth0 = 14;
        const double xsml = std::sqrt(std::numeric_limits<double>::epsilon() * 8);
        const double xmax = 0.5/std::numeric_limits<double>::epsilon();
        const double pi4 = 0.785398163397448309615660845819876;

        assert(x >= 0);
        if (x <= 4.) {
            if (x < xsml) return 1.;
            else return dcsevl(0.125*x*x-1., bj0cs, ntj0);
        } else {
            double ampl, theta;
            if (x <= 8.) {
                double z = (128. / (x * x) - 5.) / 3.;
                ampl = (dcsevl(z, bm0cs, nbm0) + 0.75) / std::sqrt(x);
                theta = x - pi4 + dcsevl(z, bt02cs, nbt02) / x;
            } else {
                if (x > xmax)
                    throw std::runtime_error("D9B0MP NO PRECISION BECAUSE X IS BIG");
                double z = 128. / (x * x) - 1.;
                ampl = (dcsevl(z, bm02cs, nbm02) + 0.75) / std::sqrt(x);
                theta = x - pi4 + dcsevl(z, bth0cs, nbth0) / x;
            }
            return ampl * std::cos(theta);
        }
    }

    double dbesj1(double x)
    {
        // ***BEGIN PROLOGUE  DBESJ1
        // ***PURPOSE  Compute the Bessel function of the first kind of order one.
        // ***LIBRARY   SLATEC (FNLIB)
        // ***CATEGORY  C10A1
        // ***TYPE      DOUBLE PRECISION (BESJ1-S, DBESJ1-D)
        // ***KEYWORDS  BESSEL FUNCTION, FIRST KIND, FNLIB, ORDER ONE,
        //             SPECIAL FUNCTIONS
        // ***AUTHOR  Fullerton, W., (LANL)
        // ***DESCRIPTION

        // DBESJ1(X) calculates the double precision Bessel function of the
        // first kind of order one for double precision argument X.

        // Series for BJ1        on the interval  0.          to  1.60000E+01
        //                                        with weighted error   1.16E-33
        //                                         log weighted error  32.93
        //                               significant figures required  32.36
        //                                    decimal places required  33.57

        // ***REFERENCES  (NONE)
        // ***ROUTINES CALLED  D1MACH, D9B1MP, DCSEVL, INITDS, XERMSG
        // ***REVISION HISTORY  (YYMMDD)
        //   780601  DATE WRITTEN
        //   890531  Changed all specific intrinsics to generic.  (WRB)
        //   890531  REVISION DATE from Version 3.2
        //   891214  Prologue converted to Version 4.0 format.  (BAB)
        //   900315  CALLs to XERROR changed to CALLs to XERMSG.  (THJ)
        //   910401  Corrected error in code which caused values to have the
        //           wrong sign for arguments less than 4.0.  (WRB)
        // ***END PROLOGUE  DBESJ1

        const double bj1cs[19] = {
            -0.117261415133327865606240574524003,
            -0.253615218307906395623030884554698,
            0.0501270809844695685053656363203743,
            -0.00463151480962508191842619728789772,
            2.47996229415914024539124064592364e-4,
            -8.67894868627882584521246435176416e-6,
            2.14293917143793691502766250991292e-7,
            -3.93609307918317979229322764073061e-9,
            5.59118231794688004018248059864032e-11,
            -6.3276164046613930247769527401488e-13,
            5.84099161085724700326945563268266e-15,
            -4.48253381870125819039135059199999e-17,
            2.90538449262502466306018688e-19,
            -1.61173219784144165412118186666666e-21,
            7.73947881939274637298346666666666e-24,
            -3.24869378211199841143466666666666e-26,
            1.2022376772274102272e-28,
            -3.95201221265134933333333333333333e-31,
            1.16167808226645333333333333333333e-33
        };
        const double bm1cs[37] = {
            0.1069845452618063014969985308538,
            0.003274915039715964900729055143445,
            -2.987783266831698592030445777938e-5,
            8.331237177991974531393222669023e-7,
            -4.112665690302007304896381725498e-8,
            2.855344228789215220719757663161e-9,
            -2.485408305415623878060026596055e-10,
            2.543393338072582442742484397174e-11,
            -2.941045772822967523489750827909e-12,
            3.743392025493903309265056153626e-13,
            -5.149118293821167218720548243527e-14,
            7.552535949865143908034040764199e-15,
            -1.169409706828846444166290622464e-15,
            1.89656244943479157172182460506e-16,
            -3.201955368693286420664775316394e-17,
            5.599548399316204114484169905493e-18,
            -1.010215894730432443119390444544e-18,
            1.873844985727562983302042719573e-19,
            -3.563537470328580219274301439999e-20,
            6.931283819971238330422763519999e-21,
            -1.376059453406500152251408930133e-21,
            2.783430784107080220599779327999e-22,
            -5.727595364320561689348669439999e-23,
            1.197361445918892672535756799999e-23,
            -2.539928509891871976641440426666e-24,
            5.461378289657295973069619199999e-25,
            -1.189211341773320288986289493333e-25,
            2.620150977340081594957824e-26,
            -5.836810774255685901920938666666e-27,
            1.313743500080595773423615999999e-27,
            -2.985814622510380355332778666666e-28,
            6.848390471334604937625599999999e-29,
            -1.58440156822247672119296e-29,
            3.695641006570938054301013333333e-30,
            -8.687115921144668243012266666666e-31,
            2.057080846158763462929066666666e-31,
            -4.905225761116225518523733333333e-32
        };
        const double bt12cs[39] = {
            0.73823860128742974662620839792764,
            -0.0033361113174483906384470147681189,
            6.1463454888046964698514899420186e-5,
            -2.4024585161602374264977635469568e-6,
            1.4663555577509746153210591997204e-7,
            -1.1841917305589180567005147504983e-8,
            1.1574198963919197052125466303055e-9,
            -1.3001161129439187449366007794571e-10,
            1.6245391141361731937742166273667e-11,
            -2.2089636821403188752155441770128e-12,
            3.2180304258553177090474358653778e-13,
            -4.9653147932768480785552021135381e-14,
            8.0438900432847825985558882639317e-15,
            -1.3589121310161291384694712682282e-15,
            2.3810504397147214869676529605973e-16,
            -4.3081466363849106724471241420799e-17,
            8.02025440327710024349935125504e-18,
            -1.5316310642462311864230027468799e-18,
            2.9928606352715568924073040554666e-19,
            -5.9709964658085443393815636650666e-20,
            1.2140289669415185024160852650666e-20,
            -2.5115114696612948901006977706666e-21,
            5.2790567170328744850738380799999e-22,
            -1.1260509227550498324361161386666e-22,
            2.43482773595763266596634624e-23,
            -5.3317261236931800130038442666666e-24,
            1.1813615059707121039205990399999e-24,
            -2.6465368283353523514856789333333e-25,
            5.9903394041361503945577813333333e-26,
            -1.3690854630829503109136383999999e-26,
            3.1576790154380228326413653333333e-27,
            -7.3457915082084356491400533333333e-28,
            1.722808148072274793070592e-28,
            -4.07169079612865079410688e-29,
            9.6934745136779622700373333333333e-30,
            -2.3237636337765716765354666666666e-30,
            5.6074510673522029406890666666666e-31,
            -1.3616465391539005860522666666666e-31,
            3.3263109233894654388906666666666e-32
        };
        const double bm12cs[40] = {
            0.09807979156233050027272093546937,
            0.001150961189504685306175483484602,
            -4.312482164338205409889358097732e-6,
            5.951839610088816307813029801832e-8,
            -1.704844019826909857400701586478e-9,
            7.798265413611109508658173827401e-11,
            -4.958986126766415809491754951865e-12,
            4.038432416421141516838202265144e-13,
            -3.993046163725175445765483846645e-14,
            4.619886183118966494313342432775e-15,
            -6.089208019095383301345472619333e-16,
            8.960930916433876482157048041249e-17,
            -1.449629423942023122916518918925e-17,
            2.546463158537776056165149648068e-18,
            -4.80947287464783644425926371862e-19,
            9.687684668292599049087275839124e-20,
            -2.067213372277966023245038117551e-20,
            4.64665155915038473180276780959e-21,
            -1.094966128848334138241351328339e-21,
            2.693892797288682860905707612785e-22,
            -6.894992910930374477818970026857e-23,
            1.83026826275206290989066855474e-23,
            -5.025064246351916428156113553224e-24,
            1.423545194454806039631693634194e-24,
            -4.152191203616450388068886769801e-25,
            1.244609201503979325882330076547e-25,
            -3.827336370569304299431918661286e-26,
            1.205591357815617535374723981835e-26,
            -3.884536246376488076431859361124e-27,
            1.278689528720409721904895283461e-27,
            -4.295146689447946272061936915912e-28,
            1.470689117829070886456802707983e-28,
            -5.128315665106073128180374017796e-29,
            1.819509585471169385481437373286e-29,
            -6.563031314841980867618635050373e-30,
            2.404898976919960653198914875834e-30,
            -8.945966744690612473234958242979e-31,
            3.37608516065723102663714897824e-31,
            -1.291791454620656360913099916966e-31,
            5.008634462958810520684951501254e-32
        };
        const double bth1cs[44] = {
            0.74749957203587276055443483969695,
            -0.0012400777144651711252545777541384,
            9.9252442404424527376641497689592e-6,
            -2.0303690737159711052419375375608e-7,
            7.5359617705690885712184017583629e-9,
            -4.1661612715343550107630023856228e-10,
            3.0701618070834890481245102091216e-11,
            -2.8178499637605213992324008883924e-12,
            3.0790696739040295476028146821647e-13,
            -3.8803300262803434112787347554781e-14,
            5.5096039608630904934561726208562e-15,
            -8.6590060768383779940103398953994e-16,
            1.4856049141536749003423689060683e-16,
            -2.7519529815904085805371212125009e-17,
            5.4550796090481089625036223640923e-18,
            -1.1486534501983642749543631027177e-18,
            2.5535213377973900223199052533522e-19,
            -5.9621490197413450395768287907849e-20,
            1.4556622902372718620288302005833e-20,
            -3.7022185422450538201579776019593e-21,
            9.7763074125345357664168434517924e-22,
            -2.6726821639668488468723775393052e-22,
            7.5453300384983271794038190655764e-23,
            -2.1947899919802744897892383371647e-23,
            6.5648394623955262178906999817493e-24,
            -2.0155604298370207570784076869519e-24,
            6.341776855677614349214466718567e-25,
            -2.0419277885337895634813769955591e-25,
            6.7191464220720567486658980018551e-26,
            -2.2569079110207573595709003687336e-26,
            7.7297719892989706370926959871929e-27,
            -2.696744451229464091321142408092e-27,
            9.5749344518502698072295521933627e-28,
            -3.4569168448890113000175680827627e-28,
            1.2681234817398436504211986238374e-28,
            -4.7232536630722639860464993713445e-29,
            1.7850008478186376177858619796417e-29,
            -6.8404361004510395406215223566746e-30,
            2.6566028671720419358293422672212e-30,
            -1.045040252791445291771416148467e-30,
            4.1618290825377144306861917197064e-31,
            -1.6771639203643714856501347882887e-31,
            6.8361997776664389173535928028528e-32,
            -2.817224786123364116673957462281e-32
        };
        const int ntj1 = 12;
        const int nbm1 = 15;
        const int nbt12 = 16;
        const int nbm12 = 13;
        const int nbth1 = 14;
        const double xsml = std::sqrt(std::numeric_limits<double>::epsilon() * 8);
        const double xmin = std::numeric_limits<double>::min() * 2.;
        const double xmax = 0.5/std::numeric_limits<double>::epsilon();
        const double pi4 = 0.785398163397448309615660845819876;

        assert(x >= 0);
        if (x <= 4.) {
            if (x <= xsml) return 0.5 * x;
            else return x * (dcsevl(0.125*x*x-1., bj1cs, ntj1) + 0.25);
        } else {
            double ampl, theta;
            if (x <= 8.) {
                double z = (128. / (x * x) - 5.) / 3.;
                ampl = (dcsevl(z, bm1cs, nbm1) + 0.75) / std::sqrt(x);
                theta = x - pi4 * 3. + dcsevl(z, bt12cs, nbt12) / x;
            } else {
                if (x > xmax)
                    throw std::runtime_error("DBESJ1 No precision because X is too big");
                double z = 128. / (x * x) - 1.;
                ampl = (dcsevl(z, bm12cs, nbm12) + 0.75) / std::sqrt(x);
                theta = x - pi4 * 3. + dcsevl(z, bth1cs, nbth1) / x;
            }
            return ampl * std::cos(theta);
        }
    }

    double dasyjy(double x, double fnu, bool is_j, double *wk, int *iflw)
    {
        // ***BEGIN PROLOGUE  DASYJY
        // ***SUBSIDIARY
        // ***PURPOSE  Subsidiary to DBESJ and DBESY
        // ***LIBRARY   SLATEC
        // ***TYPE      DOUBLE PRECISION (ASYJY-S, DASYJY-D)
        // ***AUTHOR  Amos, D. E., (SNLA)
        // ***DESCRIPTION
        //
        //                 DASYJY computes Bessel functions J and Y
        //               for arguments X.GT.0.0 and orders FNU .GE. 35.0
        //               on FLGJY = 1 and FLGJY = -1 respectively
        //
        //                                  INPUT
        //
        //      FUNJY - External subroutine JAIRY or YAIRY
        //          X - Argument, X.GT.0.0D0
        //        FNU - Order of the first Bessel function
        //      FLGJY - Selection flag
        //              FLGJY =  1.0D0 gives the J function
        //              FLGJY = -1.0D0 gives the Y function
        //         IN - Number of functions desired, IN = 1 or 2
        //
        //                                  OUTPUT
        //
        //         Y  - A vector whose first IN components contain the sequence
        //       IFLW - A flag indicating underflow or overflow
        //                    return variables for BESJ only
        //      WK(1) = 1 - (X/FNU)**2 = W**2
        //      WK(2) = SQRT(ABS(WK(1)))
        //      WK(3) = ABS(WK(2) - ATAN(WK(2)))  or
        //              ABS(LN((1 + WK(2))/(X/FNU)) - WK(2))
        //            = ABS((2/3)*ZETA**(3/2))
        //      WK(4) = FNU*WK(3)
        //      WK(5) = (1.5*WK(3)*FNU)**(1/3) = SQRT(ZETA)*FNU**(1/3)
        //      WK(6) = SIGN(1.,W**2)*WK(5)**2 = SIGN(1.,W**2)*ZETA*FNU**(2/3)
        //      WK(7) = FNU**(1/3)
        //
        //     Abstract   **** A Double Precision Routine ****
        //         DASYJY implements the uniform asymptotic expansion of
        //         the J and Y Bessel functions for FNU.GE.35 and real
        //         X.GT.0.0D0. The forms are identical except for a change
        //         in sign of some of the terms. This change in sign is
        //         accomplished by means of the flag FLGJY = 1 or -1. On
        //         FLGJY = 1 the Airy functions AI(X) and DAI(X) are
        //         supplied by the external function JAIRY, and on
        //         FLGJY = -1 the Airy functions BI(X) and DBI(X) are
        //         supplied by the external function YAIRY.
        //
        // ***SEE ALSO  DBESJ, DBESY
        // ***ROUTINES CALLED  D1MACH, I1MACH
        // ***REVISION HISTORY  (YYMMDD)
        //    750101  DATE WRITTEN
        //    890531  Changed all specific intrinsics to generic.  (WRB)
        //    890911  Removed unnecessary intrinsics.  (WRB)
        //    891004  Correction computation of ELIM.  (WRB)
        //    891009  Removed unreferenced variable.  (WRB)
        //    891214  Prologue converted to Version 4.0 format.  (BAB)
        //    900328  Added TYPE section.  (WRB)
        //    910408  Updated the AUTHOR section.  (WRB)
        //    170203  Converted to C++, and modified to only cover in==1 option. (MJ)
        // ***END PROLOGUE  DASYJY

        const double tols = -6.90775527898214;
        const double con1 = 0.666666666666667;
        const double con2 = 0.333333333333333;
        const double con548 = 0.104166666666667;
        const double ar[8] = {
            0.0835503472222222, 0.128226574556327,
            0.29184902646414, 0.881627267443758, 3.32140828186277,
            14.9957629868626, 78.9230130115865, 474.451538868264
        };
        const double br[10] = {
            -0.145833333333333, -0.0987413194444444,
            -0.143312053915895, -0.317227202678414, -0.94242914795712,
            -3.51120304082635, -15.727263620368, -82.2814390971859,
            -492.355370523671, -3316.21856854797
        };
        const double c[65] = {
            -0.208333333333333, 0.125, 0.334201388888889,
            -0.401041666666667, 0.0703125, -1.02581259645062, 1.84646267361111,
            -0.8912109375, 0.0732421875, 4.66958442342625, -11.207002616223,
            8.78912353515625, -2.3640869140625, 0.112152099609375,
            -28.2120725582002, 84.6362176746007, -91.81824154324,
            42.5349987453885, -7.36879435947963, 0.227108001708984,
            212.570130039217, -765.252468141182, 1059.990452528,
            -699.579627376133, 218.190511744212, -26.4914304869516,
            0.572501420974731, -1919.45766231841, 8061.72218173731,
            -13586.5500064341, 11655.3933368645, -5305.6469786134,
            1200.90291321635, -108.090919788395, 1.72772750258446,
            20204.2913309661, -96980.5983886375, 192547.001232532,
            -203400.177280416, 122200.464983017, -41192.6549688976,
            7109.51430248936, -493.915304773088, 6.07404200127348,
            -242919.187900551, 1311763.61466298, -2998015.91853811,
            3763271.2976564, -2813563.22658653, 1268365.27332162,
            -331645.172484564, 45218.7689813627, -2499.83048181121,
            24.3805296995561, 3284469.85307204, -19706819.1184322,
            50952602.4926646, -74105148.2115327, 66344512.274729,
            -37567176.6607634, 13288767.1664218, -2785618.12808645,
            308186.404612662, -13886.089753717, 110.017140269247
        };
        const double alfa[4][26] = {
            {
                -0.00444444444444444, -9.22077922077922e-4,
                -8.84892884892885e-5, 1.6592768783245e-4, 2.46691372741793e-4,
                2.65995589346255e-4, 2.61824297061501e-4,
                2.48730437344656e-4, 2.32721040083232e-4, 2.16362485712365e-4,
                2.00738858762752e-4, 1.86267636637545e-4,
                1.73060775917876e-4, 1.61091705929016e-4, 1.50274774160908e-4,
                1.4050349739127e-4, 1.31668816545923e-4, 1.23667445598253e-4,
                1.16405271474738e-4, 1.09798298372713e-4,
                1.03772410422993e-4, 9.82626078369363e-5, 9.32120517249503e-5,
                8.85710852478712e-5, 8.429631057157e-5, 8.03497548407791e-5
            },
            {
                6.93735541354589e-4, 2.32241745182922e-4,
                -1.41986273556691e-5, -1.16444931672049e-4,
                -1.50803558053049e-4, -1.55121924918096e-4,
                -1.46809756646466e-4, -1.33815503867491e-4,
                -1.19744975684254e-4, -1.06184319207974e-4,
                -9.37699549891194e-5, -8.26923045588193e-5,
                -7.29374348155221e-5, -6.44042357721016e-5,
                -5.69611566009369e-5, -5.04731044303562e-5,
                -4.48134868008883e-5, -3.98688727717599e-5,
                -3.55400532972042e-5, -3.17414256609022e-5,
                -2.83996793904175e-5, -2.54522720634871e-5,
                -2.28459297164725e-5, -2.05352753106481e-5,
                -1.84816217627666e-5, -1.66519330021394e-5
            },
            {
                -3.54211971457744e-4, -1.56161263945159e-4,
                3.04465503594936e-5, 1.30198655773243e-4, 1.67471106699712e-4,
                1.70222587683593e-4, 1.56501427608595e-4,
                1.36339170977445e-4, 1.14886692029825e-4, 9.45869093034688e-5,
                7.64498419250898e-5, 6.07570334965197e-5,
                4.74394299290509e-5, 3.62757512005344e-5, 2.69939714979225e-5,
                1.93210938247939e-5, 1.30056674793963e-5,
                7.82620866744497e-6, 3.59257485819352e-6, 1.44040049814252e-7,
                -2.65396769697939e-6, -4.91346867098486e-6,
                -6.72739296091248e-6, -8.17269379678658e-6,
                -9.31304715093561e-6, -1.02011418798016e-5
            },
            {
                3.78194199201773e-4, 2.02471952761816e-4,
                -6.37938506318862e-5, -2.38598230603006e-4
                    -3.10916256027362e-4, -3.13680115247576e-4,
                -2.78950273791323e-4, -2.28564082619141e-4,
                -1.75245280340847e-4, -1.2554406306069e-4,
                -8.22982872820208e-5, -4.62860730588116e-5,
                -1.72334302366962e-5, 5.60690482304602e-6,
                2.31395443148287e-5, 3.62642745856794e-5, 4.58006124490189e-5,
                5.24595294959114e-5, 5.68396208545815e-5,
                5.94349820393104e-5, 6.06478527578422e-5, 6.08023907788436e-5,
                6.0157789453946e-5, 5.89199657344698e-5, 5.72515823777593e-5,
                5.52804375585853e-5
            }
        };
        const double beta[5][26] = {
            {
                0.0179988721413553, 0.00559964911064388,
                0.00288501402231133, 0.00180096606761054, 0.00124753110589199,
                9.22878876572938e-4, 7.14430421727287e-4, 5.71787281789705e-4,
                4.69431007606482e-4, 3.93232835462917e-4,
                3.34818889318298e-4, 2.88952148495752e-4, 2.52211615549573e-4,
                2.22280580798883e-4, 1.97541838033063e-4,
                1.76836855019718e-4, 1.59316899661821e-4, 1.44347930197334e-4,
                1.31448068119965e-4, 1.20245444949303e-4,
                1.10449144504599e-4, 1.01828770740567e-4, 9.41998224204238e-5,
                8.74130545753834e-5, 8.13466262162801e-5,
                7.59002269646219e-5
            },
            {
                -0.00149282953213429,
                -8.78204709546389e-4, -5.02916549572035e-4,
                -2.94822138512746e-4, -1.75463996970783e-4,
                -1.04008550460816e-4, -5.96141953046458e-5,
                -3.12038929076098e-5, -1.2608973598023e-5,
                -2.4289260857573e-7, 8.05996165414274e-6, 1.36507009262147e-5,
                1.73964125472926e-5, 1.98672978842134e-5,
                2.14463263790823e-5, 2.23954659232457e-5, 2.28967783814713e-5,
                2.30785389811178e-5, 2.30321976080909e-5,
                2.28236073720349e-5, 2.25005881105292e-5, 2.20981015361991e-5,
                2.16418427448104e-5, 2.11507649256221e-5,
                2.06388749782171e-5, 2.01165241997082e-5
            },
            {
                5.52213076721293e-4,
                4.47932581552385e-4, 2.79520653992021e-4,
                1.52468156198447e-4, 6.93271105657044e-5, 1.76258683069991e-5,
                -1.35744996343269e-5, -3.17972413350427e-5,
                -4.18861861696693e-5, -4.69004889379141e-5,
                -4.87665447413787e-5, -4.87010031186735e-5,
                -4.74755620890087e-5, -4.55813058138628e-5,
                -4.33309644511266e-5, -4.0923019315775e-5,
                -3.84822638603221e-5, -3.60857167535411e-5,
                -3.37793306123367e-5, -3.1588856077211e-5,
                -2.95269561750807e-5, -2.75978914828336e-5,
                -2.58006174666884e-5, -2.4130835676128e-5,
                -2.25823509518346e-5, -2.11479656768913e-5
            },
            {
                -4.7461779655996e-4, -4.77864567147321e-4,
                -3.20390228067038e-4, -1.61105016119962e-4,
                -4.25778101285435e-5, 3.44571294294968e-5,
                7.97092684075675e-5, 1.03138236708272e-4, 1.12466775262204e-4,
                1.13103642108481e-4, 1.08651634848774e-4,
                1.01437951597662e-4, 9.29298396593364e-5, 8.4029313301609e-5,
                7.52727991349134e-5, 6.69632521975731e-5, 5.92564547323195e-5,
                5.22169308826976e-5, 4.58539485165361e-5,
                4.01445513891487e-5, 3.50481730031328e-5, 3.05157995034347e-5,
                2.64956119950516e-5, 2.29363633690998e-5,
                1.97893056664022e-5, 1.70091984636413e-5
            },
            {
                7.36465810572578e-4,
                8.72790805146194e-4, 6.22614862573135e-4,
                2.85998154194304e-4, 3.84737672879366e-6,
                -1.87906003636972e-4, -2.97603646594555e-4,
                -3.45998126832656e-4, -3.53382470916038e-4,
                -3.35715635775049e-4, -3.0432112478904e-4,
                -2.66722723047613e-4, -2.2765421412282e-4,
                -1.89922611854562e-4, -1.55058918599094e-4,
                -1.23778240761874e-4, -9.62926147717644e-5,
                -7.25178327714425e-5, -5.22070028895634e-5,
                -3.50347750511901e-5, -2.06489761035552e-5,
                -8.70106096849767e-6, 1.136986866751e-6, 9.16426474122779e-6,
                1.56477785428873e-5, 2.08223629482467e-5
            }
        };
        const double gama[26] = {
            0.629960524947437, 0.251984209978975,
            0.154790300415656, 0.110713062416159, 0.0857309395527395,
            0.0697161316958684, 0.0586085671893714, 0.0504698873536311,
            0.0442600580689155, 0.039372066154351, 0.0354283195924455,
            0.0321818857502098, 0.0294646240791158, 0.0271581677112934,
            0.0251768272973862, 0.0234570755306079, 0.0219508390134907,
            0.0206210828235646, 0.0194388240897881, 0.0183810633800683,
            0.0174293213231963, 0.0165685837786612, 0.0157865285987918,
            0.0150729501494096, 0.0144193250839955, 0.0138184805735342
        };

        const double tol = std::max(std::numeric_limits<double>::epsilon(), 1e-15);
        const double elim = -std::log(std::numeric_limits<double>::min() *
                                      (is_j ? 1.e3 : 1./std::numeric_limits<double>::epsilon()));

        int kmax[5];
        double cr[10];
        double dr[10];
        double upol[10];

        double fn = fnu;
        *iflw = 0;
        double xx = x / fn;
        wk[0] = 1. - xx * xx;
        double abw2 = std::abs(wk[0]);
        wk[1] = std::sqrt(abw2);
        wk[6] = std::pow(fn, con2);
        double phi, asum, bsum;
        if (abw2 <= 0.2775) {

            //     ASYMPTOTIC EXPANSION
            //     CASES NEAR X=FN, ABS(1.-(X/FN)**2).LE.0.2775
            //     COEFFICIENTS OF ASYMPTOTIC EXPANSION BY SERIES

            //     ZETA AND TRUNCATION FOR A(ZETA) AND B(ZETA) SERIES

            //     KMAX IS TRUNCATION INDEX FOR A(ZETA) AND B(ZETA) SERIES=MAX(2,SA)

            double sa = 0.;
            if (abw2 != 0.) {
                sa = tols / std::log(abw2);
            }
            double sb = sa;
            for (int i=1; i<=5; ++i) {
                double akm = std::max(sa, 2.);
                kmax[i-1] = int(akm);
                sa += sb;
            }
            int kb = kmax[4]-1;
            int klast = kb;
            sa = gama[kb];
            for (int k = 1; k <= klast; ++k) {
                --kb;
                sa = sa * wk[0] + gama[kb];
            }
            double z = wk[0] * sa;
            double az = std::abs(z);
            double rtz = std::sqrt(az);
            wk[2] = con1 * az * rtz;
            wk[3] = wk[2] * fn;
            wk[4] = rtz * wk[6];
            wk[5] = -wk[4] * wk[4];
            if (z > 0.) {
                if (wk[3] > elim) {
                    *iflw = 1;
                    return 0.;
                }
                wk[5] = -wk[5];
            }
            phi = std::sqrt(std::sqrt(sa + sa + sa + sa));

            //     B(ZETA) FOR S=0

            kb = kmax[4]-1;
            klast = kb;
            sb = beta[0][kb];
            for (int k = 1; k <= klast; ++k) {
                --kb;
                sb = sb * wk[0] + beta[0][kb];
            }
            double fn2 = fn * fn;
            double rfn2 = 1. / fn2;
            double rden = 1.;
            asum = 1.;
            double relb = tol * std::abs(sb);
            bsum = sb;
            for (int ks = 0; ks < 4; ++ks) {
                rden *= rfn2;

                //     A(ZETA) AND B(ZETA) FOR S=1,2,3,4

                kb = kmax[3-ks]-1;
                klast = kb;
                sa = alfa[ks][kb];
                sb = beta[ks+1][kb];
                for (int k = 1; k <= klast; ++k) {
                    --kb;
                    sa = sa * wk[0] + alfa[ks][kb];
                    sb = sb * wk[0] + beta[ks+1][kb];
                }
                double ta = sa * rden;
                double tb = sb * rden;
                asum += ta;
                bsum += tb;
                if (std::abs(ta) <= tol && std::abs(tb) <= relb) break;
            }
            bsum /= fn * wk[6];
        } else {
            upol[0] = 1.;
            double tau = 1. / wk[1];
            double t2 = 1. / wk[0];
            double rcz,rtz;
            if (wk[0] < 0.) {
                //     CASES FOR (X/FN).GT.SQRT(1.2775)
                wk[2] = std::abs(wk[1] - std::atan(wk[1]));
                wk[3] = wk[2] * fn;
                rcz = -con1 / wk[3];
                double z32 = wk[2] * 1.5;
                rtz = std::pow(z32, con2);
                wk[4] = rtz * wk[6];
                wk[5] = -wk[4] * wk[4];
            } else {
                //     CASES FOR (X/FN).LT.SQRT(0.7225)
                wk[2] = std::abs(std::log((wk[1] + 1.) / xx) - wk[1]);
                wk[3] = wk[2] * fn;
                rcz = con1 / wk[3];
                if (wk[3] > elim) {
                    *iflw = 1;
                    return 0.;
                }
                double z32 = wk[2] * 1.5;
                rtz = std::pow(z32, con2);
                wk[6] = std::pow(fn, con2);
                wk[4] = rtz * wk[6];
                wk[5] = wk[4] * wk[4];
            }
            phi = std::sqrt((rtz + rtz) * tau);
            double tb = 1.;
            asum = 1.;
            double tfn = tau / fn;
            double rden = 1. / fn;
            double rfn2 = rden * rden;
            rden = 1.;
            upol[1] = (c[0] * t2 + c[1]) * tfn;
            double crz32 = con548 * rcz;
            bsum = upol[1] + crz32;
            double relb = tol * std::abs(bsum);
            double ap = tfn;
            int ks = 0;
            int kp1 = 2;
            double rzden = rcz;
            int l = 2;
            int iseta = 0;
            int isetb = 0;
            for (int lr = 2; lr <= 8; lr += 2) {

                //     COMPUTE TWO U POLYNOMIALS FOR NEXT A(ZETA) AND B(ZETA)

                int lrp1 = lr + 1;
                for (int k = lr; k <= lrp1; ++k) {
                    ++ks;
                    ++kp1;
                    ++l;
                    double s1 = c[l - 1];
                    for (int j = 2; j <= kp1; ++j) {
                        ++l;
                        s1 = s1 * t2 + c[l - 1];
                    }
                    ap *= tfn;
                    upol[kp1 - 1] = ap * s1;
                    cr[ks - 1] = br[ks - 1] * rzden;
                    rzden *= rcz;
                    dr[ks - 1] = ar[ks - 1] * rzden;
                }
                double suma = upol[lrp1 - 1];
                double sumb = upol[lr + 1] + upol[lrp1 - 1] * crz32;
                int ju = lrp1;
                for (int jr = 1; jr <= lr; ++jr) {
                    --ju;
                    suma += cr[jr - 1] * upol[ju - 1];
                    sumb += dr[jr - 1] * upol[ju - 1];
                }
                rden *= rfn2;
                tb = -tb;
                if (wk[0] > 0.) tb = std::abs(tb);
                if (rden >= tol) {
                    asum += suma * tb;
                    bsum += sumb * tb;
                    continue;
                }
                if (iseta != 1) {
                    if (std::abs(suma) < tol) iseta = 1;
                    asum += suma * tb;
                }
                if (isetb != 1) {
                    if (std::abs(sumb) < relb) isetb = 1;
                    bsum += sumb * tb;
                }
                if (iseta == 1 && isetb == 1) break;
            }
            tb = wk[4];
            if (wk[0] > 0.) tb = -tb;
            bsum /= tb;
        }
        double fi, dfi;
        if (is_j) {
            djairy(wk[5], wk[4], wk[3], &fi, &dfi);
        } else {
            dyairy(wk[5], wk[4], wk[3], &fi, &dfi);
        }
        double ta = 1. / tol;
        double tb = std::numeric_limits<double>::min() * ta * 1e3;
        if (std::abs(fi) <= tb) {
            fi *= ta;
            dfi *= ta;
            phi *= tol;
        }
        double jy = phi * (fi * asum + dfi * bsum) / wk[6];
        if (!is_j) jy = -jy;
        return jy;
    }

    void djairy(double x, double rx, double c, double *ai, double *dai)
    {
        // ***BEGIN PROLOGUE  DJAIRY
        // ***SUBSIDIARY
        // ***PURPOSE  Subsidiary to DBESJ and DBESY
        // ***LIBRARY   SLATEC
        // ***TYPE      DOUBLE PRECISION (JAIRY-S, DJAIRY-D)
        // ***AUTHOR  Amos, D. E., (SNLA)
        //           Daniel, S. L., (SNLA)
        //           Weston, M. K., (SNLA)
        // ***DESCRIPTION
        //
        //                  DJAIRY computes the Airy function AI(X)
        //                   and its derivative DAI(X) for DASYJY
        //
        //                                   INPUT
        //
        //         X - Argument, computed by DASYJY, X unrestricted
        //        RX - RX=SQRT(ABS(X)), computed by DASYJY
        //         C - C=2.*(ABS(X)**1.5)/3., computed by DASYJY
        //
        //                                  OUTPUT
        //
        //        AI - Value of function AI(X)
        //       DAI - Value of the derivative DAI(X)
        //
        // ***SEE ALSO  DBESJ, DBESY
        // ***ROUTINES CALLED  (NONE)
        // ***REVISION HISTORY  (YYMMDD)
        //    750101  DATE WRITTEN
        //    890531  Changed all specific intrinsics to generic.  (WRB)
        //    891009  Removed unreferenced variable.  (WRB)
        //    891214  Prologue converted to Version 4.0 format.  (BAB)
        //    900328  Added TYPE section.  (WRB)
        //    910408  Updated the AUTHOR section.  (WRB)
        //    170203  Converted to C++. (MJ)
        // ***END PROLOGUE  DJAIRY

        const double con2 = 5.03154716196777;
        const double con3 = 0.380004589867293;
        const double con4 = 0.833333333333333;
        const double con5 = 0.866025403784439;
        const double ak1[14] = {
            0.220423090987793, -0.1252902427877,
            0.0103881163359194, 8.22844152006343e-4, -2.34614345891226e-4,
            1.63824280172116e-5, 3.06902589573189e-7, -1.29621999359332e-7,
            8.22908158823668e-9, 1.53963968623298e-11, -3.39165465615682e-11,
            2.03253257423626e-12, -1.10679546097884e-14, -5.1616949778508e-15
        };
        const double ak2[23] = {
            0.274366150869598, 0.00539790969736903,
            -0.0015733922062119, 4.2742752824875e-4, -1.12124917399925e-4,
            2.88763171318904e-5, -7.36804225370554e-6, 1.87290209741024e-6,
            -4.75892793962291e-7, 1.21130416955909e-7, -3.09245374270614e-8,
            7.92454705282654e-9, -2.03902447167914e-9, 5.26863056595742e-10,
            -1.36704767639569e-10, 3.56141039013708e-11, -9.3138829654843e-12,
            2.44464450473635e-12, -6.43840261990955e-13, 1.70106030559349e-13,
            -4.50760104503281e-14, 1.19774799164811e-14, -3.19077040865066e-15
        };
        const double ak3[14] = {
            0.280271447340791, -0.00178127042844379,
            4.03422579628999e-5, -1.63249965269003e-6, 9.21181482476768e-8,
            -6.52294330229155e-9, 5.47138404576546e-10, -5.2440825180026e-11,
            5.60477904117209e-12, -6.56375244639313e-13, 8.31285761966247e-14,
            -1.12705134691063e-14, 1.62267976598129e-15, -2.46480324312426e-16
        };
        const double ajp[19] = {
            0.0778952966437581, -0.184356363456801,
            0.0301412605216174, 0.0305342724277608, -0.00495424702513079,
            -0.00172749552563952, 2.4313763783919e-4, 5.04564777517082e-5,
            -6.16316582695208e-6, -9.03986745510768e-7, 9.70243778355884e-8,
            1.09639453305205e-8, -1.04716330588766e-9, -9.60359441344646e-11,
            8.25358789454134e-12, 6.36123439018768e-13, -4.96629614116015e-14,
            -3.29810288929615e-15, 2.35798252031104e-16
        };
        const double ajn[19] = {
            0.0380497887617242, -0.245319541845546,
            0.165820623702696, 0.0749330045818789, -0.0263476288106641,
            -0.00592535597304981, 0.00144744409589804, 2.18311831322215e-4,
            -4.10662077680304e-5, -4.66874994171766e-6, 7.1521880727716e-7,
            6.52964770854633e-8, -8.44284027565946e-9, -6.44186158976978e-10,
            7.20802286505285e-11, 4.72465431717846e-12, -4.66022632547045e-13,
            -2.67762710389189e-14, 2.36161316570019e-15
        };
        const double a[15] = {
            0.490275424742791, 0.00157647277946204,
            -9.66195963140306e-5, 1.35916080268815e-7, 2.98157342654859e-7,
            -1.86824767559979e-8, -1.03685737667141e-9, 3.28660818434328e-10,
            -2.5709141063278e-11, -2.32357655300677e-12, 9.57523279048255e-13,
            -1.20340828049719e-13, -2.90907716770715e-15, 4.55656454580149e-15,
            -9.99003874810259e-16
        };
        const double b[15] = {
            0.278593552803079, -0.00352915691882584,
            -2.31149677384994e-5, 4.7131784226356e-6, -1.12415907931333e-7,
            -2.00100301184339e-8, 2.60948075302193e-9, -3.55098136101216e-11,
            -3.50849978423875e-11, 5.83007187954202e-12, -2.04644828753326e-13,
            -1.10529179476742e-13, 2.87724778038775e-14, -2.88205111009939e-15,
            -3.32656311696166e-16
        };
        const double dak1[14] = {
            0.204567842307887, -0.0661322739905664,
            -0.00849845800989287, 0.00312183491556289, -2.70016489829432e-4,
            -6.35636298679387e-6, 3.02397712409509e-6, -2.18311195330088e-7,
            -5.36194289332826e-10, 1.1309803562231e-9, -7.43023834629073e-11,
            4.28804170826891e-13, 2.23810925754539e-13, -1.39140135641182e-14
        };
        const double dak2[24] = {
            0.29333234388323, -0.00806196784743112,
            0.0024254017233314, -6.82297548850235e-4, 1.85786427751181e-4,
            -4.97457447684059e-5, 1.32090681239497e-5, -3.49528240444943e-6,
            9.24362451078835e-7, -2.44732671521867e-7, 6.4930783764891e-8,
            -1.72717621501538e-8, 4.60725763604656e-9, -1.2324905529155e-9,
            3.30620409488102e-10, -8.89252099772401e-11, 2.39773319878298e-11,
            -6.4801392115345e-12, 1.75510132023731e-12, -4.76303829833637e-13,
            1.2949824110081e-13, -3.5267962221043e-14, 9.62005151585923e-15,
            -2.62786914342292e-15
        };
        const double dak3[14] = {
            0.284675828811349, 0.0025307307261908,
            -4.83481130337976e-5, 1.84907283946343e-6, -1.01418491178576e-7,
            7.05925634457153e-9, -5.85325291400382e-10, 5.56357688831339e-11,
            -5.908890947795e-12, 6.88574353784436e-13, -8.68588256452194e-14,
            1.17374762617213e-14, -1.68523146510923e-15, 2.55374773097056e-16
        };
        const double dajp[19] = {
            0.0653219131311457, -0.120262933688823,
            0.00978010236263823, 0.0167948429230505, -0.00197146140182132,
            -8.45560295098867e-4, 9.42889620701976e-5, 2.25827860945475e-5,
            -2.29067870915987e-6, -3.76343991136919e-7, 3.45663933559565e-8,
            4.29611332003007e-9, -3.58673691214989e-10, -3.57245881361895e-11,
            2.72696091066336e-12, 2.26120653095771e-13, -1.58763205238303e-14,
            -1.12604374485125e-15, 7.31327529515367e-17
        };
        const double dajn[19] = {
            0.0108594539632967, 0.0853313194857091,
            -0.315277068113058, -0.0878420725294257, 0.0553251906976048,
            0.00941674060503241, -0.00332187026018996, -4.11157343156826e-4,
            1.01297326891346e-4, 9.87633682208396e-6, -1.87312969812393e-6,
            -1.50798500131468e-7, 2.32687669525394e-8, 1.59599917419225e-9,
            -2.07665922668385e-10, -1.24103350500302e-11, 1.39631765331043e-12,
            7.3940097115574e-14, -7.328874756275e-15
        };
        const double da[15] = {
            0.491627321104601, 0.00311164930427489,
            8.23140762854081e-5, -4.61769776172142e-6, -6.13158880534626e-8,
            2.8729580465652e-8, -1.81959715372117e-9, -1.44752826642035e-10,
            4.53724043420422e-11, -3.99655065847223e-12, -3.24089119830323e-13,
            1.62098952568741e-13, -2.40765247974057e-14, 1.69384811284491e-16,
            8.17900786477396e-16
        };
        const double db[15] = {
            -0.277571356944231, 0.0044421283341992,
            -8.42328522190089e-5, -2.5804031841871e-6, 3.42389720217621e-7,
            -6.24286894709776e-9, -2.36377836844577e-9, 3.16991042656673e-10,
            -4.40995691658191e-12, -5.18674221093575e-12, 9.64874015137022e-13,
            -4.9019057660871e-14, -1.77253430678112e-14, 5.55950610442662e-15,
            -7.1179333757953e-16
        };
        const int n1 = 14;
        const int n2 = 23;
        const int n3 = 19;
        const int n4 = 15;
        const int n1d = 14;
        const int n2d = 24;
        const int n3d = 19;
        const int n4d = 15;
        const int m1 = 12;
        const int m2 = 21;
        const int m3 = 17;
        const int m4 = 13;
        const int m1d = 12;
        const int m2d = 22;
        const int m3d = 17;
        const int m4d = 13;
        const double fpi12 = 1.30899693899575;

        if (x < 0.) {
            if (c <= 5.) {
                double t = c * 0.4 - 1.;
                double tt = t + t;
                int j = n3-1;
                double f1 = ajp[j];
                double e1 = ajn[j];
                double f2 = 0.;
                double e2 = 0.;
                for (int i=1; i<=m3; ++i) {
                    --j;
                    double temp1 = f1;
                    double temp2 = e1;
                    f1 = tt * f1 - f2 + ajp[j];
                    e1 = tt * e1 - e2 + ajn[j];
                    f2 = temp1;
                    e2 = temp2;
                }
                *ai = t * e1 - e2 + ajn[0] - x * (t * f1 - f2 + ajp[0]);

                j = n3d-1;
                f1 = dajp[j];
                e1 = dajn[j];
                f2 = 0.;
                e2 = 0.;
                for (int i=1; i<=m3d; ++i) {
                    --j;
                    double temp1 = f1;
                    double temp2 = e1;
                    f1 = tt * f1 - f2 + dajp[j];
                    e1 = tt * e1 - e2 + dajn[j];
                    f2 = temp1;
                    e2 = temp2;
                }
                *dai = x * x * (t * f1 - f2 + dajp[0]) + (t * e1 - e2 + dajn[0]);
            } else {
                double t = 10. / c - 1.;
                double tt = t + t;
                int j = n4-1;
                double f1 = a[j];
                double e1 = b[j];
                double f2 = 0.;
                double e2 = 0.;
                for (int i=1; i<=m4; ++i) {
                    --j;
                    double temp1 = f1;
                    double temp2 = e1;
                    f1 = tt * f1 - f2 + a[j];
                    e1 = tt * e1 - e2 + b[j];
                    f2 = temp1;
                    e2 = temp2;
                }
                double temp1 = t * f1 - f2 + a[0];
                double temp2 = t * e1 - e2 + b[0];
                double rtrx = std::sqrt(rx);
                double cv = c - fpi12;
                double ccv = std::cos(cv);
                double scv = std::sin(cv);
                *ai = (temp1 * ccv - temp2 * scv) / rtrx;

                j = n4d-1;
                f1 = da[j];
                e1 = db[j];
                f2 = 0.;
                e2 = 0.;
                for (int i=1; i<=m4d; ++i) {
                    --j;
                    temp1 = f1;
                    temp2 = e1;
                    f1 = tt * f1 - f2 + da[j];
                    e1 = tt * e1 - e2 + db[j];
                    f2 = temp1;
                    e2 = temp2;
                }
                temp1 = t * f1 - f2 + da[0];
                temp2 = t * e1 - e2 + db[0];
                e1 = ccv * con5 + scv * 0.5;
                e2 = scv * con5 - ccv * 0.5;
                *dai = (temp1 * e1 - temp2 * e2) * rtrx;
            }
        } else {
            if (c > 5.) {
                double t = 10. / c - 1.;
                double tt = t + t;
                int j = n1-1;
                double f1 = ak3[j];
                double f2 = 0.;
                for (int i=1; i<=m1; ++i) {
                    --j;
                    double temp1 = f1;
                    f1 = tt * f1 - f2 + ak3[j];
                    f2 = temp1;
                }
                double rtrx = std::sqrt(rx);
                double ec = std::exp(-c);
                *ai = ec * (t * f1 - f2 + ak3[0]) / rtrx;

                j = n1d-1;
                f1 = dak3[j];
                f2 = 0.;
                for (int i=1; i<=m1d; ++i) {
                    --j;
                    double temp1 = f1;
                    f1 = tt * f1 - f2 + dak3[j];
                    f2 = temp1;
                }
                *dai = -rtrx * ec * (t * f1 - f2 + dak3[0]);
            } else if (x > 1.2) {
                double t = (x + x - con2) * con3;
                double tt = t + t;
                int j = n2-1;
                double f1 = ak2[j];
                double f2 = 0.;
                for (int i=1; i<=m2; ++i) {
                    --j;
                    double temp1 = f1;
                    f1 = tt * f1 - f2 + ak2[j];
                    f2 = temp1;
                }
                double rtrx = std::sqrt(rx);
                double ec = std::exp(-c);
                *ai = ec * (t * f1 - f2 + ak2[0]) / rtrx;

                j = n2d-1;
                f1 = dak2[j];
                f2 = 0.;
                for (int i=1; i<=m2d; ++i) {
                    --j;
                    double temp1 = f1;
                    f1 = tt * f1 - f2 + dak2[j];
                    f2 = temp1;
                }
                *dai = -ec * (t * f1 - f2 + dak2[0]) * rtrx;
            } else {
                double t = (x + x - 1.2) * con4;
                double tt = t + t;
                int j = n1-1;
                double f1 = ak1[j];
                double f2 = 0.;
                for (int i=1; i<=m1; ++i) {
                    --j;
                    double temp1 = f1;
                    f1 = tt * f1 - f2 + ak1[j];
                    f2 = temp1;
                }
                *ai = t * f1 - f2 + ak1[0];

                j = n1d-1;
                f1 = dak1[j];
                f2 = 0.;
                for (int i=1; i<=m1d; ++i) {
                    --j;
                    double temp1 = f1;
                    f1 = tt * f1 - f2 + dak1[j];
                    f2 = temp1;
                }
                *dai = -(t * f1 - f2 + dak1[0]);
            }
        }
    }


    //
    // Y_nu(x)
    //
    double dbesy(double x, double fnu)
    {
        // ***BEGIN PROLOGUE  DBESY
        // ***PURPOSE  Implement forward recursion on the three term recursion
        //            relation for a sequence of non-negative order Bessel
        //            functions Y/SUB(FNU+I-1)/(X), I=1,...,N for real, positive
        //            X and non-negative orders FNU.
        // ***LIBRARY   SLATEC
        // ***CATEGORY  C10A3
        // ***TYPE      DOUBLE PRECISION (BESY-S, DBESY-D)
        // ***KEYWORDS  SPECIAL FUNCTIONS, Y BESSEL FUNCTION
        // ***AUTHOR  Amos, D. E., (SNLA)
        // ***DESCRIPTION
        //
        //     Abstract  **** a double precision routine ****
        //         DBESY implements forward recursion on the three term
        //         recursion relation for a sequence of non-negative order Bessel
        //         functions Y/sub(FNU+I-1)/(X), I=1,N for real X .GT. 0.0D0 and
        //         non-negative orders FNU.  If FNU .LT. NULIM, orders FNU and
        //         FNU+1 are obtained from DBSYNU which computes by a power
        //         series for X .LE. 2, the K Bessel function of an imaginary
        //         argument for 2 .LT. X .LE. 20 and the asymptotic expansion for
        //         X .GT. 20.
        //
        //         If FNU .GE. NULIM, the uniform asymptotic expansion is coded
        //         in DASYJY for orders FNU and FNU+1 to start the recursion.
        //         NULIM is 70 or 100 depending on whether N=1 or N .GE. 2.  An
        //         overflow test is made on the leading term of the asymptotic
        //         expansion before any extensive computation is done.
        //
        //         The maximum number of significant digits obtainable
        //         is the smaller of 14 and the number of digits carried in
        //         double precision arithmetic.
        //
        //     Description of Arguments
        //
        //         Input
        //           X      - X .GT. 0.0D0
        //           FNU    - order of the initial Y function, FNU .GE. 0.0D0
        //           N      - number of members in the sequence, N .GE. 1
        //
        //         Output
        //           Y      - a vector whose first N components contain values
        //                    for the sequence Y(I)=Y/sub(FNU+I-1)/(X), I=1,N.
        //
        //     Error Conditions
        //         Improper input arguments - a fatal error
        //         Overflow - a fatal error
        //
        // ***REFERENCES  F. W. J. Olver, Tables of Bessel Functions of Moderate
        //                 or Large Orders, NPL Mathematical Tables 6, Her
        //                 Majesty's Stationery Office, London, 1962.
        //               N. M. Temme, On the numerical evaluation of the modified
        //                 Bessel function of the third kind, Journal of
        //                 Computational Physics 19, (1975), pp. 324-337.
        //               N. M. Temme, On the numerical evaluation of the ordinary
        //                 Bessel function of the second kind, Journal of
        //                 Computational Physics 21, (1976), pp. 343-350.
        // ***ROUTINES CALLED  D1MACH, DASYJY, DBESY0, DBESY1, DBSYNU, DYAIRY,
        //                    I1MACH, XERMSG
        // ***REVISION HISTORY  (YYMMDD)
        //    800501  DATE WRITTEN
        //    890531  Changed all specific intrinsics to generic.  (WRB)
        //    890911  Removed unnecessary intrinsics.  (WRB)
        //    890911  REVISION DATE from Version 3.2
        //    891214  Prologue converted to Version 4.0 format.  (BAB)
        //    900315  CALLs to XERROR changed to CALLs to XERMSG.  (THJ)
        //    920501  Reformatted the REFERENCES section.  (WRB)
        //    170203  Converted to C++, and modified to only cover n==1 option. (MJ)
        // ***END PROLOGUE  DBESY

        const double xlim = std::numeric_limits<double>::min() * 1.e3;
        const double elim = -std::log(xlim);
        const int nulim[2] = { 70, 100 };

        assert(fnu >= 0.);
        assert(x > 0.);

        if (x < xlim)
            throw std::runtime_error("DBESY OVERFLOW, FNU OR N TOO LARGE OR X TOO SMALL");

        if (fnu == 0.) return dbesy0(x);
        else if (fnu == 1.) return dbesy1(x);
        else if (fnu < 2.) {
            //     OVERFLOW TEST
            if (fnu > 1. && -fnu * (std::log(x) - 0.693) > elim) {
                throw std::runtime_error("DBESY OVERFLOW, FNU OR N TOO LARGE OR X TOO SMALL");
            }
            double s1;
            dbsynu(x, fnu, 1, &s1);
            return s1;
        } else {
            //     OVERFLOW TEST  (LEADING EXPONENTIAL OF ASYMPTOTIC EXPANSION)
            //     FOR THE LAST ORDER, FNU+N-1.GE.NULIM
            int nud = int(fnu);
            double dnu = fnu - nud;
            double xxn = x / fnu;
            double w2n = 1. - xxn * xxn;
            if (w2n > 0.) {
                double ran = std::sqrt(w2n);
                double azn = std::log((ran + 1.) / xxn) - ran;
                if (fnu * azn > elim)
                    throw std::runtime_error("DBESY OVERFLOW, FNU OR N TOO LARGE OR X TOO SMALL");
            }

            if (nud >= nulim[0]) {
                //     ASYMPTOTIC EXPANSION FOR ORDERS FNU AND FNU+1.GE.NULIM
                double wk[7];
                int iflw;
                double s1 = dasyjy(x, fnu, false, wk, &iflw);
                if (iflw != 0) {
                    throw std::runtime_error("DBESY OVERFLOW, FNU OR N TOO LARGE OR X TOO SMALL");
                }
                return s1;
            }

            double s1,s2;
            if (dnu == 0.) {
                s1 = dbesy0(x);
                s2 = dbesy1(x);
            } else {
                double w[2];
                dbsynu(x, dnu, (nud==0 ? 1 : 2), w);
                s1 = w[0];
                s2 = w[1];
            }
            if (nud == 0) return s1;
            double trx = 2. / x;
            double tm = (dnu + dnu + 2.) / x;
            //     FORWARD RECUR FROM DNU TO FNU+1 TO GET Y(1) AND Y(2)
            --nud;
            for (int i=0; i<nud; ++i) {
                double s = s2;
                s2 = tm * s2 - s1;
                s1 = s;
                tm += trx;
            }
            return s2;
        }
    }

    double dbesy0(double x)
    {
        // ***BEGIN PROLOGUE  DBESY0
        // ***PURPOSE  Compute the Bessel function of the second kind of order
        //            zero.
        // ***LIBRARY   SLATEC (FNLIB)
        // ***CATEGORY  C10A1
        // ***TYPE      DOUBLE PRECISION (BESY0-S, DBESY0-D)
        // ***KEYWORDS  BESSEL FUNCTION, FNLIB, ORDER ZERO, SECOND KIND,
        //             SPECIAL FUNCTIONS
        // ***AUTHOR  Fullerton, W., (LANL)
        // ***DESCRIPTION

        // DBESY0(X) calculates the double precision Bessel function of the
        // second kind of order zero for double precision argument X.

        // Series for BY0        on the interval  0.          to  1.60000E+01
        //                                        with weighted error   8.14E-32
        //                                         log weighted error  31.09
        //                               significant figures required  30.31
        //                                    decimal places required  31.73

        // ***REFERENCES  (NONE)
        // ***ROUTINES CALLED  D1MACH, D9B0MP, DBESJ0, DCSEVL, INITDS, XERMSG
        // ***REVISION HISTORY  (YYMMDD)
        //    770701  DATE WRITTEN
        //    890531  Changed all specific intrinsics to generic.  (WRB)
        //    890531  REVISION DATE from Version 3.2
        //    891214  Prologue converted to Version 4.0 format.  (BAB)
        //    900315  CALLs to XERROR changed to CALLs to XERMSG.  (THJ)
        //    170203  Converted to C++. (MJ)
        // ***END PROLOGUE  DBESY0

        const double by0cs[19] = {
            -0.01127783939286557321793980546028,
            -0.1283452375604203460480884531838,
            -0.1043788479979424936581762276618,
            0.02366274918396969540924159264613,
            -0.002090391647700486239196223950342,
            1.039754539390572520999246576381e-4,
            -3.369747162423972096718775345037e-6,
            7.729384267670667158521367216371e-8,
            -1.324976772664259591443476068964e-9,
            1.764823261540452792100389363158e-11,
            -1.881055071580196200602823012069e-13,
            1.641865485366149502792237185749e-15,
            -1.19565943860460608574599100672e-17,
            7.377296297440185842494112426666e-20,
            -3.906843476710437330740906666666e-22,
            1.79550366443615794982912e-24,
            -7.229627125448010478933333333333e-27,
            2.571727931635168597333333333333e-29,
            -8.141268814163694933333333333333e-32
        };
        const double bm0cs[37] = {
            0.09211656246827742712573767730182,
            -0.001050590997271905102480716371755,
            1.470159840768759754056392850952e-5,
            -5.058557606038554223347929327702e-7,
            2.787254538632444176630356137881e-8,
            -2.062363611780914802618841018973e-9,
            1.870214313138879675138172596261e-10,
            -1.969330971135636200241730777825e-11,
            2.325973793999275444012508818052e-12,
            -3.009520344938250272851224734482e-13,
            4.194521333850669181471206768646e-14,
            -6.219449312188445825973267429564e-15,
            9.718260411336068469601765885269e-16,
            -1.588478585701075207366635966937e-16,
            2.700072193671308890086217324458e-17,
            -4.750092365234008992477504786773e-18,
            8.61512816260437087319170374656e-19,
            -1.605608686956144815745602703359e-19,
            3.066513987314482975188539801599e-20,
            -5.987764223193956430696505617066e-21,
            1.192971253748248306489069841066e-21,
            -2.420969142044805489484682581333e-22,
            4.996751760510616453371002879999e-23,
            -1.047493639351158510095040511999e-23,
            2.227786843797468101048183466666e-24,
            -4.801813239398162862370542933333e-25,
            1.047962723470959956476996266666e-25,
            -2.3138581656786153251012608e-26,
            5.164823088462674211635199999999e-27,
            -1.164691191850065389525401599999e-27,
            2.651788486043319282958336e-28,
            -6.092559503825728497691306666666e-29,
            1.411804686144259308038826666666e-29,
            -3.298094961231737245750613333333e-30,
            7.763931143074065031714133333333e-31,
            -1.841031343661458478421333333333e-31,
            4.395880138594310737100799999999e-32
        };
        const double bth0cs[44] = {
            -0.24901780862128936717709793789967,
            4.8550299609623749241048615535485e-4,
            -5.4511837345017204950656273563505e-6,
            1.3558673059405964054377445929903e-7,
            -5.569139890222762622758321841492e-9,
            3.2609031824994335304004205719468e-10,
            -2.4918807862461341125237903877993e-11,
            2.3449377420882520554352413564891e-12,
            -2.6096534444310387762177574766136e-13,
            3.3353140420097395105869955014923e-14,
            -4.7890000440572684646750770557409e-15,
            7.5956178436192215972642568545248e-16,
            -1.3131556016891440382773397487633e-16,
            2.4483618345240857495426820738355e-17,
            -4.8805729810618777683256761918331e-18,
            1.0327285029786316149223756361204e-18,
            -2.3057633815057217157004744527025e-19,
            5.4044443001892693993017108483765e-20,
            -1.3240695194366572724155032882385e-20,
            3.3780795621371970203424792124722e-21,
            -8.9457629157111779003026926292299e-22,
            2.4519906889219317090899908651405e-22,
            -6.9388422876866318680139933157657e-23,
            2.0228278714890138392946303337791e-23,
            -6.0628500002335483105794195371764e-24,
            1.864974896403763538182378839627e-24,
            -5.8783732384849894560245036530867e-25,
            1.8958591447999563485531179503513e-25,
            -6.2481979372258858959291620728565e-26,
            2.1017901684551024686638633529074e-26,
            -7.2084300935209253690813933992446e-27,
            2.5181363892474240867156405976746e-27,
            -8.9518042258785778806143945953643e-28,
            3.2357237479762298533256235868587e-28,
            -1.1883010519855353657047144113796e-28,
            4.4306286907358104820579231941731e-29,
            -1.6761009648834829495792010135681e-29,
            6.4292946921207466972532393966088e-30,
            -2.4992261166978652421207213682763e-30,
            9.8399794299521955672828260355318e-31,
            -3.9220375242408016397989131626158e-31,
            1.5818107030056522138590618845692e-31,
            -6.4525506144890715944344098365426e-32,
            2.6611111369199356137177018346367e-32
        };
        const double bm02cs[40] = {
            0.0950041514522838136933086133556,
            -3.801864682365670991748081566851e-4,
            2.258339301031481192951829927224e-6,
            -3.895725802372228764730621412605e-8,
            1.246886416512081697930990529725e-9,
            -6.065949022102503779803835058387e-11,
            4.008461651421746991015275971045e-12,
            -3.350998183398094218467298794574e-13,
            3.377119716517417367063264341996e-14,
            -3.964585901635012700569356295823e-15,
            5.286111503883857217387939744735e-16,
            -7.852519083450852313654640243493e-17,
            1.280300573386682201011634073449e-17,
            -2.263996296391429776287099244884e-18,
            4.300496929656790388646410290477e-19,
            -8.705749805132587079747535451455e-20,
            1.86586271396209514118144277205e-20,
            -4.210482486093065457345086972301e-21,
            9.956676964228400991581627417842e-22,
            -2.457357442805313359605921478547e-22,
            6.307692160762031568087353707059e-23,
            -1.678773691440740142693331172388e-23,
            4.620259064673904433770878136087e-24,
            -1.311782266860308732237693402496e-24,
            3.834087564116302827747922440276e-25,
            -1.151459324077741271072613293576e-25,
            3.547210007523338523076971345213e-26,
            -1.119218385815004646264355942176e-26,
            3.611879427629837831698404994257e-27,
            -1.190687765913333150092641762463e-27,
            4.005094059403968131802476449536e-28,
            -1.373169422452212390595193916017e-28,
            4.794199088742531585996491526437e-29,
            -1.702965627624109584006994476452e-29,
            6.149512428936330071503575161324e-30,
            -2.255766896581828349944300237242e-30,
            8.3997075092942994860616583532e-31,
            -3.172997595562602355567423936152e-31,
            1.215205298881298554583333026514e-31,
            -4.715852749754438693013210568045e-32
        };
        const double bt02cs[39] = {
            -0.24548295213424597462050467249324,
            0.0012544121039084615780785331778299,
            -3.1253950414871522854973446709571e-5,
            1.4709778249940831164453426969314e-6,
            -9.9543488937950033643468850351158e-8,
            8.5493166733203041247578711397751e-9,
            -8.6989759526554334557985512179192e-10,
            1.0052099533559791084540101082153e-10,
            -1.2828230601708892903483623685544e-11,
            1.7731700781805131705655750451023e-12,
            -2.6174574569485577488636284180925e-13,
            4.0828351389972059621966481221103e-14,
            -6.6751668239742720054606749554261e-15,
            1.1365761393071629448392469549951e-15,
            -2.0051189620647160250559266412117e-16,
            3.6497978794766269635720591464106e-17,
            -6.83096375645823031693558437888e-18,
            1.3107583145670756620057104267946e-18,
            -2.5723363101850607778757130649599e-19,
            5.1521657441863959925267780949333e-20,
            -1.0513017563758802637940741461333e-20,
            2.1820381991194813847301084501333e-21,
            -4.6004701210362160577225905493333e-22,
            9.8407006925466818520953651199999e-23,
            -2.1334038035728375844735986346666e-23,
            4.6831036423973365296066286933333e-24,
            -1.0400213691985747236513382399999e-24,
            2.33491056773015100517777408e-25,
            -5.2956825323318615788049749333333e-26,
            1.2126341952959756829196287999999e-26,
            -2.8018897082289428760275626666666e-27,
            6.5292678987012873342593706666666e-28,
            -1.5337980061873346427835733333333e-28,
            3.6305884306364536682359466666666e-29,
            -8.6560755713629122479172266666666e-30,
            2.0779909972536284571238399999999e-30,
            -5.0211170221417221674325333333333e-31,
            1.2208360279441714184191999999999e-31,
            -2.9860056267039913454250666666666e-32
        };
        const int nty0 = 13;
        const int nbm0 = 15;
        const int nbth0 = 14;
        const int nbm02 = 13;
        const int nbt02 = 16;
        const double pi4 = 0.785398163397448309615660845819876;
        const double twodpi = 0.636619772367581343075535053490057;
        const double xsml = std::sqrt(std::numeric_limits<double>::epsilon() * 4.);
        const double xmax = 0.5/std::numeric_limits<double>::epsilon();

        assert(x>0);

        if (x < 4.) {
            double y = x > xsml ? x*x : 0.;
            return twodpi * std::log(0.5*x) * dbesj0(x) + 0.375 + dcsevl(0.125*y-1., by0cs, nty0);
        } else {
            // MJ: Note, the original code called this branch D9B0MP, but it seems short enough
            // to just include it here.
            double ampl, theta;
            if (x <= 8.) {
                double z = (128. / (x * x) - 5.) / 3.;
                ampl = (dcsevl(z, bm0cs, nbm0) + 0.75) / std::sqrt(x);
                theta = x - pi4 + dcsevl(z, bt02cs, nbt02) / x;
            } else {
                if (x > xmax)
                    throw std::runtime_error("DBESY0 NO PRECISION BECAUSE X IS BIG");
                double z = 128. / (x * x) - 1.;
                ampl = (dcsevl(z, bm02cs, nbm02) + 0.75) / std::sqrt(x);
                theta = x - pi4 + dcsevl(z, bth0cs, nbth0) / x;
            }
            return ampl * std::sin(theta);
        }
    }

    double dbesy1(double x)
    {
        // ***BEGIN PROLOGUE  DBESY1
        // ***PURPOSE  Compute the Bessel function of the second kind of order
        //            one.
        // ***LIBRARY   SLATEC (FNLIB)
        // ***CATEGORY  C10A1
        // ***TYPE      DOUBLE PRECISION (BESY1-S, DBESY1-D)
        // ***KEYWORDS  BESSEL FUNCTION, FNLIB, ORDER ONE, SECOND KIND,
        //             SPECIAL FUNCTIONS
        // ***AUTHOR  Fullerton, W., (LANL)
        // ***DESCRIPTION

        // DBESY1(X) calculates the double precision Bessel function of the
        // second kind of order for double precision argument X.

        // Series for BY1        on the interval  0.          to  1.60000E+01
        //                                        with weighted error   8.65E-33
        //                                         log weighted error  32.06
        //                               significant figures required  32.17
        //                                    decimal places required  32.71

        // ***REFERENCES  (NONE)
        // ***ROUTINES CALLED  D1MACH, D9B1MP, DBESJ1, DCSEVL, INITDS, XERMSG
        // ***REVISION HISTORY  (YYMMDD)
        //    770701  DATE WRITTEN
        //    890531  Changed all specific intrinsics to generic.  (WRB)
        //    890531  REVISION DATE from Version 3.2
        //    891214  Prologue converted to Version 4.0 format.  (BAB)
        //    900315  CALLs to XERROR changed to CALLs to XERMSG.  (THJ)
        //    170203  Converted to C++. (MJ)
        // ***END PROLOGUE  DBESY1

        const double by1cs[20] = {
            0.0320804710061190862932352018628015,
            1.26270789743350044953431725999727,
            0.00649996189992317500097490637314144,
            -0.0893616452886050411653144160009712,
            0.0132508812217570954512375510370043,
            -8.97905911964835237753039508298105e-4,
            3.64736148795830678242287368165349e-5,
            -1.00137438166600055549075523845295e-6,
            1.99453965739017397031159372421243e-8,
            -3.02306560180338167284799332520743e-10,
            3.60987815694781196116252914242474e-12,
            -3.48748829728758242414552947409066e-14,
            2.78387897155917665813507698517333e-16,
            -1.86787096861948768766825352533333e-18,
            1.06853153391168259757070336e-20,
            -5.27472195668448228943872e-23,
            2.27019940315566414370133333333333e-25,
            -8.59539035394523108693333333333333e-28,
            2.88540437983379456e-30,
            -8.64754113893717333333333333333333e-33
        };
        const double bm1cs[37] = {
            0.1069845452618063014969985308538,
            0.003274915039715964900729055143445,
            -2.987783266831698592030445777938e-5,
            8.331237177991974531393222669023e-7,
            -4.112665690302007304896381725498e-8,
            2.855344228789215220719757663161e-9,
            -2.485408305415623878060026596055e-10,
            2.543393338072582442742484397174e-11,
            -2.941045772822967523489750827909e-12,
            3.743392025493903309265056153626e-13,
            -5.149118293821167218720548243527e-14,
            7.552535949865143908034040764199e-15,
            -1.169409706828846444166290622464e-15,
            1.89656244943479157172182460506e-16,
            -3.201955368693286420664775316394e-17,
            5.599548399316204114484169905493e-18,
            -1.010215894730432443119390444544e-18,
            1.873844985727562983302042719573e-19,
            -3.563537470328580219274301439999e-20,
            6.931283819971238330422763519999e-21,
            -1.376059453406500152251408930133e-21,
            2.783430784107080220599779327999e-22,
            -5.727595364320561689348669439999e-23,
            1.197361445918892672535756799999e-23,
            -2.539928509891871976641440426666e-24,
            5.461378289657295973069619199999e-25,
            -1.189211341773320288986289493333e-25,
            2.620150977340081594957824e-26,
            -5.836810774255685901920938666666e-27,
            1.313743500080595773423615999999e-27,
            -2.985814622510380355332778666666e-28,
            6.848390471334604937625599999999e-29,
            -1.58440156822247672119296e-29,
            3.695641006570938054301013333333e-30,
            -8.687115921144668243012266666666e-31,
            2.057080846158763462929066666666e-31,
            -4.905225761116225518523733333333e-32
        };
        const double bt12cs[39] = {
            0.73823860128742974662620839792764,
            -0.0033361113174483906384470147681189,
            6.1463454888046964698514899420186e-5,
            -2.4024585161602374264977635469568e-6,
            1.4663555577509746153210591997204e-7,
            -1.1841917305589180567005147504983e-8,
            1.1574198963919197052125466303055e-9,
            -1.3001161129439187449366007794571e-10,
            1.6245391141361731937742166273667e-11,
            -2.2089636821403188752155441770128e-12,
            3.2180304258553177090474358653778e-13,
            -4.9653147932768480785552021135381e-14,
            8.0438900432847825985558882639317e-15,
            -1.3589121310161291384694712682282e-15,
            2.3810504397147214869676529605973e-16,
            -4.3081466363849106724471241420799e-17,
            8.02025440327710024349935125504e-18,
            -1.5316310642462311864230027468799e-18,
            2.9928606352715568924073040554666e-19,
            -5.9709964658085443393815636650666e-20,
            1.2140289669415185024160852650666e-20,
            -2.5115114696612948901006977706666e-21,
            5.2790567170328744850738380799999e-22,
            -1.1260509227550498324361161386666e-22,
            2.43482773595763266596634624e-23,
            -5.3317261236931800130038442666666e-24,
            1.1813615059707121039205990399999e-24,
            -2.6465368283353523514856789333333e-25,
            5.9903394041361503945577813333333e-26,
            -1.3690854630829503109136383999999e-26,
            3.1576790154380228326413653333333e-27,
            -7.3457915082084356491400533333333e-28,
            1.722808148072274793070592e-28,
            -4.07169079612865079410688e-29,
            9.6934745136779622700373333333333e-30,
            -2.3237636337765716765354666666666e-30,
            5.6074510673522029406890666666666e-31,
            -1.3616465391539005860522666666666e-31,
            3.3263109233894654388906666666666e-32
        };
        const double bm12cs[40] = {
            0.09807979156233050027272093546937,
            0.001150961189504685306175483484602,
            -4.312482164338205409889358097732e-6,
            5.951839610088816307813029801832e-8,
            -1.704844019826909857400701586478e-9,
            7.798265413611109508658173827401e-11,
            -4.958986126766415809491754951865e-12,
            4.038432416421141516838202265144e-13,
            -3.993046163725175445765483846645e-14,
            4.619886183118966494313342432775e-15,
            -6.089208019095383301345472619333e-16,
            8.960930916433876482157048041249e-17,
            -1.449629423942023122916518918925e-17,
            2.546463158537776056165149648068e-18,
            -4.80947287464783644425926371862e-19,
            9.687684668292599049087275839124e-20,
            -2.067213372277966023245038117551e-20,
            4.64665155915038473180276780959e-21,
            -1.094966128848334138241351328339e-21,
            2.693892797288682860905707612785e-22,
            -6.894992910930374477818970026857e-23,
            1.83026826275206290989066855474e-23,
            -5.025064246351916428156113553224e-24,
            1.423545194454806039631693634194e-24,
            -4.152191203616450388068886769801e-25,
            1.244609201503979325882330076547e-25,
            -3.827336370569304299431918661286e-26,
            1.205591357815617535374723981835e-26,
            -3.884536246376488076431859361124e-27,
            1.278689528720409721904895283461e-27,
            -4.295146689447946272061936915912e-28,
            1.470689117829070886456802707983e-28,
            -5.128315665106073128180374017796e-29,
            1.819509585471169385481437373286e-29,
            -6.563031314841980867618635050373e-30,
            2.404898976919960653198914875834e-30,
            -8.945966744690612473234958242979e-31,
            3.37608516065723102663714897824e-31,
            -1.291791454620656360913099916966e-31,
            5.008634462958810520684951501254e-32
        };
        const double bth1cs[44] = {
            0.74749957203587276055443483969695,
            -0.0012400777144651711252545777541384,
            9.9252442404424527376641497689592e-6,
            -2.0303690737159711052419375375608e-7,
            7.5359617705690885712184017583629e-9,
            -4.1661612715343550107630023856228e-10,
            3.0701618070834890481245102091216e-11,
            -2.8178499637605213992324008883924e-12,
            3.0790696739040295476028146821647e-13,
            -3.8803300262803434112787347554781e-14,
            5.5096039608630904934561726208562e-15,
            -8.6590060768383779940103398953994e-16,
            1.4856049141536749003423689060683e-16,
            -2.7519529815904085805371212125009e-17,
            5.4550796090481089625036223640923e-18,
            -1.1486534501983642749543631027177e-18,
            2.5535213377973900223199052533522e-19,
            -5.9621490197413450395768287907849e-20,
            1.4556622902372718620288302005833e-20,
            -3.7022185422450538201579776019593e-21,
            9.7763074125345357664168434517924e-22,
            -2.6726821639668488468723775393052e-22,
            7.5453300384983271794038190655764e-23,
            -2.1947899919802744897892383371647e-23,
            6.5648394623955262178906999817493e-24,
            -2.0155604298370207570784076869519e-24,
            6.341776855677614349214466718567e-25,
            -2.0419277885337895634813769955591e-25,
            6.7191464220720567486658980018551e-26,
            -2.2569079110207573595709003687336e-26,
            7.7297719892989706370926959871929e-27,
            -2.696744451229464091321142408092e-27,
            9.5749344518502698072295521933627e-28,
            -3.4569168448890113000175680827627e-28,
            1.2681234817398436504211986238374e-28,
            -4.7232536630722639860464993713445e-29,
            1.7850008478186376177858619796417e-29,
            -6.8404361004510395406215223566746e-30,
            2.6566028671720419358293422672212e-30,
            -1.045040252791445291771416148467e-30,
            4.1618290825377144306861917197064e-31,
            -1.6771639203643714856501347882887e-31,
            6.8361997776664389173535928028528e-32,
            -2.817224786123364116673957462281e-32
        };
        const int nty1 = 13;
        const int nbm1 = 15;
        const int nbt12 = 17;
        const int nbm12 = 13;
        const int nbth1 = 14;
        const double xmin = 1.01 * 1.571 * std::numeric_limits<double>::min();
        const double xsml = 2. * std::sqrt(std::numeric_limits<double>::epsilon());
        const double xmax = 0.5/std::numeric_limits<double>::epsilon();
        const double twodpi = 0.636619772367581343075535053490057;
        const double pi4 = 0.785398163397448309615660845819876;

        assert(x > 0);

        if (x <= 4.) {
            if (x < xmin)
                throw std::runtime_error("DBESY1 X SO SMALL Y1 OVERFLOWS");
            double y = (x > xsml) ? x*x : 0.;
            double z = 0.125*y - 1.;
            return twodpi * std::log(0.5*x) * dbesj1(x) + (dcsevl(z, by1cs, nty1) + 0.5) / x;
        } else {
            double ampl, theta;
            if (x <= 8.) {
                double z = (128. / (x * x) - 5.) / 3.;
                ampl = (dcsevl(z, bm1cs, nbm1) + 0.75) / std::sqrt(x);
                theta = x - pi4 * 3. + dcsevl(z, bt12cs, nbt12) / x;
            } else {
                if (x > xmax)
                    throw std::runtime_error("DBESY1 No precision because X is too big");
                double z = 128. / (x * x) - 1.;
                ampl = (dcsevl(z, bm12cs, nbm12) + 0.75) / std::sqrt(x);
                theta = x - pi4 * 3. + dcsevl(z, bth1cs, nbth1) / x;
            }
            return ampl * std::sin(theta);
        }
    }

    void dbsynu(double x, double fnu, int n, double *y)
    {
        // ***BEGIN PROLOGUE  DBSYNU
        // ***SUBSIDIARY
        // ***PURPOSE  Subsidiary to DBESY
        // ***LIBRARY   SLATEC
        // ***TYPE      DOUBLE PRECISION (BESYNU-S, DBSYNU-D)
        // ***AUTHOR  Amos, D. E., (SNLA)
        // ***DESCRIPTION
        //
        //     Abstract  **** A DOUBLE PRECISION routine ****
        //         DBSYNU computes N member sequences of Y Bessel functions
        //         Y/SUB(FNU+I-1)/(X), I=1,N for non-negative orders FNU and
        //         positive X. Equations of the references are implemented on
        //         small orders DNU for Y/SUB(DNU)/(X) and Y/SUB(DNU+1)/(X).
        //         Forward recursion with the three term recursion relation
        //         generates higher orders FNU+I-1, I=1,...,N.
        //
        //         To start the recursion FNU is normalized to the interval
        //         -0.5.LE.DNU.LT.0.5. A special form of the power series is
        //         implemented on 0.LT.X.LE.X1 while the Miller algorithm for the
        //         K Bessel function in terms of the confluent hypergeometric
        //         function U(FNU+0.5,2*FNU+1,I*X) is implemented on X1.LT.X.LE.X
        //         Here I is the complex number SQRT(-1.).
        //         For X.GT.X2, the asymptotic expansion for large X is used.
        //         When FNU is a half odd integer, a special formula for
        //         DNU=-0.5 and DNU+1.0=0.5 is used to start the recursion.
        //
        //         The maximum number of significant digits obtainable
        //         is the smaller of 14 and the number of digits carried in
        //         DOUBLE PRECISION arithmetic.
        //
        //         DBSYNU assumes that a significant digit SINH function is
        //         available.
        //
        //     Description of Arguments
        //
        //         INPUT
        //           X      - X.GT.0.0D0
        //           FNU    - Order of initial Y function, FNU.GE.0.0D0
        //           N      - Number of members of the sequence, N.GE.1
        //
        //         OUTPUT
        //           Y      - A vector whose first N components contain values
        //                    for the sequence Y(I)=Y/SUB(FNU+I-1), I=1,N.
        //
        //     Error Conditions
        //         Improper input arguments - a fatal error
        //         Overflow - a fatal error
        //
        // ***SEE ALSO  DBESY
        // ***REFERENCES  N. M. Temme, On the numerical evaluation of the ordinary
        //                 Bessel function of the second kind, Journal of
        //                 Computational Physics 21, (1976), pp. 343-350.
        //               N. M. Temme, On the numerical evaluation of the modified
        //                 Bessel function of the third kind, Journal of
        //                 Computational Physics 19, (1975), pp. 324-337.
        // ***ROUTINES CALLED  D1MACH, DGAMMA, XERMSG
        // ***REVISION HISTORY  (YYMMDD)
        //    800501  DATE WRITTEN
        //    890531  Changed all specific intrinsics to generic.  (WRB)
        //    890911  Removed unnecessary intrinsics.  (WRB)
        //    891214  Prologue converted to Version 4.0 format.  (BAB)
        //    900315  CALLs to XERROR changed to CALLs to XERMSG.  (THJ)
        //    900326  Removed duplicate information from DESCRIPTION section.
        //            (WRB)
        //    900328  Added TYPE section.  (WRB)
        //    900727  Added EXTERNAL statement.  (WRB)
        //    910408  Updated the AUTHOR and REFERENCES sections.  (WRB)
        //    920501  Reformatted the REFERENCES section.  (WRB)
        //    170203  Converted to C++. (MJ)
        // ***END PROLOGUE  DBSYNU

        const double x1 = 3.;
        const double x2 = 20.;
        const double pi = 3.14159265358979;
        const double rthpi = 0.797884560802865;
        const double hpi = 1.5707963267949;
        const double cc[8] = {
            0.577215664901533, -0.0420026350340952,
            -0.0421977345555443, 0.007218943246663, -2.152416741149e-4,
            -2.01348547807e-5, 1.133027232e-6, 6.116095e-9
        };

        const double tol = std::max(std::numeric_limits<double>::epsilon(), 1e-15);

        assert(x > 0);
        assert(fnu >= 0);
        assert(n >= 1);

        double rx = 2. / x;
        int inu = int(fnu + 0.5);
        double dnu = fnu - inu;
        double s1, s2;
        if (std::abs(dnu) == 0.5) {
            //     FNU=HALF ODD INTEGER CASE
            double coef = rthpi / std::sqrt(x);
            s1 = coef * std::sin(x);
            s2 = -coef * std::cos(x);
        } else {
            double dnu2 = (std::abs(dnu) >= tol) ? dnu * dnu : 0.;
            if (x <= x1) {
                //     SERIES FOR X.LE.X1
                double a1 = 1. - dnu;
                double a2 = dnu + 1.;
                double t1 = 1. / std::tgamma(a1);
                double t2 = 1. / std::tgamma(a2);
                double g1;
                if (std::abs(dnu) <= 0.1) {
                    //     SERIES FOR F0 TO RESOLVE INDETERMINACY FOR SMALL ABS(DNU)
                    double s = cc[0];
                    double ak = 1.;
                    for (int k = 1; k < 8; ++k) {
                        ak *= dnu2;
                        double tm = cc[k] * ak;
                        s += tm;
                        if (std::abs(tm) < tol) break;
                    }
                    g1 = -(s + s);
                } else {
                    g1 = (t1 - t2) / dnu;
                }
                double g2 = t1 + t2;
                double smu = 1.;
                double fc = 1. / pi;
                double flrx = std::log(rx);
                double fmu = dnu * flrx;
                double tm = 0.;
                if (dnu != 0.) {
                    tm = std::sin(dnu * hpi) / dnu;
                    tm = (dnu + dnu) * tm * tm;
                    fc = dnu / std::sin(dnu * pi);
                    if (fmu != 0.) {
                        smu = std::sinh(fmu) / fmu;
                    }
                }
                double f = fc * (g1 * std::cosh(fmu) + g2 * flrx * smu);
                double fx = std::exp(fmu);
                double p = fc * t1 * fx;
                double q = fc * t2 / fx;
                double g = f + tm * q;
                double ak = 1.;
                double ck = 1.;
                double bk = 1.;
                s1 = g;
                s2 = p;
                if (inu == 0 && n == 1) {
                    if (x < tol) {
                        y[0] = -s1;
                        return;
                    }
                    double cx = x * x * 0.25;
                    double s;
                    do {
                        f = (ak * f + p + q) / (bk - dnu2);
                        p /= ak - dnu;
                        q /= ak + dnu;
                        g = f + tm * q;
                        ck = -ck * cx / ak;
                        t1 = ck * g;
                        s1 += t1;
                        bk = bk + ak + ak + 1.;
                        ak += 1.;
                        s = std::abs(t1) / (std::abs(s1) + 1.);
                    } while (s > tol);
                }
                if (x >= tol) {
                    double cx = x * x * 0.25;
                    double s;
                    do {
                        f = (ak * f + p + q) / (bk - dnu2);
                        p /= ak - dnu;
                        q /= ak + dnu;
                        g = f + tm * q;
                        ck = -ck * cx / ak;
                        t1 = ck * g;
                        s1 += t1;
                        t2 = ck * (p - ak * g);
                        s2 += t2;
                        bk = bk + ak + ak + 1.;
                        ak += 1.;
                        s = std::abs(t1) / (std::abs(s1)+1.) + std::abs(t2) / (std::abs(s2)+1.);
                    } while (s > tol);
                }
                s2 = -s2 * rx;
                s1 = -s1;
            } else {
                double coef = rthpi / std::sqrt(x);
                if (x <= x2) {
                    //     MILLER ALGORITHM FOR X1.LT.X.LE.X2
                    double etest = std::cos(pi * dnu) / (pi * x * tol);
                    double fks = 1.;
                    double fhs = 0.25;
                    double fk = 0.;
                    double rck = 2.;
                    double cck = x + x;
                    double rp1 = 0.;
                    double cp1 = 0.;
                    double rp2 = 1.;
                    double cp2 = 0.;
                    double a[120];
                    double cb[120];
                    double rb[120];
                    int k = 0;
                    double pt;
                    do {
                        fk += 1.;
                        double ak = (fhs - dnu2) / (fks + fk);
                        pt = fk + 1.;
                        double rbk = rck / pt;
                        double cbk = cck / pt;
                        double rpt = rp2;
                        double cpt = cp2;
                        rp2 = rbk * rpt - cbk * cpt - ak * rp1;
                        cp2 = cbk * rpt + rbk * cpt - ak * cp1;
                        rp1 = rpt;
                        cp1 = cpt;
                        rb[k] = rbk;
                        cb[k] = cbk;
                        a[k] = ak;
                        rck += 2.;
                        fks = fks + fk + fk + 1.;
                        fhs = fhs + fk + fk;
                        pt = std::max(std::abs(rp1), std::abs(cp1));
                        double fc = (rp1*rp1 + cp1*cp1) / (pt*pt);
                        pt = pt * std::sqrt(fc) * fk;
                        ++k;
                    } while (etest > pt);

                    double rs = 1.;
                    double cs = 0.;
                    rp1 = 0.;
                    cp1 = 0.;
                    rp2 = 1.;
                    cp2 = 0.;
                    for (int kk=k-1; kk>=0; --kk) {
                        double rpt = rp2;
                        double cpt = cp2;
                        rp2 = (rb[kk] * rpt - cb[kk] * cpt - rp1) / a[kk];
                        cp2 = (cb[kk] * rpt + rb[kk] * cpt - cp1) / a[kk];
                        rp1 = rpt;
                        cp1 = cpt;
                        rs += rp2;
                        cs += cp2;
                    }
                    pt = std::max(std::abs(rs), std::abs(cs));
                    double fc = (rs*rs + cs*cs) / (pt*pt);
                    pt *= std::sqrt(fc);
                    double rs1 = (rp2 * (rs / pt) + cp2 * (cs / pt)) / pt;
                    double cs1 = (cp2 * (rs / pt) - rp2 * (cs / pt)) / pt;
                    fc = hpi * (dnu - 0.5) - x;
                    double p = std::cos(fc);
                    double q = std::sin(fc);
                    s1 = (cs1 * q - rs1 * p) * coef;
                    if (inu == 0 && n == 1) {
                        y[0] = s1;
                        return;
                    }

                    pt = std::max(std::abs(rp2), std::abs(cp2));
                    fc = (rp2*rp2 + cp2*cp2) / (pt*pt);
                    pt *= std::sqrt(fc);
                    double rpt = dnu + 0.5 - (rp1 * (rp2 / pt) + cp1 * (cp2 / pt)) / pt;
                    double cpt = x - (cp1 * (rp2 / pt) - rp1 * (cp2 / pt)) / pt;
                    double cs2 = cs1 * cpt - rs1 * rpt;
                    double rs2 = rpt * cs1 + rs1 * cpt;
                    s2 = (rs2 * q + cs2 * p) * coef / x;
                } else {
                    //     ASYMPTOTIC EXPANSION FOR LARGE X, X.GT.X2
                    int nn = (inu == 0 && n == 1) ? 1 : 2;
                    double dnu2 = dnu + dnu;
                    double fmu = (std::abs(dnu2) >= tol) ? dnu2 * dnu2 : 0.;
                    double arg = x - hpi * (dnu + 0.5);
                    double sa = std::sin(arg);
                    double sb = std::cos(arg);
                    double etx = x * 8.;
                    for (int k=1; k<=nn; ++k) {
                        s1 = s2;
                        double t2 = (fmu - 1.) / etx;
                        double ss = t2;
                        double relb = tol * std::abs(t2);
                        double t1 = etx;
                        double s = 1.;
                        double fn = 1.;
                        double ak = 0.;
                        for (int j = 1; j <= 13; ++j) {
                            t1 += etx;
                            ak += 8.;
                            fn += ak;
                            t2 = -t2 * (fmu - fn) / t1;
                            s += t2;
                            t1 += etx;
                            ak += 8.;
                            fn += ak;
                            t2 = t2 * (fmu - fn) / t1;
                            ss += t2;
                            if (std::abs(t2) <= relb) break;
                        }
                        s2 = coef * (s * sa + ss * sb);
                        fmu = fmu + dnu * 8. + 4.;
                        double tb = sa;
                        sa = -sb;
                        sb = tb;
                    }
                    if (nn == 1) {
                        y[0] = s2;
                        return;
                    }
                }
            }
        }

        //     FORWARD RECURSION ON THE THREE TERM RECURSION RELATION

        double ck = (dnu + dnu + 2.) / x;
        if (n == 1) --inu;
        if (inu > 0) {
            for (int i=1; i<=inu; ++i) {
                double st = s2;
                s2 = ck * s2 - s1;
                s1 = st;
                ck += rx;
            }
        }
        if (n == 1) s1 = s2;
        y[0] = s1;
        if (n == 1) return;
        y[1] = s2;
        if (n == 2) return;
        for (int i=2; i<n; ++i) {
            y[i] = ck * y[i-1] - y[i-2];
            ck += rx;
        }
    }

    void dyairy(double x, double rx, double c, double *bi, double *dbi)
    {
        // ***BEGIN PROLOGUE  DYAIRY
        // ***SUBSIDIARY
        // ***PURPOSE  Subsidiary to DBESJ and DBESY
        // ***LIBRARY   SLATEC
        // ***TYPE      DOUBLE PRECISION (YAIRY-S, DYAIRY-D)
        // ***AUTHOR  Amos, D. E., (SNLA)
        //           Daniel, S. L., (SNLA)
        // ***DESCRIPTION
        //
        //                  DYAIRY computes the Airy function BI(X)
        //                   and its derivative DBI(X) for DASYJY
        //
        //                                     INPUT
        //
        //         X  - Argument, computed by DASYJY, X unrestricted
        //        RX  - RX=SQRT(ABS(X)), computed by DASYJY
        //         C  - C=2.*(ABS(X)**1.5)/3., computed by DASYJY
        //
        //                                    OUTPUT
        //        BI  - Value of function BI(X)
        //       DBI  - Value of the derivative DBI(X)
        //
        // ***SEE ALSO  DBESJ, DBESY
        // ***ROUTINES CALLED  (NONE)
        // ***REVISION HISTORY  (YYMMDD)
        //    750101  DATE WRITTEN
        //    890531  Changed all specific intrinsics to generic.  (WRB)
        //    891214  Prologue converted to Version 4.0 format.  (BAB)
        //    900328  Added TYPE section.  (WRB)
        //    910408  Updated the AUTHOR section.  (WRB)
        //    170203  Converted to C++. (MJ)
        // ***END PROLOGUE  DYAIRY

        const double fpi12 = 1.30899693899575;
        const double spi12 = 1.83259571459405;
        const double con1 = 0.666666666666667;
        const double con2 = 7.74148278841779;
        const double con3 = 0.364766105490356;
        const double bk1[20] = {
            2.43202846447449, 2.57132009754685,
            1.02802341258616, 0.341958178205872, 0.0841978629889284,
            0.0193877282587962, 0.00392687837130335, 6.83302689948043e-4,
            1.14611403991141e-4, 1.74195138337086e-5, 2.41223620956355e-6,
            3.24525591983273e-7, 4.03509798540183e-8, 4.70875059642296e-9,
            5.35367432585889e-10, 5.70606721846334e-11, 5.80526363709933e-12,
            5.76338988616388e-13, 5.42103834518071e-14, 4.91857330301677e-15
        };
        const double bk2[20] = {
            0.574830555784088, -0.00691648648376891,
            0.00197460263052093, -5.24043043868823e-4, 1.22965147239661e-4,
            -2.27059514462173e-5, 2.23575555008526e-6, 4.15174955023899e-7,
            -2.84985752198231e-7, 8.50187174775435e-8, -1.70400826891326e-8,
            2.25479746746889e-9, -1.09524166577443e-10, -3.41063845099711e-11,
            1.11262893886662e-11, -1.75542944241734e-12, 1.36298600401767e-13,
            8.76342105755664e-15, -4.64063099157041e-15, 7.7877275873296e-16
        };
        const double bk3[20] = {
            0.566777053506912, 0.00263672828349579,
            5.1230335147313e-5, 2.10229231564492e-6, 1.4221709511389e-7,
            1.28534295891264e-8,7.28556219407507e-10, -3.45236157301011e-10,
            -2.11919115912724e-10, -6.56803892922376e-11, -8.14873160315074e-12,
            3.03177845632183e-12, 1.73447220554115e-12, 1.67935548701554e-13,
            -1.49622868806719e-13, -5.15470458953407e-14, 8.7574184185783e-15,
            7.9673555352572e-15, -1.29566137861742e-16, -1.1187879441752e-15
        };
        const double bk4[14] = {
            0.485444386705114, -0.00308525088408463,
            6.98748404837928e-5, -2.82757234179768e-6, 1.59553313064138e-7,
            -1.12980692144601e-8, 9.47671515498754e-10, -9.08301736026423e-11,
            9.70776206450724e-12, -1.13687527254574e-12, 1.43982917533415e-13,
            -1.95211019558815e-14, 2.81056379909357e-15, -4.26916444775176e-16
        };
        const double bjp[19] = {
            0.134918611457638, -0.319314588205813,
            0.0522061946276114, 0.0528869112170312, -0.0085810075607735,
            -0.00299211002025555, 4.21126741969759e-4, 8.73931830369273e-5,
            -1.06749163477533e-5, -1.56575097259349e-6, 1.68051151983999e-7,
            1.89901103638691e-8, -1.81374004961922e-9, -1.66339134593739e-10,
            1.4295633578081e-11, 1.10179811626595e-12, -8.60187724192263e-14,
            -5.71248177285064e-15, 4.08414552853803e-16
        };
        const double bjn[19] = {
            0.0659041673525697, -0.424905910566004,
            0.28720974519583, 0.129787771099606, -0.0456354317590358,
            -0.010263017598254, 0.00250704671521101, 3.78127183743483e-4,
            -7.11287583284084e-5, -8.08651210688923e-6, 1.23879531273285e-6,
            1.13096815867279e-7, -1.4623428317631e-8, -1.11576315688077e-9,
            1.24846618243897e-10, 8.18334132555274e-12, -8.07174877048484e-13,
            -4.63778618766425e-14, 4.09043399081631e-15
        };
        const double aa[14] = {
            -0.278593552803079, 0.00352915691882584,
            2.31149677384994e-5, -4.7131784226356e-6, 1.12415907931333e-7,
            2.00100301184339e-8, -2.60948075302193e-9, 3.55098136101216e-11,
            3.50849978423875e-11, -5.83007187954202e-12, 2.04644828753326e-13,
            1.10529179476742e-13, -2.87724778038775e-14, 2.88205111009939e-15
        };
        const double bb[14] = {
            -0.490275424742791, -0.00157647277946204,
            9.66195963140306e-5, -1.35916080268815e-7, -2.98157342654859e-7,
            1.86824767559979e-8, 1.03685737667141e-9, -3.28660818434328e-10,
            2.5709141063278e-11, 2.32357655300677e-12, -9.57523279048255e-13,
            1.20340828049719e-13, 2.90907716770715e-15, -4.55656454580149e-15
        };
        const double dbk1[21] = {
            2.95926143981893, 3.86774568440103,
            1.80441072356289, 0.578070764125328, 0.163011468174708,
            0.0392044409961855, 0.00790964210433812, 0.00150640863167338,
            2.56651976920042e-4, 3.93826605867715e-5, 5.81097771463818e-6,
            7.86881233754659e-7, 9.93272957325739e-8, 1.21424205575107e-8,
            1.38528332697707e-9, 1.50190067586758e-10, 1.58271945457594e-11,
            1.57531847699042e-12, 1.50774055398181e-13, 1.40594335806564e-14,
            1.24942698777218e-15
        };
        const double dbk2[20] = {
            0.549756809432471, 0.00913556983276901,
            -0.00253635048605507, 6.60423795342054e-4, -1.55217243135416e-4,
            3.00090325448633e-5, -3.76454339467348e-6, -1.33291331611616e-7,
            2.42587371049013e-7, -8.07861075240228e-8, 1.71092818861193e-8,
            -2.41087357570599e-9, 1.53910848162371e-10, 2.5646537319063e-11,
            -9.88581911653212e-12, 1.60877986412631e-12, -1.20952524741739e-13,
            -1.0697827841082e-14, 5.02478557067561e-15, -8.68986130935886e-16
        };
        const double dbk3[20] = {
            0.560598509354302, -0.00364870013248135,
            -5.98147152307417e-5, -2.33611595253625e-6, -1.64571516521436e-7,
            -2.06333012920569e-8, -4.2774543157311e-9, -1.08494137799276e-9,
            -2.37207188872763e-10, -2.22132920864966e-11, 1.07238008032138e-11,
            5.71954845245808e-12, 7.51102737777835e-13, -3.81912369483793e-13,
            -1.75870057119257e-13, 6.69641694419084e-15, 2.26866724792055e-14,
            2.69898141356743e-15, -2.67133612397359e-15, -6.54121403165269e-16
        }
        ;
        const double dbk4[14] = {
            0.493072999188036, 0.00438335419803815,
            -8.37413882246205e-5, 3.20268810484632e-6, -1.7566197954827e-7,
            1.22269906524508e-8, -1.01381314366052e-9, 9.63639784237475e-11,
            -1.02344993379648e-11, 1.19264576554355e-12, -1.50443899103287e-13,
            2.03299052379349e-14, -2.91890652008292e-15, 4.42322081975475e-16
        };
        const double dbjp[19] = {
            0.113140872390745, -0.208301511416328,
            0.0169396341953138, 0.0290895212478621, -0.00341467131311549,
            -0.00146455339197417, 1.63313272898517e-4, 3.91145328922162e-5,
            -3.96757190808119e-6, -6.51846913772395e-7, 5.9870749526928e-8,
            7.44108654536549e-9, -6.21241056522632e-10, -6.18768017313526e-11,
            4.72323484752324e-12, 3.91652459802532e-13, -2.74985937845226e-14,
            -1.9503649776275e-15, 1.26669643809444e-16
        };
        const double dbjn[19] = {
            -0.018809126006885, -0.14779818082614,
            0.546075900433171, 0.152146932663116, -0.0958260412266886,
            -0.016310273169613, 0.00575364806680105, 7.12145408252655e-4,
            -1.75452116846724e-4, -1.71063171685128e-5, 3.2443558063168e-6,
            2.61190663932884e-7, -4.03026865912779e-8, -2.76435165853895e-9,
            3.59687929062312e-10, 2.14953308456051e-11, -2.41849311903901e-12,
            -1.28068004920751e-13, 1.26939834401773e-14
        };
        const double daa[14] = {
            0.277571356944231, -0.0044421283341992,
            8.42328522190089e-5, 2.5804031841871e-6, -3.42389720217621e-7,
            6.24286894709776e-9, 2.36377836844577e-9, -3.16991042656673e-10,
            4.40995691658191e-12, 5.18674221093575e-12, -9.64874015137022e-13,
            4.9019057660871e-14, 1.77253430678112e-14, -5.55950610442662e-15
        };
        const double dbb[14] = {
            0.491627321104601, 0.00311164930427489,
            8.23140762854081e-5, -4.61769776172142e-6, -6.13158880534626e-8,
            2.8729580465652e-8, -1.81959715372117e-9, -1.44752826642035e-10,
            4.53724043420422e-11, -3.99655065847223e-12, -3.24089119830323e-13,
            1.62098952568741e-13, -2.40765247974057e-14, 1.69384811284491e-16
        };
        const int n1 = 20;
        const int n2 = 19;
        const int n3 = 14;
        const int n1d = 21;
        const int n2d = 20;
        const int n3d = 19;
        const int n4d = 14;
        const int m1 = 18;
        const int m2 = 17;
        const int m3 = 12;
        const int m1d = 19;
        const int m2d = 18;
        const int m3d = 17;
        const int m4d = 12;

        double ax = std::abs(x);
        rx = std::sqrt(ax);
        c = con1 * ax * rx;
        if (x < 0.) {
            if (c <= 5.) {
                double t = c * 0.4 - 1.;
                double tt = t + t;
                int j = n2;
                double f1 = bjp[j - 1];
                double e1 = bjn[j - 1];
                double f2 = 0.;
                double e2 = 0.;
                for (int i=1; i<=m2; ++i) {
                    --j;
                    double temp1 = f1;
                    double temp2 = e1;
                    f1 = tt * f1 - f2 + bjp[j - 1];
                    e1 = tt * e1 - e2 + bjn[j - 1];
                    f2 = temp1;
                    e2 = temp2;
                }
                *bi = t * e1 - e2 + bjn[0] - ax * (t * f1 - f2 + bjp[0]);

                j = n3d;
                f1 = dbjp[j - 1];
                e1 = dbjn[j - 1];
                f2 = 0.;
                e2 = 0.;
                for (int i=1; i<=m3d; ++i) {
                    --j;
                    double temp1 = f1;
                    double temp2 = e1;
                    f1 = tt * f1 - f2 + dbjp[j - 1];
                    e1 = tt * e1 - e2 + dbjn[j - 1];
                    f2 = temp1;
                    e2 = temp2;
                }
                *dbi = x * x * (t * f1 - f2 + dbjp[0]) + (t * e1 - e2 + dbjn[0]);
            } else {
                double rtrx = std::sqrt(rx);
                double t = 10. / c - 1.;
                double tt = t + t;
                int j = n3;
                double f1 = aa[j - 1];
                double e1 = bb[j - 1];
                double f2 = 0.;
                double e2 = 0.;
                for (int i=1; i<=m3; ++i) {
                    --j;
                    double temp1 = f1;
                    double temp2 = e1;
                    f1 = tt * f1 - f2 + aa[j - 1];
                    e1 = tt * e1 - e2 + bb[j - 1];
                    f2 = temp1;
                    e2 = temp2;
                }
                double temp1 = t * f1 - f2 + aa[0];
                double temp2 = t * e1 - e2 + bb[0];
                double cv = c - fpi12;
                *bi = (temp1 * std::cos(cv) + temp2 * std::sin(cv)) / rtrx;

                j = n4d;
                f1 = daa[j - 1];
                e1 = dbb[j - 1];
                f2 = 0.;
                e2 = 0.;
                for (int i=1; i<=m4d; ++i) {
                    --j;
                    temp1 = f1;
                    temp2 = e1;
                    f1 = tt * f1 - f2 + daa[j - 1];
                    e1 = tt * e1 - e2 + dbb[j - 1];
                    f2 = temp1;
                    e2 = temp2;
                }
                temp1 = t * f1 - f2 + daa[0];
                temp2 = t * e1 - e2 + dbb[0];
                cv = c - spi12;
                *dbi = (temp1 * std::cos(cv) - temp2 * std::sin(cv)) * rtrx;
            }
        } else {
            if (c > 8.) {
                double rtrx = std::sqrt(rx);
                double t = 16. / c - 1.;
                double tt = t + t;
                int j = n1;
                double f1 = bk3[j - 1];
                double f2 = 0.;
                for (int i=1; i<=m1; ++i) {
                    --j;
                    double temp1 = f1;
                    f1 = tt * f1 - f2 + bk3[j - 1];
                    f2 = temp1;
                }
                double s1 = t * f1 - f2 + bk3[0];
                j = n2d;
                f1 = dbk3[j - 1];
                f2 = 0.;
                for (int i=1; i<=m2d; ++i) {
                    --j;
                    double temp1 = f1;
                    f1 = tt * f1 - f2 + dbk3[j - 1];
                    f2 = temp1;
                }
                double d1 = t * f1 - f2 + dbk3[0];
                double tc = c + c;
                double ex = std::exp(c);
                if (tc > 35.) {
                    *bi = ex * s1 / rtrx;
                    *dbi = ex * rtrx * d1;
                    return;
                }
                t = 10. / c - 1.;
                tt = t + t;
                j = n3;
                f1 = bk4[j - 1];
                f2 = 0.;
                for (int i=1; i<=m3; ++i) {
                    --j;
                    double temp1 = f1;
                    f1 = tt * f1 - f2 + bk4[j - 1];
                    f2 = temp1;
                }
                double s2 = t * f1 - f2 + bk4[0];
                *bi = (s1 + std::exp(-tc) * s2) / rtrx;
                *bi *= ex;
                j = n4d;
                f1 = dbk4[j - 1];
                f2 = 0.;
                for (int i=1; i<=m4d; ++i) {
                    --j;
                    double temp1 = f1;
                    f1 = tt * f1 - f2 + dbk4[j - 1];
                    f2 = temp1;
                }
                double d2 = t * f1 - f2 + dbk4[0];
                *dbi = rtrx * (d1 + std::exp(-tc) * d2);
                *dbi *= ex;
            } else if (x > 2.5) {
                double rtrx = std::sqrt(rx);
                double t = (x + x - con2) * con3;
                double tt = t + t;
                int j = n1;
                double f1 = bk2[j - 1];
                double f2 = 0.;
                for (int i=1; i<=m1; ++i) {
                    --j;
                    double temp1 = f1;
                    f1 = tt * f1 - f2 + bk2[j - 1];
                    f2 = temp1;
                }
                *bi = (t * f1 - f2 + bk2[0]) / rtrx;
                double ex = std::exp(c);
                *bi *= ex;
                j = n2d;
                f1 = dbk2[j - 1];
                f2 = 0.;
                for (int i=1; i<=m2d; ++i) {
                    --j;
                    double temp1 = f1;
                    f1 = tt * f1 - f2 + dbk2[j - 1];
                    f2 = temp1;
                }
                *dbi = (t * f1 - f2 + dbk2[0]) * rtrx;
                *dbi *= ex;
            } else {
                double t = (x + x - 2.5) * 0.4;
                double tt = t + t;
                int j = n1;
                double f1 = bk1[j - 1];
                double f2 = 0.;
                for (int i=1; i<=m1; ++i) {
                    --j;
                    double temp1 = f1;
                    f1 = tt * f1 - f2 + bk1[j - 1];
                    f2 = temp1;
                }
                *bi = t * f1 - f2 + bk1[0];

                j = n1d;
                f1 = dbk1[j - 1];
                f2 = 0.;
                for (int i=1; i<=m1d; ++i) {
                    --j;
                    double temp1 = f1;
                    f1 = tt * f1 - f2 + dbk1[j - 1];
                    f2 = temp1;
                }
                *dbi = t * f1 - f2 + dbk1[0];
            }
        }
    }


    //
    // I_nu(x)
    //

    double dbesi(double x, double fnu)
    {
        // ***BEGIN PROLOGUE  DBESI
        // ***PURPOSE  Compute an N member sequence of I Bessel functions
        //            I/SUB(ALPHA+K-1)/(X), K=1,...,N or scaled Bessel functions
        //            EXP(-X)*I/SUB(ALPHA+K-1)/(X), K=1,...,N for nonnegative
        //            ALPHA and X.
        // ***LIBRARY   SLATEC
        // ***CATEGORY  C10B3
        // ***TYPE      DOUBLE PRECISION (BESI-S, DBESI-D)
        // ***KEYWORDS  I BESSEL FUNCTION, SPECIAL FUNCTIONS
        // ***AUTHOR  Amos, D. E., (SNLA)
        //           Daniel, S. L., (SNLA)
        // ***DESCRIPTION
        //
        //     Abstract  **** a double precision routine ****
        //         DBESI computes an N member sequence of I Bessel functions
        //         I/sub(ALPHA+K-1)/(X), K=1,...,N or scaled Bessel functions
        //         EXP(-X)*I/sub(ALPHA+K-1)/(X), K=1,...,N for nonnegative ALPHA
        //         and X.  A combination of the power series, the asymptotic
        //         expansion for X to infinity, and the uniform asymptotic
        //         expansion for NU to infinity are applied over subdivisions of
        //         the (NU,X) plane.  For values not covered by one of these
        //         formulae, the order is incremented by an integer so that one
        //         of these formulae apply.  Backward recursion is used to reduce
        //         orders by integer values.  The asymptotic expansion for X to
        //         infinity is used only when the entire sequence (specifically
        //         the last member) lies within the region covered by the
        //         expansion.  Leading terms of these expansions are used to test
        //         for over or underflow where appropriate.  If a sequence is
        //         requested and the last member would underflow, the result is
        //         set to zero and the next lower order tried, etc., until a
        //         member comes on scale or all are set to zero.  An overflow
        //         cannot occur with scaling.
        //
        //         The maximum number of significant digits obtainable
        //         is the smaller of 14 and the number of digits carried in
        //         double precision arithmetic.
        //
        //     Description of Arguments
        //
        //         Input      X,ALPHA are double precision
        //           X      - X .GE. 0.0D0
        //           ALPHA  - order of first member of the sequence,
        //                    ALPHA .GE. 0.0D0
        //           KODE   - a parameter to indicate the scaling option
        //                    KODE=1 returns
        //                           Y(K)=        I/sub(ALPHA+K-1)/(X),
        //                                K=1,...,N
        //                    KODE=2 returns
        //                           Y(K)=EXP(-X)*I/sub(ALPHA+K-1)/(X),
        //                                K=1,...,N
        //           N      - number of members in the sequence, N .GE. 1
        //
        //         Output     Y is double precision
        //           Y      - a vector whose first N components contain
        //                    values for I/sub(ALPHA+K-1)/(X) or scaled
        //                    values for EXP(-X)*I/sub(ALPHA+K-1)/(X),
        //                    K=1,...,N depending on KODE
        //           NZ     - number of components of Y set to zero due to
        //                    underflow,
        //                    NZ=0   , normal return, computation completed
        //                    NZ .NE. 0, last NZ components of Y set to zero,
        //                             Y(K)=0.0D0, K=N-NZ+1,...,N.
        //
        //     Error Conditions
        //         Improper input arguments - a fatal error
        //         Overflow with KODE=1 - a fatal error
        //         Underflow - a non-fatal error(NZ .NE. 0)
        //
        // ***REFERENCES  D. E. Amos, S. L. Daniel and M. K. Weston, CDC 6600
        //                 subroutines IBESS and JBESS for Bessel functions
        //                 I(NU,X) and J(NU,X), X .GE. 0, NU .GE. 0, ACM
        //                 Transactions on Mathematical Software 3, (1977),
        //                 pp. 76-92.
        //               F. W. J. Olver, Tables of Bessel Functions of Moderate
        //                 or Large Orders, NPL Mathematical Tables 6, Her
        //                 Majesty's Stationery Office, London, 1962.
        // ***ROUTINES CALLED  D1MACH, DASYIK, DLNGAM, I1MACH, XERMSG
        // ***REVISION HISTORY  (YYMMDD)
        //    750101  DATE WRITTEN
        //    890531  Changed all specific intrinsics to generic.  (WRB)
        //    890911  Removed unnecessary intrinsics.  (WRB)
        //    890911  REVISION DATE from Version 3.2
        //    891214  Prologue converted to Version 4.0 format.  (BAB)
        //    900315  CALLs to XERROR changed to CALLs to XERMSG.  (THJ)
        //    900326  Removed duplicate information from DESCRIPTION section.
        //            (WRB)
        //    920501  Reformatted the REFERENCES section.  (WRB)
        //    170203  Converted to C++, and modified to only cover n==1 option. (MJ)
        // ***END PROLOGUE  DBESI

        const double tol = std::max(std::numeric_limits<double>::epsilon(), 1e-15);
        const double elim = -std::log(std::numeric_limits<double>::min() * 1.e3);
        const double tolln = -std::log(tol);
        const double rttpi = 0.398942280401433;
        const int inlim = 80;

        assert(x >= 0.);
        assert(fnu >= 0.);

        if (fnu == 0.) return dbesi0(x);
        else if (fnu == 1.) return dbesi1(x);
        else if (x == 0.) return 0.;

        const double fni = std::floor(fnu);
        const double fnf = fnu - fni;
        const double xo2 = x * 0.5;
        const double sxo2 = xo2 * xo2;

        //     DECISION TREE FOR REGION WHERE SERIES, ASYMPTOTIC EXPANSION FOR X
        //     TO INFINITY AND ASYMPTOTIC EXPANSION FOR NU TO INFINITY ARE
        //     APPLIED.

        bool series;
        int ns;
        if (sxo2 <= fnu + 1.) {
            series = true;
            ns = 0;
        } else if (x <= 12.) {
            series = true;
            ns = int(sxo2 - fnu);
        } else {
            series = false;
            ns = std::max(int(36.-fnu), 0);
        }

        double fn = fnu + ns;
        int in;
        double inu;
        if (series) {
            //     SERIES FOR (X/2)**2.LE.NU+1
            double gln = std::lgamma(fn+1);
            double xo2l = std::log(xo2);
            double arg = fn * xo2l - gln;
            if (arg < -elim) return 0.;
            double earg = std::exp(arg);
            double s = 1.;
            if (x >= tol) {
                double ak = 3.;
                double t2 = 1.;
                double t = 1.;
                double s1 = fn;
                for (int k = 1; k <= 17; ++k) {
                    double s2 = t2 + s1;
                    t = t * sxo2 / s2;
                    s += t;
                    if (std::abs(t) < tol) break;
                    t2 += ak;
                    ak += 2.;
                    s1 += fn;
                }
            }
            inu = s * earg;
            if (ns == 0) return inu;

            //     BACKWARD RECURSION WITH NORMALIZATION BY
            //     ASYMPTOTIC EXPANSION FOR NU TO INFINITY OR POWER SERIES.
            //     COMPUTATION OF LAST ORDER FOR SERIES NORMALIZATION
            double akm = std::max(3.-fn,0.);
            int km = int(akm);
            double tfn = fn + km;
            double ta = (gln + tfn - 0.9189385332 - 0.0833333333 / tfn) / (tfn + 0.5);
            ta = xo2l - ta;
            double tb = -(1. - 1. / tfn) / tfn;
            double ain = tolln / (-ta + std::sqrt(ta * ta - tolln * tb)) + 1.5;
            in = int(ain) + km;
        } else {
            if (x >= std::max(17., 0.55 * fnu * fnu)) {
                //     ASYMPTOTIC EXPANSION FOR X TO INFINITY

                double earg = rttpi / std::sqrt(x);
                if (x > elim) {
                    throw std::runtime_error("DBESI OVERFLOW, X TOO LARGE FOR KODE = 1.");
                }
                earg *= std::exp(x);
                double etx = x * 8.;
                in = 0;
                double fn = fnu;
                double dx = fni + fni;
                double tm = fnf * 4. * (fni + fni + fnf);
                double dtm = dx * dx;
                double s1 = etx;
                double trx = dtm - 1.;
                dx = -(trx + tm) / etx;
                double t = dx;
                double s = dx + 1.;
                double atol = tol * std::abs(s);
                double s2 = 1.;
                double ak = 8.;
                for (int k = 1; k <= 25; ++k) {
                    s1 += etx;
                    s2 += ak;
                    dx = dtm - s2;
                    double ap = dx + tm;
                    t = -t * ap / s1;
                    s += t;
                    if (std::abs(t) <= atol) break;
                    ak += 8.;
                }
                return s * earg;
            }
            //     OVERFLOW TEST ON UNIFORM ASYMPTOTIC EXPANSION
            if (fnu < 1.) {
                if (x > elim) {
                    throw std::runtime_error("DBESI OVERFLOW, X TOO LARGE FOR KODE = 1.");
                }
            } else {
                double z = x / fnu;
                double ra = std::sqrt(z * z + 1.);
                double gln = std::log((ra + 1.) / z);
                double t = ra;
                double arg = fnu * (t - gln);
                if (arg > elim) {
                    throw std::runtime_error("DBESI OVERFLOW, X TOO LARGE FOR KODE = 1.");
                }
                if (ns == 0 && arg < -elim) return 0.;
            }

            //     UNDERFLOW TEST ON UNIFORM ASYMPTOTIC EXPANSION
            double z = x / fn;
            double ra = std::sqrt(z * z + 1.);
            double gln = std::log((ra + 1.) / z);
            double t = ra;
            double arg = fn * (t - gln);
            if (arg < -elim) return 0.;

            inu = dasyik(x, fn, true);
            if (ns == 0) return inu;

            //     COMPUTATION OF LAST ORDER FOR ASYMPTOTIC EXPANSION NORMALIZATION
            t = 1. / (fn * ra);
            double ain = tolln / (gln + std::sqrt(gln * gln + t * tolln)) + 1.5;
            in = int(ain);
        }

        double trx = 2. / x;
        double tm = (fn + in) * trx;
        double ta = 0.;
        double tb = tol;
        //     BACKWARD RECUR UNINDEXED
        for (int kk=1; kk<=2; ++kk) {
            for (int i=1; i<=in; ++i) {
                double s = tb;
                tb = tm * tb + ta;
                ta = s;
                tm -= trx;
            }
            //     NORMALIZATION
            if (kk == 2) break;
            ta = ta / tb * inu;
            tb = inu;
            in = ns;
        }
        return tb;
    }

    double dbesi0(double x)
    {
        // ***BEGIN PROLOGUE  DBESI0
        // ***PURPOSE  Compute the hyperbolic Bessel function of the first kind
        //            of order zero.
        // ***LIBRARY   SLATEC (FNLIB)
        // ***CATEGORY  C10B1
        // ***TYPE      DOUBLE PRECISION (BESI0-S, DBESI0-D)
        // ***KEYWORDS  FIRST KIND, FNLIB, HYPERBOLIC BESSEL FUNCTION,
        //             MODIFIED BESSEL FUNCTION, ORDER ZERO, SPECIAL FUNCTIONS
        // ***AUTHOR  Fullerton, W., (LANL)
        // ***DESCRIPTION
        //
        // DBESI0(X) calculates the double precision modified (hyperbolic)
        // Bessel function of the first kind of order zero and double
        // precision argument X.
        //
        // Series for BI0        on the interval  0.          to  9.00000E+00
        //                                        with weighted error   9.51E-34
        //                                         log weighted error  33.02
        //                               significant figures required  33.31
        //                                    decimal places required  33.65
        //
        // ***REFERENCES  (NONE)
        // ***ROUTINES CALLED  D1MACH, DBSI0E, DCSEVL, INITDS, XERMSG
        // ***REVISION HISTORY  (YYMMDD)
        //    770701  DATE WRITTEN
        //    890531  Changed all specific intrinsics to generic.  (WRB)
        //    890531  REVISION DATE from Version 3.2
        //    891214  Prologue converted to Version 4.0 format.  (BAB)
        //    900315  CALLs to XERROR changed to CALLs to XERMSG.  (THJ)
        //    170203  Converted to C++. (MJ)
        // ***END PROLOGUE  DBESI0

        const double bi0cs[18] = {
            -0.07660547252839144951081894976243285,
            1.927337953993808269952408750881196,
            0.2282644586920301338937029292330415,
            0.01304891466707290428079334210691888,
            4.344270900816487451378682681026107e-4,
            9.422657686001934663923171744118766e-6,
            1.434006289510691079962091878179957e-7,
            1.613849069661749069915419719994611e-9,
            1.396650044535669699495092708142522e-11,
            9.579451725505445344627523171893333e-14,
            5.333981859862502131015107744e-16,
            2.458716088437470774696785919999999e-18,
            9.535680890248770026944341333333333e-21,
            3.154382039721427336789333333333333e-23,
            9.004564101094637431466666666666666e-26,
            2.240647369123670016e-28,
            4.903034603242837333333333333333333e-31,
            9.508172606122666666666666666666666e-34
        };
        const int nti0 = 11;
        const double xsml = std::sqrt(std::numeric_limits<double>::epsilon() * 4.5);

        assert(x > 0.);
        if (x <= 3.) {
            return (x > xsml) ? (2.75 + dcsevl(x*x/4.5-1., bi0cs, nti0)) : 1.;
        } else {
            return std::exp(x) * dbsi0e(x);
        }
    }

    double dbsi0e(double x)
    {
        // ***BEGIN PROLOGUE  DBSI0E
        // ***PURPOSE  Compute the exponentially scaled modified (hyperbolic)
        //            Bessel function of the first kind of order zero.
        // ***LIBRARY   SLATEC (FNLIB)
        // ***CATEGORY  C10B1
        // ***TYPE      DOUBLE PRECISION (BESI0E-S, DBSI0E-D)
        // ***KEYWORDS  EXPONENTIALLY SCALED, FIRST KIND, FNLIB,
        //             HYPERBOLIC BESSEL FUNCTION, MODIFIED BESSEL FUNCTION,
        //             ORDER ZERO, SPECIAL FUNCTIONS
        // ***AUTHOR  Fullerton, W., (LANL)
        // ***DESCRIPTION
        //
        // DBSI0E(X) calculates the double precision exponentially scaled
        // modified (hyperbolic) Bessel function of the first kind of order
        // zero for double precision argument X.  The result is the Bessel
        // function I0(X) multiplied by EXP(-ABS(X)).
        //
        // Series for BI0        on the interval  0.          to  9.00000E+00
        //                                        with weighted error   9.51E-34
        //                                         log weighted error  33.02
        //                               significant figures required  33.31
        //                                    decimal places required  33.65
        //
        // Series for AI0        on the interval  1.25000E-01 to  3.33333E-01
        //                                        with weighted error   2.74E-32
        //                                         log weighted error  31.56
        //                               significant figures required  30.15
        //                                    decimal places required  32.39
        //
        // Series for AI02       on the interval  0.          to  1.25000E-01
        //                                        with weighted error   1.97E-32
        //                                         log weighted error  31.71
        //                               significant figures required  30.15
        //                                    decimal places required  32.63
        //
        // ***REFERENCES  (NONE)
        // ***ROUTINES CALLED  D1MACH, DCSEVL, INITDS
        // ***REVISION HISTORY  (YYMMDD)
        //    770701  DATE WRITTEN
        //    890531  Changed all specific intrinsics to generic.  (WRB)
        //    890531  REVISION DATE from Version 3.2
        //    891214  Prologue converted to Version 4.0 format.  (BAB)
        //    170203  Converted to C++. (MJ)
        // ***END PROLOGUE  DBSI0E

        const double bi0cs[18] = {
            -0.07660547252839144951081894976243285,
            1.927337953993808269952408750881196,
            0.2282644586920301338937029292330415,
            0.01304891466707290428079334210691888,
            4.344270900816487451378682681026107e-4,
            9.422657686001934663923171744118766e-6,
            1.434006289510691079962091878179957e-7,
            1.613849069661749069915419719994611e-9,
            1.396650044535669699495092708142522e-11,
            9.579451725505445344627523171893333e-14,
            5.333981859862502131015107744e-16,
            2.458716088437470774696785919999999e-18,
            9.535680890248770026944341333333333e-21,
            3.154382039721427336789333333333333e-23,
            9.004564101094637431466666666666666e-26,
            2.240647369123670016e-28,
            4.903034603242837333333333333333333e-31,
            9.508172606122666666666666666666666e-34
        };
        const double ai0cs[46] = {
            0.07575994494023795942729872037438,
            0.007591380810823345507292978733204,
            4.153131338923750501863197491382e-4,
            1.07007646343907307358242970217e-5,
            -7.90117997921289466075031948573e-6,
            -7.826143501438752269788989806909e-7,
            2.783849942948870806381185389857e-7,
            8.252472600612027191966829133198e-9,
            -1.204463945520199179054960891103e-8,
            1.559648598506076443612287527928e-9,
            2.292556367103316543477254802857e-10,
            -1.191622884279064603677774234478e-10,
            1.757854916032409830218331247743e-11,
            1.128224463218900517144411356824e-12,
            -1.146848625927298877729633876982e-12,
            2.715592054803662872643651921606e-13,
            -2.415874666562687838442475720281e-14,
            -6.084469888255125064606099639224e-15,
            3.145705077175477293708360267303e-15,
            -7.172212924871187717962175059176e-16,
            7.874493403454103396083909603327e-17,
            1.004802753009462402345244571839e-17,
            -7.56689536535053485342843588881e-18,
            2.150380106876119887812051287845e-18,
            -3.754858341830874429151584452608e-19,
            2.354065842226992576900757105322e-20,
            1.11466761204792853022637335511e-20,
            -5.398891884396990378696779322709e-21,
            1.439598792240752677042858404522e-21,
            -2.591916360111093406460818401962e-22,
            2.23813318399858390743409229824e-23,
            5.250672575364771172772216831999e-24,
            -3.249904138533230784173432285866e-24,
            9.9242141032050379278572847104e-25,
            -2.164992254244669523146554299733e-25,
            3.233609471943594083973332991999e-26,
            -1.184620207396742489824733866666e-27,
            -1.281671853950498650548338687999e-27,
            5.827015182279390511605568853333e-28,
            -1.668222326026109719364501503999e-28,
            3.6253095105415699757006848e-29,
            -5.733627999055713589945958399999e-30,
            3.736796722063098229642581333333e-31,
            1.602073983156851963365512533333e-31,
            -8.700424864057229884522495999999e-32,
            2.741320937937481145603413333333e-32
        };
        const double ai02cs[69] = {
            0.0544904110141088316078960962268,
            0.003369116478255694089897856629799,
            6.889758346916823984262639143011e-5,
            2.891370520834756482966924023232e-6,
            2.048918589469063741827605340931e-7,
            2.266668990498178064593277431361e-8,
            3.396232025708386345150843969523e-9,
            4.940602388224969589104824497835e-10,
            1.188914710784643834240845251963e-11,
            -3.149916527963241364538648629619e-11,
            -1.321581184044771311875407399267e-11,
            -1.794178531506806117779435740269e-12,
            7.180124451383666233671064293469e-13,
            3.852778382742142701140898017776e-13,
            1.540086217521409826913258233397e-14,
            -4.150569347287222086626899720156e-14,
            -9.554846698828307648702144943125e-15,
            3.811680669352622420746055355118e-15,
            1.772560133056526383604932666758e-15,
            -3.425485619677219134619247903282e-16,
            -2.827623980516583484942055937594e-16,
            3.461222867697461093097062508134e-17,
            4.465621420296759999010420542843e-17,
            -4.830504485944182071255254037954e-18,
            -7.233180487874753954562272409245e-18,
            9.92147541217369859888046093981e-19,
            1.193650890845982085504399499242e-18,
            -2.488709837150807235720544916602e-19,
            -1.938426454160905928984697811326e-19,
            6.444656697373443868783019493949e-20,
            2.886051596289224326481713830734e-20,
            -1.601954907174971807061671562007e-20,
            -3.270815010592314720891935674859e-21,
            3.686932283826409181146007239393e-21,
            1.268297648030950153013595297109e-23,
            -7.549825019377273907696366644101e-22,
            1.502133571377835349637127890534e-22,
            1.265195883509648534932087992483e-22,
            -6.100998370083680708629408916002e-23,
            -1.268809629260128264368720959242e-23,
            1.661016099890741457840384874905e-23,
            -1.585194335765885579379705048814e-24,
            -3.302645405968217800953817667556e-24,
            1.313580902839239781740396231174e-24,
            3.689040246671156793314256372804e-25,
            -4.210141910461689149219782472499e-25,
            4.79195459108286578063171401373e-26,
            8.459470390221821795299717074124e-26,
            -4.03980094087283249314607937181e-26,
            -6.434714653650431347301008504695e-27,
            1.225743398875665990344647369905e-26,
            -2.934391316025708923198798211754e-27,
            -1.961311309194982926203712057289e-27,
            1.503520374822193424162299003098e-27,
            -9.588720515744826552033863882069e-29,
            -3.483339380817045486394411085114e-28,
            1.690903610263043673062449607256e-28,
            1.982866538735603043894001157188e-29,
            -5.317498081491816214575830025284e-29,
            1.803306629888392946235014503901e-29,
            6.213093341454893175884053112422e-30,
            -7.69218929277216186320072806673e-30,
            1.858252826111702542625560165963e-30,
            1.237585142281395724899271545541e-30,
            -1.102259120409223803217794787792e-30,
            1.886287118039704490077874479431e-31,
            2.16019687224365891314903141406e-31,
            -1.605454124919743200584465949655e-31,
            1.965352984594290603938848073318e-32
        };
        const int nti0 = 11;
        const int ntai0 = 23;
        const int ntai02 = 25;
        const double xsml = std::sqrt(std::numeric_limits<double>::epsilon() * 4.5);

        assert(x > 0.);
        if (x <= 3.) {
            return (x > xsml) ? std::exp(-x) * (2.75 + dcsevl(x*x/4.5-1., bi0cs, nti0)) : 1.-x;
        } else if (x <= 8.) {
            return (dcsevl((48./x-11.)/5., ai0cs, ntai0) + 0.375) / std::sqrt(x);
        } else {
            return (dcsevl(16./x-1., ai02cs, ntai02) + 0.375) / std::sqrt(x);
        }
    }

    double dbesi1(double x)
    {
        // ***BEGIN PROLOGUE  DBESI1
        // ***PURPOSE  Compute the modified (hyperbolic) Bessel function of the
        //            first kind of order one.
        // ***LIBRARY   SLATEC (FNLIB)
        // ***CATEGORY  C10B1
        // ***TYPE      DOUBLE PRECISION (BESI1-S, DBESI1-D)
        // ***KEYWORDS  FIRST KIND, FNLIB, HYPERBOLIC BESSEL FUNCTION,
        //             MODIFIED BESSEL FUNCTION, ORDER ONE, SPECIAL FUNCTIONS
        // ***AUTHOR  Fullerton, W., (LANL)
        // ***DESCRIPTION
        //
        // DBESI1(X) calculates the double precision modified (hyperbolic)
        // Bessel function of the first kind of order one and double precision
        // argument X.
        //
        // Series for BI1        on the interval  0.          to  9.00000E+00
        //                                        with weighted error   1.44E-32
        //                                         log weighted error  31.84
        //                               significant figures required  31.45
        //                                    decimal places required  32.46
        //
        // ***REFERENCES  (NONE)
        // ***ROUTINES CALLED  D1MACH, DBSI1E, DCSEVL, INITDS, XERMSG
        // ***REVISION HISTORY  (YYMMDD)
        //    770701  DATE WRITTEN
        //    890531  Changed all specific intrinsics to generic.  (WRB)
        //    890531  REVISION DATE from Version 3.2
        //    891214  Prologue converted to Version 4.0 format.  (BAB)
        //    900315  CALLs to XERROR changed to CALLs to XERMSG.  (THJ)
        //    170203  Converted to C++. (MJ)
        // ***END PROLOGUE  DBESI1

        const double bi1cs[17] = {
            -0.0019717132610998597316138503218149,
            0.40734887667546480608155393652014,
            0.034838994299959455866245037783787,
            0.0015453945563001236038598401058489,
            4.188852109837778412945883200412e-5,
            7.6490267648362114741959703966069e-7,
            1.0042493924741178689179808037238e-8,
            9.9322077919238106481371298054863e-11,
            7.6638017918447637275200171681349e-13,
            4.741418923816739498038809194816e-15,
            2.4041144040745181799863172032e-17,
            1.0171505007093713649121100799999e-19,
            3.6450935657866949458491733333333e-22,
            1.1205749502562039344810666666666e-24,
            2.9875441934468088832e-27,
            6.9732310939194709333333333333333e-30,
            1.43679482206208e-32
        };
        const int nti1 = 11;
        const double xmin = 2. * std::numeric_limits<double>::min();
        const double xsml = std::sqrt(std::numeric_limits<double>::epsilon() * 4.5);
        const double xmax = std::log(std::numeric_limits<double>::max());

        assert(x > 0.);
        if (x <= 3.) {
            if (x < xsml) return 0.5 * x;
            else return x * (dcsevl(x*x/4.5-1., bi1cs, nti1) + 0.875);
        } else {
            return std::exp(x) * dbsi1e(x);
        }
    }

    double dbsi1e(double x)
    {
        // ***BEGIN PROLOGUE  DBSI1E
        // ***PURPOSE  Compute the exponentially scaled modified (hyperbolic)
        //            Bessel function of the first kind of order one.
        // ***LIBRARY   SLATEC (FNLIB)
        // ***CATEGORY  C10B1
        // ***TYPE      DOUBLE PRECISION (BESI1E-S, DBSI1E-D)
        // ***KEYWORDS  EXPONENTIALLY SCALED, FIRST KIND, FNLIB,
        //             HYPERBOLIC BESSEL FUNCTION, MODIFIED BESSEL FUNCTION,
        //             ORDER ONE, SPECIAL FUNCTIONS
        // ***AUTHOR  Fullerton, W., (LANL)
        // ***DESCRIPTION
        //
        // DBSI1E(X) calculates the double precision exponentially scaled
        // modified (hyperbolic) Bessel function of the first kind of order
        // one for double precision argument X.  The result is I1(X)
        // multiplied by EXP(-ABS(X)).
        //
        // Series for BI1        on the interval  0.          to  9.00000E+00
        //                                        with weighted error   1.44E-32
        //                                         log weighted error  31.84
        //                               significant figures required  31.45
        //                                    decimal places required  32.46
        //
        // Series for AI1        on the interval  1.25000E-01 to  3.33333E-01
        //                                        with weighted error   2.81E-32
        //                                         log weighted error  31.55
        //                               significant figures required  29.93
        //                                    decimal places required  32.38
        //
        // Series for AI12       on the interval  0.          to  1.25000E-01
        //                                        with weighted error   1.83E-32
        //                                         log weighted error  31.74
        //                               significant figures required  29.97
        //                                    decimal places required  32.66
        //
        // ***REFERENCES  (NONE)
        // ***ROUTINES CALLED  D1MACH, DCSEVL, INITDS, XERMSG
        // ***REVISION HISTORY  (YYMMDD)
        //    770701  DATE WRITTEN
        //    890531  Changed all specific intrinsics to generic.  (WRB)
        //    890531  REVISION DATE from Version 3.2
        //    891214  Prologue converted to Version 4.0 format.  (BAB)
        //    900315  CALLs to XERROR changed to CALLs to XERMSG.  (THJ)
        //    170203  Converted to C++. (MJ)
        // ***END PROLOGUE  DBSI1E

        const double bi1cs[17] = {
            -0.0019717132610998597316138503218149,
            0.40734887667546480608155393652014,
            0.034838994299959455866245037783787,
            0.0015453945563001236038598401058489,
            4.188852109837778412945883200412e-5,
            7.6490267648362114741959703966069e-7,
            1.0042493924741178689179808037238e-8,
            9.9322077919238106481371298054863e-11,
            7.6638017918447637275200171681349e-13,
            4.741418923816739498038809194816e-15,
            2.4041144040745181799863172032e-17,
            1.0171505007093713649121100799999e-19,
            3.6450935657866949458491733333333e-22,
            1.1205749502562039344810666666666e-24,
            2.9875441934468088832e-27,
            6.9732310939194709333333333333333e-30,
            1.43679482206208e-32
        };
        const double ai1cs[46] = {
            -0.02846744181881478674100372468307,
            -0.01922953231443220651044448774979,
            -6.115185857943788982256249917785e-4,
            -2.069971253350227708882823777979e-5,
            8.585619145810725565536944673138e-6,
            1.04949824671159086251745399786e-6,
            -2.918338918447902202093432326697e-7,
            -1.559378146631739000160680969077e-8,
            1.318012367144944705525302873909e-8,
            -1.448423418183078317639134467815e-9,
            -2.90851224399314209482504099301e-10,
            1.266388917875382387311159690403e-10,
            -1.66494777291922067062417839858e-11,
            -1.666653644609432976095937154999e-12,
            1.242602414290768265232168472017e-12,
            -2.731549379672432397251461428633e-13,
            2.023947881645803780700262688981e-14,
            7.307950018116883636198698126123e-15,
            -3.332905634404674943813778617133e-15,
            7.17534655851295374354225466567e-16,
            -6.982530324796256355850629223656e-17,
            -1.299944201562760760060446080587e-17,
            8.12094286424279889205467834286e-18,
            -2.194016207410736898156266643783e-18,
            3.630516170029654848279860932334e-19,
            -1.695139772439104166306866790399e-20,
            -1.288184829897907807116882538222e-20,
            5.694428604967052780109991073109e-21,
            -1.459597009090480056545509900287e-21,
            2.514546010675717314084691334485e-22,
            -1.844758883139124818160400029013e-23,
            -6.339760596227948641928609791999e-24,
            3.46144110203101111110814662656e-24,
            -1.017062335371393547596541023573e-24,
            2.149877147090431445962500778666e-25,
            -3.045252425238676401746206173866e-26,
            5.238082144721285982177634986666e-28,
            1.443583107089382446416789503999e-27,
            -6.121302074890042733200670719999e-28,
            1.700011117467818418349189802666e-28,
            -3.596589107984244158535215786666e-29,
            5.448178578948418576650513066666e-30,
            -2.731831789689084989162564266666e-31,
            -1.858905021708600715771903999999e-31,
            9.212682974513933441127765333333e-32,
            -2.813835155653561106370833066666e-32
        };
        const double ai12cs[69] = {
            0.02857623501828012047449845948469,
            -0.009761097491361468407765164457302,
            -1.105889387626237162912569212775e-4,
            -3.882564808877690393456544776274e-6,
            -2.512236237870208925294520022121e-7,
            -2.631468846889519506837052365232e-8,
            -3.835380385964237022045006787968e-9,
            -5.589743462196583806868112522229e-10,
            -1.897495812350541234498925033238e-11,
            3.252603583015488238555080679949e-11,
            1.412580743661378133163366332846e-11,
            2.03562854414708950722452613684e-12,
            -7.198551776245908512092589890446e-13,
            -4.083551111092197318228499639691e-13,
            -2.101541842772664313019845727462e-14,
            4.272440016711951354297788336997e-14,
            1.042027698412880276417414499948e-14,
            -3.814403072437007804767072535396e-15,
            -1.880354775510782448512734533963e-15,
            3.308202310920928282731903352405e-16,
            2.962628997645950139068546542052e-16,
            -3.209525921993423958778373532887e-17,
            -4.650305368489358325571282818979e-17,
            4.414348323071707949946113759641e-18,
            7.517296310842104805425458080295e-18,
            -9.314178867326883375684847845157e-19,
            -1.242193275194890956116784488697e-18,
            2.414276719454848469005153902176e-19,
            2.026944384053285178971922860692e-19,
            -6.394267188269097787043919886811e-20,
            -3.049812452373095896084884503571e-20,
            1.612841851651480225134622307691e-20,
            3.56091396430992505451027090462e-21,
            -3.752017947936439079666828003246e-21,
            -5.787037427074799345951982310741e-23,
            7.759997511648161961982369632092e-22,
            -1.452790897202233394064459874085e-22,
            -1.318225286739036702121922753374e-22,
            6.116654862903070701879991331717e-23,
            1.376279762427126427730243383634e-23,
            -1.690837689959347884919839382306e-23,
            1.430596088595433153987201085385e-24,
            3.409557828090594020405367729902e-24,
            -1.309457666270760227845738726424e-24,
            -3.940706411240257436093521417557e-25,
            4.277137426980876580806166797352e-25,
            -4.424634830982606881900283123029e-26,
            -8.734113196230714972115309788747e-26,
            4.045401335683533392143404142428e-26,
            7.067100658094689465651607717806e-27,
            -1.249463344565105223002864518605e-26,
            2.867392244403437032979483391426e-27,
            2.04429289250429267028177957421e-27,
            -1.518636633820462568371346802911e-27,
            8.110181098187575886132279107037e-29,
            3.58037935477358609112717370327e-28,
            -1.692929018927902509593057175448e-28,
            -2.222902499702427639067758527774e-29,
            5.424535127145969655048600401128e-29,
            -1.787068401578018688764912993304e-29,
            -6.56547906872281493882392943788e-30,
            7.807013165061145280922067706839e-30,
            -1.816595260668979717379333152221e-30,
            -1.287704952660084820376875598959e-30,
            1.114548172988164547413709273694e-30,
            -1.808343145039336939159368876687e-31,
            -2.231677718203771952232448228939e-31,
            1.619029596080341510617909803614e-31,
            -1.83407990880494141390130843921e-32
        };
        const int nti1 = 11;
        const int ntai1 = 23;
        const int ntai12 = 25;
        const double xmin = 2. * std::numeric_limits<double>::min();
        const double xsml = std::sqrt(std::numeric_limits<double>::epsilon() * 4.5);

        assert(x > 0.);
        if (x <= 3.) {
            if (x < xsml) return std::exp(-x) * (0.5 * x);
            else return std::exp(-x) * (x * (dcsevl(x*x/4.5-1., bi1cs, nti1) + 0.875));
        } else if (x <= 8.) {
            return (dcsevl((48./x-11.)/5., ai1cs, ntai1) + 0.375) / std::sqrt(x);
        } else {
            return (dcsevl(16./x-1., ai12cs, ntai12) + 0.375) / std::sqrt(x);
        }
    }

    double dasyik(double x, double fnu, bool is_i)
    {
        // ***BEGIN PROLOGUE  DASYIK
        // ***SUBSIDIARY
        // ***PURPOSE  Subsidiary to DBESI and DBESK
        // ***LIBRARY   SLATEC
        // ***TYPE      DOUBLE PRECISION (ASYIK-S, DASYIK-D)
        // ***AUTHOR  Amos, D. E., (SNLA)
        // ***DESCRIPTION
        //
        //                    DASYIK computes Bessel functions I and K
        //                  for arguments X.GT.0.0 and orders FNU.GE.35
        //                  on FLGIK = 1 and FLGIK = -1 respectively.
        //
        //                                    INPUT
        //
        //      X    - Argument, X.GT.0.0D0
        //      FNU  - Order of first Bessel function
        //      KODE - A parameter to indicate the scaling option
        //             KODE=1 returns Y(I)=        I/SUB(FNU+I-1)/(X), I=1,IN
        //                    or      Y(I)=        K/SUB(FNU+I-1)/(X), I=1,IN
        //                    on FLGIK = 1.0D0 or FLGIK = -1.0D0
        //             KODE=2 returns Y(I)=EXP(-X)*I/SUB(FNU+I-1)/(X), I=1,IN
        //                    or      Y(I)=EXP( X)*K/SUB(FNU+I-1)/(X), I=1,IN
        //                    on FLGIK = 1.0D0 or FLGIK = -1.0D0
        //     FLGIK - Selection parameter for I or K FUNCTION
        //             FLGIK =  1.0D0 gives the I function
        //             FLGIK = -1.0D0 gives the K function
        //        RA - SQRT(1.+Z*Z), Z=X/FNU
        //       ARG - Argument of the leading exponential
        //        IN - Number of functions desired, IN=1 or 2
        //
        //                                    OUTPUT
        //
        //         Y - A vector whose first IN components contain the sequence
        //
        //     Abstract  **** A double precision routine ****
        //         DASYIK implements the uniform asymptotic expansion of
        //         the I and K Bessel functions for FNU.GE.35 and real
        //         X.GT.0.0D0. The forms are identical except for a change
        //         in sign of some of the terms. This change in sign is
        //         accomplished by means of the FLAG FLGIK = 1 or -1.
        //
        // ***SEE ALSO  DBESI, DBESK
        // ***ROUTINES CALLED  D1MACH
        // ***REVISION HISTORY  (YYMMDD)
        //    750101  DATE WRITTEN
        //    890531  Changed all specific intrinsics to generic.  (WRB)
        //    890911  Removed unnecessary intrinsics.  (WRB)
        //    891214  Prologue converted to Version 4.0 format.  (BAB)
        //    900328  Added TYPE section.  (WRB)
        //    910408  Updated the AUTHOR section.  (WRB)
        //    170203  Converted to C++, and modified to only cover in==1 option. (MJ)
        // ***END PROLOGUE  DASYIK

        const double con[2] = { 0.398942280401432678, 1.25331413731550025 };
        const double c[65] = {
            -0.208333333333333, 0.125, 0.334201388888889,
            -0.401041666666667, 0.0703125, -1.02581259645062, 1.84646267361111,
            -0.8912109375, 0.0732421875, 4.66958442342625, -11.207002616223,
            8.78912353515625, -2.3640869140625, 0.112152099609375,
            -28.2120725582002, 84.6362176746007, -91.81824154324,
            42.5349987453885, -7.36879435947963, 0.227108001708984,
            212.570130039217, -765.252468141182, 1059.990452528,
            -699.579627376133, 218.190511744212, -26.4914304869516,
            0.572501420974731, -1919.45766231841, 8061.72218173731,
            -13586.5500064341, 11655.3933368645, -5305.6469786134,
            1200.90291321635, -108.090919788395, 1.72772750258446,
            20204.2913309661, -96980.5983886375, 192547.001232532,
            -203400.177280416, 122200.464983017, -41192.6549688976,
            7109.51430248936, -493.915304773088, 6.07404200127348,
            -242919.187900551, 1311763.61466298, -2998015.91853811,
            3763271.2976564, -2813563.22658653, 1268365.27332162,
            -331645.172484564, 45218.7689813627, -2499.83048181121,
            24.3805296995561, 3284469.85307204, -19706819.1184322,
            50952602.4926646, -74105148.2115327, 66344512.274729,
            -37567176.6607634, 13288767.1664218, -2785618.12808645,
            308186.404612662, -13886.089753717, 110.017140269247
        };

        const double tol = std::max(std::numeric_limits<double>::epsilon(), 1e-15);

        double flgik = is_i ? 1. : -1.;
        double fn = fnu;
        int kk = is_i ? 0 : 1;
        double z = x / fn;
        double ra = std::sqrt(z * z + 1.);
        double gln = std::log((ra + 1.) / z);
        double arg = fn * (ra - gln) * flgik;
        double coef = std::exp(arg);
        double t = 1. / ra;
        double t2 = t * t;
        t /= fn;
        t = std::copysign(t, flgik);
        double s2 = 1.;
        double ap = 1.;
        int l = 0;
        for (int k = 2; k <= 11; ++k) {
            double s1 = c[l++];
            for (int j=2; j<=k; ++j) {
                s1 = s1 * t2 + c[l++];
            }
            ap *= t;
            double ak = ap * s1;
            s2 += ak;
            if (std::max(std::abs(ak), std::abs(ap)) < tol) break;
        }
        t = std::abs(t);
        return s2 * coef * std::sqrt(t) * con[kk];
    }

    //
    // K_nu(x)
    //

    double dbesk(double x, double fnu)
    {
        // ***BEGIN PROLOGUE  DBESK
        // ***PURPOSE  Implement forward recursion on the three term recursion
        //            relation for a sequence of non-negative order Bessel
        //            functions K/SUB(FNU+I-1)/(X), or scaled Bessel functions
        //            EXP(X)*K/SUB(FNU+I-1)/(X), I=1,...,N for real, positive
        //            X and non-negative orders FNU.
        // ***LIBRARY   SLATEC
        // ***CATEGORY  C10B3
        // ***TYPE      DOUBLE PRECISION (BESK-S, DBESK-D)
        // ***KEYWORDS  K BESSEL FUNCTION, SPECIAL FUNCTIONS
        // ***AUTHOR  Amos, D. E., (SNLA)
        // ***DESCRIPTION
        //
        //     Abstract  **** a double precision routine ****
        //         DBESK implements forward recursion on the three term
        //         recursion relation for a sequence of non-negative order Bessel
        //         functions K/sub(FNU+I-1)/(X), or scaled Bessel functions
        //         EXP(X)*K/sub(FNU+I-1)/(X), I=1,..,N for real X .GT. 0.0D0 and
        //         non-negative orders FNU.  If FNU .LT. NULIM, orders FNU and
        //         FNU+1 are obtained from DBSKNU to start the recursion.  If
        //         FNU .GE. NULIM, the uniform asymptotic expansion is used for
        //         orders FNU and FNU+1 to start the recursion.  NULIM is 35 or
        //         70 depending on whether N=1 or N .GE. 2.  Under and overflow
        //         tests are made on the leading term of the asymptotic expansion
        //         before any extensive computation is done.
        //
        //         The maximum number of significant digits obtainable
        //         is the smaller of 14 and the number of digits carried in
        //         double precision arithmetic.
        //
        //     Description of Arguments
        //
        //         Input      X,FNU are double precision
        //           X      - X .GT. 0.0D0
        //           FNU    - order of the initial K function, FNU .GE. 0.0D0
        //           KODE   - a parameter to indicate the scaling option
        //                    KODE=1 returns Y(I)=       K/sub(FNU+I-1)/(X),
        //                                        I=1,...,N
        //                    KODE=2 returns Y(I)=EXP(X)*K/sub(FNU+I-1)/(X),
        //                                        I=1,...,N
        //           N      - number of members in the sequence, N .GE. 1
        //
        //         Output     Y is double precision
        //           Y      - a vector whose first N components contain values
        //                    for the sequence
        //                    Y(I)=       k/sub(FNU+I-1)/(X), I=1,...,N  or
        //                    Y(I)=EXP(X)*K/sub(FNU+I-1)/(X), I=1,...,N
        //                    depending on KODE
        //           NZ     - number of components of Y set to zero due to
        //                    underflow with KODE=1,
        //                    NZ=0   , normal return, computation completed
        //                    NZ .NE. 0, first NZ components of Y set to zero
        //                             due to underflow, Y(I)=0.0D0, I=1,...,NZ
        //
        //     Error Conditions
        //         Improper input arguments - a fatal error
        //         Overflow - a fatal error
        //         Underflow with KODE=1 -  a non-fatal error (NZ .NE. 0)
        //
        // ***REFERENCES  F. W. J. Olver, Tables of Bessel Functions of Moderate
        //                 or Large Orders, NPL Mathematical Tables 6, Her
        //                 Majesty's Stationery Office, London, 1962.
        //               N. M. Temme, On the numerical evaluation of the modified
        //                 Bessel function of the third kind, Journal of
        //                 Computational Physics 19, (1975), pp. 324-337.
        // ***ROUTINES CALLED  D1MACH, DASYIK, DBESK0, DBESK1, DBSK0E, DBSK1E,
        //                    DBSKNU, I1MACH, XERMSG
        // ***REVISION HISTORY  (YYMMDD)
        //    790201  DATE WRITTEN
        //    890531  Changed all specific intrinsics to generic.  (WRB)
        //    890911  Removed unnecessary intrinsics.  (WRB)
        //    890911  REVISION DATE from Version 3.2
        //    891214  Prologue converted to Version 4.0 format.  (BAB)
        //    900315  CALLs to XERROR changed to CALLs to XERMSG.  (THJ)
        //    920501  Reformatted the REFERENCES section.  (WRB)
        //    170203  Converted to C++, and modified to only cover n==1 option. (MJ)
        // ***END PROLOGUE  DBESK

        const int nulim[2] = { 35,70 };

        const double elim = -std::log(std::numeric_limits<double>::min() * 1.e3);
        const double xlim = std::numeric_limits<double>::min() * 1.e3;

        assert(fnu >= 0.);
        assert(x > 0.);

        if (x < xlim)
            throw std::runtime_error("DBESK OVERFLOW, FNU OR N TOO LARGE OR X TOO SMALL");

        int nud = int(fnu);
        double dnu = fnu - nud;
        double fn = fnu;
        double fnn = fn;

        if (fnu == 0.) return dbesk0(x);
        else if (fnu == 1.) return dbesk1(x);
        else if (fnu < 2.) {
            //     UNDERFLOW TEST FOR KODE=1
            if (x > elim) return 0.;
            //     OVERFLOW TEST
            if (fnu > 1.  && -fnu * (std::log(x) - 0.693) > elim) {
                throw std::runtime_error("DBESK OVERFLOW, FNU OR N TOO LARGE OR X TOO SMALL");
            }
            double knu;
            dbsknu(x, fnu, 1, &knu);
            return knu;
        } else {
            //     OVERFLOW TEST  (LEADING EXPONENTIAL OF ASYMPTOTIC EXPANSION)
            //     FOR THE LAST ORDER, FNU+N-1.GE.NULIM
            double zn = x / fnu;
            if (zn == 0.) {
                throw std::runtime_error("DBESK OVERFLOW, FNU OR N TOO LARGE OR X TOO SMALL");
            }
            double rtz = std::sqrt(zn * zn + 1.);
            double gln = std::log((rtz + 1.) / zn);
            double t = rtz;
            double cn = -fn * (t - gln);
            if (cn > elim) {
                throw std::runtime_error("DBESK OVERFLOW, FNU OR N TOO LARGE OR X TOO SMALL");
            }
            if (nud > nulim[0]) {
                if (cn < -elim) return 0.;
                //     ASYMPTOTIC EXPANSION FOR ORDERS FNU AND FNU+1.GE.NULIM
                return dasyik(x, fnu, false);
            }

            //     UNDERFLOW TEST (LEADING EXPONENTIAL OF ASYMPTOTIC EXPANSION IN X)
            //     FOR ORDER DNU

            if (x > elim) return 0.;

            double s1,s2;
            if (dnu == 0.) {
                s1 = dbesk0(x);
                s2 = dbesk1(x);
            } else {
                double w[2];
                dbsknu(x, dnu, 2, w);
                s1 = w[0];
                s2 = w[1];
            }
            double trx = 2. / x;
            double tm = (dnu + dnu + 2.) / x;
            //     FORWARD RECUR FROM DNU TO FNU+1 TO GET Y(1) AND Y(2)
            for (int i=1; i<nud; ++i) {
                double s = s2;
                s2 = tm * s2 + s1;
                s1 = s;
                tm += trx;
            }
            return s2;
        }
    }

    double dbesk0(double x)
    {
        // ***BEGIN PROLOGUE  DBESK0
        // ***PURPOSE  Compute the modified (hyperbolic) Bessel function of the
        //            third kind of order zero.
        // ***LIBRARY   SLATEC (FNLIB)
        // ***CATEGORY  C10B1
        // ***TYPE      DOUBLE PRECISION (BESK0-S, DBESK0-D)
        // ***KEYWORDS  FNLIB, HYPERBOLIC BESSEL FUNCTION,
        //             MODIFIED BESSEL FUNCTION, ORDER ZERO, SPECIAL FUNCTIONS,
        //             THIRD KIND
        // ***AUTHOR  Fullerton, W., (LANL)
        // ***DESCRIPTION
        //
        // DBESK0(X) calculates the double precision modified (hyperbolic)
        // Bessel function of the third kind of order zero for double
        // precision argument X.  The argument must be greater than zero
        // but not so large that the result underflows.
        //
        // Series for BK0        on the interval  0.          to  4.00000E+00
        //                                        with weighted error   3.08E-33
        //                                         log weighted error  32.51
        //                               significant figures required  32.05
        //                                    decimal places required  33.11
        //
        // ***REFERENCES  (NONE)
        // ***ROUTINES CALLED  D1MACH, DBESI0, DBSK0E, DCSEVL, INITDS, XERMSG
        // ***REVISION HISTORY  (YYMMDD)
        //    770701  DATE WRITTEN
        //    890531  Changed all specific intrinsics to generic.  (WRB)
        //    890531  REVISION DATE from Version 3.2
        //    891214  Prologue converted to Version 4.0 format.  (BAB)
        //    900315  CALLs to XERROR changed to CALLs to XERMSG.  (THJ)
        // ***END PROLOGUE  DBESK0
        
        const double bk0cs[16] = {
            -0.0353273932339027687201140060063153,
            0.344289899924628486886344927529213,
            0.0359799365153615016265721303687231,
            0.00126461541144692592338479508673447,
            2.28621210311945178608269830297585e-5,
            2.53479107902614945730790013428354e-7,
            1.90451637722020885897214059381366e-9,
            1.03496952576336245851008317853089e-11,
            4.25981614279108257652445327170133e-14,
            1.3744654358807508969423832544e-16,
            3.57089652850837359099688597333333e-19,
            7.63164366011643737667498666666666e-22,
            1.36542498844078185908053333333333e-24,
            2.07527526690666808319999999999999e-27,
            2.7128142180729856e-30,
            3.08259388791466666666666666666666e-33
        };
        const int ntk0 = 10;
        const double xsml = std::sqrt(std::numeric_limits<double>::epsilon() * 4.);
        const double xmaxt = -std::log(std::numeric_limits<double>::min());
        const double xmax = xmaxt * (1. - 0.5 * std::log(xmaxt) / (xmaxt + 0.5));

        assert(x > 0);
        if (x <= 2.) {
            double y = (x > xsml) ? x*x : 0.;
            return -std::log(0.5*x) * dbesi0(x) - 0.25 + dcsevl(0.5*y-1., bk0cs, ntk0);
        } else {
            if (x > xmax) return 0.;
            else return std::exp(-x) * dbsk0e(x);
        }
    }

    double dbsk0e(double x)
    {
        // ***BEGIN PROLOGUE  DBSK0E
        // ***PURPOSE  Compute the exponentially scaled modified (hyperbolic)
        //            Bessel function of the third kind of order zero.
        // ***LIBRARY   SLATEC (FNLIB)
        // ***CATEGORY  C10B1
        // ***TYPE      DOUBLE PRECISION (BESK0E-S, DBSK0E-D)
        // ***KEYWORDS  EXPONENTIALLY SCALED, FNLIB, HYPERBOLIC BESSEL FUNCTION,
        //             MODIFIED BESSEL FUNCTION, ORDER ZERO, SPECIAL FUNCTIONS,
        //             THIRD KIND
        // ***AUTHOR  Fullerton, W., (LANL)
        // ***DESCRIPTION
        //
        // DBSK0E(X) computes the double precision exponentially scaled
        // modified (hyperbolic) Bessel function of the third kind of
        // order zero for positive double precision argument X.
        //
        // Series for BK0        on the interval  0.          to  4.00000E+00
        //                                        with weighted error   3.08E-33
        //                                         log weighted error  32.51
        //                               significant figures required  32.05
        //                                    decimal places required  33.11
        //
        // Series for AK0        on the interval  1.25000E-01 to  5.00000E-01
        //                                        with weighted error   2.85E-32
        //                                         log weighted error  31.54
        //                               significant figures required  30.19
        //                                    decimal places required  32.33
        //
        // Series for AK02       on the interval  0.          to  1.25000E-01
        //                                        with weighted error   2.30E-32
        //                                         log weighted error  31.64
        //                               significant figures required  29.68
        //                                    decimal places required  32.40
        //
        // ***REFERENCES  (NONE)
        // ***ROUTINES CALLED  D1MACH, DBESI0, DCSEVL, INITDS, XERMSG
        // ***REVISION HISTORY  (YYMMDD)
        //    770701  DATE WRITTEN
        //    890531  Changed all specific intrinsics to generic.  (WRB)
        //    890531  REVISION DATE from Version 3.2
        //    891214  Prologue converted to Version 4.0 format.  (BAB)
        //    900315  CALLs to XERROR changed to CALLs to XERMSG.  (THJ)
        //    170203  Converted to C++. (MJ)
        // ***END PROLOGUE  DBSK0E

        const double bk0cs[16] = {
            -0.0353273932339027687201140060063153,
            0.344289899924628486886344927529213,
            0.0359799365153615016265721303687231,
            0.00126461541144692592338479508673447,
            2.28621210311945178608269830297585e-5,
            2.53479107902614945730790013428354e-7,
            1.90451637722020885897214059381366e-9,
            1.03496952576336245851008317853089e-11,
            4.25981614279108257652445327170133e-14,
            1.3744654358807508969423832544e-16,
            3.57089652850837359099688597333333e-19,
            7.63164366011643737667498666666666e-22,
            1.36542498844078185908053333333333e-24,
            2.07527526690666808319999999999999e-27,
            2.7128142180729856e-30,
            3.08259388791466666666666666666666e-33
        };
        const double ak0cs[38] = {
            -0.07643947903327941424082978270088,
            -0.02235652605699819052023095550791,
            7.734181154693858235300618174047e-4,
            -4.281006688886099464452146435416e-5,
            3.08170017386297474365001482666e-6,
            -2.639367222009664974067448892723e-7,
            2.563713036403469206294088265742e-8,
            -2.742705549900201263857211915244e-9,
            3.169429658097499592080832873403e-10,
            -3.902353286962184141601065717962e-11,
            5.068040698188575402050092127286e-12,
            -6.889574741007870679541713557984e-13,
            9.744978497825917691388201336831e-14,
            -1.427332841884548505389855340122e-14,
            2.156412571021463039558062976527e-15,
            -3.34965425514956277218878205853e-16,
            5.335260216952911692145280392601e-17,
            -8.693669980890753807639622378837e-18,
            1.446404347862212227887763442346e-18,
            -2.452889825500129682404678751573e-19,
            4.2337545262321715728217063424e-20,
            -7.427946526454464195695341294933e-21,
            1.3231505293926668662779674624e-21,
            -2.390587164739649451335981465599e-22,
            4.376827585923226140165712554666e-23,
            -8.113700607345118059339011413333e-24,
            1.521819913832172958310378154666e-24,
            -2.886041941483397770235958613333e-25,
            5.530620667054717979992610133333e-26,
            -1.070377329249898728591633066666e-26,
            2.091086893142384300296328533333e-27,
            -4.121713723646203827410261333333e-28,
            8.19348397112130764013568e-29,
            -1.642000275459297726780757333333e-29,
            3.316143281480227195890346666666e-30,
            -6.746863644145295941085866666666e-31,
            1.382429146318424677635413333333e-31,
            -2.851874167359832570811733333333e-32
        };
        const double ak02cs[33] = {
            -0.01201869826307592239839346212452,
            -0.009174852691025695310652561075713,
            1.444550931775005821048843878057e-4,
            -4.013614175435709728671021077879e-6,
            1.567831810852310672590348990333e-7,
            -7.77011043852173771031579975446e-9,
            4.611182576179717882533130529586e-10,
            -3.158592997860565770526665803309e-11,
            2.435018039365041127835887814329e-12,
            -2.074331387398347897709853373506e-13,
            1.925787280589917084742736504693e-14,
            -1.927554805838956103600347182218e-15,
            2.062198029197818278285237869644e-16,
            -2.341685117579242402603640195071e-17,
            2.805902810643042246815178828458e-18,
            -3.530507631161807945815482463573e-19,
            4.645295422935108267424216337066e-20,
            -6.368625941344266473922053461333e-21,
            9.0695213109865155676223488e-22,
            -1.337974785423690739845005311999e-22,
            2.03983602185995231552208896e-23,
            -3.207027481367840500060869973333e-24,
            5.189744413662309963626359466666e-25,
            -8.629501497540572192964607999999e-26,
            1.4721611831025598552080384e-26,
            -2.573069023867011283812351999999e-27,
            4.60177408664351658737664e-28,
            -8.411555324201093737130666666666e-29,
            1.569806306635368939301546666666e-29,
            -2.988226453005757788979199999999e-30,
            5.796831375216836520618666666666e-31,
            -1.145035994347681332155733333333e-31,
            2.301266594249682802005333333333e-32
        };

        const int ntk0 = 11;
        const int ntak0 = 18;
        const int ntak02 = 14;
        const double xsml = 2. * std::sqrt(std::numeric_limits<double>::epsilon());

        assert(x > 0.);
        if (x <= 2.) {
            double y = (x > xsml) ? x*x : 0.;
            return std::exp(x) * (-std::log(0.5*x) * dbesi0(x) - 0.25 +
                                  dcsevl(0.5*y-1., bk0cs, ntk0));
        } else if (x <= 8.) {
            return (dcsevl((16./x-5.)/3., ak0cs, ntak0) + 1.25) / std::sqrt(x);
        } else {
            return (dcsevl(16./x-1., ak02cs, ntak02) + 1.25) / std::sqrt(x);
        }
    }

    double dbesk1(double x)
    {
        // ***BEGIN PROLOGUE  DBESK1
        // ***PURPOSE  Compute the modified (hyperbolic) Bessel function of the
        //            third kind of order one.
        // ***LIBRARY   SLATEC (FNLIB)
        // ***CATEGORY  C10B1
        // ***TYPE      DOUBLE PRECISION (BESK1-S, DBESK1-D)
        // ***KEYWORDS  FNLIB, HYPERBOLIC BESSEL FUNCTION,
        //             MODIFIED BESSEL FUNCTION, ORDER ONE, SPECIAL FUNCTIONS,
        //             THIRD KIND
        // ***AUTHOR  Fullerton, W., (LANL)
        // ***DESCRIPTION
        //
        // DBESK1(X) calculates the double precision modified (hyperbolic)
        // Bessel function of the third kind of order one for double precision
        // argument X.  The argument must be large enough that the result does
        // not overflow and small enough that the result does not underflow.
        //
        // Series for BK1        on the interval  0.          to  4.00000E+00
        //                                        with weighted error   9.16E-32
        //                                         log weighted error  31.04
        //                               significant figures required  30.61
        //                                    decimal places required  31.64
        //
        // ***REFERENCES  (NONE)
        // ***ROUTINES CALLED  D1MACH, DBESI1, DBSK1E, DCSEVL, INITDS, XERMSG
        // ***REVISION HISTORY  (YYMMDD)
        //    770701  DATE WRITTEN
        //    890531  Changed all specific intrinsics to generic.  (WRB)
        //    890531  REVISION DATE from Version 3.2
        //    891214  Prologue converted to Version 4.0 format.  (BAB)
        //    900315  CALLs to XERROR changed to CALLs to XERMSG.  (THJ)
        // ***END PROLOGUE  DBESK1

        const double bk1cs[16] = {
            0.025300227338947770532531120868533,
            -0.35315596077654487566723831691801,
            -0.12261118082265714823479067930042,
            -0.0069757238596398643501812920296083,
            -1.7302889575130520630176507368979e-4,
            -2.4334061415659682349600735030164e-6,
            -2.2133876307347258558315252545126e-8,
            -1.4114883926335277610958330212608e-10,
            -6.6669016941993290060853751264373e-13,
            -2.4274498505193659339263196864853e-15,
            -7.023863479386287597178379712e-18,
            -1.6543275155100994675491029333333e-20,
            -3.2338347459944491991893333333333e-23,
            -5.3312750529265274999466666666666e-26,
            -7.5130407162157226666666666666666e-29,
            -9.1550857176541866666666666666666e-32
        };
        const int ntk1 = 11;
        const double xmin = 1.01 * std::numeric_limits<double>::min();
        const double xsml = 2. * std::sqrt(std::numeric_limits<double>::epsilon());
        const double xmaxt = -std::log(std::numeric_limits<double>::min());
        const double xmax = xmaxt * (1. - 0.5 * std::log(xmaxt) / (xmaxt + 0.5));
 
        assert(x > 0.);
        if (x <= 2.) {
            if (x < xmin) 
                throw std::runtime_error("DBESK1 X SO SMALL K1 OVERFLOWS");
            double y = (x > xsml) ? x*x : 0.;
            return log(0.5*x) * dbesi1(x) + (dcsevl(0.5*y-1., bk1cs, ntk1) + 0.75) / x;
        } else {
            if (x > xmax) return 0.;
            else return exp(-x) * dbsk1e(x);
        }
    }

    double dbsk1e(double x)
    {
        // ***BEGIN PROLOGUE  DBSK1E
        // ***PURPOSE  Compute the exponentially scaled modified (hyperbolic)
        //            Bessel function of the third kind of order one.
        // ***LIBRARY   SLATEC (FNLIB)
        // ***CATEGORY  C10B1
        // ***TYPE      DOUBLE PRECISION (BESK1E-S, DBSK1E-D)
        // ***KEYWORDS  EXPONENTIALLY SCALED, FNLIB, HYPERBOLIC BESSEL FUNCTION,
        //             MODIFIED BESSEL FUNCTION, ORDER ONE, SPECIAL FUNCTIONS,
        //             THIRD KIND
        // ***AUTHOR  Fullerton, W., (LANL)
        // ***DESCRIPTION
        //
        // DBSK1E(S) computes the double precision exponentially scaled
        // modified (hyperbolic) Bessel function of the third kind of order
        // one for positive double precision argument X.
        //
        // Series for BK1        on the interval  0.          to  4.00000E+00
        //                                        with weighted error   9.16E-32
        //                                         log weighted error  31.04
        //                               significant figures required  30.61
        //                                    decimal places required  31.64
        //
        // Series for AK1        on the interval  1.25000E-01 to  5.00000E-01
        //                                        with weighted error   3.07E-32
        //                                         log weighted error  31.51
        //                               significant figures required  30.71
        //                                    decimal places required  32.30
        //
        // Series for AK12       on the interval  0.          to  1.25000E-01
        //                                        with weighted error   2.41E-32
        //                                         log weighted error  31.62
        //                               significant figures required  30.25
        //                                    decimal places required  32.38
        //
        // ***REFERENCES  (NONE)
        // ***ROUTINES CALLED  D1MACH, DBESI1, DCSEVL, INITDS, XERMSG
        // ***REVISION HISTORY  (YYMMDD)
        //    770701  DATE WRITTEN
        //    890531  Changed all specific intrinsics to generic.  (WRB)
        //    890531  REVISION DATE from Version 3.2
        //    891214  Prologue converted to Version 4.0 format.  (BAB)
        //    900315  CALLs to XERROR changed to CALLs to XERMSG.  (THJ)
        //    170203  Converted to C++. (MJ)
        // ***END PROLOGUE  DBSK1E

        const double bk1cs[16] = {
            0.025300227338947770532531120868533,
            -0.35315596077654487566723831691801,
            -0.12261118082265714823479067930042,
            -0.0069757238596398643501812920296083,
            -1.7302889575130520630176507368979e-4,
            -2.4334061415659682349600735030164e-6,
            -2.2133876307347258558315252545126e-8,
            -1.4114883926335277610958330212608e-10,
            -6.6669016941993290060853751264373e-13,
            -2.4274498505193659339263196864853e-15,
            -7.023863479386287597178379712e-18,
            -1.6543275155100994675491029333333e-20,
            -3.2338347459944491991893333333333e-23,
            -5.3312750529265274999466666666666e-26,
            -7.5130407162157226666666666666666e-29,
            -9.1550857176541866666666666666666e-32
        };
        const double ak1cs[38] = {
            0.27443134069738829695257666227266,
            0.07571989953199367817089237814929,
            -0.0014410515564754061229853116175625,
            6.6501169551257479394251385477036e-5,
            -4.3699847095201407660580845089167e-6,
            3.5402774997630526799417139008534e-7,
            -3.3111637792932920208982688245704e-8,
            3.4459775819010534532311499770992e-9,
            -3.8989323474754271048981937492758e-10,
            4.7208197504658356400947449339005e-11,
            -6.047835662875356234537359156289e-12,
            8.1284948748658747888193837985663e-13,
            -1.1386945747147891428923915951042e-13,
            1.654035840846228232597294820509e-14,
            -2.4809025677068848221516010440533e-15,
            3.8292378907024096948429227299157e-16,
            -6.0647341040012418187768210377386e-17,
            9.8324256232648616038194004650666e-18,
            -1.6284168738284380035666620115626e-18,
            2.7501536496752623718284120337066e-19,
            -4.7289666463953250924281069568e-20,
            8.2681500028109932722392050346666e-21,
            -1.4681405136624956337193964885333e-21,
            2.6447639269208245978085894826666e-22,
            -4.82901575648563878979698688e-23,
            8.9293020743610130180656332799999e-24,
            -1.6708397168972517176997751466666e-24,
            3.1616456034040694931368618666666e-25,
            -6.0462055312274989106506410666666e-26,
            1.1678798942042732700718421333333e-26,
            -2.277374158265399623286784e-27,
            4.4811097300773675795305813333333e-28,
            -8.8932884769020194062336e-29,
            1.7794680018850275131392e-29,
            -3.5884555967329095821994666666666e-30,
            7.2906290492694257991679999999999e-31,
            -1.4918449845546227073024e-31,
            3.0736573872934276300799999999999e-32
        };
        const double ak12cs[33] = {
            0.06379308343739001036600488534102,
            0.02832887813049720935835030284708,
            -2.475370673905250345414545566732e-4,
            5.771972451607248820470976625763e-6,
            -2.068939219536548302745533196552e-7,
            9.739983441381804180309213097887e-9,
            -5.585336140380624984688895511129e-10,
            3.732996634046185240221212854731e-11,
            -2.825051961023225445135065754928e-12,
            2.372019002484144173643496955486e-13,
            -2.176677387991753979268301667938e-14,
            2.157914161616032453939562689706e-15,
            -2.290196930718269275991551338154e-16,
            2.582885729823274961919939565226e-17,
            -3.07675264126846318762109817344e-18,
            3.851487721280491597094896844799e-19,
            -5.0447948976415289771172825088e-20,
            6.888673850418544237018292223999e-21,
            -9.77504154195011830300213248e-22,
            1.437416218523836461001659733333e-22,
            -2.185059497344347373499733333333e-23,
            3.4262456218092206316453888e-24,
            -5.531064394246408232501248e-25,
            9.176601505685995403782826666666e-26,
            -1.562287203618024911448746666666e-26,
            2.725419375484333132349439999999e-27,
            -4.865674910074827992378026666666e-28,
            8.879388552723502587357866666666e-29,
            -1.654585918039257548936533333333e-29,
            3.145111321357848674303999999999e-30,
            -6.092998312193127612416e-31,
            1.202021939369815834623999999999e-31,
            -2.412930801459408841386666666666e-32
        };

        const int ntk1 = 11;
        const int ntak1 = 18;
        const int ntak12 = 14;
        const double xmin = 1.01 * std::numeric_limits<double>::min();
        const double xsml = 2. * std::sqrt(std::numeric_limits<double>::epsilon());

        assert(x > 0.);
        if (x <= 2.) {
            if (x < xmin)
                throw std::runtime_error("DBSK1E X SO SMALL K1 OVERFLOWS");
            double y = (x > xsml) ? x*x : 0.;
            return std::exp(x) * (std::log(0.5*x) * dbesi1(x) +
                                  (0.75 + dcsevl(0.5*y-1., bk1cs, ntk1)) / x);
        } else if (x <= 8.) {
            return (dcsevl((16./x-5.)/3., ak1cs, ntak1) + 1.25) / std::sqrt(x);
        } else {
            return (dcsevl(16./x-1., ak12cs, ntak12) + 1.25) / std::sqrt(x);
        }
    }

    void dbsknu(double x, double fnu, int n, double *y)
    {
        // ***BEGIN PROLOGUE  DBSKNU
        // ***SUBSIDIARY
        // ***PURPOSE  Subsidiary to DBESK
        // ***LIBRARY   SLATEC
        // ***TYPE      DOUBLE PRECISION (BESKNU-S, DBSKNU-D)
        // ***AUTHOR  Amos, D. E., (SNLA)
        // ***DESCRIPTION
        //
        //     Abstract  **** A DOUBLE PRECISION routine ****
        //         DBSKNU computes N member sequences of K Bessel functions
        //         K/SUB(FNU+I-1)/(X), I=1,N for non-negative orders FNU and
        //         positive X. Equations of the references are implemented on
        //         small orders DNU for K/SUB(DNU)/(X) and K/SUB(DNU+1)/(X).
        //         Forward recursion with the three term recursion relation
        //         generates higher orders FNU+I-1, I=1,...,N. The parameter
        //         KODE permits K/SUB(FNU+I-1)/(X) values or scaled values
        //         EXP(X)*K/SUB(FNU+I-1)/(X), I=1,N to be returned.
        //
        //         To start the recursion FNU is normalized to the interval
        //         -0.5.LE.DNU.LT.0.5. A special form of the power series is
        //         implemented on 0.LT.X.LE.X1 while the Miller algorithm for the
        //         K Bessel function in terms of the confluent hypergeometric
        //         function U(FNU+0.5,2*FNU+1,X) is implemented on X1.LT.X.LE.X2.
        //         For X.GT.X2, the asymptotic expansion for large X is used.
        //         When FNU is a half odd integer, a special formula for
        //         DNU=-0.5 and DNU+1.0=0.5 is used to start the recursion.
        //
        //         The maximum number of significant digits obtainable
        //         is the smaller of 14 and the number of digits carried in
        //         DOUBLE PRECISION arithmetic.
        //
        //         DBSKNU assumes that a significant digit SINH function is
        //         available.
        //
        //     Description of Arguments
        //
        //         INPUT      X,FNU are DOUBLE PRECISION
        //           X      - X.GT.0.0D0
        //           FNU    - Order of initial K function, FNU.GE.0.0D0
        //           N      - Number of members of the sequence, N.GE.1
        //           KODE   - A parameter to indicate the scaling option
        //                    KODE= 1  returns
        //                             Y(I)=       K/SUB(FNU+I-1)/(X)
        //                                  I=1,...,N
        //                        = 2  returns
        //                             Y(I)=EXP(X)*K/SUB(FNU+I-1)/(X)
        //                                  I=1,...,N
        //
        //         OUTPUT     Y is DOUBLE PRECISION
        //           Y      - A vector whose first N components contain values
        //                    for the sequence
        //                    Y(I)=       K/SUB(FNU+I-1)/(X), I=1,...,N or
        //                    Y(I)=EXP(X)*K/SUB(FNU+I-1)/(X), I=1,...,N
        //                    depending on KODE
        //           NZ     - Number of components set to zero due to
        //                    underflow,
        //                    NZ= 0   , normal return
        //                    NZ.NE.0 , first NZ components of Y set to zero
        //                              due to underflow, Y(I)=0.0D0,I=1,...,NZ
        //
        //     Error Conditions
        //         Improper input arguments - a fatal error
        //         Overflow - a fatal error
        //         Underflow with KODE=1 - a non-fatal error (NZ.NE.0)
        //
        // ***SEE ALSO  DBESK
        // ***REFERENCES  N. M. Temme, On the numerical evaluation of the modified
        //                 Bessel function of the third kind, Journal of
        //                 Computational Physics 19, (1975), pp. 324-337.
        // ***ROUTINES CALLED  D1MACH, DGAMMA, I1MACH, XERMSG
        // ***REVISION HISTORY  (YYMMDD)
        //    790201  DATE WRITTEN
        //    890531  Changed all specific intrinsics to generic.  (WRB)
        //    890911  Removed unnecessary intrinsics.  (WRB)
        //    891214  Prologue converted to Version 4.0 format.  (BAB)
        //    900315  CALLs to XERROR changed to CALLs to XERMSG.  (THJ)
        //    900326  Removed duplicate information from DESCRIPTION section.
        //            (WRB)
        //    900328  Added TYPE section.  (WRB)
        //    900727  Added EXTERNAL statement.  (WRB)
        //    910408  Updated the AUTHOR and REFERENCES sections.  (WRB)
        //    920501  Reformatted the REFERENCES section.  (WRB)
        // ***END PROLOGUE  DBSKNU

        const double x1 = 2.;
        const double x2 = 17.;
        const double pi = 3.14159265358979;
        const double rthpi = 1.2533141373155;
        const double cc[8] = {
            0.577215664901533, -0.0420026350340952,
            -.0421977345555443, 0.007218943246663, -2.152416741149e-4,
            -2.01348547807e-5, 1.133027232e-6, 6.116095e-9
        };

        const double elim = -std::log(std::numeric_limits<double>::min() * 1.e3);
        const double tol = std::max(std::numeric_limits<double>::epsilon(), 1e-15);

        assert(x > 0.);
        assert(fnu >= 0.);
        assert(n >= 1);

        bool iflag = false;
        double rx = 2. / x;
        int inu = int(fnu + .5);
        double dnu = fnu - inu;
        double dnu2 = (std::abs(dnu) >= tol) ? dnu * dnu : 0.;

        bool recurse = true;
        double ck, s1, s2;
        if (std::abs(dnu) == 0.5 || x > x1) {
            double coef = rthpi / std::sqrt(x);

            if (x > elim) iflag = true;
            else coef *= std::exp(-x);

            if (std::abs(dnu) == .5) {
                //     FNU=HALF ODD INTEGER CASE
                s1 = coef;
                s2 = coef;
            } else if (x > x2) {
                //     ASYMPTOTIC EXPANSION FOR LARGE X, X.GT.X2

                //     IFLAG=0 MEANS NO UNDERFLOW OCCURRED
                //     IFLAG=1 MEANS AN UNDERFLOW OCCURRED- COMPUTATION PROCEEDS WITH
                //     KODED=2 AND A TEST FOR ON SCALE VALUES IS MADE DURING FORWARD
                //     RECURSION
                int nn = (inu == 0 && n == 1) ? 1 : 2;
                double dnu2 = dnu + dnu;
                double fmu = (std::abs(dnu2) >= tol) ? dnu2 * dnu2 : 0.;
                double ex = x * 8.;
                s2 = 0.;
                for (int k=1; k<=nn; ++k) {
                    s1 = s2;
                    double s = 1.;
                    double ak = 0.;
                    double ck = 1.;
                    double sqk = 1.;
                    double dk = ex;
                    for (int j=0; j < 30; ++j) {
                        ck = ck * (fmu - sqk) / dk;
                        s += ck;
                        dk += ex;
                        ak += 8.;
                        sqk += ak;
                        if (std::abs(ck) < tol) break;
                    }
                    s2 = s * coef;
                    fmu = fmu + dnu * 8. + 4.;
                }
                if (nn == 1) {
                    s1 = s2;
                    recurse = false;
                }
            } else {
                //     MILLER ALGORITHM FOR X1.LT.X.LE.X2

                double etest = std::cos(pi * dnu) / (pi * x * tol);
                double fks = 1.;
                double fhs = .25;
                double fk = 0.;
                double ck = x + x + 2.;
                double p1 = 0.;
                double p2 = 1.;
                double a[160], b[160];
                int k = 0;
                do {
                    fk += 1.;
                    double ak = (fhs - dnu2) / (fks + fk);
                    double bk = ck / (fk + 1.);
                    double pt = p2;
                    p2 = bk * p2 - ak * p1;
                    p1 = pt;
                    a[k] = ak;
                    b[k] = bk;
                    ck += 2.;
                    fks += fk + fk + 1.;
                    fhs += fk + fk;
                    ++k;
                } while (etest > fk * p1);

                double s = 1.;
                p1 = 0.;
                p2 = 1.;
                int kk = k-1;
                for (int i=1; i<=k; ++i) {
                    double pt = p2;
                    p2 = (b[kk] * p2 - p1) / a[kk];
                    p1 = pt;
                    s += p2;
                    --kk;
                }
                s1 = coef * (p2 / s);
                if (inu == 0 && n == 1) {
                    recurse = false;
                } else {
                    s2 = s1 * (x + dnu + .5 - p1 / p2) / x;
                }
            }
        } else {
            //     SERIES FOR X.LE.X1
            double a1 = 1. - dnu;
            double a2 = dnu + 1.;
            double t1 = 1. / std::tgamma(a1);
            double t2 = 1. / std::tgamma(a2);
            double g1;
            if (std::abs(dnu) > .1) {
                g1 = (t1 - t2) / (dnu + dnu);
            } else {
                //     SERIES FOR F0 TO RESOLVE INDETERMINACY FOR SMALL ABS(DNU)
                double s = cc[0];
                double ak = 1.;
                for (int k = 1; k < 8; ++k) {
                    ak *= dnu2;
                    double tm = cc[k] * ak;
                    s += tm;
                    if (std::abs(tm) < tol) break;
                }
                g1 = -s;
            }
            double g2 = (t1 + t2) * .5;
            double smu = 1.;
            double fc = 1.;
            double flrx = std::log(rx);
            double fmu = dnu * flrx;
            if (dnu != 0.) {
                fc = dnu * pi;
                fc /= std::sin(fc);
                if (fmu != 0.) {
                    smu = std::sinh(fmu) / fmu;
                }
            }
            double f = fc * (g1 * std::cosh(fmu) + g2 * flrx * smu);
            fc = std::exp(fmu);
            double p = fc * .5 / t2;
            double q = .5 / (fc * t1);
            double ak = 1.;
            double ck = 1.;
            double bk = 1.;
            s1 = f;
            s2 = p;
            if (inu == 0 && n == 1) {
                if (x >= tol) {
                    double cx = x * x * .25;
                    double s;
                    do {
                        f = (ak * f + p + q) / (bk - dnu2);
                        p /= ak - dnu;
                        q /= ak + dnu;
                        ck = ck * cx / ak;
                        double t1 = ck * f;
                        s1 += t1;
                        bk = bk + ak + ak + 1.;
                        ak += 1.;
                        s = std::abs(t1) / (std::abs(s1) + 1.);
                    } while (s > tol);
                }
                if (iflag) y[0] = s1 * std::exp(x);
                else y[0] = s1;
                return;
            }
            if (x >= tol) {
                double cx = x * x * .25;
                double s;
                do {
                    f = (ak * f + p + q) / (bk - dnu2);
                    p /= ak - dnu;
                    q /= ak + dnu;
                    ck = ck * cx / ak;
                    double t1 = ck * f;
                    s1 += t1;
                    double t2 = ck * (p - ak * f);
                    s2 += t2;
                    bk = bk + ak + ak + 1.;
                    ak += 1.;
                    s = std::abs(t1) / (std::abs(s1) + 1.) + std::abs(t2) / (std::abs(s2) + 1.);
                } while (s > tol);
            }
            s2 *= rx;
            if (iflag) {
                f = std::exp(x);
                s1 *= f;
                s2 *= f;
            }
        }
        if (recurse) {
            //     FORWARD RECURSION ON THE THREE TERM RECURSION RELATION */
            double ck = (dnu + dnu + 2.) / x;
            if (n == 1) --inu;
            if (inu > 0) {
                for (int i=1; i<=inu; ++i) {
                    double st = s2;
                    s2 = ck * s2 + s1;
                    s1 = st;
                    ck += rx;
                }
            }
            if (n == 1) s1 = s2;
        }
        if (iflag) {
            //     IFLAG=1 CASES
            double s = -x + std::log(s1);
            int nz = 0;
            if (s >= -elim) {
                y[0] = std::exp(s);
            } else {
                y[0] = 0.;
                ++nz;
            }
            if (n == 1) return;
            s = -x + std::log(s2);
            if (s >= -elim) {
                y[1] = std::exp(s);
            } else {
                y[1] = 0.;
                ++nz;
            }
            if (n == 2) return;
            int kk = 2;
            if (nz == 2) {
                for (int i=2; i<n; ++i) {
                    kk = i+1;
                    double st = s2;
                    double s2 = ck * s2 + s1;
                    double s1 = st;
                    ck += rx;
                    s = -x + std::log(s2);
                    if (s < -elim) {
                        ++nz;
                        y[i] = 0.;
                        continue;
                    } else {
                        y[i] = std::exp(s);
                        break;
                    }
                }
            }
            if (kk == n) return;
            s2 = s2 * ck + s1;
            ck += rx;
            y[kk++] = std::exp(-x + std::log(s2));
            for (int i=kk; i<=n; ++i) {
                y[i] = ck * y[i-1] + y[i-2];
                ck += rx;
            }
        } else {
            y[0] = s1;
            if (n == 1) return;
            y[1] = s2;
            if (n == 2) return;
            for (int i=2; i<=n; ++i) {
                y[i] = ck * y[i-1] + y[i-2];
                ck += rx;
            }
        }
    }

    // An auxiliary function used by many of the above functions.
    double dcsevl(double x, const double* cs, int n)
    {
        // ***BEGIN PROLOGUE  DCSEVL
        // ***PURPOSE  Evaluate a Chebyshev series.
        // ***LIBRARY   SLATEC (FNLIB)
        // ***CATEGORY  C3A2
        // ***TYPE      DOUBLE PRECISION (CSEVL-S, DCSEVL-D)
        // ***KEYWORDS  CHEBYSHEV SERIES, FNLIB, SPECIAL FUNCTIONS
        // ***AUTHOR  Fullerton, W., (LANL)
        // ***DESCRIPTION
        //
        //  Evaluate the N-term Chebyshev series CS at X.  Adapted from
        //  a method presented in the paper by Broucke referenced below.
        //
        //       Input Arguments --
        //  X    value at which the series is to be evaluated.
        //  CS   array of N terms of a Chebyshev series.  In evaluating
        //       CS, only half the first coefficient is summed.
        //  N    number of terms in array CS.
        //
        // ***REFERENCES  R. Broucke, Ten subroutines for the manipulation of
        //                 Chebyshev series, Algorithm 446, Communications of
        //                 the A.C.M. 16, (1973) pp. 254-256.
        //               L. Fox and I. B. Parker, Chebyshev Polynomials in
        //                 Numerical Analysis, Oxford University Press, 1968,
        //                 page 56.
        // ***ROUTINES CALLED  D1MACH, XERMSG
        // ***REVISION HISTORY  (YYMMDD)
        //    770401  DATE WRITTEN
        //    890831  Modified array declarations.  (WRB)
        //    890831  REVISION DATE from Version 3.2
        //    891214  Prologue converted to Version 4.0 format.  (BAB)
        //    900315  CALLs to XERROR changed to CALLs to XERMSG.  (THJ)
        //    900329  Prologued revised extensively and code rewritten to allow
        //            X to be slightly outside interval (-1,+1).  (WRB)
        //    920501  Reformatted the REFERENCES section.  (WRB)
        //    170203  Converted to C++. (MJ)
        // ***END PROLOGUE  DCSEVL

        const double onepl = 1. + 2.*std::numeric_limits<double>::epsilon();
        assert(n >= 1);
        assert(n <= 1000);
        if (std::abs(x) > onepl)
            throw std::runtime_error("DCSEVL X OUTSIDE THE INTERVAL (-1,+1)");

        double b0 = 0.;
        double b1 = 0.;
        double b2 = 0.;
        double twox = x * 2.;
        for (int i=n-1; i >= 0; --i) {
            b2 = b1;
            b1 = b0;
            b0 = twox * b1 - b2 + cs[i];
        }
        return (b0 - b2) * 0.5;
    }


} }

