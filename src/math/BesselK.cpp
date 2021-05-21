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
#include <cstdlib>
#include <stdexcept>
#include <limits>
#include <algorithm>

#include "math/Gamma.h"
#include "Std.h"

// The functions in this file and the other Bessel?.cpp files are manual conversions from the
// public domain fortran code here:
//
//     http://www.netlib.org/slatec/fnlib/
//
// to C++ (guided by f2c, but then manually edited).
// I left the original PROLOGUEs from the fortran code intact, but added a line to their
// revision histories that I converted them to C++.  In some cases, I also changed the
// functionality slightly to make it easier to clean up some of the spaghetti code.

namespace galsim {
namespace math {

    // Routines ported from netlib, defined below.
    double dbesk(double x, double fnu);
    double dbesk0(double x);
    double dbsk0e(double x);
    double dbesk1(double x);
    double dbsk1e(double x);
    void dbsknu(double x, double fnu, int n, double *y);

    // Defined in BesselI.cpp
    double dbesi0(double x);
    double dbesi1(double x);
    double dasyik(double x, double fnu, bool is_i);

    // Defined in BesselJ.cpp
    double dcsevl(double x, const double* cs, int n);

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
        //    170203  Converted to C++. (MJ)
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
        //    170203  Converted to C++. (MJ)
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
        //    170203  Converted to C++. (MJ)
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
                    ck = 1.;
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
                ck = x + x + 2.;
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
                s2 = 0.;
                if (inu == 0 && n == 1) {
                    recurse = false;
                    s2 = 0;  // Unused in this case, but saves a maybe-uninitialized warning.
                } else {
                    s2 = s1 * (x + dnu + .5 - p1 / p2) / x;
                }
            }
        } else {
            //     SERIES FOR X.LE.X1
            double a1 = 1. - dnu;
            double a2 = dnu + 1.;
            double t1 = 1. / math::tgamma(a1);
            double t2 = 1. / math::tgamma(a2);
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
            ck = 1.;
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
            ck = (dnu + dnu + 2.) / x;
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
                    s2 = ck * s2 + s1;
                    s1 = st;
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

} }

