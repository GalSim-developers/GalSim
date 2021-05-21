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
    double dbesi(double x, double fnu);
    double dbesi0(double x);
    double dbesi1(double x);
    double dbsi0e(double x);
    double dbsi1e(double x);
    double dasyik(double x, double fnu, bool is_i);

    // Defined in BesselJ.cpp
    double dcsevl(double x, const double* cs, int n);

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
            double gln = math::lgamma(fn+1);
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
        const double xsml = std::sqrt(std::numeric_limits<double>::epsilon() * 4.5);

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
        t *= flgik;
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

} }

