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
// dbesj in particular was far and away the worst case of spaghetti code I've ever seen.
// Removing the option of doing n>1 made it much easier to refactor it into a modern code style.

namespace galsim {
namespace math {

    // Routines ported from netlib, defined below.
    double dbesj(double x, double fnu);
    double dbesj0(double x);
    double dbesj1(double x);
    double dasyjy(double x, double fnu, bool is_j, double *wk, int* iflw);
    void djairy(double x, double rx, double c, double *ai, double *dai);
    double dcsevl(double x, const double* cs, int n);

    // Defined in BesselY.cpp
    void dyairy(double x, double rx, double c, double *ai, double *dai);

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
            double gln = math::lgamma(fn+1);
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
            jnu = dasyjy(x, fn, true, wk, &iflw);
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
        //    770701  DATE WRITTEN
        //    890531  Changed all specific intrinsics to generic.  (WRB)
        //    890531  REVISION DATE from Version 3.2
        //    891214  Prologue converted to Version 4.0 format.  (BAB)
        //    170203  Converted to C++. (MJ)
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
        //    780601  DATE WRITTEN
        //    890531  Changed all specific intrinsics to generic.  (WRB)
        //    890531  REVISION DATE from Version 3.2
        //    891214  Prologue converted to Version 4.0 format.  (BAB)
        //    900315  CALLs to XERROR changed to CALLs to XERMSG.  (THJ)
        //    910401  Corrected error in code which caused values to have the
        //            wrong sign for arguments less than 4.0.  (WRB)
        //    170203  Converted to C++. (MJ)
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

