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
    double dbesy(double x, double fnu);
    double dbesy0(double x);
    double dbesy1(double x);
    void dbsynu(double x, double fnu, int n, double *y);
    void dyairy(double x, double rx, double c, double *ai, double *dai);

    // Defined in BesselJ.cpp
    double dbesj0(double x);
    double dbesj1(double x);
    double dasyjy(double x, double fnu, bool is_j, double *wk, int* iflw);
    double dcsevl(double x, const double* cs, int n);

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
                double t1 = 1. / math::tgamma(a1);
                double t2 = 1. / math::tgamma(a2);
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
                    s2 = 0.;
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

} }

