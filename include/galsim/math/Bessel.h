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

#ifndef GalSim_Bessel_H
#define GalSim_Bessel_H
/**
 * @file math/Bessel.h
 * @brief Contains implementations of some bessel functions ported from netlib.
 */

#include <cmath>

//#define TEST // Uncomment this to turn on testing of this code against boost code.
#ifdef TEST
#include <boost/math/special_functions/bessel.hpp>
#include <iostream>
#endif

namespace galsim {
namespace math {

    double dbsk0e(double x);
    double dbsk1e(double x);
    double dbesi0(double x);
    double dbsi0e(double x);
    double dbesi1(double x);
    double dbsi1e(double x);
    double d9knus(double xnu, double x, double* bknu1=0);
    double dcsevl(double x, double* cs, int n);

    inline double cyl_bessel_k(double nu, double x)
    {
        const double sqrteps = std::sqrt(std::numeric_limits<double>::epsilon());

        // Identity: K_-nu(x) = K_nu(x)
        nu = std::abs(nu);

        // Negative arguments yield complex values, so don't do that.
        // And K_nu(0) = inf.
        if (x <= 0)
            throw std::runtime_error("cyl_bessel_k x must be > 0");

        // The implementation here is based on the netlib fortran code found here:
        //   http://www.netlib.org/slatec/fnlib/gamit.f
        // but reimplemented in C++.
        //
        // The fundamental recursion for K(nu,x) is
        //    K(nu+1,x) = 2nu K(nu,x)/x + K(nu-1,x)
        // which is stable for forward recursion.  The netlib code computes this in an
        // array of values with the desired nu at the last position, but it's easier
        // to just use direct iteration from either (0,1) or (nu0,nu0+1) where 0<nu0<1.
        double knu;
        if (nu == 0.)
            knu = dbsk0e(x);
        else if (nu == 1.)
            knu = dbsk1e(x);
        else if (nu < 1.) 
            knu = d9knus(nu, x);
        else if (x < sqrteps) {
            // Then better off just using the direct series representation, since the 
            // below iteration has round off erorrs.
            // Knu(x) ~= 1/2 gamma(nu) * (x/2)^-nu (1 - (x/2)^2/(nu-1) + ...)
            knu = 0.5 * std::tgamma(nu) * std::pow(x/2., -nu) * ( 1. - (nu-1.)*x*x/4. );
        } else {
            double nu0 = nu - std::floor(nu);
            int niter = int(std::floor(nu)) - 1;
            double knu1;
            if (nu0 == 0.) {
                knu = dbsk0e(x);
                knu1 = dbsk1e(x);
            } else {
                knu = d9knus(nu0, x, &knu1);
            }

            for (; niter; --niter) {
                nu0 += 1.;
                knu += 2. * nu0 * knu1 / x;
                std::swap(knu,knu1);
            }
            knu = knu1;
        }
        // So far everything has actually been calculating exp(x)*K(nu,x), so we need
        // to divide by exp(x) now.
        knu *= std::exp(-x);
#ifdef TEST
        double knu2 = boost::math::cyl_bessel_k(nu,x);
        if (std::abs(knu-knu2)/std::abs(knu2) > 1.e-12) {
            std::cerr.precision(16);
            std::cerr<<"K("<<nu<<","<<x<<") = "<<knu2<<"  =? "<<knu<<std::endl;
            std::cerr<<"diff = "<<knu-knu2<<std::endl;
            std::cerr<<"rel diff = "<<(knu-knu2)/std::abs(knu2)<<std::endl;
            throw std::runtime_error("cyl_bessel_k doesn't agree with boost cyl_bessel_k");
        }
#endif
        return knu;
    }

    // The below functions are manual conversions from the public domain fortran code here:
    //   http://www.netlib.org/slatec/fnlib/
    // to C++ (guided by f2c, but then manually edited).
    // I left the original PROLOGUEs from the fortran code intact, but added a line to their
    // revision histories that I converted them to C++.

    inline double dbsk0e(double x)
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
        //    170203  Converted to C++ (MJ)
        // ***END PROLOGUE  DBSK0E

        static double bk0cs[16] = {
            -.0353273932339027687201140060063153,
            .344289899924628486886344927529213,
            .0359799365153615016265721303687231,
            .00126461541144692592338479508673447,
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
        static double ak0cs[38] = {
            -.07643947903327941424082978270088,
            -.02235652605699819052023095550791,
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
        static double ak02cs[33] = {
            -.01201869826307592239839346212452,
            -.009174852691025695310652561075713,
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

    inline double dbesi0(double x)
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
        //    170203  Converted to C++ (MJ)
        // ***END PROLOGUE  DBESI0

        static double bi0cs[18] = {
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

    inline double dbsi0e(double x)
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
        //    170203  Converted to C++ (MJ)
        // ***END PROLOGUE  DBSI0E
        static double bi0cs[18] = {
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
        static double ai0cs[46] = {
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
        static double ai02cs[69] = {
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

    inline double dbsk1e(double x)
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
        //    170203  Converted to C++ (MJ)
        // ***END PROLOGUE  DBSK1E

        static double bk1cs[16] = {
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
        static double ak1cs[38] = {
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
        static double ak12cs[33] = {
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

    inline double dbesi1(double x)
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
        //    170203  Converted to C++ (MJ)
        // ***END PROLOGUE  DBESI1

        static double bi1cs[17] = {
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
            if (x > xsml) 
                return x * (dcsevl(x*x/4.5-1., bi1cs, nti1) + .875);
            else {
                if (x == 0.) return 0.;
                if (x < xmin)
                    throw std::runtime_error("DBESI1 ABS(X) SO SMALL I1 UNDERFLOWS");
                return x * 0.5;
            }
        } else {
            return std::exp(x) * dbsi1e(x);
        }
    }

    inline double dbsi1e(double x)
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
        //    170203  Converted to C++ (MJ)
        // ***END PROLOGUE  DBSI1E

        static double bi1cs[17] = {
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
        static double ai1cs[46] = {
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
        static double ai12cs[69] = {
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
            if (x > xsml) 
                return std::exp(-x) * (x * (dcsevl(x*x/4.5-1., bi1cs, nti1) + .875));
            else {
                if (x == 0.) return 1.;
                if (x < xmin)
                    throw std::runtime_error("DBSI1E ABS(X) SO SMALL I1 UNDERFLOWS");
                return std::exp(-x) * (x * 0.5);
            }
        } else if (x <= 8.) {
            return (dcsevl((48./x-11.)/5., ai1cs, ntai1) + .375) / std::sqrt(x);
        } else {
            return (dcsevl(16./x-1., ai12cs, ntai12) + .375) / std::sqrt(x);
        }
    }

    inline double d9knus(double xnu, double x, double* bknu1)
    {
        // ***BEGIN PROLOGUE  D9KNUS
        // ***SUBSIDIARY
        // ***PURPOSE  Compute Bessel functions EXP(X)*K-SUB-XNU(X) and EXP(X)*
        //            K-SUB-XNU+1(X) for 0.0 .LE. XNU .LT. 1.0.
        // ***LIBRARY   SLATEC (FNLIB)
        // ***CATEGORY  C10B3
        // ***TYPE      DOUBLE PRECISION (R9KNUS-S, D9KNUS-D)
        // ***KEYWORDS  BESSEL FUNCTION, FNLIB, SPECIAL FUNCTIONS
        // ***AUTHOR  Fullerton, W., (LANL)
        // ***DESCRIPTION
        //
        // Compute Bessel functions EXP(X) * K-sub-XNU (X)  and
        // EXP(X) * K-sub-XNU+1 (X) for 0.0 .LE. XNU .LT. 1.0 .
        //
        // Series for C0K        on the interval  0.          to  2.50000E-01
        //                                        with weighted error   2.16E-32
        //                                         log weighted error  31.67
        //                               significant figures required  30.86
        //                                    decimal places required  32.40
        //
        // Series for ZNU1       on the interval -7.00000E-01 to  0.
        //                                        with weighted error   2.45E-33
        //                                         log weighted error  32.61
        //                               significant figures required  31.85
        //                                    decimal places required  33.26
        //
        // ***REFERENCES  (NONE)
        // ***ROUTINES CALLED  D1MACH, DCSEVL, DGAMMA, INITDS, XERMSG
        // ***REVISION HISTORY  (YYMMDD)
        //    770601  DATE WRITTEN
        //    890531  Changed all specific intrinsics to generic.  (WRB)
        //    890911  Removed unnecessary intrinsics.  (WRB)
        //    890911  REVISION DATE from Version 3.2
        //    891214  Prologue converted to Version 4.0 format.  (BAB)
        //    900315  CALLs to XERROR changed to CALLs to XERMSG.  (THJ)
        //    900720  Routine changed from user-callable to subsidiary.  (WRB)
        //    900727  Added EXTERNAL statement.  (WRB)
        //    920618  Removed space from variable names.  (RWC, WRB)
        //    170203  Converted to C++ (MJ)
        // ***END PROLOGUE  D9KNUS

        static double c0kcs[29] = {
            0.060183057242626108387577445180329,
            -0.15364871433017286092959755943124,
            -0.011751176008210492040068229226213,
            -8.5248788891979509827048401550987e-4,
            -6.1329838767496791874098176922111e-5,
            -4.4052281245510444562679889548505e-6,
            -3.1631246728384488192915445892199e-7,
            -2.2710719382899588330673771793396e-8,
            -1.630564460807760955227462051536e-9,
            -1.170693929941477656875604404313e-10,
            -8.4052063786464437174546593413792e-12,
            -6.0346670118979991487096050737198e-13,
            -4.3326960335681371952045997366903e-14,
            -3.1107358030203546214634697772237e-15,
            -2.233407822673698225448613340984e-16,
            -1.603514671686422630063579152861e-17,
            -1.1512717363666556196035697705305e-18,
            -8.2657591746836959105169479089258e-20,
            -5.9345480806383948172333436695984e-21,
            -4.2608138196467143926499613023976e-22,
            -3.0591266864812876299263698370542e-23,
            -2.1963541426734575224975501815516e-24,
            -1.576911326149583607110575068476e-25,
            -1.1321713935950320948757731048056e-26,
            -8.1286248834598404082792349714433e-28,
            -5.8360900893453226552829349315949e-29,
            -4.1901241623610922519452337780905e-30,
            -3.0083737960206435069530504212862e-31,
            -2.1599152067808647728342168089832e-32
        };
        static double znu1cs[20] = {
            0.203306756994191729674444001216911,
            0.140077933413219771062943670790563,
            0.0079167969610016135284097224197232,
            3.3980118253210404535293009220575e-4,
            1.1741975688989336666450722835269e-5,
            3.39357570612261680333825865475121e-7,
            8.42594176976219910194629891264803e-9,
            1.8333667702485008918474815090009e-10,
            3.54969844704416310863007064469557e-12,
            6.19032496469887332205244342078407e-14,
            9.81964535680439424960346115456527e-16,
            1.42851314396490474211473563005985e-17,
            1.91894921887825298966162467488436e-19,
            2.39430979739498914162313140597128e-21,
            2.78890246815347354835870465474995e-23,
            3.04606650633033442582845214092865e-25,
            3.13173237042191815771564260932089e-27,
            3.04133098987854951645174908005034e-29,
            2.79840384636833084343185097659733e-31,
            2.44637186274497596485238794922666e-33
        };
        static double euler = 0.5772156649015328606065120900824;
        static double sqpi2 = 1.2533141373155002512078826424055;
        static double aln2 = 0.69314718055994530941723212145818;

        const int ntc0k = 16;
        const int ntznu1 = 12;

        const double xnusml = std::sqrt(std::numeric_limits<double>::epsilon()/8.);
        const double xsml = std::numeric_limits<double>::epsilon() * 0.1;
        const double alnsml = std::log(std::numeric_limits<double>::min());
        const double alnbig = std::log(std::numeric_limits<double>::max());
        const double alneps = std::log(xsml);

        assert(xnu >= 0.);
        assert(xnu < 1.);
        assert(x > 0.);

        if (x <= 2.) {
            // X IS SMALL.  COMPUTE K-SUB-XNU (X) AND THE DERIVATIVE OF K-SUB-XNU (X)
            // THEN FIND K-SUB-XNU+1 (X).  XNU IS REDUCED TO THE INTERVAL (-.5,+.5)
            // THEN TO (0., .5), BECAUSE K OF NEGATIVE ORDER (-NU) = K OF POSITIVE
            // ORDER (+NU).
            double v = xnu > 0.5 ? 1.-xnu : xnu;

            // CAREFULLY FIND (X/2)**XNU AND Z**XNU WHERE Z = X*X/4.
            double alnz = 2. * (std::log(x) - aln2);

            if (x <= xnu && xnu * -0.5 * alnz - aln2 - log(xnu) > alnbig)
                throw std::runtime_error("D9KNUS X SO SMALL BESSEL K-SUB-XNU OVERFLOWS");

            double vlnz = v * alnz;
            double x2tov = std::exp(0.5*vlnz);
            double ztov = (vlnz > alnsml) ? x2tov * x2tov : 0.;

            double a0 = 0.5 * std::tgamma(1.+v);
            double b0 = 0.5 * std::tgamma(1.-v);

            int nterms = std::max(2, 11+int((8.*alnz - 25.19 - alneps) / (4.28 - alnz)));
            double alpha[nterms];
            double beta[nterms];

            if (ztov <= 0.5) 
                alpha[0] = (a0 - ztov * b0) / v;
            else {
                double c0 = (v > xnusml) ? -0.75 + dcsevl(8.*v*v-1., c0kcs, ntc0k) : -euler;
                alpha[0] = c0 - alnz * (0.75 + dcsevl(vlnz/0.35+1., znu1cs, ntznu1)) * b0;
            }
            beta[0] = -0.5 * (a0 + ztov * b0);

            for (int i=1; i<nterms; ++i) {
                double xi(i);
                a0 /= xi * (xi - v);
                b0 /= xi * (xi + v);
                alpha[i] = (alpha[i-1] + 2.*xi*a0) / (xi * (xi + v));
                beta[i] = (xi - 0.5 * v) * alpha[i] - ztov * b0;
            }

            double z = (x > xsml) ? 0.25 * x*x : 0.;
            double bknu = alpha[nterms-1];
            double bknud = beta[nterms-1];
            for (int i=nterms-2; i>=0; --i) {
                bknu = alpha[i] + bknu * z;
                bknud = beta[i] + bknud * z;
            }

            double expx = std::exp(x);
            bknu *= expx / x2tov;

            if (-0.5 * (xnu + 1.) * alnz - 2. * aln2 > alnbig) {
                if (bknu1)
                    throw std::runtime_error("DBSKES X SO SMALL BESSEL K-SUB-XNU+1 OVERFLOWS");
                return bknu;
            }
            bknud *= 2. * expx / (x2tov * x);

            if (xnu <= 0.5) {
                if (bknu1) *bknu1 = v * bknu / x - bknud;
                return bknu;
            } else {
                double bknu0 = bknu;
                bknu = -v * bknu / x - bknud;
                if (bknu1) *bknu1 = 2. * xnu * bknu / x + bknu0;
                return bknu;
            }

        } else {
            // X IS LARGE.  FIND K-SUB-XNU (X) AND K-SUB-XNU+1 (X) WITH Y. L. LUKE-S
            // RATIONAL EXPANSION.

            double bknu;
            double sqrtx = std::sqrt(x);
            if (x > 1. / xsml) {
                bknu = sqpi2 / sqrtx;
                if (bknu1) *bknu1 = bknu;
                return bknu;
            }
            double an = -0.6 - 1.02/x;
            double bn = -0.27 - 0.53/x;
            int nterms = std::min(32, std::max(3, int(an + bn * alneps)));

            double alpha[nterms];
            double beta[nterms];
            double a[nterms];

            for (int inu=0; inu <= (bknu1 ? 1 : 0); ++inu, xnu += 1.) {

                double xmu = (xnu > xnusml) ? 4. * xnu * xnu : 0.;

                a[0] = 1. - xmu;
                a[1] = 9. - xmu;
                a[2] = 25. - xmu;

                double result;
                if (a[1] == 0.) {
                    result = sqpi2 * (16. * x + xmu + 7.) / (16. * x * sqrtx);
                } else {
                    alpha[0] = 1.;
                    alpha[1] = (16. * x + a[1]) / a[1];
                    alpha[2] = ((768. * x + 48. * a[2]) * x + a[1] * a[2]) / (a[1] * a[2]);

                    beta[0] = 1.;
                    beta[1] = (16. * x + xmu + 7.) / a[1];
                    beta[2] = ((768. * x + 48. * (xmu + 23.)) * x + ((xmu + 62.) * xmu + 129.)) / 
                        (a[1] * a[2]);

                    for (int n=3; n<nterms; ++n) {
                        double x2n = 2.*n - 1.;
                        a[n] = (x2n+2.) * (x2n+2.) - xmu;
                        double qq = 16. * x2n / a[n];
                        double p1 = -x2n * (12.*n*n - 20.*n - a[0]) / ((x2n - 2.) * a[n]) - qq * x;
                        double p2 = (12.*n*n - 28.*n + 8. - a[0]) / a[n] - qq * x;
                        double p3 = -x2n * a[n-3] / ((x2n - 2.) * a[n]);

                        alpha[n] = -p1 * alpha[n-1] - p2 * alpha[n-2] - p3 * alpha[n-3];
                        beta[n] = -p1 * beta[n-1] - p2 * beta[n-2] - p3 * beta[n-3];
                    }

                    result = sqpi2 * beta[nterms-1] / (sqrtx * alpha[nterms-1]);
                }
                if (inu == 0) bknu = result;
                if (inu == 1) *bknu1 = result;
            }
            return bknu;
        }
    }

    inline double dcsevl(double x, double* cs, int n)
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
        //    170203  Converted to C++ (MJ)
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

#endif

