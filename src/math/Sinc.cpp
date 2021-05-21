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
#include "math/Sinc.h"
#include "math/Angle.h"

namespace galsim {
namespace math {

    // sinc(x) is defined here as sin(Pi x) / (Pi x)
    double sinc(double x)
    {
        if (std::abs(x) < 1.e-4) return 1.- (M_PI*M_PI/6.)*x*x;
        else return std::sin(M_PI*x)/(M_PI*x);
    }

    // Utility for calculating the integral of sin(t)/t from 0 to x.  Note the official definition
    // does not have pi multiplying t.
    double Si(double x)
    {
#if 0
        // These are the version Gary had taken from Abramowitz & Stegun's formulae.
        // Unfortunately, they don't seem to be quite accurate enough for our needs.
        // They are accurate to better than 1.e-6, but since our calculation of the
        // fourier transform of Lanczon(n,x) involves subtracting multiples of these
        // from each other, there is a lot of cancelling, and the resulting relative
        // accuracy for uval is much worse than 1.e-6.
        double x2=x*x;
        if(x2>=3.8) {
            // Use rational approximation from Abramowitz & Stegun
            // cf. Eqns. 5.2.38, 5.2.39, 5.2.8 - where it says it's good to <1e-6.
            // ain't this pretty?
            return (M_PI/2.)*((x>0.)?1.:-1.)
                - (38.102495+x2*(335.677320+x2*(265.187033+x2*(38.027264+x2))))
                / (x* (157.105423+x2*(570.236280+x2*(322.624911+x2*(40.021433+x2)))) )*std::cos(x)
                - (21.821899+x2*(352.018498+x2*(302.757865+x2*(42.242855+x2))))
                / (x2*(449.690326+x2*(1114.978885+x2*(482.485984+x2*(48.196927+x2)))))*std::sin(x);

        } else {
            // x2<3.8: the series expansion is the better approximation, A&S 5.2.14
            dbg<<"Calculate Si(x) for x = "<<x<<std::endl;
            double n1=1.;
            double n2=1.;
            double tt=x;
            double t=0;
            for(int i=1; i<7; i++) {
                t += tt/(n1*n2);
                tt = -tt*x2;
                n1 = 2.*double(i)+1.;
                n2*= n1*2.*double(i);
            }
            return t;
        }
#else
        double x2 = x*x;
        if (x2 > 16.) {
            // For |x| > 4, we use the asymptotic formula:
            //
            // Si(x) = pi/2 - f(x) cos(x) - g(x) sin(x)
            //
            // where f(x) = int(sin(t)/(x+t),t=0..inf)
            //       g(x) = int(cos(t)/(x+t),t=0..inf)
            //
            // (By asymptotic, I mean that f and g approach 1/x and 1/x^2 respectively as x -> inf.
            //  The formula as given is exact.)
            //
            // I used Maple to calculate a Chebyshev-Pade approximation of 1/sqrt(y) f(1/sqrt(y))
            // from 0..1/4^2, which leads to the following formula for f(x).  It is accurate to
            // better than 1.e-16 for x > 4.
            double y=1./x2;
            double f =
                (1. +
                 y*(7.44437068161936700618e2 +
                    y*(1.96396372895146869801e5 +
                       y*(2.37750310125431834034e7 +
                          y*(1.43073403821274636888e9 +
                             y*(4.33736238870432522765e10 +
                                y*(6.40533830574022022911e11 +
                                   y*(4.20968180571076940208e12 +
                                      y*(1.00795182980368574617e13 +
                                         y*(4.94816688199951963482e12 +
                                            y*(-4.94701168645415959931e11)))))))))))
                / (x*(1. +
                      y*(7.46437068161927678031e2 +
                         y*(1.97865247031583951450e5 +
                            y*(2.41535670165126845144e7 +
                               y*(1.47478952192985464958e9 +
                                  y*(4.58595115847765779830e10 +
                                     y*(7.08501308149515401563e11 +
                                        y*(5.06084464593475076774e12 +
                                           y*(1.43468549171581016479e13 +
                                              y*(1.11535493509914254097e13)))))))))));

            // Similarly, a Chebyshev-Pade approximation of 1/y g(1/sqrt(y)) from 0..1/4^2
            // leads to the following formula for g(x), which is also accurate to better than
            // 1.e-16 for x > 4.
            double g =
                y*(1. +
                   y*(8.1359520115168615e2 +
                      y*(2.35239181626478200e5 +
                         y*(3.12557570795778731e7 +
                            y*(2.06297595146763354e9 +
                               y*(6.83052205423625007e10 +
                                  y*(1.09049528450362786e12 +
                                     y*(7.57664583257834349e12 +
                                        y*(1.81004487464664575e13 +
                                           y*(6.43291613143049485e12 +
                                              y*(-1.36517137670871689e12)))))))))))
                / (1. +
                   y*(8.19595201151451564e2 +
                      y*(2.40036752835578777e5 +
                         y*(3.26026661647090822e7 +
                            y*(2.23355543278099360e9 +
                               y*(7.87465017341829930e10 +
                                  y*(1.39866710696414565e12 +
                                     y*(1.17164723371736605e13 +
                                        y*(4.01839087307656620e13 +
                                           y*(3.99653257887490811e13))))))))));

            double sinx,cosx;
            math::sincos(x, sinx, cosx);
            return ((x>0.)?(M_PI/2.):(-M_PI/2.)) - f*cosx - g*sinx;
        } else {
            // Here I used Maple to calculate the Pade approximation for Si(x), which is accurate
            // to better than 1.e-16 for x < 4:
            return
                x*(1. +
                   x2*(-4.54393409816329991e-2 +
                       x2*(1.15457225751016682e-3 +
                           x2*(-1.41018536821330254e-5 +
                               x2*(9.43280809438713025e-8 +
                                   x2*(-3.53201978997168357e-10 +
                                       x2*(7.08240282274875911e-13 +
                                           x2*(-6.05338212010422477e-16))))))))
                / (1. +
                   x2*(1.01162145739225565e-2 +
                       x2*(4.99175116169755106e-5 +
                           x2*(1.55654986308745614e-7 +
                               x2*(3.28067571055789734e-10 +
                                   x2*(4.5049097575386581e-13 +
                                       x2*(3.21107051193712168e-16)))))));

        }
        // Note: I also put these formulae on wikipedia, so other people can use them.
        //     http://en.wikipedia.org/wiki/Trigonometric_integral
        // There was a notable lack of information online about how to efficiently calculate
        // Si(x), so hopefully this will help people in the future to not have to reproduce
        // my work.  -MJ
#endif
    }

}
}
