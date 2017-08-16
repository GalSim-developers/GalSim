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
#include <stdexcept>
#include "math/Bessel.h"

//#define TEST // Uncomment this to turn on testing of this code against boost code.
#ifdef TEST
#include <boost/math/special_functions/bessel.hpp>
#include <iostream>
#endif

namespace galsim {
namespace math {

    // Routines ported from netlib, defined in BesselJ.cpp
    double dbesj(double x, double fnu);
    double dbesj0(double x);
    double dbesj1(double x);
    double dasyjy(double x, double fnu, bool is_j, double *wk, int* iflw);
    void djairy(double x, double rx, double c, double *ai, double *dai);
    double dcsevl(double x, const double* cs, int n);

    // Routines ported from netlib, defined in BesselY.cpp
    double dbesy(double x, double fnu);
    double dbesy0(double x);
    double dbesy1(double x);
    void dbsynu(double x, double fnu, int n, double *y);
    void dyairy(double x, double rx, double c, double *ai, double *dai);

    // Routines ported from netlib, defined in BesselI.cpp
    double dbesi(double x, double fnu);
    double dbesi0(double x);
    double dbesi1(double x);
    double dbsi0e(double x);
    double dbsi1e(double x);
    double dasyik(double x, double fnu, bool is_i);

    // Routines ported from netlib, defined in BesselK.cpp
    double dbesk(double x, double fnu);
    double dbesk0(double x);
    double dbsk0e(double x);
    double dbesk1(double x);
    double dbsk1e(double x);
    void dbsknu(double x, double fnu, int n, double *y);

    double BesselJ(double nu, double x)
    {
        // Negative arguments yield complex values, so don't do that.
        if (x < 0)
            throw std::runtime_error("BesselJ x must be >= 0");

        // Identity: J_-nu(x) = cos(pi nu) J_nu(x) - sin(pi nu) Y_nu(x)
        if (nu < 0) {
            nu = -nu;
            if (int(nu) == nu) {
                if (int(nu) % 2 == 0) return BesselJ(nu, x);
                else return -BesselJ(nu, x);
            } else {
                double c = std::cos(M_PI * nu);
                double s = std::sin(M_PI * nu);
                return c * BesselJ(nu, x) - s * BesselY(nu, x);
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
            throw std::runtime_error("BesselJ doesn't agree with boost::cyl_bessel_j");
        }
#endif
        return jnu;
    }

    // It's not much faster to have these specializations, but people will expect them.
    // We only do it for J though.  It there is demand, it would be trivial to add the others.
    double BesselJ0(double x)
    {
        if (x < 0)
            throw std::runtime_error("BesselJ0 x must be >= 0");
        return dbesj0(x);
    }

    double BesselJ1(double x)
    {
        if (x < 0)
            throw std::runtime_error("BesselJ1 x must be >= 0");
        return dbesj1(x);
    }

    double BesselY(double nu, double x)
    {
        // Negative arguments yield complex values, so don't do that.
        // And Y_nu(0) = inf.
        if (x <= 0)
            throw std::runtime_error("BesselY x must be > 0");

        // Identity: Y_-nu(x) = cos(pi nu) Y_nu(x) + sin(pi nu) J_nu(x)
        if (nu < 0) {
            nu = -nu;
            if (int(nu) == nu) {
                if (int(nu) % 2 == 1) return -BesselY(nu, x);
                else return BesselY(nu, x);
            } else {
                double c = std::cos(M_PI * nu);
                double s = std::sin(M_PI * nu);
                return c * BesselY(nu, x) + s * BesselJ(nu, x);
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
            throw std::runtime_error("BesselY doesn't agree with boost::cyl_bessel_y");
        }
#endif
        return ynu;
    }

    double BesselI(double nu, double x)
    {
        // Negative arguments yield complex values, so don't do that.
        if (x < 0)
            throw std::runtime_error("BesselI x must be >= 0");

        // Identity: I_−nu(x) = I_nu(z) + 2/pi sin(pi nu) K_nu(x)
        if (nu < 0)
            return BesselI(-nu, x) + 2./M_PI * std::sin(-M_PI*nu) * BesselK(-nu,x);

        double inu = dbesi(x, nu);
#ifdef TEST
        double inu2 = boost::math::cyl_bessel_i(nu, x);
        if (std::abs(inu-inu2)/std::abs(inu2) > 1.e-10) {
            std::cerr.precision(16);
            std::cerr<<"K("<<nu<<","<<x<<") = "<<inu2<<"  =? "<<inu<<std::endl;
            std::cerr<<"diff = "<<inu-inu2<<std::endl;
            std::cerr<<"rel diff = "<<(inu-inu2)/std::abs(inu2)<<std::endl;
            throw std::runtime_error("BesselI doesn't agree with boost::cyl_bessel_i");
        }
#endif
        return inu;
    }

    double BesselK(double nu, double x)
    {
        // Identity: K_-nu(x) = K_nu(x)
        nu = std::abs(nu);

        // Negative arguments yield complex values, so don't do that.
        // And K_nu(0) = inf.
        if (x <= 0)
            throw std::runtime_error("BesselK x must be > 0");

        double knu = dbesk(x, nu);
#ifdef TEST
        double knu2 = boost::math::cyl_bessel_k(nu, x);
        if (std::abs(knu-knu2)/std::abs(knu2) > 1.e-10) {
            std::cerr.precision(16);
            std::cerr<<"K("<<nu<<","<<x<<") = "<<knu2<<"  =? "<<knu<<std::endl;
            std::cerr<<"diff = "<<knu-knu2<<std::endl;
            std::cerr<<"rel diff = "<<(knu-knu2)/std::abs(knu2)<<std::endl;
            throw std::runtime_error("BesselK doesn't agree with boost::cyl_bessel_k");
        }
#endif
        return knu;
    }

}}
