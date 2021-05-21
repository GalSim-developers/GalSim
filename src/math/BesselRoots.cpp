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

//#define DEBUGLOGGING

#include <cmath>
#include <stdexcept>
#include "math/Bessel.h"
#include "Std.h"

namespace galsim {
namespace math {

    const int n_roots_j0 = 40;

    // These values are from:
    //   http://keisan.casio.com/exec/system/1180573472
    // If we need zeros for other values of nu, it can calculate them too.

    const double root_j0[n_roots_j0] = {
        2.404825557695772768622,
        5.520078110286310649597,
        8.653727912911012216954,
        11.79153443901428161374,
        14.93091770848778594776,
        18.07106396791092254315,
        21.21163662987925895908,
        24.35247153074930273706,
        27.49347913204025479588,
        30.63460646843197511755,
        33.77582021357356868424,
        36.91709835366404397977,
        40.0584257646282392948,
        43.19979171317673035752,
        46.34118837166181401869,
        49.4826098973978171736,
        52.62405184111499602925,
        55.76551075501997931168,
        58.90698392608094213283,
        62.04846919022716988285,
        65.18996480020686044064,
        68.33146932985679827099,
        71.47298160359373282506,
        74.61450064370183788382,
        77.75602563038805503774,
        80.89755587113762786377,
        84.03909077693819015788,
        87.18062984364115365126,
        90.32217263721048005572,
        93.46371878194477417119,
        96.60526795099626877812,
        99.74681985868059647028,
        102.8883742541947945964,
        106.0299309164516155102,
        109.1714896498053835521,
        112.3130502804949096275,
        115.4546126536669396281,
        118.5961766308725317156,
        121.7377420879509629652,
        124.8793089132329460453,
    };

    double getBesselRoot0(int s)
    {
        if (s <= 0)
            throw std::runtime_error("s must be > 0");
        if (s <= n_roots_j0) return root_j0[s-1];
        else {
            // Above this value, the asymptotic formula from Abramowitz and Stegun 9.5.12
            // is accurate to better than 1.e-16, so essentially exact.
            // b - (m-1)/8b - 4(m-1)(7m-31)/3(8b)^3 - 32(m-1)(83m^2-982m+3779)/15(8b)^5
            //   - 64(m-1)(6949m^3-153855m^2+1585743m-6277237)/105(8b)^7
            // where m = 4nu^2 and b = (s + nu/2 - 1/4)pi.
            //
            // For nu = 0, this simplifies considerably to:
            // b + 1/8b - 4*31/3(8b)^3 + 32*3779/15(8b)^5 - 64*6277237/105(8b)^7
            double b = (s - 0.25) * M_PI;
            double temp = 0.125/b;
            double inv8bsq = temp*temp;

            // From here on, b will be the running total, not the original b, and temp will be
            // the part of each coefficient that is in common with the next term.
            b += temp;

            temp *= (4./3.) * inv8bsq;
            b -= 31. * temp;

            temp *= (8./5.) * inv8bsq;
            b += 3779. * temp;

            temp *= (2./7.) * inv8bsq;
            b -= 6277237. * temp;

            return b;
        }
    }

    double getBesselRoot(double nu, int s)
    {
        // For orders other than 0, we follow the method that boost uses, described here:
        // https://www.boost.org/doc/libs/1_59_0/libs/math/doc/html/math_toolkit/bessel/bessel_root.html
        // The basic idea is use some fitting formula to get pretty close, and then
        // refine it with a few steps of Newton's method to get to the zero.
        // The boost code has a lot more sophisticated intial estimate.  However, the above
        // Abramowitz and Stegun approximation for large s is actually pretty decent for
        // all s, at least for nu >= 0.  (Negative nu has other issues...)
        // Se we just use that and rely on Newton refinement to tune it up.
        if (nu == 0) {
            return getBesselRoot0(s);
        } else {
            // This doesn't work for negative nu.
            if (nu < 0)
                throw std::runtime_error("nu must be >= 0.");
            if (s <= 0)
                throw std::runtime_error("s must be > 0");
            dbg<<"Bessel Root for nu="<<nu<<" s = "<<s<<std::endl;
            double m = 4.*nu*nu;
            double b = (s + nu/2. - 0.25)*M_PI;
            double temp = 0.125/b;
            double inv8bsq = temp*temp;
            temp *= (m-1);

            // As above, now b is the running total.
            // We compute the following approximation:
            // b - (m-1)/8b - 4(m-1)(7m-31)/3(8b)^3 - 32(m-1)(83m^2-982m+3779)/15(8b)^5
            //   - 64(m-1)(6949m^3-153855m^2+1585743m-6277237)/105(8b)^7
            b -= temp;

            temp *= (4./3.) * inv8bsq;
            b -= (7.*m - 31.) * temp;

            temp *= (8./5.) * inv8bsq;
            b -= ((83.*m - 982)*m + 3779.) * temp;

            temp *= (2./7.) * inv8bsq;
            b -= (((6949.*m - 153855)*m + 1585743)*m-6277237.) * temp;

            dbg<<"Initial estimate is "<<b<<std::endl;
            double y = cyl_bessel_j(nu,b);
            dbg<<"J(nu,b) = "<<y<<std::endl;

            // Now the Newton refinement.  Typically only takes a few iterations at most.
            while (std::abs(y) > 1.e-14) {
                double dydb = y*nu/b - cyl_bessel_j(nu+1,b);
                double db = y/dydb;
                dbg<<"db = "<<db<<std::endl;
                b -= db;
                dbg<<"b -> "<<b<<std::endl;
                y = cyl_bessel_j(nu,b);
                dbg<<"J(nu,b) -> "<<y<<std::endl;
            }
            return b;
        }
    }

}}
