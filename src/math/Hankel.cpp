/* -*- c++ -*-
 * Copyright (c) 2012-2020 by the GalSim developers team on GitHub
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
#include <functional>
#include "math/Bessel.h"
#include "math/BesselRoots.h"
#include "integ/Int.h"
#include "Std.h"

namespace galsim {
namespace math {

    // Integrand class for the Hankel transform
    class Integrand : public std::function<double(double)>
    {
    public:
        Integrand(const std::function<double(double)> f, double k) : _f(f), _k(k)
        {
            dbg<<"Make Integrand: "<<k<<std::endl;
        }
        double operator()(double r) const
        { 
            dbg<<"Call integrand: "<<r<<std::endl;
            return r*_f(r) * math::j0(_k*r);
        }

    private:
        const std::function<double(double)> _f;
        double _k;
    };

    double hankel(const std::function<double(double)> f, double k, double maxr,
                  double relerr, double abserr, int nzeros)
        {
            dbg<<"Start hankel: "<<k<<"  "<<maxr<<std::endl;
            Integrand I(f, k);

    #ifdef DEBUGLOGGING
            std::ostream* integ_dbgout = verbose_level >= 3 ? &Debugger::instance().get_dbgout() : 0;
            integ::IntRegion<double> reg(0, maxr, integ_dbgout);
    #else
            integ::IntRegion<double> reg(0, maxr);
    #endif
            // Add explicit splits at first several roots of J0.
            // This tends to make the integral more accurate.
            for (int s=1; s<=nzeros; ++s) {
                double root = math::getBesselRoot0(s);
                if (root > k * maxr) break;
                reg.addSplit(root/k);
            }
            return integ::int1d(I, reg, relerr, abserr);
        }

        double hankel_inf(const std::function<double(double)> f, double k,
                      double relerr, double abserr, int nzeros)
    {
        dbg<<"Start hankel_inf: "<<k<<std::endl;
        const double inf = integ::MOCK_INF;
        return hankel(f, k, inf, relerr, abserr, nzeros);
    }

}
}
