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
#include <vector>
#include <map>
#include <functional>
#include "integ/Int.h"
#include "math/Bessel.h"
#include "math/BesselRoots.h"
#include "Std.h"

namespace galsim {
namespace math {

    // Integrand class for the Hankel transform
    class Integrand : public std::function<double(double)>
    {
    public:
        Integrand(const std::function<double(double)> f, double k) : _f(f), _k(k)
        {
            xdbg<<"Make Integrand: "<<k<<std::endl;
        }
        double operator()(double r) const
        {
            xdbg<<"Call integrand: "<<r<<std::endl;
            return r*_f(r) * math::j0(_k*r);
        }

    private:
        const std::function<double(double)> _f;
        double _k;
    };

    // This is the straightforward GKP method for doing the Hankel integral.
    double hankel_gkp(const std::function<double(double)> f, double k, double maxr,
                      double relerr, double abserr, int nzeros)
    {
        xdbg<<"Start hankel: "<<k<<"  "<<maxr<<std::endl;
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

    class HankelIntegrator
    {
    public:
        // Helper class that can perform a Hankel integral for a specific choice of h
        // using the method of Ogata (2005):
        // http://www.kurims.kyoto-u.ac.jp/~prims/pdf/41-4/41-4-40.pdf
        HankelIntegrator(double h, double batch=256) :
            _h(h), _Nmax(long(std::min(M_PI/h,1.e6))), _batch(batch), _N(0)
        {
            dbg<<"Setup HankelIntegrator for h="<<h<<std::endl;
            dbg<<"Nmax = "<<_Nmax<<std::endl;
            // Rarely will we need more than a few hundred values in the sum, even with very
            // small values of h (since when that is needed, usually the function is highly
            // concentrated at small values of x).  So do this in batches of 256.
            // Only make the next batch if we end up needing it.
            setWeightsBatch();
        }

        void setWeightsBatch()
        {
            // Set up a batch of the weights and kernel values for this choice of h.
            long N1 = _N;  // Number of values already set.
            _N += _batch;
            if (_N > _Nmax) _N = _Nmax;
            _w.resize(_N);
            _x.resize(_N);
            for (long i=N1; i<_N; ++i) {
                double xi = math::getBesselRoot0(i+1)/M_PI;
                double t = _h * xi;
                _x[i] = M_PI/_h * psi(t);
                _w[i] = math::cyl_bessel_y(0,M_PI*xi) / math::j1(M_PI*xi);
                _w[i] *= M_PI * _x[i] * math::j0(_x[i]) * dpsi(t);
                xdbg<<i<<"  "<<xi<<"  "<<t<<"  "<<_x[i]<<"  "<<_w[i]<<std::endl;
            }
            dbg<<"Done setWeightsBatch: _N = "<<_N<<std::endl;
        }

        inline double psi(double t)
        {
            return t * std::tanh(M_PI/2. * std::sinh(t));
        }

        inline double SQR(double x)
        {
            return x*x;
        }

        inline double dpsi(double t)
        {
            return t * M_PI/2. * std::cosh(t) / SQR(std::cosh(M_PI/2. * std::sinh(t))) + psi(t)/t;
        }

        double integrate(const std::function<double(double)> f, double k)
        {
            xdbg<<"start integrate for k = "<<k<<std::endl;
            xdbg<<"h, N = "<<_h<<"  "<<_N<<std::endl;
            assert(_N == long(_w.size()));
            assert(_N == long(_x.size()));
            double ans = 0.;
            long N1 = 0;
            bool done = false;
            do {
                double step;
                for (long i=N1; i<_N; ++i) {
                    step = _w[i] * f(_x[i]/k);
                    ans += step;
                    xdbg<<i<<"  "<<step<<"  "<<ans<<std::endl;
                    if (std::abs(step) < 1.e-15 * std::abs(ans)) {
                        xdbg<<"Break at i = "<<i<<std::endl;
                        xdbg<<"step = "<<step<<", ans = "<<ans<<std::endl;
                        done = true;
                        break;
                    }
                }
                N1 = _N;
                if (_N == _Nmax) done = true;  // Can't go higher than this.
                if (!done) {
                    dbg<<"Didn't converge with current values.  N="<<_N<<std::endl;
                    dbg<<"current ans = "<<ans<<", last step = "<<step<<std::endl;
                    setWeightsBatch();
                }
            } while (!done);

            // We omitted a factor of 1/k^2, so apply that now.
            ans /= k*k;
            xdbg<<"ans = "<<ans<<std::endl;
            return ans;
        }
    private:
        double _h;
        long _Nmax;
        long _batch;
        long _N;
        std::vector<double> _w;
        std::vector<double> _x;
    };

    class AdaptiveHankelIntegrator
    {
    public:
        // Wraps the above HankelIntegrator to get requested rel/abs errors.
        AdaptiveHankelIntegrator(double h0=1./32.) : _h0(h0)
        {
            _integrators[h0] = std::unique_ptr<HankelIntegrator>(new HankelIntegrator(h0));
            double h1 = 0.5 * h0;
            _integrators[h1] = std::unique_ptr<HankelIntegrator>(new HankelIntegrator(h1));
        }

        HankelIntegrator* get_integrator(double h)
        {
            if (_integrators.count(h) == 0) {
                _integrators[h] = std::unique_ptr<HankelIntegrator>(new HankelIntegrator(h));
            }
            return _integrators[h].get();
        }

        double integrate(const std::function<double(double)> f, double k,
                         double relerr, double abserr)
        {
            dbg<<"start adaptive integrate for k = "<<k<<std::endl;

            double h0 = _h0;
            while (h0 > k) h0 *= 0.5;

            double ans0 = get_integrator(h0)->integrate(f,k);
            double h1 = 0.5 * h0;
            double ans1 = get_integrator(h1)->integrate(f,k);
            double err = std::abs(ans1-ans0);
            dbg<<"first answers = "<<ans0<<", "<<ans1<<"  diff = "<<err<<std::endl;
            while (err > abserr && err > relerr * ans1) {
                h0 = h1;
                ans0 = ans1;
                h1 *= 0.5;
                ans1 = get_integrator(h1)->integrate(f,k);
                err = std::abs(ans1 - ans0);
                dbg<<"answers = "<<ans0<<", "<<ans1<<"  diff = "<<err<<std::endl;
            }
            return ans1;
        }
    private:
        double _h0;
        std::map<double, std::unique_ptr<HankelIntegrator> > _integrators;
    };

    double hankel_inf(const std::function<double(double)> f, double k,
                      double relerr, double abserr, int nzeros)
    {
        static AdaptiveHankelIntegrator H;
        dbg<<"Start hankel_inf: "<<k<<std::endl;
        if (k == 0.) {
            // If k = 0, can't do the Ogata method, since it integrates f(x/k) J(x).
            return hankel_gkp(f, k, integ::MOCK_INF, relerr, abserr, nzeros);
        } else {
            return H.integrate(f, k, relerr, abserr);
        }
    }

    double hankel(const std::function<double(double)> f, double k, double maxr,
                  double relerr, double abserr, int nzeros)
    { return hankel_gkp(f, k, maxr, relerr, abserr, nzeros); }

}
}
