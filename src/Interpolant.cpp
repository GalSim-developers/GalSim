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

#include "Interpolant.h"
#include "integ/Int.h"
#include "SBProfile.h"
#include "math/Sinc.h"
#include "math/Angle.h"

// Gary's original code used a lot of lookup tables, but most of these have analytic formulae
// that seem to be generally faster than the lookup table.  Part of this is probably because
// our lookup table isn't super fast, so I'm leaving the code in, but disabled.  If we manage
// to massively speed up the lookup table, it might be worth re-enabling this code with the
// following #define.
//
//#define USE_TABLES

// Gary's Quintic interpolant was designed to exactly interpolate up to 4th order of a Taylor
// series expansion.  This implies F'(j) = F''(j) = F'''(j) = F''''(j) = 0.  However, it
// doesn't have a continuous second derivative.  I (MJ) derived an alternate version that does
// have a continuous second derivative, but doesn't have F''''(j) = 0.  Gary thinks the
// F''''(j) = 0 constraint is more important than the continuous second derivatives, since:
//
// "The most important characteristic for the k interpolation is to have F(k) as close to zero as
// possible in the vicinity of the aliasing frequency.  It's these components of F in the vicinity
// of 2 pi that cause the ghost images after interpolation. In this application I'm less worried
// about the continuous derivatives because the ringing at frequencies beyond the vicinity of
// k=2 pi does not affect the interpolated image if we know that we have zero-padded it before
// transforming."
//
// However, I have the alternate version coded up, so I figured I'd leave it in as an option.
// The two functions are actually extremely similar though, so I doubt it really matters that
// much which one we use in practice.
//
//#define ALT_QUINTIC


// For grins, I also figured out the Septimic formulae while I was at it:
//
// The version that correctly interpolates up to 6th order in the Taylor series (but has
// discontinuous 2nd and 3rd derivatives) is:
//
// |x| < 1: 1 + |x|^4 (-3899/144 + 9233/144 |x| - 7669/144 |x|^2 + 2191/144 |x|^3)
// |x| < 2: (|x|-1) (|x|-2) (481/10 - 7369/40 |x| + 1379/5 |x|^2 - 9517/48 |x|^3 + 2739/40 |x|^4
//                           - 2191/240 |x|^5)
// |x| < 3: (|x|-2) (|x|-3) (-1567/6 + 4401/8 |x| - 1373/3 |x|^2 + 27049/144 |x|^3 - 913/24 |x|^4
//                           + 2191/720 |x|^5)
// |x| < 4: (|x|-3) (|x|-4)^2 (-3211/60 + 781/12 |x| - 7067/240 |x|^2 + 2113/360 |x|^3
//                             - 313/720 |x|^4)
//
// F(u) = (1/45) s^7 (6780 c piu^2 + 98595 s - 98550 c - 39570 s piu^2 + 112 c piu^4)
//
// The version with continuous 1st, 2nd, and 3rd derivatives (but only accurately interpolates
// up to 5th order in a Taylor series) is:
//
// |x| < 1: 1 + |x|^4 (-203/8 + 2849/48 |x| - 293/6 |x|^2 + 665/48 |x|^3)
// |x| < 2: (|x|-1) (|x|-2) (913/20 - 10441/60 |x| + 5173/20 |x|^2 - 11051/60 |x|^3 + 5037/80 |x|^4
//                           - 133/16 |x|^5)
// |x| < 3: (|x|-2) (|x|-3) (-2987/12 + 31219/60 |x| - 5149/12 |x|^2 + 5233/30 |x|^3
//                           - 1679/48 |x|^4 + 133/48 |x|^5)
// |x| < 4: (|x|-3) (|x|-4)^4 (-383/120 + 539/240 |x| - 19/48 |x|^2)
//
// F(u) = s^6 (1995 s^2 - 98 - 702 piu^2 s^2 + 104 c s piu^2 - 1896 c s)
//
// Just in case we ever decide we want to go to the next order of polynomial interpolation...

namespace galsim {

    double InterpolantFunction::operator()(double x) const  { return _interp.xval(x); }

    double Interpolant::getPositiveFlux2d() const
    {
        double p = getPositiveFlux();
        double n = getNegativeFlux();
        return p*p + n*n;
    }

    double Interpolant::getNegativeFlux2d() const
    {
        double p = getPositiveFlux();
        double n = getNegativeFlux();
        return 2*p*n;
    }

    double Interpolant::xvalWrapped(double x, int N) const
    {
        // sum over all arguments x+jN that are within range.
        // Start by finding x+jN closest to zero
        double xdown = x - N*std::floor(x/N + 0.5);
        xassert(std::abs(xdown) <= N);
        if (xrange() <= N) {
            // This is the usual case.
            return xval(xdown);
        } else {
            double xup = xdown+N;
            double sum = 0.;
            while (std::abs(xdown) <= xrange()) {
                sum += xval(xdown);
                xdown -= N;
            }
            while (xup <= xrange()) {
                sum += xval(xup);
                xup += N;
            }
            return sum;
        }
    }

    void Interpolant::xvalMany(double* x, int N) const
    {
        // x is both input and output here.
        // x_i <- xval(x_i)
        for (; N; --N, ++x) *x = xval(*x);
    }

    void Interpolant::uvalMany(double* u, int N) const
    {
        // u is both input and output here.
        // u_i <- uval(u_i)
        for (; N; --N, ++u) *u = uval(*u);
    }

    //
    // Delta
    //

    void Delta::shoot(PhotonArray& photons, UniformDeviate ud) const
    {
        const int N = photons.size();
        dbg<<"Delta shoot: N = "<<N<<std::endl;
        dbg<<"Target flux = 1.\n";
        double fluxPerPhoton = 1./N;
        for (int i=0; i<N; i++)  {
            photons.setPhoton(i, 0., 0., fluxPerPhoton);
        }
        dbg<<"Delta Realized flux = "<<photons.getTotalFlux()<<std::endl;
    }

    std::string Delta::makeStr() const
    {
        std::ostringstream oss(" ");
        oss.precision(std::numeric_limits<double>::digits10 + 4);
        oss << "galsim._galsim.Delta(";
        oss << "galsim._galsim.GSParams("<<_gsparams<<"))";
        return oss.str();
    }


    //
    // Nearest
    //

    double Nearest::xval(double x) const
    {
        return std::abs(x)>0.5 ? 0. : 1.;
    }

    double Nearest::uval(double u) const { return math::sinc(u); }

    void Nearest::shoot(PhotonArray& photons, UniformDeviate ud) const
    {
        const int N = photons.size();
        dbg<<"Nearest shoot: N = "<<N<<std::endl;
        dbg<<"Target flux = 1.\n";
        double fluxPerPhoton = 1./N;
        for (int i=0; i<N; i++)  {
            photons.setPhoton(i, ud()-0.5, ud()-0.5, fluxPerPhoton);
        }
        dbg<<"Nearest Realized flux = "<<photons.getTotalFlux()<<std::endl;
    }

    std::string Nearest::makeStr() const
    {
        std::ostringstream oss(" ");
        oss.precision(std::numeric_limits<double>::digits10 + 4);
        oss << "galsim._galsim.Nearest(";
        oss << "galsim._galsim.GSParams("<<_gsparams<<"))";
        return oss.str();
    }


    //
    // SincInterpolant
    //

    double SincInterpolant::uval(double u) const
    {
        if (std::abs(u)>0.5) return 0.;
        else if (std::abs(u)<0.5) return 1.;
        else return 0.5;
    }

    double SincInterpolant::xval(double x) const { return math::sinc(x); }

    double SincInterpolant::xvalWrapped(double x, int N) const
    {
        // Magic formula:
        x *= M_PI;
        if (N%2==0) {
            if (std::abs(x) < 1.e-4) return 1. - x*x*(1/6.+1/2.-1./(6.*N*N));
            return std::sin(x) * std::cos(x/N) / (N*std::sin(x/N));
        } else {
            if (std::abs(x) < 1.e-4) return 1. - (1./6.)*x*x*(1-1./(N*N));
            return std::sin(x) / (N*std::sin(x/N));
        }
    }

    void SincInterpolant::shoot(PhotonArray& photons, UniformDeviate ud) const
    {
        throw std::runtime_error("Photon shooting is not practical with sinc Interpolant");
    }

    std::string SincInterpolant::makeStr() const
    {
        std::ostringstream oss(" ");
        oss.precision(std::numeric_limits<double>::digits10 + 4);
        oss << "galsim._galsim.SincInterpolant(";
        oss << "galsim._galsim.GSParams("<<_gsparams<<"))";
        return oss.str();
    }


    //
    // Linear
    //

    double Linear::xval(double x) const
    {
        x = std::abs(x);
        if (x > 1.) return 0.;
        else return 1.-x;
    }
    double Linear::uval(double u) const
    {
        double s = math::sinc(u);
        return s*s;
    }

    void Linear::shoot(PhotonArray& photons, UniformDeviate ud) const
    {
        const int N = photons.size();
        dbg<<"Linear shoot: N = "<<N<<std::endl;
        dbg<<"Target flux = 1.\n";
        double fluxPerPhoton = 1./N;
        for (int i=0; i<N; i++) {
            // *** Guessing here that 2 random draws is faster than a sqrt:
            photons.setPhoton(i, ud() + ud() - 1., ud() + ud() - 1., fluxPerPhoton);
        }
        dbg<<"Linear Realized flux = "<<photons.getTotalFlux()<<std::endl;
    }

    std::string Linear::makeStr() const
    {
        std::ostringstream oss(" ");
        oss.precision(std::numeric_limits<double>::digits10 + 4);
        oss << "galsim._galsim.Linear(";
        oss << "galsim._galsim.GSParams("<<_gsparams<<"))";
        return oss.str();
    }


    //
    // Cubic
    //

    double Cubic::xval(double x) const
    {
        x = std::abs(x);
        if (x < 1.) return 1. + x*x*(1.5*x-2.5);
        else if (x < 2.) return -0.5*(x-1.)*(x-2.)*(x-2.);
        else return 0.;
    }

    double Cubic::uval(double u) const
    {
        u = std::abs(u);
#ifdef USE_TABLES
        return u>_uMax ? 0. : (*_tab)(u);
#else
        double s = math::sinc(u);
        double c = cos(M_PI*u);
        return s*s*s*(3.*s-2.*c);
#endif
    }

    class CubicIntegrand : public std::unary_function<double,double>
    {
    public:
        CubicIntegrand(double u, const Cubic& c): _u(u), _c(c) {}
        double operator()(double x) const { return _c.xval(x)*std::cos(2*M_PI*_u*x); }

    private:
        double _u;
        const Cubic& _c;
    };

    double Cubic::uCalc(double u, double tolerance) const
    {
        CubicIntegrand ci(u, *this);
        return 2.*( integ::int1d(ci, 0., 1., 0.1*tolerance, 0.1*tolerance)
                    + integ::int1d(ci, 1., 2., 0.1*tolerance, 0.1*tolerance));
    }

    Cubic::Cubic(const GSParams& gsparams) : Interpolant(gsparams)
    {
        dbg<<"Start Cubic\n";
        _range = 2.;

#ifdef USE_TABLES
        double tol = gsparams.kvalue_accuracy;
        // Strangely, not all compilers correctly setup an empty map when it is a
        // static variable, so you can get seg faults using it.
        // Doing an explicit clear fixes the problem.
        if (_cache_umax.size() == 0) { _cache_umax.clear(); _cache_tab.clear(); }

        if (_cache_umax.count(tol)) {
            // Then uMax and tab are already cached.
            _tab = _cache_tab[tol];
            _uMax = _cache_umax[tol];
        } else {
            // Then need to do the calculation and then cache it.
            const double uStep =
                gsparams.table_spacing * std::pow(gsparams.kvalue_accuracy/10.,0.25);
            _uMax = 0.;
            _tab.reset(new TableBuilder(Table::spline));
            for (double u=0.; u - _uMax < 1. || u<1.1; u+=uStep) {
                double ft = uCalc(u);
#ifdef DEBUGLOGGING
                double s = math::sinc(u);
                double c = cos(M_PI*u);
                double ft2 = s*s*s*(3.*s-2.*c);
                dbg<<"u = "<<u<<", ft = "<<ft<<"  "<<ft2<<"  diff = "<<ft-ft2<<std::endl;
#endif
                _tab->addEntry(u, ft);
                if (std::abs(ft) > tol) _uMax = u;
            }
            _tab->finalize();
            // Save these values in the cache.
            _cache_tab[tol] = _tab;
            _cache_umax[tol] = _uMax;
            dbg<<"umax = "<<_uMax<<", alt umax = "<<
                std::pow((3.*sqrt(3.)/8.)/tol, 1./3.) / M_PI <<std::endl;
        }
#else
        // uMax is the value where |ft| <= tolerance
        // ft = sin(pi u)^3/(pi u)^3 * (3*sin(pi u)/(pi u) - 2*cos(pi u))
        // |ft| < 2 max[sin(x)^3 cos(x)] / (pi u)^3
        //      = 2 (3sqrt(3)/16) / (pi u)^3
        // umax = (3sqrt(3)/8 tol)^1/3 / pi
        _uMax = std::pow((3.*sqrt(3.)/8.)/gsparams.kvalue_accuracy, 1./3.) / M_PI;
#endif
    }

    std::map<double,shared_ptr<TableBuilder> > Cubic::_cache_tab;
    std::map<double,double> Cubic::_cache_umax;

    std::string Cubic::makeStr() const
    {
        std::ostringstream oss(" ");
        oss.precision(std::numeric_limits<double>::digits10 + 4);
        oss << "galsim._galsim.Cubic(";
        oss << "galsim._galsim.GSParams("<<_gsparams<<"))";
        return oss.str();
    }


    //
    // Quintic
    //

    double Quintic::xval(double x) const
    {
        x = std::abs(x);
#ifdef ALT_QUINTIC
        // Gary claims in http://arxiv.org/abs/1401.2636 that his quintic function (below) has the
        // following properties:
        //
        // f(0) = 1
        // f(1) = f(2) = f(3) = 0
        // f'(0) = 0
        // f'(1)_left = f'(1)_right
        // f'(2)_left = f'(2)_right
        // f'(3)_left = 0
        // f''(0) = 0
        // (*) f''(1)_left = f'(1)_right
        // (*) f''(2)_left = f'(2)_right
        // (*) f''(3)_left = 0
        // f(x-3)+f(x-2) + f(x-1) + f(x) + f(x+1) + f(x+2) = 1 from 0..1
        // F'(j) = F''(j) = F'''(j) = F''''(j) = 0
        //
        // However, it turns out that the second derivative continuity equations (marked * above)
        // are not actually satisfied.  I (MJ) tried to derive a version that does satisfy all
        // the constraints and discovered that the system is over-constrained.  If I keep the
        // second derivative constraints and drop F''''(j) = 0, I get the following:
        if (x <= 1.)
            return 1. + x*x*x*(-15./2. + x*(32./3. + x*(-25./6.)));
        else if (x <= 2.)
            return (x-1.)*(x-2.)*(-23./4. + x*(169./12. + x*(-39./4. + x*(25./12.))));
        else if (x <= 3.)
            return (x-2.)*(x-3.)*(x-3.)*(x-3.)*(3./4. + x*(-5./12.));
        else
            return 0.;
#else
        // This is Gary's original version with F''''(j) = 0, but f''(x) is not continuous at
        // x = 1,2,3.
        if (x <= 1.)
            return 1. + x*x*x*(-95./12. + x*(23./2. + x*(-55./12.)));
        else if (x <= 2.)
            return (x-1.)*(x-2.)*(-23./4. + x*(29./2. + x*(-83./8. + x*(55./24.))));
        else if (x <= 3.)
            return (x-2.)*(x-3.)*(x-3.)*(-9./4. + x*(25./12. + x*(-11./24.)));
        else
            return 0.;
#endif
    }

    double Quintic::uval(double u) const
    {
        u = std::abs(u);
#ifdef USE_TABLES
        return u>_uMax ? 0. : (*_tab)(u);
#else
        double s = math::sinc(u);
        double piu = M_PI*u;
        double c = cos(piu);
        double ssq = s*s;
        double piusq = piu*piu;
#ifdef ALT_QUINTIC
        return ssq*ssq*(ssq*(12.*piusq-50.) + 44.*s*c + 5.);
#else
        return s*ssq*ssq*(s*(55.-19.*piusq) + 2.*c*(piusq-27.));
#endif
#endif
    }

    class QuinticIntegrand : public std::unary_function<double,double>
    {
    public:
        QuinticIntegrand(double u, const Quintic& q): _u(u), _q(q) {}
        double operator()(double x) const { return _q.xval(x)*std::cos(2*M_PI*_u*x); }
    private:
        double _u;
        const Quintic& _q;
    };

    double Quintic::uCalc(double u, double tolerance) const
    {
        QuinticIntegrand qi(u, *this);
        return 2.*( integ::int1d(qi, 0., 1., 0.1*tolerance, 0.1*tolerance)
                    + integ::int1d(qi, 1., 2., 0.1*tolerance, 0.1*tolerance)
                    + integ::int1d(qi, 2., 3., 0.1*tolerance, 0.1*tolerance));
    }

    Quintic::Quintic(const GSParams& gsparams) : Interpolant(gsparams)
    {
        dbg<<"Start Quintic\n";
        _range = 3.;

#ifdef USE_TABLES
        double tol = gsparams.kvalue_accuracy;
        // Strangely, not all compilers correctly setup an empty map when it is a
        // static variable, so you can get seg faults using it.
        // Doing an explicit clear fixes the problem.
        if (_cache_umax.size() == 0) { _cache_umax.clear(); _cache_tab.clear(); }

        if (_cache_umax.count(tol)) {
            // Then uMax and tab are already cached.
            _tab = _cache_tab[tol];
            _uMax = _cache_umax[tol];
        } else {
            // Then need to do the calculation and then cache it.
            const double uStep =
                gsparams.table_spacing * std::pow(gsparams.kvalue_accuracy/10.,0.25);
            _uMax = 0.;
            _tab.reset(new TableBuilder(Table::spline));
            for (double u=0.; u - _uMax < 1. || u<1.1; u+=uStep) {
                dbg<<"u = "<<u<<std::endl;
                double ft = uCalc(u);
                _tab->addEntry(u, ft);
#ifdef DEBUGLOGGING
                double s = math::sinc(u);
                double piu = M_PI*u;
                double c = cos(piu);
                double ssq = s*s;
                double piusq = piu*piu;
#ifdef ALT_QUINTIC
                double ft2 = ssq*ssq*(ssq*(12.*piusq-50.) + 44.*s*c+5.);
#else
                double ft2 = s*ssq*ssq*(s*(55.-19.*piusq) + 2.*c*(piusq-27.));
#endif
                dbg<<"u = "<<u<<", ft = "<<ft<<"  "<<ft2<<"  diff = "<<ft-ft2<<std::endl;
#endif
                if (std::abs(ft) > tol) _uMax = u;
            }
            _tab->finalize();
            // Save these values in the cache.
            _cache_tab[tol] = _tab;
            _cache_umax[tol] = _uMax;
            dbg<<"umax = "<<_uMax<<", alt umax = "<<
                std::pow((25.*sqrt(5.)/108.)/tol, 1./3.) / M_PI <<std::endl;
        }
#else
        // uMax is the value where |ft| <= tolerance
        // ft = sin(pi u)^5/(pi u)^5 * (sin(pi u)/(pi u)*(55.-19 pi^2 u^2)
        //                              + 2*cos(pi u)*(pi^2 u^2-27)))
        // |ft| < 2 max[sin(x)^5 cos(x))] / (pi u)^3
        //      = 2 (25sqrt(5)/216) / (pi u)^3
        // umax = (25sqrt(5)/108 tol)^1/3 / pi
        _uMax = std::pow((25.*sqrt(5.)/108.)/gsparams.kvalue_accuracy, 1./3.) / M_PI;
#endif
    }

    // Override default sampler configuration because Quintic filter has sign change in
    // outer interval
    void Quintic::checkSampler() const
    {
        if (_sampler.get()) return;
        std::vector<double> ranges(8);
        ranges[0] = -3.;
        ranges[1] = -(1./11.)*(25.+sqrt(31.));  // This is the extra zero-crossing
        ranges[2] = -2.;
        ranges[3] = -1.;
        for (int i=0; i<4; i++)
            ranges[7-i] = -ranges[i];
        _sampler.reset(new OneDimensionalDeviate(_interp, ranges, false, 1.0, _gsparams));
    }

    std::map<double,shared_ptr<TableBuilder> > Quintic::_cache_tab;
    std::map<double,double> Quintic::_cache_umax;

    std::string Quintic::makeStr() const
    {
        std::ostringstream oss(" ");
        oss.precision(std::numeric_limits<double>::digits10 + 4);
        oss << "galsim._galsim.Quintic(";
        oss << "galsim._galsim.GSParams("<<_gsparams<<"))";
        return oss.str();
    }


    //
    // Lanczos
    //

    double Lanczos::xCalc(double x) const
    {
        xassert(x >= 0);
        xassert(x <= _nd);

        double res; // res will be the result to return.
        double s;   // s will be sin(pi x) which we save for the flux conservation correction.
        if (x > 1.e-4) {
            // For low values of n, we can save some time by calculating sin(pi x)
            // from the value of sin(pi x / n) using trig identities.
            //
            // At some point it might be worth implementing the same trick as we did with
            // SBMoffat's kValue and pow functions, making these different cases all different
            // functions and having the constructor just set the function once.  Then calls to
            // xval wouldn't have any jumps from the case or (if you wanted) even the
            // _conserve_dc check.
            switch (_n) {
              case 1 : {
                  // Then xval = 1/pi^2 sin(pi x)^2 / x^2
                  s = sin(M_PI*x);
                  double temp = s/(M_PI * x);
                  res = temp*temp;
                  break;
              }
              case 2 : {
                  // Then xval = 2/pi^2 sin(pi x) sin(pi x/2) / x^2
                  // Let sn = sin(pi x/2), cn = cos(pi x/2)
                  // Then sin(pi x) = 2 * sn * cn
                  // xval = 4/pi^2 sn^2 cn / x^2
                  double sn, cn;
                  math::sincos(x * M_PI/2., sn, cn);
                  s = 2.*sn*cn;
                  res = (2./(M_PI*M_PI)) * s*sn/(x*x);
                  break;
              }
              case 3 : {
                  // Then xval = 3/pi^2 sin(pi x) sin(pi x/3) / x^2
                  // Let sn = sin(pi x/3)
                  // Then sin(pi x) = sn*(3-4sn^2)
                  // xval = 3/pi^2 sn^2*(3-4sn) / x^2
                  double sn = sin((M_PI/3.)*x);
                  s = sn*(3.-4.*sn*sn);
                  res = (3./(M_PI*M_PI)) * s*sn/(x*x);
                  break;
              }
              case 4 : {
                  double sn, cn;
                  math::sincos(x * M_PI/4, sn, cn);
                  s = sn*cn*(4.-8.*sn*sn);
                  res = (4./(M_PI*M_PI)) * s*sn/(x*x);
                  break;
              }
              case 5 : {
                  double sn = sin((M_PI/5.)*x);
                  double snsq = sn*sn;
                  s = sn*(5.-snsq*(20.-16.*snsq));
                  res = (5./(M_PI*M_PI)) * s*sn/(x*x);
                  break;
              }
              case 6 : {
                  double sn, cn;
                  math::sincos(x * M_PI/6., sn, cn);
                  double snsq = sn*sn;
                  s = sn*cn*(6.-32.*snsq*(1.-snsq));
                  res = (6./(M_PI*M_PI)) * s*sn/(x*x);
                  break;
              }
              case 7 : {
                  double sn = sin((M_PI/7.)*x);
                  double snsq = sn*sn;
                  s = sn*(7.-snsq*(56.-snsq*(112.-64.*snsq)));
                  res = (7./(M_PI*M_PI)) * s*sn/(x*x);
              }
              default : {
                  // Above n=7, there isn't much advantage anymore to specialization.
                  // The second sin call isn't much slower than the multiplications
                  // required to get sin(pi x) from sin(pi x/n)
                  s = sin(M_PI*x);
                  double sn = sin(M_PI*x/_nd);
                  res = (_nd/(M_PI*M_PI)) * s*sn/(x*x);
                  break;
              }
            }
        } else { // x < 1.e-4
            // res = n/(pi x)^2 * sin(pi x) * sin(pi x / n)
            //     ~= (1 - 1/6 pix^2) * (1 - 1/6 pix^2 / n^2)
            //     = 1 - 1/6 pix^2 ( 1 + 1/n^2 )
            double pix = M_PI*x;
            double temp = (1./6.) * pix*pix;
            s = pix * (1. - temp);
            res = 1. - temp * (1. + 1./(_nd*_nd));
            // For x < 1.e-4, the errors in this approximation are less than 1.e-16.
        }

        // Gary's write up about this is in http://arxiv.org/abs/1401.2636.
        // We start with Gary's eqn 22, and extend the subsequent derivation to 3rd order.
        // (More in uCalc below than here...)
        //
        // An image with uniform f(x) = 1 when interpolated with Lanczos will have an error of:
        // E(x) = 2 * Sum_j K(j) (cos(2 pi j x) - 1)
        //      = -2 K(1) (1-cos(2pix)) - 2 K(2) (1-cos(4pix)) - 2 K(3) (1-cos(6pix)) ...
        //
        // To preserve a uniform flux, we want to divide by (1 + the above value) to correct
        // for the error.
        //
        // Unfortunately, it turns out that while K(1) << 1, the series from there on starts
        // to converge more slowly, so the gains from each subsequent term become less.
        // For n=3, the values of K(1)..K(4) are: 1.416e-3, 4.390e-5, 7.716e-6, 2.343e-6.
        // Thus, it would be hard to use this method to get to significantly better accuracy
        // than about 1.e-6.
        //
        // To give feel for how this correction goes, a 2-d unit flux field interpolated
        // with Lanczos, n=3, has the following maximum errors:
        //
        // With no correction: 1.13e-2
        // With _K1:           3.98e-4
        // With _K2:           8.36e-5
        // With _K3:           3.02e-5
        // With _K4:           1.39e-5
        // With _K5:           7.27e-6
        //
        // I stopped here, since we have other approximations that are only accurate to 1.e-5.
        // But certainly, it will be hard to get much more accurate that this, at least with
        // this framework for the correction.

        // res /= 1. - 2.*_K1*(1.-cos(2.*M_PI*x)) - 2*_K2*(1.-cos(4.*M_PI*x)) - ...;
        if (_conserve_dc) {
            dbg<<"xCalc for x = "<<x<<std::endl;
            dbg<<"res = "<<res<<" / ";
            double ssq = s*s;
            double factor = (1.
                             - 4.*_K[1]*ssq
                             - 16.*_K[2]*ssq*(1.-ssq)
                             - 4.*_K[3]*ssq*(9.-ssq*(24.-16.*ssq))
                             - 64.*_K[4]*ssq*(1.-ssq*(5.-ssq*(8.-4.*ssq)))
                             - 4.*_K[5]*ssq*(25.-ssq*(200.-ssq*(560.-ssq*(640.-256.*ssq)))));
            res /= factor;
#ifdef DEBUGLOGGING
            dbg<<factor<<" = "<<res<<std::endl;
            dbg<<"factor = 1 - "<<2.*_K[1]*(1.-std::cos(2.*M_PI*x))
                <<" - "<<2.*_K[2]*(1.-std::cos(4.*M_PI*x))
                <<" - "<<2.*_K[3]*(1.-std::cos(6.*M_PI*x))
                <<" - "<<2.*_K[4]*(1.-std::cos(8.*M_PI*x))
                <<" - "<<2.*_K[5]*(1.-std::cos(10.*M_PI*x))<<" = "
                << (1.
                    - 2.*_K[1]*(1.-std::cos(2.*M_PI*x))
                    - 2.*_K[2]*(1.-std::cos(4.*M_PI*x))
                    - 2.*_K[3]*(1.-std::cos(6.*M_PI*x))
                    - 2.*_K[4]*(1.-std::cos(8.*M_PI*x))
                    - 2.*_K[5]*(1.-std::cos(10.*M_PI*x)))
                <<" = "<<factor<<std::endl;
#endif
        }
        return res;
    }

    double Lanczos::uCalcRaw(double u) const
    {
        // F(u) = ( (vp+1) Si((vp+1)pi) - (vp-1) Si((vp-1)pi) +
        //          (vm-1) Si((vm-1)pi) - (vm+1) Si((vm+1)pi) ) / 2pi
        double vp=_nd*(2.*u+1.);
        double vm=_nd*(2.*u-1.);
        double retval = (vm-1.)*math::Si(M_PI*(vm-1.))
            -(vm+1.)*math::Si(M_PI*(vm+1.))
            -(vp-1.)*math::Si(M_PI*(vp-1.))
            +(vp+1.)*math::Si(M_PI*(vp+1.));
        return retval/(2.*M_PI);
    }

    double Lanczos::uCalc(double u) const
    {
        double retval = uCalcRaw(u);
        // The correction (described in xCalc) to preserve a uniform flux profile can be
        // approximate by its series approximation, where I throw out terms that are 3rd
        // order or higher in the coefficients (K1^3 ~ 3.e-9, so negligible), and the only
        // 2nd order terms I keep have K1 as one of the terms (K2^2 ~ 2.e-9).
        //
        // (1+E(x))^-1 ~= 1 + 2K(1) (1-cos(2pix)) + 4K(1)^2 (1-cos(2pix))^2
        //                  + 2K(2) (1-cos(4pix)) + 4K(1)K(2) (1-cos(2pix)) (1-cos(4pix))
        //                  + 2K(3) (1-cos(6pix)) + 4K(1)K(3) (1-cos(2pix)) (1-cos(6pix))
        //                  + 2K(4) (1-cos(8pix)) + 2K(5) (1-cos(10pix))
        //
        // The effect in the Fourier transform will then be a convolution by the fourier transform
        // of (1+E(x))^-1:
        //
        // F[(1+E(x))^-1] = 2pi (
        //     D(k)
        //     + K(1) (-D(k-2pi) + 2 D(k) - D(k+2pi))
        //     + K(1)^2 (D(k-4pi) - 4 D(k-2pi) + 6 D(k) - 4 D(k+2pi) + D(k+4pi))
        //     + K(2) (-D(k-4pi) + 2 D(k) - D(k+4pi))
        //     + K(1) K(2) (D(k-6pi) - 2 D(k-4pi) - D(k-2pi) + 4 D(k) - D(k+2pi)
        //                  - 2 D(k+4pi) + D(k+6pi))
        //     + K(3) (-D(k-6pi) + 2 D(k) - D(k+6pi))
        //     + K(1) K(3) (D(k-8pi) - 2 D(k-6pi) + D(k-4pi) - 2 D(k-2pi) + 4 D(k)
        //                  - 2 D(k+2pi) + D(k+4pi) - 2 D(k+6pi) + D(k+8pi))
        //     + K(4) (-D(k-8pi) + 2 D(k) - D(k+8pi))
        //     + K(5) (-D(k-10pi) + 2 D(k) - D(k+10pi))
        //     )
        //
        // where D(k) is the Dirac delta function.
        //
        // When convolved with the original F(u) (since we are multiplying in real space, it
        // becomes a convolution in k-space), we get:
        //
        // (1 + 2K1 + 6K1^2 + 2K2 + 2K1 K2 + 2K3 + 2K1 K3 + 2K4 + 2K5) F(u)
        // + (-K1 - 4K1^2 - K1 K2 - 2K1 K3) ( F(u-1) + F(u+1) )
        // + (K1^2 - K2 - 2K1 K2 + K1 K3) ( F(u-2) + F(u+2) )
        // + (K1 K2 - K3 - 2K1 K3) ( F(u-3) + F(u+3) )
        // + (K1 K3 - K4) ( F(u-4) + F(u+4) )
        // + (-K5) ( F(u-5) + F(u+5) )
        //
        // These coefficients are constant, so they are stored in _C.

        if (_conserve_dc) {
            retval *= _C[0];
            retval += _C[1] * (uCalcRaw(u+1.) + uCalcRaw(u-1.));
            retval += _C[2] * (uCalcRaw(u+2.) + uCalcRaw(u-2.));
            retval += _C[3] * (uCalcRaw(u+3.) + uCalcRaw(u-3.));
            retval += _C[4] * (uCalcRaw(u+4.) + uCalcRaw(u-4.));
            retval += _C[5] * (uCalcRaw(u+5.) + uCalcRaw(u-5.));
        }
        return retval;
    }

    Lanczos::Lanczos(int n, bool conserve_dc, const GSParams& gsparams) :
        Interpolant(gsparams), _n(n), _nd(n), _conserve_dc(conserve_dc)
    {
        dbg<<"Start constructor for Lanczos n = "<<n<<std::endl;
        // Reduce range slightly from n so we're not including points with zero weight in
        // interpolations:
        _range = _nd;
        double tol = gsparams.kvalue_accuracy;

        for(double u=0.;u<=10.;u+=0.1) dbg<<"F("<<u<<") = "<<uCalcRaw(u)<<std::endl;

        _K.resize(6);
        _K[1] = uCalcRaw(1.);
        _K[2] = uCalcRaw(2.);
        _K[3] = uCalcRaw(3.);
        _K[4] = uCalcRaw(4.);
        _K[5] = uCalcRaw(5.);
        dbg<<"K1,2,3,4,5 = "<<_K[1]<<','<<_K[2]<<','<<_K[3]<<','<<_K[4]<<','<<_K[5]<<std::endl;

        // See comments in _uCalc above.
        // C0 = 1 + 2K1 + 6K1^2 + 2K2 + 2K1 K2 + 2K3 + 2K1 K3 + 2K4 + 2K5
        // C1 = -K1 - 4K1^2 - K1 K2 - 2K1 K3
        // C2 = K1^2 - K2 - 2K1 K2 + K1 K3
        // C3 = K1 K2 - K3 - 2K1 K3
        // C4 = K1 K3 - K4
        // C5 = -K5
        _C.resize(6);
        _C[0] = 1. + 2.*(_K[1]*(1. + 3.*_K[1] + _K[2] + _K[3]) + _K[2] + _K[3] + _K[4] + _K[5]);
        _C[1] = -_K[1] * (1. + 4.*_K[1] + _K[2] + 2.*_K[3]);
        _C[2] = _K[1]*(_K[1] - 2.*_K[2] + _K[3]) - _K[2];
        _C[3] = _K[1]*(_K[1] - 2.*_K[3]) - _K[3];
        _C[4] = _K[1]*_K[3] - _K[4];
        _C[5] = -_K[5];
        dbg<<"C0,1,2,3,4,5 = "<<_C[0]<<','<<_C[1]<<','<<_C[2]<<','<<_C[3]<<','<<_C[4]
            <<','<<_C[5]<<std::endl;

        for (double x=0.; x<1.; x+=0.1) {
            dbg<<"S("<<x<<") = ";
            double sum = 0.;
            for (int i=-_n;i<_n;++i) {
                double val = math::sinc(x+i)*math::sinc((x+i)/_nd);
                sum += val;
                dbg<<val<<" + ";
            }
            dbg<<" = "<<sum<<std::endl;
            dbg<<"Nominal S("<<x<<") = "<<
                (1.
                 - 2.*_K[1]*(1.-std::cos(2.*M_PI*x))
                 - 2.*_K[2]*(1.-std::cos(4.*M_PI*x))
                 - 2.*_K[3]*(1.-std::cos(6.*M_PI*x))
                 - 2.*_K[4]*(1.-std::cos(8.*M_PI*x))
                 - 2.*_K[5]*(1.-std::cos(10.*M_PI*x))) << std::endl;
        }

        // Strangely, not all compilers correctly setup an empty map when it is a
        // static variable, so you can get seg faults using it.
        // Doing an explicit clear fixes the problem.
        if (_cache_umax.size() == 0) {
            _cache_umax.clear();
#ifdef USE_TABLES
            _cache_xtab.clear();
#endif
            _cache_utab.clear();
        }

        KeyType key(n,std::pair<bool,double>(_conserve_dc,tol));

        if (_cache_umax.count(key)) {
            // Then uMax and tab are already cached.
#ifdef USE_TABLES
            _xtab = _cache_xtab[key];
#endif
            _utab = _cache_utab[key];
            _uMax = _cache_umax[key];
        } else {
#ifdef USE_TABLES
            // Build xtab = table of x values
            _xtab.reset(new TableBuilder(Table::spline));
            // Spline is accurate to O(dx^3), so errors should be ~dx^4.
            const double xStep1 =
                gsparams.table_spacing * std::pow(gsparams.xvalue_accuracy/10.,0.25);
            // Make sure steps hit the integer values exactly.
            const double xStep = 1. / std::ceil(1./xStep1);
            for(double x=0.; x<_nd; x+=xStep) _xtab->addEntry(x, xCalc(x));
            _xtab->finalize();
#endif

            // Build utab = table of u values
            _utab.reset(new TableBuilder(Table::spline));
            // The peak second derivative of the Lanczos kernel if Fourier space empirically
            // seems to a bit over ~100.  Use 200 to be conservative, so this mean use
            // h = (kvalue_accuracy/200)**0.25
            const double uStep =
                gsparams.table_spacing * std::pow(gsparams.kvalue_accuracy/200.,0.25) / _nd;
            _uMax = 0.;
            for (double u=0.; u - _uMax < 1./_nd || u<1.1; u+=uStep) {
                double uval = uCalc(u);
                _utab->addEntry(u, uval);
                if (std::abs(uval) > gsparams.kvalue_accuracy) _uMax = u;
            }
            _utab->finalize();
            // Save these values in the cache.
#ifdef USE_TABLES
            _cache_xtab[key] = _xtab;
#endif
            _cache_utab[key] = _utab;
            _cache_umax[key] = _uMax;
        }
    }

    std::map<Lanczos::KeyType,shared_ptr<TableBuilder> > Lanczos::_cache_xtab;
    std::map<Lanczos::KeyType,shared_ptr<TableBuilder> > Lanczos::_cache_utab;
    std::map<Lanczos::KeyType,double> Lanczos::_cache_umax;

    double Lanczos::xval(double x) const
    {
        x = std::abs(x);
        if (x >= _nd) return 0.;
        else {
#ifdef USE_TABLES
            return (*_xtab)(x);
#else
            return xCalc(x);
#endif
        }
    }

    double Lanczos::uval(double u) const
    {
        // For this one, we always use the lookup table.
        u = std::abs(u);
        return u>_uMax ? 0. : (*_utab)(u);
    }

    std::string Lanczos::makeStr() const
    {
        std::ostringstream oss(" ");
        oss.precision(std::numeric_limits<double>::digits10 + 4);
        oss << "galsim._galsim.Lanczos("<<_n<<", ";
        if (_conserve_dc) oss << "True, ";
        else oss << "False, ";
        oss << "galsim._galsim.GSParams("<<_gsparams<<"))";
        return oss.str();
    }

}
