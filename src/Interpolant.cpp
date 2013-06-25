// -*- c++ -*-
/*
 * Copyright 2012, 2013 The GalSim developers:
 * https://github.com/GalSim-developers
 *
 * This file is part of GalSim: The modular galaxy image simulation toolkit.
 *
 * GalSim is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * GalSim is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with GalSim.  If not, see <http://www.gnu.org/licenses/>
 */

//#define DEBUGLOGGING

#include "Interpolant.h"
#include "integ/Int.h"
#include "SBProfile.h"

#ifdef DEBUGLOGGING
#include <fstream>
//std::ostream* dbgout = new std::ofstream("debug.out");
//int verbose_level = 2;
#endif

// It turns out the Lanczos xval is faster to do directly with trig than with a lookup table.
// I left the original lookup-table code in here in case we find that this isn't true for
// some situation.  But I think this next line should always be enabled. -MJ
#define USE_DIRECT_LANCZOS_XVAL

namespace galsim {


    //
    // Some auxilliary functions we will want: 
    //

    // sinc(x) is defined here as sin(Pi x) / (Pi x)
    static double sinc(double x) 
    {
        if (std::abs(x) < 1.e-4) return 1.- (M_PI*M_PI/6.)*x*x; 
        else return std::sin(M_PI*x)/(M_PI*x);
    }

    // Utility for calculating the integral of sin(t)/t from 0 to x.  Note the official definition
    // does not have pi multiplying t.
    static double Si(double x) 
    {
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
    }

    //
    // Generic InterpolantXY class methods
    //
    
    double InterpolantFunction::operator()(double x) const  { return _interp.xval(x); }

    double InterpolantXY::getPositiveFlux() const 
    {
        return _i1d->getPositiveFlux()*_i1d->getPositiveFlux()
            + _i1d->getNegativeFlux()*_i1d->getNegativeFlux();
    }

    double InterpolantXY::getNegativeFlux() const 
    { return 2.*_i1d->getPositiveFlux()*_i1d->getNegativeFlux(); }

    boost::shared_ptr<PhotonArray> InterpolantXY::shoot(int N, UniformDeviate ud) const 
    {
        dbg<<"InterpolantXY shoot: N = "<<N<<std::endl;
        dbg<<"Target flux = 1.\n";
        // Going to assume here that there is not a need to randomize any Interpolant
        boost::shared_ptr<PhotonArray> result = _i1d->shoot(N, ud);   // get X coordinates
        result->takeYFrom(*_i1d->shoot(N, ud));
        dbg<<"InterpolantXY Realized flux = "<<result->getTotalFlux()<<std::endl;
        return result;
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


    //
    // Delta
    //
    
    boost::shared_ptr<PhotonArray> Delta::shoot(int N, UniformDeviate ud) const 
    {
        dbg<<"InterpolantXY shoot: N = "<<N<<std::endl;
        dbg<<"Target flux = 1.\n";
        boost::shared_ptr<PhotonArray> result(new PhotonArray(N));
        double fluxPerPhoton = 1./N;
        for (int i=0; i<N; i++)  {
            result->setPhoton(i, 0., 0., fluxPerPhoton);
        }
        dbg<<"Delta Realized flux = "<<result->getTotalFlux()<<std::endl;
        return result;
    }


    //
    // Nearest
    //
    
    double Nearest::xval(double x) const 
    {
        if (std::abs(x)>0.5) return 0.;
        else if (std::abs(x)<0.5) return 1.;
        else return 0.5;
    }

    double Nearest::uval(double u) const { return sinc(u); }

    boost::shared_ptr<PhotonArray> Nearest::shoot(int N, UniformDeviate ud) const 
    {
        dbg<<"InterpolantXY shoot: N = "<<N<<std::endl;
        dbg<<"Target flux = 1.\n";
        boost::shared_ptr<PhotonArray> result(new PhotonArray(N));
        double fluxPerPhoton = 1./N;
        for (int i=0; i<N; i++)  {
            result->setPhoton(i, ud()-0.5, 0., fluxPerPhoton);
        }
        dbg<<"Nearest Realized flux = "<<result->getTotalFlux()<<std::endl;
        return result;
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

    double SincInterpolant::xval(double x) const { return sinc(x); }

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

    boost::shared_ptr<PhotonArray> SincInterpolant::shoot(int N, UniformDeviate ud) const 
    {
        throw std::runtime_error("Photon shooting is not practical with sinc Interpolant");
        return boost::shared_ptr<PhotonArray>();
    }


    //
    // Linear
    //

    double Linear::xval(double x) const 
    {
        x=std::abs(x);
        if (x > 1.) return 0.;
        else return 1.-x;
    }
    double Linear::uval(double u) const 
    { 
        double sincu = sinc(u);
        return sincu*sincu;
    }

    boost::shared_ptr<PhotonArray> Linear::shoot(int N, UniformDeviate ud) const 
    {
        dbg<<"InterpolantXY shoot: N = "<<N<<std::endl;
        dbg<<"Target flux = 1.\n";
        boost::shared_ptr<PhotonArray> result(new PhotonArray(N));
        double fluxPerPhoton = 1./N;
        for (int i=0; i<N; i++) {
            // *** Guessing here that 2 random draws is faster than a sqrt:
            result->setPhoton(i, ud() + ud() - 1., 0., fluxPerPhoton);
        }
        dbg<<"Linear Realized flux = "<<result->getTotalFlux()<<std::endl;
        return result;
    }


    //
    // Cubic
    //

    double Cubic::xval(double x) const 
    { 
        x = std::abs(x);
        if (x < 1) return 1. + x*x*(1.5*x-2.5);
        else if (x < 2) return -0.5*(x-1.)*(x-2.)*(x-2.);
        else return 0.;
    }

    double Cubic::uval(double u) const 
    {
        u = std::abs(u);
        return u>_uMax ? 0. : (*_tab)(u);
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

    double Cubic::uCalc(double u) const 
    {
        CubicIntegrand ci(u, *this);
        return 2.*( integ::int1d(ci, 0., 1., 0.1*_tolerance, 0.1*_tolerance)
                    + integ::int1d(ci, 1., 2., 0.1*_tolerance, 0.1*_tolerance));
    }

    Cubic::Cubic(double tol, const GSParamsPtr& gsparams) : 
        Interpolant(gsparams), _tolerance(tol)
    {
        // Reduce range slightly from n so we're not including points with zero weight in
        // interpolations:
        _range = 2.-0.1*_tolerance;

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
                gsparams->table_spacing * std::pow(gsparams->kvalue_accuracy/10.,0.25);
            _uMax = 0.;
            _tab.reset(new Table<double,double>(Table<double,double>::spline));
            for (double u=0.; u - _uMax < 1. || u<1.1; u+=uStep) {
                double ft = uCalc(u);
                _tab->addEntry(u, ft);
                if (std::abs(ft) > _tolerance) _uMax = u;
            }
            // Save these values in the cache.
            _cache_tab[tol] = _tab;
            _cache_umax[tol] = _uMax;
        }
    }

    std::map<double,boost::shared_ptr<Table<double,double> > > Cubic::_cache_tab;
    std::map<double,double> Cubic::_cache_umax;


    //
    // Quintic
    //
 
    double Quintic::xval(double x) const 
    {
        x = std::abs(x);
        if (x <= 1.)
            return 1. + (1./12.)*x*x*x*(-95.+x*(138.-55.*x));
        else if (x <= 2.)
            return (1./24.)*(x-1.)*(x-2.)*(-138.+x*(348.+x*(-249.+55.*x)));
        else if (x <= 3.)
            return (1./24.)*(x-2.)*(x-3.)*(x-3.)*(-54.+x*(50.-11.*x));
        else 
            return 0.;
    }

    double Quintic::uval(double u) const 
    {
        u = std::abs(u);
        return u>_uMax ? 0. : (*_tab)(u);
    }

    class QuinticIntegrand : public std::unary_function<double,double>
    {
    public:
        QuinticIntegrand(double u, const Quintic& c): _u(u), _c(c) {}
        double operator()(double x) const { return _c.xval(x)*std::cos(2*M_PI*_u*x); }
    private:
        double _u;
        const Quintic& _c;
    };

    double Quintic::uCalc(double u) const 
    {
        QuinticIntegrand qi(u, *this);
        return 2.*( integ::int1d(qi, 0., 1., 0.1*_tolerance, 0.1*_tolerance)
                    + integ::int1d(qi, 1., 2., 0.1*_tolerance, 0.1*_tolerance)
                    + integ::int1d(qi, 2., 3., 0.1*_tolerance, 0.1*_tolerance));
    }

    Quintic::Quintic(double tol, const GSParamsPtr& gsparams) :
        Interpolant(gsparams), _tolerance(tol)
    {
        // Reduce range slightly from n so we're not including points with zero weight in
        // interpolations:
        _range = 3.-0.1*_tolerance;

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
                gsparams->table_spacing * std::pow(gsparams->kvalue_accuracy/10.,0.25);
            _uMax = 0.;
            _tab.reset(new Table<double,double>(Table<double,double>::spline));
            for (double u=0.; u - _uMax < 1. || u<1.1; u+=uStep) {
                double ft = uCalc(u);
                _tab->addEntry(u, ft);
                if (std::abs(ft) > _tolerance) _uMax = u;
            }
            // Save these values in the cache.
            _cache_tab[tol] = _tab;
            _cache_umax[tol] = _uMax;
        }
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
        _sampler.reset(new OneDimensionalDeviate(_interp, ranges, false, _gsparams));
    }

    std::map<double,boost::shared_ptr<Table<double,double> > > Quintic::_cache_tab;
    std::map<double,double> Quintic::_cache_umax;


    //
    // Lanczos
    //
    
    double Lanczos::xCalc(double x) const
    {
        double retval = sinc(x)*sinc(x/_n);
        if (_fluxConserve) retval *= 1. + 2.*_u1*(1.-std::cos(2.*M_PI*x));
        return retval;
    }

    double Lanczos::uCalc(double u) const 
    {
        double vp=_n*(2.*u+1.);
        double vm=_n*(2.*u-1.);
        double retval = (vm-1.)*Si(M_PI*(vm-1.))
            -(vm+1.)*Si(M_PI*(vm+1.))
            -(vp-1.)*Si(M_PI*(vp-1.))
            +(vp+1.)*Si(M_PI*(vp+1.));
        return retval/(2.*M_PI);
    }

    Lanczos::Lanczos(int n, bool fluxConserve, double tol,
                     const GSParamsPtr& gsparams) :  
        Interpolant(gsparams), _in(n), _n(n), _fluxConserve(fluxConserve), _tolerance(tol)
    {
        // Reduce range slightly from n so we're not including points with zero weight in
        // interpolations:
        _range = _n*(1-0.1*std::sqrt(_tolerance));

        _u1 = uCalc(1.);

        // Strangely, not all compilers correctly setup an empty map when it is a 
        // static variable, so you can get seg faults using it.
        // Doing an explicit clear fixes the problem.
        if (_cache_umax.size() == 0) {
            _cache_umax.clear();
#ifndef USE_DIRECT_LANCZOS_XVAL
            _cache_xtab.clear();
#endif
            _cache_utab.clear();  
        }

        KeyType key(n,std::pair<bool,double>(fluxConserve,tol));

        if (_cache_umax.count(key)) {
            // Then uMax and tab are already cached.
#ifndef USE_DIRECT_LANCZOS_XVAL
            _xtab = _cache_xtab[key];
#endif
            _utab = _cache_utab[key];
            _uMax = _cache_umax[key];
        } else {
#ifndef USE_DIRECT_LANCZOS_XVAL
            // Build xtab = table of x values
            _xtab.reset(new Table<double,double>(Table<double,double>::spline));
            // Spline is accurate to O(dx^3), so errors should be ~dx^4.
            const double xStep1 = 
                gsparams->table_spacing * std::pow(gsparams->xvalue_accuracy/10.,0.25);
            // Make sure steps hit the integer values exactly.
            const double xStep = 1. / std::ceil(1./xStep1);
            for(double x=0.; x<_n; x+=xStep) _xtab->addEntry(x, xCalc(x));
#endif

            // Build utab = table of u values
            _utab.reset(new Table<double,double>(Table<double,double>::spline));
            const double uStep = 
                gsparams->table_spacing * std::pow(gsparams->kvalue_accuracy/10.,0.25) / _n;
            _uMax = 0.;
            if (_fluxConserve) {
                for (double u=0.; u - _uMax < 1./_n || u<1.1; u+=uStep) {
                    double uval = uCalc(u);
                    uval *= 1.+2.*_u1;
                    uval -= _u1*uCalc(u+1.);
                    uval -= _u1*uCalc(u-1.);
                    _utab->addEntry(u, uval);
                    if (std::abs(uval) > _tolerance) _uMax = u;
                }
            } else {
                for (double u=0.; u - _uMax < 1./_n || u<1.1; u+=uStep) {
                    double uval = uCalc(u);
                    _utab->addEntry(u, uval);
                    if (std::abs(uval) > _tolerance) _uMax = u;
                }
            }
            // Save these values in the cache.
#ifndef USE_DIRECT_LANCZOS_XVAL
            _cache_xtab[key] = _xtab;
#endif
            _cache_utab[key] = _utab;
            _cache_umax[key] = _uMax;
        }
    }

    std::map<Lanczos::KeyType,boost::shared_ptr<Table<double,double> > > Lanczos::_cache_xtab;
    std::map<Lanczos::KeyType,boost::shared_ptr<Table<double,double> > > Lanczos::_cache_utab;
    std::map<Lanczos::KeyType,double> Lanczos::_cache_umax;

    double Lanczos::xval(double x) const
    {
        x = std::abs(x);
        if (x >= _n) return 0.;
        else {
#ifdef USE_DIRECT_LANCZOS_XVAL
            switch (_in) {
                // TODO: We usually only use n=3,5,7.  If we start using any other values on a
                // regular basis, it's worth it to specialize that case here.
                // Also, at some point it might be worth doing the same trick we did with 
                // SBMoffat's kValue and pow functions, making these different cases all different 
                // functions and having the constructor just set the function once.  Then calls to 
                // xval wouldn't have any jumps from the case or (if you wanted) even the
                // _fluxConserve check.
              case 3 : {
                  if (x < 1.e-6) {
                      double xsq = x*x;
                      double res = 1. - 5./3. * xsq;
                      if (_fluxConserve) res *= 1. + 4.*M_PI*M_PI*_u1*xsq;
                      return res;
                  } else {
                      // Then xval = 3/pi^2 sin(pi x) sin(pi x /3) / x^2
                      // Let s = sin(pi x /3)
                      // Then sin(pi x) = s*(3-4s^2)
                      // xval = 3/pi^2 s^2*(3-4s) / x^2
                      double sn = sin((M_PI/3.)*x);
                      double s = sn*(3.-4.*sn*sn);
                      double res = (3./(M_PI*M_PI)) * s*sn/(x*x);
                      if (_fluxConserve) res *= 1. + 4.*_u1*s*s;
                      return res;
                  }
                  break;
              }
              case 5 : {
                  if (x < 1.e-6) {
                      double xsq = x*x;
                      double res = 1. - 13./3. * xsq;
                      if (_fluxConserve) res *= 1. + 4.*M_PI*M_PI*_u1*xsq;
                      return res;
                  } else {
                      double sn = sin((M_PI/5.)*x);
                      double snsq = sn*sn;
                      double s = sn*(5.-snsq*(20.-16.*snsq));
                      double res = (5./(M_PI*M_PI)) * s*sn/(x*x);
                      if (_fluxConserve) res *= 1. + 4.*_u1*s*s;
                      return res;
                  }
                  break;
              }
              case 7 : {
                  if (x < 1.e-6) {
                      double xsq = x*x;
                      double res = 1. - 25./3. * xsq;
                      if (_fluxConserve) res *= 1. + 4.*M_PI*M_PI*_u1*xsq;
                      return res;
                  } else {
                      double sn = sin((M_PI/7.)*x);
                      double snsq = sn*sn;
                      double s = sn*(7.-snsq*(56.-snsq*(112.-64.*snsq)));
                      double res = (7./(M_PI*M_PI)) * s*sn/(x*x);
                      if (_fluxConserve) res *= 1. + 4.*_u1*s*s;
                      return res;
                  }
                  break;
              }
              default : {
                  if (x < 1.e-6) {
                      double xsq = x*x;
                      double res = 1. - (_n*_n+1.)/6. * xsq;
                      if (_fluxConserve) res *= 1. + 4.*M_PI*M_PI*_u1*xsq;
                      return res;
                  } else {
                      // xval = n/pi^2 sin(pi x) sin(pi x /n) / x^2
                      double s = sin(M_PI*x);
                      double sn = sin(M_PI*x/_n);
                      double res = (_n/(M_PI*M_PI)) * s*sn/(x*x);
                      // res *= 1. + 2.*_u1*(1.-cos(2.*M_PI*x));
                      if (_fluxConserve) res *= 1. + 4.*_u1*s*s;
                      return res;
                  }
                  break;
              }
            }
#else
            return (*_xtab)(x);
#endif
        }
    }

    double Lanczos::uval(double u) const
    {
        u = std::abs(u);
        return u>_uMax ? 0. : (*_utab)(u);
    }

}
