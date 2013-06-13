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

namespace galsim {

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

    double Lanczos::xCalc(double x) const
    {
        double retval = sinc(x)*sinc(x/_n);
        if (_fluxConserve) retval *= 1. + 2.*_k1*(1.-std::cos(2.*M_PI*x));
        return retval;
    }

    double Lanczos::kCalc(double k) const 
    {
        double vp=_n*(2*k+1);
        double vm=_n*(2*k-1);
        double retval = (vm-1.)*Si(M_PI*(vm-1.))
            -(vm+1.)*Si(M_PI*(vm+1.))
            -(vp-1.)*Si(M_PI*(vp-1.))
            +(vp+1.)*Si(M_PI*(vp+1.));
        return retval/(2.*M_PI);
    }

    Lanczos::Lanczos(int n, bool fluxConserve, double tol,
                     const GSParamsPtr& gsparams) :  
        Interpolant(gsparams), _n(n), _fluxConserve(fluxConserve), _tolerance(tol)
    {
        // Reduce range slightly from n so we're not including points with zero weight in
        // interpolations:
        _range = _n*(1-0.1*std::sqrt(_tolerance));

        _k1 = kCalc(1.);

        // Strangely, not all compilers correctly setup an empty map when it is a 
        // static variable, so you can get seg faults using it.
        // Doing an explicit clear fixes the problem.
        if (_cache_kmax.size() == 0) {
            _cache_kmax.clear();
            _cache_xtab.clear();
            _cache_ktab.clear();  
        }

        KeyType key(n,std::pair<bool,double>(fluxConserve,tol));

        if (_cache_kmax.count(key)) {
            // Then uMax and tab are already cached.
            _xtab = _cache_xtab[key];
            _ktab = _cache_ktab[key];
            _kMax = _cache_kmax[key];
        } else {
            _xtab.reset(new Table<double,double>(Table<double,double>::spline));
            _ktab.reset(new Table<double,double>(Table<double,double>::spline));

            // Build xtab = table of x values
            // Spline is accurate to O(dx^3), so errors should be ~dx^4.
            const double xStep1 = 
                gsparams->table_spacing * std::pow(gsparams->xvalue_accuracy/10.,0.25);
            // Make sure steps hit the integer values exactly.
            const double xStep = 1. / std::ceil(1./xStep1);
            for(double x=0.; x<_n; x+=xStep) _xtab->addEntry(x, xCalc(x));

            // Build ktab = table of u values
            const double kStep = 
                gsparams->table_spacing * std::pow(gsparams->kvalue_accuracy/10.,0.25) / _n;
            _kMax = 0.;
            if (_fluxConserve) {
                for (double k=0.; k - _kMax < 1./_n || k<1.1; k+=kStep) {
                    double kval = kCalc(k);
                    kval *= 1.+2.*_k1;
                    kval -= _k1*kCalc(k+1.);
                    kval -= _k1*kCalc(k-1.);
                    _ktab->addEntry(k, kval);
                    if (std::abs(kval) > _tolerance) _kMax = k;
                }
            } else {
                for (double k=0.; k - _kMax < 1./_n || k<1.1; k+=kStep) {
                    double kval = kCalc(k);
                    _ktab->addEntry(k, kval);
                    if (std::abs(kval) > _tolerance) _kMax = k;
                }
            }
            // Save these values in the cache.
            _cache_xtab[key] = _xtab;
            _cache_ktab[key] = _ktab;
            _cache_kmax[key] = _kMax;
        }
    }

    std::map<Lanczos::KeyType,boost::shared_ptr<Table<double,double> > > Lanczos::_cache_xtab;
    std::map<Lanczos::KeyType,boost::shared_ptr<Table<double,double> > > Lanczos::_cache_ktab;
    std::map<Lanczos::KeyType,double> Lanczos::_cache_kmax;

    double Lanczos::xval(double x) const
    {
        x = std::abs(x);
        return x>=_n ? 0. : (*_xtab)(x);
    }

    double Lanczos::kval(double k) const
    {
        k = std::abs(k);
        return k>_kMax ? 0. : (*_ktab)(k);
    }

    class CubicIntegrand : public std::unary_function<double,double>
    {
    public:
        CubicIntegrand(double k, const Cubic& c): _k(k), _c(c) {}
        double operator()(double x) const { return _c.xval(x)*std::cos(2*M_PI*_k*x); }

    private:
        double _k;
        const Cubic& _c;
    };

    double Cubic::kCalc(double k) const 
    {
        CubicIntegrand ci(k, *this);
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
        if (_cache_kmax.size() == 0) { _cache_kmax.clear(); _cache_tab.clear(); }

        if (_cache_kmax.count(tol)) {
            // Then uMax and tab are already cached.
            _tab = _cache_tab[tol];
            _kMax = _cache_kmax[tol];
        } else {
            // Then need to do the calculation and then cache it.
            const double kStep = 
                gsparams->table_spacing * std::pow(gsparams->kvalue_accuracy/10.,0.25);
            _kMax = 0.;
            _tab.reset(new Table<double,double>(Table<double,double>::spline));
            for (double k=0.; k - _kMax < 1. || k<1.1; k+=kStep) {
                double ft = kCalc(k);
                _tab->addEntry(k, ft);
                if (std::abs(ft) > _tolerance) _kMax = k;
            }
            // Save these values in the cache.
            _cache_tab[tol] = _tab;
            _cache_kmax[tol] = _kMax;
        }
    }

    std::map<double,boost::shared_ptr<Table<double,double> > > Cubic::_cache_tab;
    std::map<double,double> Cubic::_cache_kmax;


    class QuinticIntegrand : public std::unary_function<double,double>
    {
    public:
        QuinticIntegrand(double k, const Quintic& c): _k(k), _c(c) {}
        double operator()(double x) const { return _c.xval(x)*std::cos(2*M_PI*_k*x); }
    private:
        double _k;
        const Quintic& _c;
    };

    double Quintic::kCalc(double k) const 
    {
        QuinticIntegrand qi(k, *this);
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
        if (_cache_kmax.size() == 0) { _cache_kmax.clear(); _cache_tab.clear(); }

        if (_cache_kmax.count(tol)) {
            // Then uMax and tab are already cached.
            _tab = _cache_tab[tol];
            _kMax = _cache_kmax[tol];
        } else {
            // Then need to do the calculation and then cache it.
            const double kStep = 
                gsparams->table_spacing * std::pow(gsparams->kvalue_accuracy/10.,0.25);
            _kMax = 0.;
            _tab.reset(new Table<double,double>(Table<double,double>::spline));
            for (double k=0.; k - _kMax < 1. || k<1.1; k+=kStep) {
                double ft = kCalc(k);
                _tab->addEntry(k, ft);
                if (std::abs(ft) > _tolerance) _kMax = k;
            }
            // Save these values in the cache.
            _cache_tab[tol] = _tab;
            _cache_kmax[tol] = _kMax;
        }
    }

    std::map<double,boost::shared_ptr<Table<double,double> > > Quintic::_cache_tab;
    std::map<double,double> Quintic::_cache_kmax;
}

