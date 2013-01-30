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
        assert(std::abs(xdown) <= N);
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
        if (_fluxConserve) retval *= 1. + 2.*_u1*(1.-std::cos(2.*M_PI*x));
        return retval;
    }

    double Lanczos::uCalc(double u) const 
    {
        double vp=_n*(2*u+1);
        double vm=_n*(2*u-1);
        double retval = (vm-1.)*Si(M_PI*(vm-1.))
            -(vm+1.)*Si(M_PI*(vm+1.))
            -(vp-1.)*Si(M_PI*(vp-1.))
            +(vp+1.)*Si(M_PI*(vp+1.));
        return retval/(2.*M_PI);
    }

    void Lanczos::setup() 
    {
        // Reduce range slightly from n so we're not including points with zero weight in
        // interpolations:
        _range = _n*(1-0.1*std::sqrt(_tolerance));

        _u1 = uCalc(1.);

        // Build utab = table of u values
        // Spline is accurate to O(dx^3), so errors should be ~dx^4.
        const double xStep1 = std::pow(sbp::xvalue_accuracy,0.25);
        // Make sure steps hit the integer values exactly.
        const double xStep = 1. / std::ceil(1./xStep1);
        for(double x = 0.; x<_n; x+=xStep) _xtab.addEntry(x, xCalc(x));

        // Build utab = table of u values
        const double uStep = std::pow(sbp::kvalue_accuracy,0.25) / _n;
        _uMax = 0.;
        double u = _utab.size()>0 ? _utab.argMax() + uStep : 0.;
        if (_fluxConserve) {
            while ( u - _uMax < 1./_n || u<1.1) {
                double uval = uCalc(u);
                uval *= 1.+2.*_u1;
                uval -= _u1*uCalc(u+1.);
                uval -= _u1*uCalc(u-1.);
                _utab.addEntry(u, uval);
                if (std::abs(uval) > _tolerance) _uMax = u;
                u += uStep;
            }
        } else {
            while ( u - _uMax < 1./_n || u<1.1) {
                double uval = uCalc(u);
                _utab.addEntry(u, uval);
                if (std::abs(uval) > _tolerance) _uMax = u;
                u += uStep;
            }
        }
    }

    double Lanczos::xval(double x) const
    {
        x = std::abs(x);
        return x>=_n ? 0. : _xtab(x);
    }

    double Lanczos::uval(double u) const
    {
        u = std::abs(u);
        return u>_uMax ? 0. : _utab(u);
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

    void Cubic::setup() 
    {
        // Reduce range slightly from n so we're not including points with zero weight in
        // interpolations:
        _range = 2.-0.1*_tolerance;
        const double uStep = 0.001;
        _uMax = 0.;
        double u = _tab.size()>0 ? _tab.argMax() + uStep : 0.;
        while ( u - _uMax < 1. || u<1.1) {
            double ft = uCalc(u);
            _tab.addEntry(u, ft);
            if (std::abs(ft) > _tolerance) _uMax = u;
            u += uStep;
        }
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

    void Quintic::setup() 
    {
        // Reduce range slightly from n so we're not including points with zero weight in
        // interpolations:
        _range = 3.-0.1*_tolerance;
        const double uStep = 0.001;
        _uMax = 0.;
        double u = _tab.size()>0 ? _tab.argMax() + uStep : 0.;
        while ( u - _uMax < 1. || u<1.1) {
            double ft = uCalc(u);
            _tab.addEntry(u, ft);
            if (std::abs(ft) > _tolerance) _uMax = u;
            u += uStep;
        }
    }

}

