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

#include "SBExponential.h"
#include "SBExponentialImpl.h"

// Define this variable to find azimuth (and sometimes radius within a unit disc) of 2d photons by 
// drawing a uniform deviate for theta, instead of drawing 2 deviates for a point on the unit 
// circle and rejecting corner photons.
// The relative speed of the two methods was tested as part of issue #163, and the results
// are collated in devutils/external/time_photon_shooting.
// The conclusion was that using sin/cos was faster for icpc, but not g++ or clang++.
#ifdef _INTEL_COMPILER
#define USE_COS_SIN
#endif

// Define this use the Newton-Raphson method for solving the radial value in SBExponential::shoot
// rather than using OneDimensionalDeviate.
// The relative speed of the two methods was tested as part of issue #163, and the results
// are collated in devutils/external/time_photon_shooting.
// The conclusion was that using OneDimensionalDeviate was universally quite a bit faster.
// However, we leave this option here in case someone has an idea for massively speeding up
// the solution that might be faster than the table lookup.
//#define USE_NEWTON_RAPHSON

#ifdef DEBUGLOGGING
#include <fstream>
std::ostream* dbgout = new std::ofstream("debug.out");
int verbose_level = 2;
#endif

namespace galsim {

    SBExponential::SBExponential(double r0, double flux) :
        SBProfile(new SBExponentialImpl(r0, flux)) {}

    SBExponential::SBExponential(const SBExponential& rhs) : SBProfile(rhs) {}

    SBExponential::~SBExponential() {}

    double SBExponential::getScaleRadius() const 
    { 
        assert(dynamic_cast<const SBExponentialImpl*>(_pimpl.get()));
        return static_cast<const SBExponentialImpl&>(*_pimpl).getScaleRadius(); 
    }

    SBExponential::SBExponentialImpl::SBExponentialImpl(double r0, double flux) :
        _flux(flux), _r0(r0), _r0_sq(_r0*_r0), _inv_r0(1./r0), _inv_r0_sq(_inv_r0*_inv_r0)
    {
        // For large k, we clip the result of kValue to 0.
        // We do this when the correct answer is less than kvalue_accuracy.
        // (1+k^2 r0^2)^-1.5 = kvalue_accuracy
        _ksq_max = (std::pow(sbp::kvalue_accuracy,-1./1.5)-1.);

        // For small k, we can use up to quartic in the taylor expansion to avoid the sqrt.
        // This is acceptable when the next term is less than kvalue_accuracy.
        // 35/16 (k^2 r0^2)^3 = kvalue_accuracy
        _ksq_min = std::pow(sbp::kvalue_accuracy * 16./35., 1./3.);

        _flux_over_2pi = _flux / (2. * M_PI);
        _norm = _flux_over_2pi * _inv_r0_sq;

        dbg<<"Exponential:\n";
        dbg<<"_flux = "<<_flux<<std::endl;
        dbg<<"_r0 = "<<_r0<<std::endl;
        dbg<<"_ksq_max = "<<_ksq_max<<std::endl;
        dbg<<"_ksq_min = "<<_ksq_min<<std::endl;
        dbg<<"_norm = "<<_norm<<std::endl;
        dbg<<"maxK() = "<<maxK()<<std::endl;
        dbg<<"stepK() = "<<stepK()<<std::endl;
    }

    double SBExponential::SBExponentialImpl::maxK() const 
    { return SBExponential::_info.maxK() * _inv_r0; }
    double SBExponential::SBExponentialImpl::stepK() const 
    { return SBExponential::_info.stepK() * _inv_r0; }

    double SBExponential::SBExponentialImpl::xValue(const Position<double>& p) const
    {
        double r = sqrt(p.x*p.x + p.y*p.y);
        return _norm * std::exp(-r*_inv_r0);
    }

    std::complex<double> SBExponential::SBExponentialImpl::kValue(const Position<double>& k) const 
    {
        double ksq = (k.x*k.x+k.y*k.y)*_r0_sq;

        if (ksq > _ksq_max) {
            return 0.;
        } else if (ksq < _ksq_min) {
            return _flux*(1. - 1.5*ksq*(1. - 1.25*ksq));
        } else {
            double temp = 1. + ksq;
            return _flux/(temp*sqrt(temp));
            // NB: flux*std::pow(temp,-1.5) is slower.
        }
    }

    void SBExponential::SBExponentialImpl::xValue(
        tmv::VectorView<double> x, tmv::VectorView<double> y,
        tmv::MatrixView<double> val) const
    {
        assert(x.step() == 1);
        assert(y.step() == 1);
        assert(val.stepi() == 1);
        assert(val.canLinearize());
        assert(x.size() == val.colsize());
        assert(y.size() == val.rowsize());
        const int m = val.colsize();
        const int n = val.rowsize();
        typedef tmv::VIt<double,1,tmv::NonConj> It;
        x *= _inv_r0;
        x = ElemProd(x,x);
        y *= _inv_r0;
        y = ElemProd(y,y);
        It yit = y.begin();
        It valit = val.linearView().begin();
        for (int j=0;j<n;++j,++yit) {
            It xit = x.begin();
            for (int i=0;i<m;++i)  {
                *valit++ = _norm * std::exp(-sqrt(*xit++ + *yit));
            }
        }
     }

    void SBExponential::SBExponentialImpl::kValue(
        tmv::VectorView<double> kx, tmv::VectorView<double> ky,
        tmv::MatrixView<std::complex<double> > kval) const
    { 
        assert(kx.step() == 1);
        assert(ky.step() == 1);
        assert(kval.stepi() == 1);
        assert(kval.canLinearize());
        assert(kx.size() == kval.colsize());
        assert(ky.size() == kval.rowsize());
        const int m = kval.colsize();
        const int n = kval.rowsize();
        typedef tmv::VIt<double,1,tmv::NonConj> It;
        typedef tmv::VIt<std::complex<double>,1,tmv::NonConj> CIt;
        kx *= _r0;
        kx = ElemProd(kx,kx);
        ky *= _r0;
        ky = ElemProd(ky,ky);
        It kyit = ky.begin();
        CIt kvalit(kval.linearView().begin().getP(),1);
        for (int j=0;j<n;++j,++kyit) {
            It kxit = kx.begin();
            for (int i=0;i<m;++i)  {
                double ksq = *kxit++ + *kyit;
                if (ksq > _ksq_max) {
                    *kvalit++ = 0.;
                } else if (ksq < _ksq_min) {
                    *kvalit++ = _flux * (1. - 1.5*ksq*(1. - 1.25*ksq));
                } else {
                    double temp = 1. + ksq;
                    *kvalit++ =  _flux/(temp*sqrt(temp));
                }
            }
        }
    }

    void SBExponential::SBExponentialImpl::xValue(
        tmv::MatrixView<double> x, tmv::MatrixView<double> y,
        tmv::MatrixView<double> val) const
    { 
        assert(x.stepi() == 1);
        assert(y.stepi() == 1);
        assert(val.stepi() == 1);
        assert(val.canLinearize());
        assert(x.colsize() == val.colsize());
        assert(x.rowsize() == val.rowsize());
        assert(y.colsize() == val.colsize());
        assert(y.rowsize() == val.rowsize());
        const int m = val.colsize();
        const int n = val.rowsize();
        typedef tmv::VIt<double,1,tmv::NonConj> It;
        x *= _inv_r0;
        x = ElemProd(x,x);
        y *= _inv_r0;
        y = ElemProd(y,y);
        x += y;
        It xit = x.linearView().begin();
        It valit = val.linearView().begin();
        const int ntot = m*n;
        for (int i=0;i<ntot;++i) *valit++ = _norm * std::exp(-sqrt(*xit++));
     }

    void SBExponential::SBExponentialImpl::kValue(
        tmv::MatrixView<double> kx, tmv::MatrixView<double> ky,
        tmv::MatrixView<std::complex<double> > kval) const
    { 
        assert(kx.stepi() == 1);
        assert(ky.stepi() == 1);
        assert(kval.stepi() == 1);
        assert(kx.canLinearize());
        assert(ky.canLinearize());
        assert(kval.canLinearize());
        assert(kx.colsize() == kval.colsize());
        assert(kx.rowsize() == kval.rowsize());
        assert(ky.colsize() == kval.colsize());
        assert(ky.rowsize() == kval.rowsize());
        const int m = kval.colsize();
        const int n = kval.rowsize();
        typedef tmv::VIt<double,1,tmv::NonConj> It;
        typedef tmv::VIt<std::complex<double>,1,tmv::NonConj> CIt;
        kx *= _r0;
        kx = ElemProd(kx,kx);
        ky *= _r0;
        ky = ElemProd(ky,ky);
        kx += ky;
        It kxit = kx.linearView().begin();
        CIt kvalit(kval.linearView().begin().getP(),1);
        const int ntot = m*n;
        for (int i=0;i<ntot;++i)  {
            double ksq = *kxit++;
            if (ksq > _ksq_max) {
                *kvalit++ = 0.;
            } else if (ksq < _ksq_min) {
                *kvalit++ = _flux * (1. - 1.5*ksq*(1. - 1.25*ksq));
            } else {
                double temp = 1. + ksq;
                *kvalit++ =  _flux/(temp*sqrt(temp));
            }
        }
    }

    // Constructor to initialize Exponential functions for 1D deviate photon shooting
    SBExponential::ExponentialInfo::ExponentialInfo()
    {
#ifndef USE_NEWTON_RAPHSON
        // Next, set up the classes for photon shooting
        _radial.reset(new ExponentialRadialFunction());
        std::vector<double> range(2,0.);
        range[1] = -std::log(sbp::shoot_flux_accuracy);
        _sampler.reset(new OneDimensionalDeviate( *_radial, range, true));
#endif

        // Calculate maxk:
        _maxk = std::pow(sbp::maxk_threshold, -1./3.);

        // Calculate stepk:
        // int( exp(-r) r, r=0..R) = (1 - exp(-R) - Rexp(-R))
        // Fraction excluded is thus (1+R) exp(-R)
        // A fast solution to (1+R)exp(-R) = x:
        // log(1+R) - R = log(x)
        // R = log(1+R) - log(x)
        double logx = std::log(sbp::alias_threshold);
        double R = -logx;
        for (int i=0; i<3; i++) R = std::log(1.+R) - logx;
        // Make sure it is at least 6 scale radii.
        R = std::max(6., R);
        _stepk = M_PI / R;
    }

    // Set maxK to the value where the FT is down to maxk_threshold
    double SBExponential::ExponentialInfo::maxK() const 
    { return _maxk; }

    // The amount of flux missed in a circle of radius pi/stepk should miss at 
    // most alias_threshold of the flux.
    double SBExponential::ExponentialInfo::stepK() const
    { return _stepk; }

    boost::shared_ptr<PhotonArray> SBExponential::ExponentialInfo::shoot(
        int N, UniformDeviate ud) const
    {
        dbg<<"ExponentialInfo shoot: N = "<<N<<std::endl;
        dbg<<"Target flux = 1.0\n";
        assert(_sampler.get());
        boost::shared_ptr<PhotonArray> result = _sampler->shoot(N,ud);
        dbg<<"ExponentialInfo Realized flux = "<<result->getTotalFlux()<<std::endl;
        return result;
    }

    SBExponential::ExponentialInfo SBExponential::_info;

    boost::shared_ptr<PhotonArray> SBExponential::SBExponentialImpl::shoot(
        int N, UniformDeviate u) const
    {
        dbg<<"Exponential shoot: N = "<<N<<std::endl;
        dbg<<"Target flux = "<<getFlux()<<std::endl;
#ifdef USE_NEWTON_RAPHSON
        // The cumulative distribution of flux is 1-(1+r)exp(-r).
        // Here is a way to solve for r by an initial guess followed
        // by Newton-Raphson iterations.  Probably not
        // the most efficient thing since there are logs in the iteration.

        // Accuracy to which to solve for (log of) cumulative flux distribution:
        const double Y_TOLERANCE=sbp::shoot_flux_accuracy;

        double fluxPerPhoton = _flux / N;
        boost::shared_ptr<PhotonArray> result(new PhotonArray(N));

        for (int i=0; i<N; i++) {
            double y = u();
            if (y==0.) {
                // In case of infinite radius - just set to origin:
                result->setPhoton(i,0.,0.,fluxPerPhoton);
                continue;
            }
            // Initial guess
            y = -std::log(y);
            double r = y>2. ? y : sqrt(2.*y);
            double dy = y - r + std::log(1.+r);
            while ( std::abs(dy) > Y_TOLERANCE) {
                r = r + (1.+r)*dy/r;
                dy = y - r + std::log(1.+r);
            }
            // Draw another (or multiple) randoms for azimuthal angle 
#ifdef USE_COS_SIN
            double theta = 2. * M_PI * u();
#ifdef _GLIBCXX_HAVE_SINCOS
            // Most optimizing compilers will do this automatically, but just in case...
            double sint,cost;
            sincos(theta,&sint,&cost);
#else
            double cost = std::cos(theta);
            double sint = std::sin(theta);
#endif
            double rFactor = r * _r0;
            result->setPhoton(i, rFactor * cost, rFactor * sint, fluxPerPhoton);
#else
            double xu, yu, rsq;
            do {
                xu = 2. * u() - 1.;
                yu = 2. * u() - 1.;
                rsq = xu*xu+yu*yu;
             } while (rsq >= 1. || rsq == 0.);
            double rFactor = r * _r0 / std::sqrt(rsq);
            result->setPhoton(i, rFactor * xu, rFactor * yu, fluxPerPhoton);
#endif
        }
#else
        // Get photons from the ExponentialInfo structure, rescale flux and size for this instance
        boost::shared_ptr<PhotonArray> result = SBExponential::_info.shoot(N,u);
        result->scaleFlux(_flux_over_2pi);
        result->scaleXY(_r0);
#endif
        dbg<<"Exponential Realized flux = "<<result->getTotalFlux()<<std::endl;
        return result;
    }
}
