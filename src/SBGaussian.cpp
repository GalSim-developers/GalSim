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

#include "SBGaussian.h"
#include "SBGaussianImpl.h"

// Define this variable to find azimuth (and sometimes radius within a unit disc) of 2d photons by 
// drawing a uniform deviate for theta, instead of drawing 2 deviates for a point on the unit 
// circle and rejecting corner photons.
// The relative speed of the two methods was tested as part of issue #163, and the results
// are collated in devutils/external/time_photon_shooting.
// The conclusion was that using sin/cos was faster for icpc, but not g++ or clang++.
#ifdef _INTEL_COMPILER
#define USE_COS_SIN
#endif

#ifdef DEBUGLOGGING
#include <fstream>
//std::ostream* dbgout = new std::ofstream("debug.out");
//int verbose_level = 2;
#endif

namespace galsim {


    SBGaussian::SBGaussian(double sigma, double flux) : 
        SBProfile(new SBGaussianImpl(sigma, flux)) {}

    SBGaussian::SBGaussian(const SBGaussian& rhs) : SBProfile(rhs) {}

    SBGaussian::~SBGaussian() {}

    double SBGaussian::getSigma() const 
    { 
        assert(dynamic_cast<const SBGaussianImpl*>(_pimpl.get()));
        return static_cast<const SBGaussianImpl&>(*_pimpl).getSigma(); 
    }

    SBGaussian::SBGaussianImpl::SBGaussianImpl(double sigma, double flux) :
        _flux(flux), _sigma(sigma), _sigma_sq(_sigma*_sigma),
        _inv_sigma(1./_sigma), _inv_sigma_sq(_inv_sigma*_inv_sigma)
    {
        // For large k, we clip the result of kValue to 0.
        // We do this when the correct answer is less than kvalue_accuracy.
        // exp(-k^2*sigma^2/2) = kvalue_accuracy
        _ksq_max = -2. * std::log(sbp::kvalue_accuracy);

        // For small k, we can use up to quartic in the taylor expansion to avoid the exp.
        // This is acceptable when the next term is less than kvalue_accuracy.
        // 1/48 (k^2 r0^2)^3 = kvalue_accuracy
        _ksq_min = std::pow(sbp::kvalue_accuracy * 48., 1./3.);

        _norm = _flux * _inv_sigma_sq / (2. * M_PI);

        dbg<<"Gaussian:\n";
        dbg<<"_flux = "<<_flux<<std::endl;
        dbg<<"_sigma = "<<_sigma<<std::endl;
        dbg<<"_ksq_max = "<<_ksq_max<<std::endl;
        dbg<<"_ksq_min = "<<_ksq_min<<std::endl;
        dbg<<"_norm = "<<_norm<<std::endl;
        dbg<<"maxK() = "<<maxK()<<std::endl;
        dbg<<"stepK() = "<<stepK()<<std::endl;
    }

    // Set maxK to the value where the FT is down to maxk_threshold
    double SBGaussian::SBGaussianImpl::maxK() const 
    { return sqrt(-2.*std::log(sbp::maxk_threshold))*_inv_sigma; }

    // The amount of flux missed in a circle of radius pi/stepk should miss at 
    // most alias_threshold of the flux.
    double SBGaussian::SBGaussianImpl::stepK() const
    {
        // int( exp(-r^2/2) r, r=0..R) = 1 - exp(-R^2/2)
        // exp(-R^2/2) = alias_threshold
        double R = sqrt(-2.*std::log(sbp::alias_threshold));
        // Make sure it is at least 4 sigma;
        R = std::max(4., R);
        return M_PI / (R*_sigma);
    }

    double SBGaussian::SBGaussianImpl::xValue(const Position<double>& p) const
    {
        double rsq = p.x*p.x + p.y*p.y;
        return _norm * std::exp( -0.5 * rsq * _inv_sigma_sq );
    }

    std::complex<double> SBGaussian::SBGaussianImpl::kValue(const Position<double>& k) const
    {
        double ksq = (k.x*k.x+k.y*k.y)*_sigma_sq;

        if (ksq > _ksq_max) {
            return 0.;
        } else if (ksq < _ksq_min) {
            return _flux*(1. - 0.5*ksq*(1. - 0.25*ksq));
        } else {
            return _flux * std::exp(-0.5*ksq);
        }
    }

    void SBGaussian::SBGaussianImpl::xValue(
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
        x *= _inv_sigma;
        x = ElemProd(x,x);
        y *= _inv_sigma;
        y = ElemProd(y,y);
        It xit = x.begin();
        for (int i=0;i<m;++i,++xit) *xit = exp(-0.5* *xit);
        It yit = y.begin();
        for (int j=0;j<n;++j,++yit) *yit = exp(-0.5* *yit);
        val = _norm * x ^ y;
    }

    void SBGaussian::SBGaussianImpl::kValue(
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
        kx *= _sigma;
        kx = ElemProd(kx,kx);
        ky *= _sigma;
        ky = ElemProd(ky,ky);
        It kxit = kx.begin();
        for (int i=0;i<m;++i,++kxit) *kxit = exp(-0.5* *kxit);
        It kyit = ky.begin();
        for (int j=0;j<n;++j,++kyit) *kyit = exp(-0.5* *kyit);
        kval = _flux * kx ^ ky;
    }

    void SBGaussian::SBGaussianImpl::xValue(
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
        x *= _inv_sigma;
        x = ElemProd(x,x);
        y *= _inv_sigma;
        y = ElemProd(y,y);
        x += y;
        It xit = x.linearView().begin();
        It valit = val.linearView().begin();
        const int ntot = m*n;
        for (int i=0;i<ntot;++i) *valit++ = _norm * std::exp(-0.5* *xit++);
     }

    void SBGaussian::SBGaussianImpl::kValue(
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
        kx *= _sigma;
        kx = ElemProd(kx,kx);
        ky *= _sigma;
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
                *kvalit++ = _flux*(1. - 0.5*ksq*(1. - 0.25*ksq));
            } else {
                *kvalit++ = _flux * std::exp(-0.5*ksq);
            }
        }
    }

    boost::shared_ptr<PhotonArray> SBGaussian::SBGaussianImpl::shoot(int N, UniformDeviate u) const 
    {
        dbg<<"Gaussian shoot: N = "<<N<<std::endl;
        dbg<<"Target flux = "<<getFlux()<<std::endl;
        boost::shared_ptr<PhotonArray> result(new PhotonArray(N));
        double fluxPerPhoton = _flux/N;
        for (int i=0; i<N; i++) {
            // First get a point uniformly distributed on unit circle
#ifdef USE_COS_SIN
            double theta = 2.*M_PI*u();
            double rsq = u(); // cumulative dist function P(<r) = r^2 for unit circle
#ifdef _GLIBCXX_HAVE_SINCOS
            // Most optimizing compilers will do this automatically, but just in case...
            double sint,cost;
            sincos(theta,&sint,&cost);
#else
            double cost = std::cos(theta);
            double sint = std::sin(theta);
#endif
            // Then map radius to the desired Gaussian with analytic transformation
            double rFactor = _sigma * std::sqrt( -2. * std::log(rsq));
            result->setPhoton(i, rFactor*cost, rFactor*sint, fluxPerPhoton);
#else
            double xu, yu, rsq;
            do {
                xu = 2.*u()-1.;
                yu = 2.*u()-1.;
                rsq = xu*xu+yu*yu;
            } while (rsq>=1. || rsq==0.);
            // Then map radius to the desired Gaussian with analytic transformation
            double rFactor = _sigma * std::sqrt( -2. * std::log(rsq) / rsq);
            result->setPhoton(i, rFactor*xu, rFactor*yu, fluxPerPhoton);
#endif
        }
        dbg<<"Gaussian Realized flux = "<<result->getTotalFlux()<<std::endl;
        return result;
    }
}
