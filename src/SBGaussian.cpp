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
        return dynamic_cast<const SBGaussianImpl&>(*_pimpl).getSigma(); 
    }

    SBGaussian::SBGaussianImpl::SBGaussianImpl(double sigma, double flux) :
        _flux(flux), _sigma(sigma), _sigma_sq(sigma*sigma)
    {
        // For large k, we clip the result of kValue to 0.
        // We do this when the correct answer is less than kvalue_accuracy.
        // exp(-k^2*sigma^2/2) = kvalue_accuracy
        _ksq_max = -2. * std::log(sbp::kvalue_accuracy) / _sigma_sq;

        // For small k, we can use up to quartic in the taylor expansion to avoid the exp.
        // This is acceptable when the next term is less than kvalue_accuracy.
        // 1/48 (k^2 r0^2)^3 = kvalue_accuracy
        _ksq_min = std::pow(sbp::kvalue_accuracy * 48., 1./3.) / _sigma_sq;

        _norm = _flux / (_sigma_sq * 2. * M_PI);

        dbg<<"Gaussian:\n";
        dbg<<"_flux = "<<_flux<<std::endl;
        dbg<<"_sigma = "<<_sigma<<std::endl;
        dbg<<"_sigma_sq = "<<_sigma_sq<<std::endl;
        dbg<<"_ksq_max = "<<_ksq_max<<std::endl;
        dbg<<"_ksq_min = "<<_ksq_min<<std::endl;
        dbg<<"_norm = "<<_norm<<std::endl;
        dbg<<"maxK() = "<<maxK()<<std::endl;
        dbg<<"stepK() = "<<stepK()<<std::endl;
    }

    // Set maxK to the value where the FT is down to maxk_threshold
    double SBGaussian::SBGaussianImpl::maxK() const 
    { return sqrt(-2.*std::log(sbp::maxk_threshold))/_sigma; }

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
        return _norm * std::exp( -rsq/(2.*_sigma_sq) );
    }

    std::complex<double> SBGaussian::SBGaussianImpl::kValue(const Position<double>& k) const
    {
        double ksq = k.x*k.x+k.y*k.y;

        if (ksq > _ksq_max) {
            return 0.;
        } else if (ksq < _ksq_min) {
            ksq *= _sigma_sq;
            return _flux*(1. - 0.5*ksq*(1. - 0.25*ksq));
        } else {
            return _flux * std::exp(-ksq * _sigma_sq/2.);
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
