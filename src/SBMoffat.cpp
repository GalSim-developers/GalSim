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

#include "SBMoffat.h"
#include "SBMoffatImpl.h"
#include "integ/Int.h"
#include "Solve.h"

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
std::ostream* dbgout = new std::ofstream("debug.out");
int verbose_level = 2;
#endif

namespace galsim {

    SBMoffat::SBMoffat(double beta, double size, RadiusType rType, double trunc, double flux) :
        SBProfile(new SBMoffatImpl(beta, size, rType, trunc, flux)) {}

    SBMoffat::SBMoffat(const SBMoffat& rhs) : SBProfile(rhs) {}

    SBMoffat::~SBMoffat() {}

    double SBMoffat::getBeta() const 
    {
        assert(dynamic_cast<const SBMoffatImpl*>(_pimpl.get()));
        return static_cast<const SBMoffatImpl&>(*_pimpl).getBeta(); 
    }

    double SBMoffat::getFWHM() const 
    {
        assert(dynamic_cast<const SBMoffatImpl*>(_pimpl.get()));
        return static_cast<const SBMoffatImpl&>(*_pimpl).getFWHM(); 
    }

    double SBMoffat::getScaleRadius() const 
    {
        assert(dynamic_cast<const SBMoffatImpl*>(_pimpl.get()));
        return static_cast<const SBMoffatImpl&>(*_pimpl).getScaleRadius();
    }

    double SBMoffat::getHalfLightRadius() const 
    {
        assert(dynamic_cast<const SBMoffatImpl*>(_pimpl.get()));
        return static_cast<const SBMoffatImpl&>(*_pimpl).getHalfLightRadius();
    }

    class MoffatScaleRadiusFunc 
    {
    public:
        MoffatScaleRadiusFunc(double re, double rm, double beta) :
            _re(re), _rm(rm), _beta(beta) {}
        double operator()(double rd) const
        {
            double fre = 1.-std::pow(1.+(_re*_re)/(rd*rd), 1.-_beta);
            double frm = 1.-std::pow(1.+(_rm*_rm)/(rd*rd), 1.-_beta);
            xdbg<<"func("<<rd<<") = 2*"<<fre<<" - "<<frm<<" = "<<2.*fre-frm<<std::endl;
            return 2.*fre-frm;
        }
    private:
        double _re,_rm,_beta;
    };

    double MoffatCalculateScaleRadiusFromHLR(double re, double rm, double beta)
    {
        dbg<<"Start MoffatCalculateScaleRadiusFromHLR\n";
        // The basic equation that is relevant here is the flux of a Moffat profile
        // out to some radius.
        // flux(R) = int( (1+r^2/rd^2 )^(-beta) 2pi r dr, r=0..R )
        //         = (pi rd^2 / (beta-1)) (1 - (1+R^2/rd^2)^(1-beta) )
        // For now, we can ignore the first factor.  We call the second factor fluxfactor below,
        // or in this function f(R).
        // 
        // We are given two values of R for which we know that the ratio of their fluxes is 1/2:
        // f(re) = 0.5 * f(rm)
        //
        if (rm == 0.) {
            // If rm = infinity (which we actually indicate with rm=0), then we can solve for 
            // rd analytically:
            //
            // f(rm) = 1
            // f(re) = 0.5 = 1 - (1+re^2/rd^2)^(1-beta)
            // re^2/rd^2 = 0.5^(1/(1-beta)) - 1
            double rerd = std::sqrt( std::pow(0.5, 1./(1.-beta)) - 1.);
            dbg<<"rm = 0, so analytic.\n";
            xdbg<<"rd = re/rerd = "<<re<<" / "<<rerd<<" = "<<re/rerd<<std::endl;
            return re / rerd;
        } else {
            // If trunc < infinity, then the equations are slightly circular:
            // f(rm) = 1 - (1 + rm^2/rd^2)^(1-beta)
            // 2*f(re) = 2 - 2*(1 + re^2/rd^2)^(1-beta)
            // 2*(1+re^2/rd^2)^(1-beta) = 1 + (1+rm^2/rd^2)^(1-beta)
            // 
            // As rm decreases, rd increases.  
            // Eventually rd increases to infinity.  When does that happen:
            // Take the limit as rd->infinity in the above equation:
            // 2 + 2*(1-beta) re^2/rd^2) = 1 + 1 + (1-beta) rm^2/rd^2
            // 2 re^2 = rm^2
            // rm = sqrt(2) * re
            // So this is the limit for how low rm is allowed to be for a given re
            if (rm <= std::sqrt(2.) * re)
                throw SBError("Moffat truncation radius must be > sqrt(2) * half_light_radius.");

            dbg<<"rm != 0, so not analytic.\n";
            MoffatScaleRadiusFunc func(re,rm,beta);
            // For the lower bound of rd, we can use the untruncated value:
            double r1 = re / std::sqrt( std::pow(0.5, 1./(1.-beta)) - 1.);
            xdbg<<"r1 = "<<r1<<std::endl;
            // For the upper bound, we don't really have a good choice, so start with 2*r1
            // and we'll expand it if necessary.
            double r2 = 2. * r1;
            xdbg<<"r2 = "<<r2<<std::endl;
            Solve<MoffatScaleRadiusFunc> solver(func,r1,r2);
            solver.setMethod(Brent);
            solver.bracketUpper();
            xdbg<<"After bracket, range is "<<solver.getLowerBound()<<" .. "<<
                solver.getUpperBound()<<std::endl;
            double rd = solver.root();
            xdbg<<"Root is "<<rd<<std::endl;
            return rd;
        }
    }

    SBMoffat::SBMoffatImpl::SBMoffatImpl(double beta, double size, RadiusType rType,
                                         double trunc, double flux) : 
        _beta(beta), _flux(flux), _trunc(trunc), _ft(Table<double,double>::spline),
        _re(0.) // initially set to zero, may be updated by size or getHalfLightRadius()
    {
        xdbg<<"Start SBMoffat constructor: \n";
        xdbg<<"beta = "<<_beta<<"\n";
        xdbg<<"flux = "<<_flux<<"\n";
        xdbg<<"trunc = "<<_trunc<<"\n";

        if (_trunc == 0. && beta <= 1.) 
            throw SBError("Moffat profiles with beta <= 1 must be truncated.");

        // First, relation between FWHM and rD:
        double FWHMrD = 2.* std::sqrt(std::pow(2., 1./_beta)-1.);
        xdbg<<"FWHMrD = "<<FWHMrD<<"\n";

        // Set size of this instance according to type of size given in constructor:
        switch (rType) {
          case FWHM:
               _rD = size / FWHMrD;
               break;
          case HALF_LIGHT_RADIUS: 
               {
                   _re = size;
                   // This is a bit complicated, so break it out into its own function.
                   _rD = MoffatCalculateScaleRadiusFromHLR(_re,trunc,_beta);
               }
               break;
          case SCALE_RADIUS:
               _rD = size;
               break;
          default:
               throw SBError("Unknown SBMoffat::RadiusType");
        }

        double maxRrD;
        if (trunc > 0.) {
            maxRrD = trunc / _rD;  // note new usage of trunc in physical units requires _rD here
            xdbg<<"maxRrD = "<<maxRrD<<"\n";

            // Analytic integration of total flux:
            _fluxFactor = 1. - std::pow( 1+maxRrD*maxRrD, (1.-_beta));
        } else {
            _fluxFactor = 1.;

            // Set maxRrD to the radius where surface brightness is kvalue_accuracy
            // of center value.  (I know this isn't  a kvalue, but the same level 
            // is probably appropriate here.)
            // (1+R^2)^-beta = kvalue_accuracy
            // And ignore the 1+ part of (1+R^2), so
            maxRrD = std::pow(sbp::kvalue_accuracy,-1./(2.*_beta));
            xdbg<<"Not truncate.  Calculated maxRrD = "<<maxRrD<<"\n";
        }

        _FWHM = FWHMrD * _rD;
        _maxR = maxRrD * _rD;
        _maxR_sq = _maxR * _maxR;
        _rD_sq = _rD * _rD;
        _norm = _flux * (_beta-1.) / (M_PI * _fluxFactor * _rD_sq);

        dbg << "Moffat rD " << _rD << " fluxFactor " << _fluxFactor
            << " norm " << _norm << " maxRrD " << _maxR << std::endl;

        if (_beta == 1) pow_beta = &SBMoffat::pow_1;
        else if (_beta == 2) pow_beta = &SBMoffat::pow_2;
        else if (_beta == 3) pow_beta = &SBMoffat::pow_3;
        else if (_beta == 4) pow_beta = &SBMoffat::pow_4;
        else if (_beta == int(_beta)) pow_beta = &SBMoffat::pow_int;
        else pow_beta = &SBMoffat::pow_gen;
    }

    double SBMoffat::SBMoffatImpl::getHalfLightRadius() const 
    {
        // Done here since _re depends on _fluxFactor and thus requires _rD in advance, so this 
        // needs to happen largely post setup. Doesn't seem efficient to ALWAYS calculate it above,
        // so we'll just calculate it once if requested and store it.
        if (_re == 0.) {
            _re = _rD * std::sqrt(std::pow(1.-0.5*_fluxFactor , 1./(1.-_beta)) - 1.);
        }
        return _re;
    }

    double SBMoffat::SBMoffatImpl::xValue(const Position<double>& p) const 
    {
        double rsq = p.x*p.x + p.y*p.y;
        if (rsq > _maxR_sq) return 0.;
        else return _norm / pow_beta(1.+rsq/_rD_sq, _beta);
    }

    std::complex<double> SBMoffat::SBMoffatImpl::kValue(const Position<double>& k) const 
    {
        setupFT();
        double ksq = k.x*k.x + k.y*k.y;
        if (ksq > _ft.argMax()) return 0.;
        else return _ft(ksq);
    }

    // Set maxK to the value where the FT is down to maxk_threshold
    double SBMoffat::SBMoffatImpl::maxK() const 
    {
        // _maxK is determined during setupFT() as the last k value to have a  kValue > 1.e-3.
        setupFT();
        return _maxK;
    }

    // The amount of flux missed in a circle of radius pi/stepk should miss at 
    // most alias_threshold of the flux.
    double SBMoffat::SBMoffatImpl::stepK() const
    {
        dbg<<"Find Moffat stepK\n";
        dbg<<"beta = "<<_beta<<std::endl;
        
        // The fractional flux out to radius R is (if not truncated)
        // 1 - (1+R^2)^(1-beta)
        // So solve (1+R^2)^(1-beta) = alias_threshold
        if (_beta <= 1.1) {
            // Then flux never converges (or nearly so), so just use truncation radius
            return M_PI / _maxR;
        } else {
            // Ignore the 1 in (1+R^2), so approximately:
            double R = std::pow(sbp::alias_threshold, 0.5/(1.-_beta)) * _rD;
            dbg<<"R = "<<R<<std::endl;
            // If it is truncated at less than this, drop to that value.
            if (R > _maxR) R = _maxR;
            dbg<<"_maxR = "<<_maxR<<std::endl;
            dbg<<"R => "<<R<<std::endl;
            dbg<<"stepk = "<<(M_PI/R)<<std::endl;
            return M_PI / R;
        }
    }

    // Integrand class for the Hankel transform of Moffat
    class MoffatIntegrand : public std::unary_function<double,double>
    {
    public:
        MoffatIntegrand(double beta, double k, double (*pb)(double, double)) : 
            _beta(beta), _k(k), pow_beta(pb) {}
        double operator()(double r) const 
        { return r/pow_beta(1.+r*r, _beta)*j0(_k*r); }

    private:
        double _beta;
        double _k;
        double (*pow_beta)(double x, double beta);
    };

    void SBMoffat::SBMoffatImpl::setupFT() const
    {
        if (_ft.size() > 0) return;

        // Do a Hankel transform and store the results in a lookup table.

        double nn = _norm * 2.*M_PI * _rD_sq;
        //double maxR = _fluxFactor == 1. ? integ::MOCK_INF : _maxR / _rD;
        double maxR = _maxR / _rD;

        // Along the way, find the last k that has a kValue > 1.e-3
        double maxK_val = sbp::maxk_threshold * _flux;
        dbg<<"Looking for maxK_val = "<<maxK_val<<std::endl;
        // Keep going until at least 5 in a row have kvalues below kvalue_accuracy.
        // (It's oscillatory, so want to make sure not to stop at a zero crossing.)
        double thresh = sbp::kvalue_accuracy * _flux;

        // These are dimensionless k values for doing the integral.
        double dk = 0.1;
        dbg<<"dk = "<<dk<<std::endl;
        int n_below_thresh = 0;
        // Don't go past k = 50
        for(double k=0.; k < 50; k += dk) {
            MoffatIntegrand I(_beta, k, pow_beta);
            double val = integ::int1d(
                I, 0., maxR, sbp::integration_relerr, sbp::integration_abserr);
            val *= nn;

            double kreal = k / _rD;
            xdbg<<"ft("<<kreal<<") = "<<val<<std::endl;
            _ft.addEntry( kreal*kreal, val );

            if (std::abs(val) > maxK_val) _maxK = kreal;

            if (std::abs(val) > thresh) n_below_thresh = 0;
            else ++n_below_thresh;
            if (n_below_thresh == 5) break;
        }
        dbg<<"maxK = "<<_maxK<<std::endl;
    }

    boost::shared_ptr<PhotonArray> SBMoffat::SBMoffatImpl::shoot(int N, UniformDeviate u) const
    {
        dbg<<"Moffat shoot: N = "<<N<<std::endl;
        dbg<<"Target flux = "<<getFlux()<<std::endl;
        // Moffat has analytic inverse-cumulative-flux function.
        boost::shared_ptr<PhotonArray> result(new PhotonArray(N));
        double fluxPerPhoton = _flux/N;
        for (int i=0; i<N; i++) {
#ifdef USE_COS_SIN
            // First get a point uniformly distributed on unit circle
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
            // Then map radius to the Moffat flux distribution
            double newRsq = std::pow(1. - rsq * _fluxFactor, 1. / (1. - _beta)) - 1.;
            double rFactor = _rD * std::sqrt(newRsq);
            result->setPhoton(i, rFactor*cost, rFactor*sint, fluxPerPhoton);
#else
            // First get a point uniformly distributed on unit circle
            double xu, yu, rsq;
            do {
                xu = 2.*u()-1.;
                yu = 2.*u()-1.;
                rsq = xu*xu+yu*yu;
            } while (rsq>=1. || rsq==0.);
            // Then map radius to the Moffat flux distribution
            double newRsq = std::pow(1. - rsq * _fluxFactor, 1. / (1. - _beta)) - 1.;
            double rFactor = _rD * std::sqrt(newRsq / rsq);
            result->setPhoton(i, rFactor*xu, rFactor*yu, fluxPerPhoton);
#endif
        }
        dbg<<"Moffat Realized flux = "<<result->getTotalFlux()<<std::endl;
        return result;
    }

}
