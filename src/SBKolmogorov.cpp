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

#include "SBKolmogorov.h"
#include "SBKolmogorovImpl.h"

#ifdef DEBUGLOGGING
#include <fstream>
std::ostream* dbgout = new std::ofstream("debug.out");
int verbose_level = 2;
#endif

// Uncomment this to do the calculation that solves for the conversion between lam_over_r0
// and fwhm and hlr.
// (Solved values are put into Kolmogorov class in galsim/base.py = 0.975865, 0.554811)
//#define SOLVE_FWHM_HLR

#ifdef SOLVE_FWHM_HLR
#include "Solve.h"
#endif

namespace galsim {

    // A static variable for the SBKolmogorov class:
    KolmogorovInfo SBKolmogorov::SBKolmogorovImpl::_info;

    SBKolmogorov::SBKolmogorov(double lam_over_r0, double flux) :
        SBProfile(new SBKolmogorovImpl(lam_over_r0, flux)) {}

    SBKolmogorov::SBKolmogorov(const SBKolmogorov& rhs) : SBProfile(rhs) {}

    SBKolmogorov::~SBKolmogorov() {}

    double SBKolmogorov::getLamOverR0() const 
    {
        assert(dynamic_cast<const SBKolmogorovImpl*>(_pimpl.get()));
        return dynamic_cast<const SBKolmogorovImpl&>(*_pimpl).getLamOverR0(); 
    }

    // The "magic" number 2.992934 below comes from the standard form of the Kolmogorov spectrum
    // from Racine, 1996 PASP, 108, 699 (who in turn is quoting Fried, 1966, JOSA, 56, 1372):
    // T(k) = exp(-1/2 D(k)) 
    // D(k) = 6.8839 (lambda/r0 k/2Pi)^(5/3)
    //
    // We convert this into T(k) = exp(-(k/k0)^5/3) for efficiency,
    // which implies 1/2 6.8839 (lambda/r0 / 2Pi)^5/3 = (1/k0)^5/3
    // k0 * lambda/r0 = 2Pi * (6.8839 / 2)^-3/5 = 2.992934
    //
    SBKolmogorov::SBKolmogorovImpl::SBKolmogorovImpl(double lam_over_r0, double flux) :
        _lam_over_r0(lam_over_r0), 
        _k0(2.992934 / lam_over_r0), 
        _k0sq(_k0*_k0),
        _inv_k0sq(1./_k0sq),
        _flux(flux), 
        _xnorm(_flux * _k0sq)
    {
        dbg<<"SBKolmogorov:\n";
        dbg<<"lam_over_r0 = "<<_lam_over_r0<<std::endl;
        dbg<<"k0 = "<<_k0<<std::endl;
        dbg<<"k0sq = "<<_k0sq<<std::endl;
        dbg<<"inv_k0sq = "<<_inv_k0sq<<std::endl;
        dbg<<"flux = "<<_flux<<std::endl;
        dbg<<"xnorm = "<<_xnorm<<std::endl;
    }

    double SBKolmogorov::SBKolmogorovImpl::xValue(const Position<double>& p) const 
    {
        double r = sqrt(p.x*p.x+p.y*p.y) * _k0;
#ifdef DEBUGLOGGING
        xdbg<<"xValue: p = "<<p<<std::endl;
        xdbg<<"r = "<<sqrt(p.x*p.x+p.y*p.y)<<" * "<<_k0<<" = "<<r<<std::endl;
        xdbg<<"return "<<_flux<<" * "<<_k0sq<<" * "<<_info.xValue(r)<<" = "<<
            (_xnorm * _info.xValue(r))<<std::endl;
#endif
        return _xnorm * _info.xValue(r);
    }

    double KolmogorovInfo::xValue(double r) const 
    { return r < _radial.argMax() ? _radial(r) : 0.; }

    std::complex<double> SBKolmogorov::SBKolmogorovImpl::kValue(const Position<double>& k) const
    {
        double ksq = (k.x*k.x+k.y*k.y) * _inv_k0sq;
#ifdef DEBUGLOGGING
        xdbg<<"Kolmogorov kValue: ksq = "<<(k.x*k.x + k.y*k.y)<<" * "<<_inv_k0sq<<" = "<<ksq<<std::endl;
        xdbg<<"flux = "<<_flux<<std::endl;
        xdbg<<"info.kval = "<<_info.kValue(ksq)<<std::endl;
        xdbg<<"return "<<_flux * _info.kValue(ksq)<<std::endl;
        double k1 = sqrt(k.x*k.x+k.y*k.y);
        double dk = 6.8839 * std::pow(_lam_over_r0 * k1 / (2.*M_PI),5./3.);
        double tk = exp(-0.5*dk);
        xdbg<<"k = "<<k1<<", D(k) = "<<dk<<", T(k) = "<<tk<<std::endl;
#endif
        return _flux * _info.kValue(ksq);
    }

    // Set maxK to where kValue drops to maxk_threshold
    double SBKolmogorov::SBKolmogorovImpl::maxK() const 
    { return _info.maxK() * _k0; }

    // The amount of flux missed in a circle of radius pi/stepk should miss at 
    // most alias_threshold of the flux.
    double SBKolmogorov::SBKolmogorovImpl::stepK() const
    { return _info.stepK() * _k0; }

    // f(k) = exp(-(k/k0)^5/3)
    // The input value should already be (k/k0)^2
    double KolmogorovInfo::kValue(double ksq) const 
    { return exp(-std::pow(ksq,5./6.)); }

    // Integrand class for the Hankel transform of Kolmogorov
    class KolmIntegrand : public std::unary_function<double,double>
    {
    public:
        KolmIntegrand(double r) : _r(r) {}
        double operator()(double k) const
        { return k*std::exp(-std::pow(k, 5./3.))*j0(k*_r); }

    private:
        double _r;
    };

    // Perform the integral
    class KolmXValue : public std::unary_function<double,double>
    {
    public:
        double operator()(double r) const
        { 
            const double integ_maxK = integ::MOCK_INF;
            KolmIntegrand I(r);
            return integ::int1d(I, 0., integ_maxK,
                                sbp::integration_relerr, sbp::integration_abserr);
        }
    };

#ifdef SOLVE_FWHM_HLR
    // XValue - target  (used for solving for fwhm)
    class KolmTargetValue : public std::unary_function<double,double>
    {
    public:
        KolmTargetValue(double target) : _target(target) {}
        double operator()(double r) const { return f(r) - _target; }
    private:
        KolmXValue f;
        double _target;
    };

    class KolmXValueTimes2piR : public std::unary_function<double,double>
    {
    public:
        double operator()(double r) const
        { return f(r) * r; }
    private:
        KolmXValue f;
    };

    class KolmEnclosedFlux : public std::unary_function<double,double>
    {
    public:
        double operator()(double r) const 
        {
            return integ::int1d(f, 0., r, sbp::integration_relerr, sbp::integration_abserr);
        }
    private:
        KolmXValueTimes2piR f;
    };

    class KolmTargetFlux : public std::unary_function<double,double>
    {
    public:
        KolmTargetFlux(double target) : _target(target) {}
        double operator()(double r) const { return f(r) - _target; }
    private:
        KolmEnclosedFlux f;
        double _target;
    };
#endif
     
    // Constructor to initialize Kolmogorov constants and xvalue lookup table
    KolmogorovInfo::KolmogorovInfo() : _radial(TableDD::spline)
    {
        dbg<<"Initializing KolmogorovInfo\n";

        // Calculate maxK:
        // exp(-k^5/3) = kvalue_accuracy
        _maxk = std::pow(-std::log(sbp::kvalue_accuracy),3./5.);
        dbg<<"maxK = "<<_maxk<<std::endl;

        // Build the table for the radial function.
        double dr = 0.5/_maxk;
        // Start with f(0), which is analytic:
        // According to Wolfram Alpha:
        // Integrate[k*exp(-k^5/3),{k,0,infinity}] = 3/5 Gamma(6/5)
        //    = 0.55090124543985636638457099311149824;
        double val = 0.55090124543985636638457099311149824 / (2.*M_PI);
        _radial.addEntry(0.,val);
        xdbg<<"f(0) = "<<val<<std::endl;
        // Along the way accumulate the flux integral to determine the radius
        // that encloses (1-alias_threshold) of the flux.
        double sum = 0.;
        double thresh1 = (1.-sbp::alias_threshold) / (2.*M_PI*dr);
        double thresh2 = 0.999 / (2.*M_PI*dr);
        double R = 0.;
        // Continue until accumulate 0.999 of the flux
        KolmXValue xval_func;
        for (double r = dr; sum < thresh2; r += dr) {
            val = xval_func(r) / (2.*M_PI);
            xdbg<<"f("<<r<<") = "<<val<<std::endl;
            _radial.addEntry(r,val);

            // Accumulate int(r*f(r)) / dr  (i.e. don't include 2*pi*dr factor as part of sum)
            sum += r * val;
            xdbg<<"sum = "<<sum<<"  thresh1 = "<<thresh1<<"  thesh2 = "<<thresh2<<std::endl;
            xdbg<<"sum*2*pi*dr "<<sum*2.*M_PI*dr<<std::endl;
            if (R == 0. && sum > thresh1) R = r;
        }
        dbg<<"Done loop to build radial function.\n";
        dbg<<"R = "<<R<<std::endl;
        _stepk = M_PI/R;
        dbg<<"stepk = "<<_stepk<<std::endl;
        dbg<<"sum*2*pi*dr = "<<sum*2.*M_PI*dr<<"   (should ~= 0.999)\n";

        // Next, set up the sampler for photon shooting
        std::vector<double> range(2,0.);
        range[1] = _radial.argMax();
        _sampler.reset(new OneDimensionalDeviate(_radial, range, true));

#ifdef SOLVE_FWHM_HLR
        // Improve upon the conversion between lam_over_r0 and fwhm:
        KolmTargetValue fwhm_func(0.55090124543985636638457099311149824 / 2.);
        double r1 = 1.4;
        double r2 = 1.5;
        Solve<KolmTargetValue> fwhm_solver(fwhm_func,r1,r2);
        fwhm_solver.setMethod(Brent);
        double rd = fwhm_solver.root();
        xdbg<<"Root is "<<rd<<std::endl;
        // This is in units of 1/k0.  k0 = 2.992934 / lam_over_r0
        // It's also the half-width hal-max, so * 2 to get fwhm.
        xdbg<<"fwhm = "<<rd * 2. / 2.992934<<" * lam_over_r0\n";

        // Confirm that flux function gets unit flux when integrated to infinity:
        KolmEnclosedFlux enc_flux;
        for(double rmax = 0.; rmax < 20.; rmax += 1.) {
            dbg<<"Flux enclosed by r="<<rmax<<" = "<<enc_flux(rmax)<<std::endl;
        }

        // Next find the conversion between lam_over_r0 and hlr:
        KolmTargetFlux hlr_func(0.5);
        r1 = 1.6;
        r2 = 1.7;
        Solve<KolmTargetFlux> hlr_solver(hlr_func,r1,r2);
        hlr_solver.setMethod(Brent);
        rd = hlr_solver.root();
        xdbg<<"Root is "<<rd<<std::endl;
        dbg<<"Flux enclosed by r="<<rd<<" = "<<enc_flux(rd)<<std::endl;
        // This is in units of 1/k0.  k0 = 2.992934 / lam_over_r0
        xdbg<<"hlr = "<<rd / 2.992934<<" * lam_over_r0\n";
#endif
    }

    boost::shared_ptr<PhotonArray> KolmogorovInfo::shoot(int N, UniformDeviate ud) const
    {
        dbg<<"KolmogorovInfo shoot: N = "<<N<<std::endl;
        dbg<<"Target flux = 1.0\n";
        assert(_sampler.get());
        boost::shared_ptr<PhotonArray> result = _sampler->shoot(N,ud);
        //result->scaleFlux(_norm);
        dbg<<"KolmogorovInfo Realized flux = "<<result->getTotalFlux()<<std::endl;
        return result;
    }

    boost::shared_ptr<PhotonArray> SBKolmogorov::SBKolmogorovImpl::shoot(
        int N, UniformDeviate ud) const
    {
        dbg<<"Kolmogorov shoot: N = "<<N<<std::endl;
        dbg<<"Target flux = "<<getFlux()<<std::endl;
        // Get photons from the KolmogorovInfo structure, rescale flux and size for this instance
        boost::shared_ptr<PhotonArray> result = _info.shoot(N,ud);
        result->scaleFlux(_flux);
        result->scaleXY(1./_k0);
        dbg<<"Kolmogorov Realized flux = "<<result->getTotalFlux()<<std::endl;
        return result;
    }
}
