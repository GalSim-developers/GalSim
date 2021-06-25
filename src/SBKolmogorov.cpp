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

#include "SBKolmogorov.h"
#include "SBKolmogorovImpl.h"
#include "math/Bessel.h"
#include "math/Hankel.h"
#include "fmath/fmath.hpp"

// Uncomment this to do the calculation that solves for the conversion between lam_over_r0
// and fwhm and hlr.
// (Solved values are put into Kolmogorov class in galsim/base.py = 0.9758634299, 0.5548101137)
//#define SOLVE_FWHM_HLR

#ifdef SOLVE_FWHM_HLR
#include <iomanip>
#include "Solve.h"
#endif

// This is used in two places below, so store it up here.
// See KolmogorovInfo constructor for the explanation of this number.
// It is equal to 3/5 Gamma(6/5) / 2pi
#define XVAL_ZERO 0.0876786563672346

// Another magic number.  This one turns out to be
// 2Pi (24 Gamma(6/5) / 5)^(-1/2)
#define K0_FACTOR 2.992939911888651

namespace galsim {

    inline double fast_pow(double x, double y)
    { return fmath::expd(y * std::log(x)); }

    SBKolmogorov::SBKolmogorov(double lam_over_r0, double flux, const GSParams& gsparams) :
        SBProfile(new SBKolmogorovImpl(lam_over_r0, flux, gsparams)) {}

    SBKolmogorov::SBKolmogorov(const SBKolmogorov& rhs) : SBProfile(rhs) {}

    SBKolmogorov::~SBKolmogorov() {}

    double SBKolmogorov::getLamOverR0() const
    {
        assert(dynamic_cast<const SBKolmogorovImpl*>(_pimpl.get()));
        return static_cast<const SBKolmogorovImpl&>(*_pimpl).getLamOverR0();
    }

    LRUCache<GSParamsPtr, KolmogorovInfo> SBKolmogorov::SBKolmogorovImpl::cache(
        sbp::max_kolmogorov_cache);

    // The "magic" number we call K0_FACTOR omes from the standard form of the Kolmogorov spectrum
    // from Racine, 1996 PASP, 108, 699 (who in turn is quoting Fried, 1966, JOSA, 56, 1372):
    // T(k) = exp(-1/2 D(k))
    // D(k) = 6.8839 (lambda/r0 k/2Pi)^(5/3)
    //
    // We convert this into T(k) = exp(-(k/k0)^5/3) for efficiency,
    // which implies 1/2 6.8839 (lambda/r0 / 2Pi)^5/3 = (1/k0)^5/3
    // k0 * lambda/r0 = 2Pi * (6.8839 / 2)^-3/5 = 2.992934
    //
    // Update: It turns out that 6.8839/2 is actually (24 Gamma(6/5) / 5)^(5/6)
    // Which in turn makes the constant factor above
    // 2Pi (24 Gamma(6/5) / 5)^(5/6)^-(3/5)
    // = 2Pi (24 Gamma(6/5) / 5)^(-1/2)
    // = 2.992939911888651
    // (Not that we need this many digits, but hey, why not?)
    SBKolmogorov::SBKolmogorovImpl::SBKolmogorovImpl(
        double lam_over_r0, double flux, const GSParams& gsparams) :
        SBProfileImpl(gsparams),
        _lam_over_r0(lam_over_r0),
        _k0(K0_FACTOR / lam_over_r0),
        _k0sq(_k0*_k0),
        _inv_k0(1./_k0),
        _inv_k0sq(1./_k0sq),
        _flux(flux),
        _xnorm(_flux * _k0sq),
        _info(cache.get(GSParamsPtr(this->gsparams)))
    {
        dbg<<"SBKolmogorov:\n";
        dbg<<"lam_over_r0 = "<<_lam_over_r0<<std::endl;
        dbg<<"k0 = "<<_k0<<std::endl;
        dbg<<"flux = "<<_flux<<std::endl;
        dbg<<"xnorm = "<<_xnorm<<std::endl;
    }

    double SBKolmogorov::SBKolmogorovImpl::maxSB() const
    {
        // _info->xValue(r) is just XVAL_ZERO, defined above.
        return _xnorm * XVAL_ZERO;
    }

    double SBKolmogorov::SBKolmogorovImpl::xValue(const Position<double>& p) const
    {
        double r = sqrt(p.x*p.x+p.y*p.y) * _k0;
        return _xnorm * _info->xValue(r);
    }

    double KolmogorovInfo::xValue(double r) const
    { return r < _radial.argMax() ? _radial(r) : 0.; }

    std::complex<double> SBKolmogorov::SBKolmogorovImpl::kValue(const Position<double>& k) const
    {
        double ksq = (k.x*k.x+k.y*k.y) * _inv_k0sq;
        return _flux * _info->kValue(ksq);
    }

    template <typename T>
    void SBKolmogorov::SBKolmogorovImpl::fillXImage(ImageView<T> im,
                                                    double x0, double dx, int izero,
                                                    double y0, double dy, int jzero) const
    {
        dbg<<"SBKolmogorov fillXImage\n";
        dbg<<"x = "<<x0<<" + i * "<<dx<<", izero = "<<izero<<std::endl;
        dbg<<"y = "<<y0<<" + j * "<<dy<<", jzero = "<<jzero<<std::endl;
        if (izero != 0 || jzero != 0) {
            xdbg<<"Use Quadrant\n";
            fillXImageQuadrant(im,x0,dx,izero,y0,dy,jzero);
        } else {
            xdbg<<"Non-Quadrant\n";
            const int m = im.getNCol();
            const int n = im.getNRow();
            T* ptr = im.getData();
            const int skip = im.getNSkip();
            assert(im.getStep() == 1);

            x0 *= _k0;
            dx *= _k0;
            y0 *= _k0;
            dy *= _k0;

            for (int j=0; j<n; ++j,y0+=dy,ptr+=skip) {
                double x = x0;
                double ysq = y0*y0;
                for (int i=0; i<m; ++i,x+=dx)
                    *ptr++ = _xnorm * _info->xValue(sqrt(x*x + ysq));
            }
        }
    }

    template <typename T>
    void SBKolmogorov::SBKolmogorovImpl::fillXImage(ImageView<T> im,
                                                    double x0, double dx, double dxy,
                                                    double y0, double dy, double dyx) const
    {
        dbg<<"SBKolmogorov fillXImage\n";
        dbg<<"x = "<<x0<<" + i * "<<dx<<" + j * "<<dxy<<std::endl;
        dbg<<"y = "<<y0<<" + i * "<<dyx<<" + j * "<<dy<<std::endl;
        const int m = im.getNCol();
        const int n = im.getNRow();
        T* ptr = im.getData();
        const int skip = im.getNSkip();
        assert(im.getStep() == 1);

        x0 *= _k0;
        dx *= _k0;
        dxy *= _k0;
        y0 *= _k0;
        dy *= _k0;
        dyx *= _k0;

        for (int j=0; j<n; ++j,x0+=dxy,y0+=dy,ptr+=skip) {
            double x = x0;
            double y = y0;
            for (int i=0; i<m; ++i,x+=dx,y+=dyx)
                *ptr++ = _xnorm * _info->xValue(sqrt(x*x + y*y));
        }
    }

    template <typename T>
    void SBKolmogorov::SBKolmogorovImpl::fillKImage(ImageView<std::complex<T> > im,
                                                double kx0, double dkx, int izero,
                                                double ky0, double dky, int jzero) const
    {
        dbg<<"SBKolmogorov fillKImage\n";
        dbg<<"kx = "<<kx0<<" + i * "<<dkx<<", izero = "<<izero<<std::endl;
        dbg<<"ky = "<<ky0<<" + j * "<<dky<<", jzero = "<<jzero<<std::endl;
        if (izero != 0 || jzero != 0) {
            xdbg<<"Use Quadrant\n";
            fillKImageQuadrant(im,kx0,dkx,izero,ky0,dky,jzero);
        } else {
            xdbg<<"Non-Quadrant\n";
            const int m = im.getNCol();
            const int n = im.getNRow();
            std::complex<T>* ptr = im.getData();
            int skip = im.getNSkip();
            assert(im.getStep() == 1);

            kx0 *= _inv_k0;
            dkx *= _inv_k0;
            ky0 *= _inv_k0;
            dky *= _inv_k0;

            for (int j=0; j<n; ++j,ky0+=dky,ptr+=skip) {
                double kx = kx0;
                double kysq = ky0*ky0;
                for (int i=0;i<m;++i,kx+=dkx)
                    *ptr++ = _flux * _info->kValue(kx*kx+kysq);
            }
        }
    }

    template <typename T>
    void SBKolmogorov::SBKolmogorovImpl::fillKImage(ImageView<std::complex<T> > im,
                                                    double kx0, double dkx, double dkxy,
                                                    double ky0, double dky, double dkyx) const
    {
        dbg<<"SBKolmogorov fillKImage\n";
        dbg<<"kx = "<<kx0<<" + i * "<<dkx<<" + j * "<<dkxy<<std::endl;
        dbg<<"ky = "<<ky0<<" + i * "<<dkyx<<" + j * "<<dky<<std::endl;
        const int m = im.getNCol();
        const int n = im.getNRow();
        std::complex<T>* ptr = im.getData();
        int skip = im.getNSkip();
        assert(im.getStep() == 1);

        kx0 *= _inv_k0;
        dkx *= _inv_k0;
        dkxy *= _inv_k0;
        ky0 *= _inv_k0;
        dky *= _inv_k0;
        dkyx *= _inv_k0;

        for (int j=0; j<n; ++j,kx0+=dkxy,ky0+=dky,ptr+=skip) {
            double kx = kx0;
            double ky = ky0;
            for (int i=0; i<m; ++i,kx+=dkx,ky+=dkyx)
                *ptr++ = _flux * _info->kValue(kx*kx+ky*ky);
        }
    }

    // Set maxK to where kValue drops to maxk_threshold
    double SBKolmogorov::SBKolmogorovImpl::maxK() const
    { return _info->maxK() * _k0; }

    // The amount of flux missed in a circle of radius pi/stepk should be at
    // most folding_threshold of the flux.
    double SBKolmogorov::SBKolmogorovImpl::stepK() const
    { return _info->stepK() * _k0; }

    // f(k) = exp(-(k/k0)^5/3)
    // The input value should already be (k/k0)^2
    double KolmogorovInfo::kValue(double ksq) const
    { return fmath::expd(-fast_pow(ksq,5./6.)); }

    class KolmKValue : public std::function<double(double)>
    {
    public:
        double operator()(double k) const
        { return fmath::expd(-fast_pow(k, 5./3.)); }
    };

    // Perform the integral
    class KolmXValue : public std::function<double(double)>
    {
    public:
        KolmXValue(const GSParams& gsparams) : _gsparams(gsparams) {}

        double operator()(double r) const
        {
            KolmKValue kvalue;
            return math::hankel_inf(kvalue, r, 0,
                                    _gsparams.integration_relerr,
                                    _gsparams.integration_abserr);
        }
    private:
        const GSParams& _gsparams;
    };

#ifdef SOLVE_FWHM_HLR
    // XValue - target  (used for solving for fwhm)
    class KolmTargetValue : public std::function<double(double)>
    {
    public:
        KolmTargetValue(double target, const GSParams& gsparams) :
            f(gsparams), _target(target) {}
        double operator()(double r) const { return f(r) - _target; }
    private:
        KolmXValue f;
        double _target;
    };

    class KolmXValueTimes2piR : public std::function<double(double)>
    {
    public:
        KolmXValueTimes2piR(const GSParams& gsparams) : f(gsparams) {}

        double operator()(double r) const
        { return f(r) * r; }
    private:
        KolmXValue f;
    };

    class KolmEnclosedFlux : public std::function<double(double)>
    {
    public:
        KolmEnclosedFlux(const GSParams& gsparams) :
            f(gsparams), _gsparams(gsparams) {}
        double operator()(double r) const
        {
            return integ::int1d(f, 0., r,
                                _gsparams.integration_relerr,
                                _gsparams.integration_abserr);
        }
    private:
        KolmXValueTimes2piR f;
        const GSParams& _gsparams;
    };

    class KolmTargetFlux : public std::function<double(double)>
    {
    public:
        KolmTargetFlux(double target, const GSParams& gsparams) :
            f(gsparams), _target(target) {}
        double operator()(double r) const { return f(r) - _target; }
    private:
        KolmEnclosedFlux f;
        double _target;
    };
#endif

    // Constructor to initialize Kolmogorov constants and xvalue lookup table
    KolmogorovInfo::KolmogorovInfo(const GSParamsPtr& gsparams) :
        _radial(Table::spline)
    {
        dbg<<"Initializing KolmogorovInfo\n";

        // Calculate maxK:
        // exp(-k^5/3) = kvalue_accuracy
        _maxk = std::pow(-std::log(gsparams->kvalue_accuracy),3./5.);
        dbg<<"maxK = "<<_maxk<<std::endl;

        // Build the table for the radial function.

        // Start with f(0), which is analytic:
        // According to Wolfram Alpha:
        // Integrate[k*exp(-k^5/3),{k,0,infinity}] = 3/5 Gamma(6/5)
        // The value we want is this / 2pi, which we define as XVAL_ZERO above.
        double val = XVAL_ZERO;
        _radial.addEntry(0.,val);
        xdbg<<"f(0) = "<<val<<std::endl;

        // We use a cubic spline for the interpolation, which has an error of O(h^4) max(f'''').
        // As with Sersic (since Kolmogorov is just a Sersic with n=0.6 in reverse), we use
        // max(f'''') ~= 10.  So:
        // 10 h^4 <= xvalue_accuracy
        // h = (xvalue_accuracy/10)^0.25
        double dlogr = gsparams->table_spacing * sqrt(sqrt(gsparams->xvalue_accuracy / 10.));
        xdbg<<"dlogr = "<<dlogr<<std::endl;

        // Continue until the missing flux is less than shoot_accuracy.
        double thresh = gsparams->shoot_accuracy / (2.*M_PI);
        xdbg<<"thresh  = "<<thresh<<std::endl;
        KolmXValue xval_func(*gsparams);

        // Don't go over r=1.e4.  F(1.e4) ~ 1.e-14, so if we haven't stopped by then,
        // we're probably hitting numerical precision issues.
        for (double logr=-3.; logr < std::log(1.e4); logr += dlogr) {
            double r = std::exp(logr);
            val = xval_func(r) / (2.*M_PI);
            dbg<<"f("<<r<<") = "<<val<<std::endl;
            _radial.addEntry(r,val);

            // At high r, the profile is well approximated by a power law, F ~ r^-3.67
            // The integral of the missing flux out to infinity is int_r^inf F(r) r dr = F r^2/1.67
            xdbg<<"F r^2/1.67 = "<<val*r*r/1.67<<"  thresh = "<<thresh<<std::endl;
            if (val * r * r / 1.67 < thresh) break;
        }
        _radial.finalize();
        dbg<<"Done loop to build radial function.\n";

        // The large r behavior of F(r) is well approximated by a power law, F ~ r^-3.67
        // This affords an easier calculation of R for stepk than numerically accumulating
        // the integral.
        // F(r) = F1 (r/r1)^-n
        // int_r^inf F(r) 2pi r dr = folding_threshold
        // 2pi F1 r1^n / (n-2) f_t = R^(n-2)
        double r1 = _radial.argMax();
        double F1 = _radial.lookup(r1);
#ifdef DEBUGLOGGING
        double r2 = r1 * (1-dlogr);
        double F2 = _radial.lookup(r2);
        dbg<<"r1,F1 = "<<r1<<','<<F1<<std::endl;
        dbg<<"r2,F2 = "<<r2<<','<<F2<<std::endl;
        // power law index = dlog(F)/dlog(r)
        double n_emp = -(std::log(F2)-std::log(F1)) / (std::log(r2)-std::log(r1));
        dbg<<"Empirical n = "<<n_emp<<std::endl;
#endif
        // Emprically n is very close to 11/3.  This is probably exact, since it's the kind of
        // fraction that probably comes out of the Komogorov turbulence.  But I (MJ) haven't
        // figured out how to prove this.
        // Regardless, let's just always use this for the purpose of estimating stepk,
        // since any deviations from the exactly correct answer don't matter much.
        double n = 11./3.;
        double R = fast_pow(2.*M_PI*F1*fast_pow(r1,n)/((n-2)*gsparams->folding_threshold),
                            1./(n-2));
        dbg<<"R = "<<R<<std::endl;

        // Make sure it is at least 5 hlr
        double hlr = K0_FACTOR * 0.5548101137;  // (k0_factor * hlr_factor)
        dbg<<"hlr = "<<hlr<<std::endl;
        R = std::max(R,gsparams->stepk_minimum_hlr*hlr);
        _stepk = M_PI / R;
        dbg<<"stepk = "<<_stepk<<std::endl;

        // Next, set up the sampler for photon shooting
        std::vector<double> range(2,0.);
        range[1] = _radial.argMax();
        _sampler.reset(new OneDimensionalDeviate(_radial, range, true, 1.0, *gsparams));
        dbg<<"made sampler\n";

#ifdef SOLVE_FWHM_HLR
        // Improve upon the conversion between lam_over_r0 and fwhm:
        KolmTargetValue fwhm_func(0.55090124543985636638457099311149824 / 2., *gsparams);
        double r1 = 1.4;
        double r2 = 1.5;
        Solve<KolmTargetValue> fwhm_solver(fwhm_func,r1,r2);
        fwhm_solver.setMethod(Brent);
        double rd = fwhm_solver.root();
        xdbg<<"Root is "<<rd<<std::endl;
        // This is in units of 1/k0.  k0 = 2.992934 / lam_over_r0
        // It's also the half-width hal-max, so * 2 to get fwhm.
        dbg<<"fwhm = "<<std::setprecision(10)<<rd * 2. / K0_FACTOR<<" * lam_over_r0\n";

        // Confirm that flux function gets unit flux when integrated to infinity:
        KolmEnclosedFlux enc_flux(*gsparams);
        for(double rmax = 0.; rmax < 20.; rmax += 1.) {
            xdbg<<"Flux enclosed by r="<<rmax<<" = "<<enc_flux(rmax)<<std::endl;
        }

        // Next find the conversion between lam_over_r0 and hlr:
        KolmTargetFlux hlr_func(0.5, *gsparams);
        r1 = 1.6;
        r2 = 1.7;
        Solve<KolmTargetFlux> hlr_solver(hlr_func,r1,r2);
        hlr_solver.setMethod(Brent);
        rd = hlr_solver.root();
        xdbg<<"Root is "<<rd<<std::endl;
        xdbg<<"Flux enclosed by r="<<rd<<" = "<<enc_flux(rd)<<std::endl;
        // This is in units of 1/k0.  k0 = 2.992934 / lam_over_r0
        dbg<<"hlr = "<<std::setprecision(10)<<rd / K0_FACTOR<<" * lam_over_r0\n";
#endif
    }

    void KolmogorovInfo::shoot(PhotonArray& photons, UniformDeviate ud) const
    {
        const int N = photons.size();
        dbg<<"KolmogorovInfo shoot: N = "<<N<<std::endl;
        dbg<<"Target flux = 1.0\n";
        assert(_sampler.get());
        _sampler->shoot(photons,ud);
        dbg<<"KolmogorovInfo Realized flux = "<<photons.getTotalFlux()<<std::endl;
    }

    void SBKolmogorov::SBKolmogorovImpl::shoot(PhotonArray& photons, UniformDeviate ud) const
    {
        const int N = photons.size();
        dbg<<"Kolmogorov shoot: N = "<<N<<std::endl;
        dbg<<"Target flux = "<<getFlux()<<std::endl;
        // Get photons from the KolmogorovInfo structure, rescale flux and size for this instance
        _info->shoot(photons,ud);
        photons.scaleFlux(_flux);
        photons.scaleXY(1./_k0);
        dbg<<"Kolmogorov Realized flux = "<<photons.getTotalFlux()<<std::endl;
    }
}
