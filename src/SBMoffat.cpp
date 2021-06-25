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

#include <cmath>

#include "SBMoffat.h"
#include "SBMoffatImpl.h"
#include "integ/Int.h"
#include "Solve.h"
#include "math/Bessel.h"
#include "math/Gamma.h"
#include "math/Angle.h"
#include "math/Hankel.h"
#include "fmath/fmath.hpp"

// Define this variable to find azimuth (and sometimes radius within a unit disc) of 2d photons by
// drawing a uniform deviate for theta, instead of drawing 2 deviates for a point on the unit
// circle and rejecting corner photons.
// The relative speed of the two methods was tested as part of issue #163, and the results
// are collated in devutils/external/time_photon_shooting.
// The conclusion was that using sin/cos was faster for icpc, but not g++ or clang++.
#ifdef _INTEL_COMPILER
#define USE_COS_SIN
#endif

namespace galsim {

    inline double fast_pow(double x, double y)
    { return fmath::expd(y * std::log(x)); }

    SBMoffat::SBMoffat(double beta, double scale_radius, double trunc, double flux,
                       const GSParams& gsparams) :
        SBProfile(new SBMoffatImpl(beta, scale_radius, trunc, flux, gsparams)) {}

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

    double SBMoffat::getTrunc() const
    {
        assert(dynamic_cast<const SBMoffatImpl*>(_pimpl.get()));
        return static_cast<const SBMoffatImpl&>(*_pimpl).getTrunc();
    }

    class MoffatScaleRadiusFunc
    {
    public:
        MoffatScaleRadiusFunc(double re, double rm, double beta) :
            _re(re), _rm(rm), _beta(beta) {}
        double operator()(double rd) const
        {
            double fre = 1.-fast_pow(1.+(_re*_re)/(rd*rd), 1.-_beta);
            double frm = 1.-fast_pow(1.+(_rm*_rm)/(rd*rd), 1.-_beta);
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

    SBMoffat::SBMoffatImpl::SBMoffatImpl(double beta, double scale_radius,
                                         double trunc, double flux,
                                         const GSParams& gsparams) :
        SBProfileImpl(gsparams),
        _beta(beta), _flux(flux), _rD(scale_radius),
        _rD_sq(_rD * _rD), _inv_rD(1./_rD), _inv_rD_sq(_inv_rD*_inv_rD),
        _trunc(trunc),
        _ft(Table::spline),
        _stepk(0.), // calculated by stepK() and stored.
        _maxk(0.) // calculated by maxK() and stored.
    {
        xdbg<<"Start SBMoffat constructor: \n";
        xdbg<<"beta = "<<_beta<<"\n";
        xdbg<<"flux = "<<_flux<<"\n";
        xdbg<<"trunc = "<<_trunc<<"\n";

        if (_trunc == 0. && beta <= 1.1)
            throw SBError("Moffat profiles with beta <= 1.1 must be truncated.");

        if (_trunc < 0.)
            throw SBError("Invalid negative truncation radius provided to SBMoffat.");

        if (_trunc > 0.) {
            _maxRrD = _trunc * _inv_rD;
            xdbg<<"maxRrD = "<<_maxRrD<<"\n";

            // Analytic integration of total flux:
            _fluxFactor = 1. - std::pow( 1+_maxRrD*_maxRrD, (1.-_beta));
        } else {
            _fluxFactor = 1.;

            // Set maxRrD to the radius where missing fractional flux is xvalue_accuracy
            // (1+R^2)^(1-beta) = xvalue_accuracy
            _maxRrD = std::sqrt(std::pow(this->gsparams.xvalue_accuracy, 1. / (1. - _beta))- 1.);
            xdbg<<"Not truncated.  Calculated maxRrD = "<<_maxRrD<<"\n";
        }

        _maxR = _maxRrD * _rD;
        _maxR_sq = _maxR * _maxR;
        _maxRrD_sq = _maxRrD * _maxRrD;
        _norm = _flux * (_beta-1.) / (M_PI * _fluxFactor * _rD_sq);
        _knorm = _flux;

        dbg << "Moffat rD " << _rD << " fluxFactor " << _fluxFactor
            << " norm " << _norm << " maxR " << _maxR << std::endl;

        if (std::abs(_beta-1) < this->gsparams.xvalue_accuracy)
            _pow_mbeta = &SBMoffatImpl::pow_1;
        else if (std::abs(_beta-1.5) < this->gsparams.xvalue_accuracy)
            _pow_mbeta = &SBMoffatImpl::pow_15;
        else if (std::abs(_beta-2) < this->gsparams.xvalue_accuracy)
            _pow_mbeta = &SBMoffatImpl::pow_2;
        else if (std::abs(_beta-2.5) < this->gsparams.xvalue_accuracy)
            _pow_mbeta = &SBMoffatImpl::pow_25;
        else if (std::abs(_beta-3) < this->gsparams.xvalue_accuracy)
            _pow_mbeta = &SBMoffatImpl::pow_3;
        else if (std::abs(_beta-3.5) < this->gsparams.xvalue_accuracy)
            _pow_mbeta = &SBMoffatImpl::pow_35;
        else if (std::abs(_beta-4) < this->gsparams.xvalue_accuracy)
            _pow_mbeta = &SBMoffatImpl::pow_4;
        else _pow_mbeta = &SBMoffatImpl::pow_gen;

        if (_trunc > 0.) _kV = &SBMoffatImpl::kV_trunc;
        else if (std::abs(_beta-1.5) < this->gsparams.kvalue_accuracy)
            _kV = &SBMoffatImpl::kV_15;
        else if (std::abs(_beta-2) < this->gsparams.kvalue_accuracy)
            _kV = &SBMoffatImpl::kV_2;
        else if (std::abs(_beta-2.5) < this->gsparams.kvalue_accuracy)
            _kV = &SBMoffatImpl::kV_25;
        else if (std::abs(_beta-3) < this->gsparams.kvalue_accuracy) {
            _kV = &SBMoffatImpl::kV_3; _knorm /= 2.;
        } else if (std::abs(_beta-3.5) < this->gsparams.kvalue_accuracy) {
            _kV = &SBMoffatImpl::kV_35; _knorm /= 3.;
        } else if (std::abs(_beta-4) < this->gsparams.kvalue_accuracy) {
            _kV = &SBMoffatImpl::kV_4; _knorm /= 8.;
        } else {
            _kV = &SBMoffatImpl::kV_gen;
            _knorm *= 4. / (math::tgamma(beta-1.) * std::pow(2.,beta));
        }
    }

    double SBMoffat::SBMoffatImpl::getHalfLightRadius() const
    {
        return _rD * std::sqrt(std::pow(1.-0.5*_fluxFactor , 1./(1.-_beta)) - 1.);
    }

    double SBMoffat::SBMoffatImpl::getFWHM() const
    {
        return _rD * 2.* std::sqrt(std::pow(2., 1./_beta)-1.);
    }

    double SBMoffat::SBMoffatImpl::xValue(const Position<double>& p) const
    {
        double rsq = (p.x*p.x + p.y*p.y)*_inv_rD_sq;
        if (rsq > _maxRrD_sq) return 0.;
        else return _norm * _pow_mbeta(1.+rsq, _beta);
    }

    // Specialized functions for x**-beta for some probably common choices for beta, which
    // can be done faster than using fast_pow(x,-beta).
    double SBMoffat::SBMoffatImpl::pow_1(double x, double ) { return 1./x; }
    double SBMoffat::SBMoffatImpl::pow_15(double x, double ) { return 1./(x * std::sqrt(x)); }
    double SBMoffat::SBMoffatImpl::pow_2(double x, double ) { return 1./(x*x); }
    double SBMoffat::SBMoffatImpl::pow_25(double x, double ) { return 1./(x*x * std::sqrt(x)); }
    double SBMoffat::SBMoffatImpl::pow_3(double x, double ) { return 1./(x*x*x); }
    double SBMoffat::SBMoffatImpl::pow_35(double x, double ) { return 1./(x*x*x * std::sqrt(x)); }
    double SBMoffat::SBMoffatImpl::pow_4(double x, double ) { double xsq=x*x; return 1./(xsq*xsq); }
    double SBMoffat::SBMoffatImpl::pow_gen(double x, double beta) { return fast_pow(x,-beta); }

    std::complex<double> SBMoffat::SBMoffatImpl::kValue(const Position<double>& k) const
    {
        double ksq = (k.x*k.x + k.y*k.y)*_rD_sq;
        return _knorm * (this->*_kV)(ksq);
    }

    double SBMoffat::SBMoffatImpl::kV_15(double ksq) const
    {
        double k = sqrt(ksq);
        return fmath::expd(-k);
    }

    double SBMoffat::SBMoffatImpl::kV_2(double ksq) const
    {
        if (ksq == 0.) return 1.;
        else {
            double k = sqrt(ksq);
            return math::cyl_bessel_k(1,k) * k;
        }
    }

    double SBMoffat::SBMoffatImpl::kV_25(double ksq) const
    {
        double k = sqrt(ksq);
        return fmath::expd(-k)*(1.+k);
    }

    double SBMoffat::SBMoffatImpl::kV_3(double ksq) const
    {
        if (ksq == 0.) return 2.;
        else {
            double k = sqrt(ksq);
            return math::cyl_bessel_k(2,k) * ksq;
        }
    }

    double SBMoffat::SBMoffatImpl::kV_35(double ksq) const
    {
        double k = sqrt(ksq);
        return fmath::expd(-k)*(3.+(3.+k)*k);
    }

    double SBMoffat::SBMoffatImpl::kV_4(double ksq) const
    {
        if (ksq == 0.) return 8.;
        else {
            double k = sqrt(ksq);
            return math::cyl_bessel_k(3,k) * k*ksq;
        }
    }

    double SBMoffat::SBMoffatImpl::kV_gen(double ksq) const
    {
        if (ksq == 0.) return _flux/_knorm;
        else {
            double k = sqrt(ksq);
            return math::cyl_bessel_k(_beta-1,k) * fast_pow(k,_beta-1);
        }
    }

    double SBMoffat::SBMoffatImpl::kV_trunc(double ksq) const
    {
        setupFT();
        if (ksq > _ft.argMax()) return 0.;
        else return _ft(ksq);
    }

    template <typename T>
    void SBMoffat::SBMoffatImpl::fillXImage(ImageView<T> im,
                                            double x0, double dx, int izero,
                                            double y0, double dy, int jzero) const
    {
        dbg<<"SBMoffat fillXImage\n";
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

            x0 *= _inv_rD;
            dx *= _inv_rD;
            y0 *= _inv_rD;
            dy *= _inv_rD;

            for (int j=0; j<n; ++j,y0+=dy,ptr+=skip) {
                double x = x0;
                double ysq = y0*y0;
                for (int i=0; i<m; ++i,x+=dx) {
                    double rsq = x*x + ysq;
                    if (rsq <= _maxRrD_sq)
                        *ptr++ = _norm * _pow_mbeta(1.+rsq, _beta);
                    else
                        *ptr++ = T(0);
                }
            }
        }
    }

    template <typename T>
    void SBMoffat::SBMoffatImpl::fillXImage(ImageView<T> im,
                                            double x0, double dx, double dxy,
                                            double y0, double dy, double dyx) const
    {
        dbg<<"SBMoffat fillXImage\n";
        dbg<<"x = "<<x0<<" + i * "<<dx<<" + j * "<<dxy<<std::endl;
        dbg<<"y = "<<y0<<" + i * "<<dyx<<" + j * "<<dy<<std::endl;
        const int m = im.getNCol();
        const int n = im.getNRow();
        T* ptr = im.getData();
        const int skip = im.getNSkip();
        assert(im.getStep() == 1);

        x0 *= _inv_rD;
        dx *= _inv_rD;
        dxy *= _inv_rD;
        y0 *= _inv_rD;
        dy *= _inv_rD;
        dyx *= _inv_rD;

        for (int j=0; j<n; ++j,x0+=dxy,y0+=dy,ptr+=skip) {
            double x = x0;
            double y = y0;
            for (int i=0; i<m; ++i,x+=dx,y+=dyx) {
                double rsq = x*x + y*y;
                if (rsq <= _maxRrD_sq)
                    *ptr++ = _norm * _pow_mbeta(1.+rsq, _beta);
                else
                    *ptr++ = T(0);
            }
        }
    }

    template <typename T>
    void SBMoffat::SBMoffatImpl::fillKImage(ImageView<std::complex<T> > im,
                                                double kx0, double dkx, int izero,
                                                double ky0, double dky, int jzero) const
    {
        dbg<<"SBMoffat fillKImage\n";
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

            kx0 *= _rD;
            dkx *= _rD;
            ky0 *= _rD;
            dky *= _rD;

            for (int j=0; j<n; ++j,ky0+=dky,ptr+=skip) {
                double kx = kx0;
                double kysq = ky0*ky0;
                for (int i=0;i<m;++i,kx+=dkx)
                    *ptr++ = _knorm * (this->*_kV)(kx*kx + kysq);
            }
        }
    }

    template <typename T>
    void SBMoffat::SBMoffatImpl::fillKImage(ImageView<std::complex<T> > im,
                                            double kx0, double dkx, double dkxy,
                                            double ky0, double dky, double dkyx) const
    {
        dbg<<"SBMoffat fillKImage\n";
        dbg<<"kx = "<<kx0<<" + i * "<<dkx<<" + j * "<<dkxy<<std::endl;
        dbg<<"ky = "<<ky0<<" + i * "<<dkyx<<" + j * "<<dky<<std::endl;
        const int m = im.getNCol();
        const int n = im.getNRow();
        std::complex<T>* ptr = im.getData();
        int skip = im.getNSkip();
        assert(im.getStep() == 1);

        kx0 *= _rD;
        dkx *= _rD;
        dkxy *= _rD;
        ky0 *= _rD;
        dky *= _rD;
        dkyx *= _rD;

        for (int j=0; j<n; ++j,kx0+=dkxy,ky0+=dky,ptr+=skip) {
            double kx = kx0;
            double ky = ky0;
            for (int i=0; i<m; ++i,kx+=dkx,ky+=dkyx)
                *ptr++ = _knorm * (this->*_kV)(kx*kx + ky*ky);
        }
    }

    // Set maxK to the value where the FT is down to maxk_threshold
    double SBMoffat::SBMoffatImpl::maxK() const
    {
        if (_maxk == 0.) {
            if (_trunc == 0.) {
                // f(k) = 4 K(beta-1,k) (k/2)^beta / Gamma(beta-1)
                //
                // The asymptotic formula for K(beta-1,k) is
                //     K(beta-1,k) ~= sqrt(pi/(2k)) exp(-k)
                //
                // So f(k) becomes
                //
                // f(k) ~= 2 sqrt(pi) (k/2)^(beta-1/2) exp(-k) / Gamma(beta-1)
                //
                // Solve for f(k) = maxk_threshold
                //
                double temp = (this->gsparams.maxk_threshold
                               * math::tgamma(_beta-1.)
                               * std::pow(2.,_beta-0.5)
                               / (2. * sqrt(M_PI)));
                // Solve k^(beta-1/2) exp(-k) = temp
                // (beta-1/2) log(k) - k = log(temp)
                // k = (beta-1/2) log(k) - log(temp)
                temp = std::log(temp);
                _maxk = -temp;
                dbg<<"temp = "<<temp<<std::endl;
                for (int i=0;i<5;++i) {
                    _maxk = (_beta-0.5) * std::log(_maxk) - temp;
                    dbg<<"_maxk = "<<_maxk<<std::endl;
                }
            } else {
                // _maxk is determined during setupFT() as the last k value to have a
                // kValue > 1.e-3.
                setupFT();
            }
        }
        return _maxk*_inv_rD;
    }

    // The amount of flux missed in a circle of radius pi/stepk should be at
    // most folding_threshold of the flux.
    double SBMoffat::SBMoffatImpl::stepK() const
    {
        dbg<<"Find Moffat stepK\n";
        dbg<<"beta = "<<_beta<<std::endl;

        if (_stepk == 0.) {
            // The fractional flux out to radius R is (if not truncated)
            // 1 - (1+R^2)^(1-beta)
            // So solve (1+R^2)^(1-beta) = folding_threshold
            if (_beta <= 1.1) {
                // Then flux never converges (or nearly so), so just use truncation radius
                _stepk = M_PI / _maxR;
            } else {
                // Ignore the 1 in (1+R^2), so approximately:
                double R = std::pow(this->gsparams.folding_threshold, 0.5/(1.-_beta)) * _rD;
                dbg<<"R = "<<R<<std::endl;
                // If it is truncated at less than this, drop to that value.
                if (R > _maxR) R = _maxR;
                dbg<<"_maxR = "<<_maxR<<std::endl;
                dbg<<"R => "<<R<<std::endl;
                dbg<<"stepk = "<<(M_PI/R)<<std::endl;
                // Make sure it is at least 5 hlr
                R = std::max(R,gsparams.stepk_minimum_hlr*getHalfLightRadius());
                _stepk = M_PI / R;
            }
        }
        return _stepk;
    }

    // Integrand class for the Hankel transform of Moffat
    class MoffatIntegrand : public std::function<double(double)>
    {
    public:
        MoffatIntegrand(double beta, double (*pb)(double, double)) :
            _beta(beta), _pow_mbeta(pb) {}
        double operator()(double r) const
        { return _pow_mbeta(1.+r*r, _beta); }

    private:
        double _beta;
        double (*_pow_mbeta)(double x, double beta);
    };

    void SBMoffat::SBMoffatImpl::setupFT() const
    {
        assert(_trunc > 0.);
        if (_ft.finalized()) return;

        // Do a Hankel transform and store the results in a lookup table.

        double prefactor = 2. * (_beta-1.) / (_fluxFactor);

        // Along the way, find the last k that has a kValue > 1.e-3
        double maxk_val = this->gsparams.maxk_threshold;
        dbg<<"Looking for maxk_val = "<<maxk_val<<std::endl;
        // Keep going until at least 5 in a row have kvalues below kvalue_accuracy.
        // (It's oscillatory, so want to make sure not to stop at a zero crossing.)

        // We use a cubic spline for the interpolation, which has an error of O(h^4) max(f'''').
        // I have no idea what range the fourth derivative can take for the hankel transform,
        // so let's take the completely arbitrary value of 10.  (This value was found to be
        // conservative for Sersic, but I haven't investigated here.)
        // 10 h^4 <= kvalue_accuracy
        // h = (kvalue_accuracy/10)^0.25
        double dk = gsparams.table_spacing * sqrt(sqrt(gsparams.kvalue_accuracy / 10.));
        dbg<<"dk = "<<dk<<std::endl;
        int n_below_thresh = 0;
        MoffatIntegrand I(_beta, _pow_mbeta);
        // Don't go past k = 50
        for(double k=0.; k < 50; k += dk) {

            double val;
            if (_trunc > 0) {
                val = math::hankel_trunc(I, k, 0., _maxRrD,
                                         this->gsparams.integration_relerr,
                                         this->gsparams.integration_abserr);
            } else {
                val = math::hankel_inf(I, k, 0.,
                                       this->gsparams.integration_relerr,
                                       this->gsparams.integration_abserr);
            }
            val *= prefactor;

            xdbg<<"ft("<<k<<") = "<<val<<std::endl;
            _ft.addEntry(k*k, val);

            if (std::abs(val) > maxk_val) _maxk = k;

            if (std::abs(val) > this->gsparams.kvalue_accuracy) n_below_thresh = 0;
            else ++n_below_thresh;
            if (n_below_thresh == 5) break;
        }
        _ft.finalize();
        dbg<<"maxk = "<<_maxk<<std::endl;
    }

    void SBMoffat::SBMoffatImpl::shoot(PhotonArray& photons, UniformDeviate ud) const
    {
        const int N = photons.size();
        dbg<<"Moffat shoot: N = "<<N<<std::endl;
        dbg<<"Target flux = "<<getFlux()<<std::endl;
        // Moffat has analytic inverse-cumulative-flux function.
        double fluxPerPhoton = _flux/N;
        for (int i=0; i<N; i++) {
#ifdef USE_COS_SIN
            // First get a point uniformly distributed on unit circle
            double theta = 2.*M_PI*ud();
            double rsq = ud(); // cumulative dist function P(<r) = r^2 for unit circle
            double sint,cost;
            math::sincos(theta, sint, cost);
            // Then map radius to the Moffat flux distribution
            double newRsq = fast_pow(1. - rsq * _fluxFactor, 1. / (1. - _beta)) - 1.;
            double rFactor = _rD * std::sqrt(newRsq);
            photons.setPhoton(i, rFactor*cost, rFactor*sint, fluxPerPhoton);
#else
            // First get a point uniformly distributed on unit circle
            double xu, yu, rsq;
            do {
                xu = 2.*ud()-1.;
                yu = 2.*ud()-1.;
                rsq = xu*xu+yu*yu;
            } while (rsq>=1. || rsq==0.);
            // Then map radius to the Moffat flux distribution
            double newRsq = fast_pow(1. - rsq * _fluxFactor, 1. / (1. - _beta)) - 1.;
            double rFactor = _rD * std::sqrt(newRsq / rsq);
            photons.setPhoton(i, rFactor*xu, rFactor*yu, fluxPerPhoton);
#endif
        }
        dbg<<"Moffat Realized flux = "<<photons.getTotalFlux()<<std::endl;
    }

}
