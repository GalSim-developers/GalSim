/* -*- c++ -*-
 * Copyright (c) 2012-2017 by the GalSim developers team on GitHub
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

// #define DEBUGLOGGING

#include "galsim/IgnoreWarnings.h"

#define BOOST_NO_CXX11_SMART_PTR
#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/special_functions/bessel.hpp>

#include "SBSecondKick.h"
#include "SBSecondKickImpl.h"
#include "fmath/fmath.hpp"
#include "Solve.h"
#include "bessel/Roots.h"
#include <ctime>

namespace galsim {

    const double ARCSEC2RAD = 180.*60*60/M_PI;  // ~206265
    const double MOCK_INF = 1.e300;

    inline double fast_pow(double x, double y)
    { return fmath::expd(y * std::log(x)); }

    //
    //
    //
    //SBSecondKick
    //
    //
    //

    SBSecondKick::SBSecondKick(double lam, double r0, double diam, double obscuration, double L0,
                               double kcrit, double flux, double scale,
                               const GSParamsPtr& gsparams) :
        SBProfile(new SBSecondKickImpl(lam, r0, diam, obscuration, L0, kcrit, flux, scale,
                                       gsparams)) {}

    SBSecondKick::SBSecondKick(const SBSecondKick &rhs) : SBProfile(rhs) {}

    SBSecondKick::~SBSecondKick() {}

    double SBSecondKick::getLam() const
    {
        assert(dynamic_cast<const SBSecondKickImpl*>(_pimpl.get()));
        return static_cast<const SBSecondKickImpl&>(*_pimpl).getLam();
    }

    double SBSecondKick::getR0() const
    {
        assert(dynamic_cast<const SBSecondKickImpl*>(_pimpl.get()));
        return static_cast<const SBSecondKickImpl&>(*_pimpl).getR0();
    }

    double SBSecondKick::getDiam() const
    {
        assert(dynamic_cast<const SBSecondKickImpl*>(_pimpl.get()));
        return static_cast<const SBSecondKickImpl&>(*_pimpl).getDiam();
    }

    double SBSecondKick::getObscuration() const
    {
        assert(dynamic_cast<const SBSecondKickImpl*>(_pimpl.get()));
        return static_cast<const SBSecondKickImpl&>(*_pimpl).getObscuration();
    }

    double SBSecondKick::getL0() const
    {
        assert(dynamic_cast<const SBSecondKickImpl*>(_pimpl.get()));
        return static_cast<const SBSecondKickImpl&>(*_pimpl).getL0();
    }

    double SBSecondKick::getKCrit() const
    {
        assert(dynamic_cast<const SBSecondKickImpl*>(_pimpl.get()));
        return static_cast<const SBSecondKickImpl&>(*_pimpl).getKCrit();
    }

    double SBSecondKick::getScale() const
    {
        assert(dynamic_cast<const SBSecondKickImpl*>(_pimpl.get()));
        return static_cast<const SBSecondKickImpl&>(*_pimpl).getScale();
    }

    double SBSecondKick::getHalfLightRadius() const
    {
        assert(dynamic_cast<const SBSecondKickImpl*>(_pimpl.get()));
        return static_cast<const SBSecondKickImpl&>(*_pimpl).getHalfLightRadius();
    }

    double SBSecondKick::structureFunction(double rho) const
    {
        assert(dynamic_cast<const SBSecondKickImpl*>(_pimpl.get()));
        return static_cast<const SBSecondKickImpl&>(*_pimpl).structureFunction(rho);
    }

    double SBSecondKick::kValue(double k) const
    {
        assert(dynamic_cast<const SBSecondKickImpl*>(_pimpl.get()));
        return static_cast<const SBSecondKickImpl&>(*_pimpl).kValue(k);
    }

    double SBSecondKick::kValueRaw(double k) const
    {
        assert(dynamic_cast<const SBSecondKickImpl*>(_pimpl.get()));
        return static_cast<const SBSecondKickImpl&>(*_pimpl).kValueRaw(k);
    }

    double SBSecondKick::xValue(double k) const
    {
        assert(dynamic_cast<const SBSecondKickImpl*>(_pimpl.get()));
        return static_cast<const SBSecondKickImpl&>(*_pimpl).xValue(k);
    }

    double SBSecondKick::xValueRaw(double k) const
    {
        assert(dynamic_cast<const SBSecondKickImpl*>(_pimpl.get()));
        return static_cast<const SBSecondKickImpl&>(*_pimpl).xValueRaw(k);
    }

    double SBSecondKick::xValueExact(double k) const
    {
        assert(dynamic_cast<const SBSecondKickImpl*>(_pimpl.get()));
        return static_cast<const SBSecondKickImpl&>(*_pimpl).xValueExact(k);
    }

    //
    //
    //
    //SKInfo
    //
    //
    //

    class SKIkValueResid {
    public:
        SKIkValueResid(const SKInfo& ski, double thresh) : _ski(ski), _thresh(thresh) {}
        double operator()(double k) const {
            double val = _ski.kValueRaw(k)-_thresh;
            xdbg<<"resid(k="<<k<<")="<<val<<'\n';
            return val;
        }
    private:
        const SKInfo& _ski;
        const double _thresh;
    };

    SKInfo::SKInfo(double lam, double r0, double diam, double obscuration, double L0,
                   double kcrit, const GSParamsPtr& gsparams) :
        _lam(lam), _lam_factor(lam*ARCSEC2RAD/(2*M_PI)), _r0(r0), _r0m53(pow(r0, -5./3)),
        _diam(diam), _obscuration(obscuration), _L0(L0),
        _L0invsq(1/L0/L0), _r0L0m53(pow(r0/L0, -5./3)), _kmin(kcrit/r0),
        _knorm(1./(M_PI*(1.-obscuration*obscuration))),
        _4_over_diamsq(4.0/diam/diam),
        _gsparams(gsparams),
        _airy_info((obscuration==0.0) ?
                   boost::movelib::unique_ptr<AiryInfo>(new AiryInfoNoObs(gsparams)) :
                   boost::movelib::unique_ptr<AiryInfo>(new AiryInfoObs(obscuration,gsparams))),
        _radial(TableDD::spline),
        _kvLUT(TableDD::spline)
    {
        _maxk = _diam/_lam_factor;
        _stepk = 0;

        // build the radial function, and along the way, set _stepk, _hlr.
        std::clock_t t0 = std::clock();
        _buildKVLUT();
        std::clock_t t1 = std::clock();
        _buildRadial();
        std::clock_t t2 = std::clock();
        std::cout << "buildKV time = " << (double)(t1-t0)/CLOCKS_PER_SEC << '\n';
        std::cout << "buildRad time = " << (double)(t2-t1)/CLOCKS_PER_SEC << '\n';

        // Find a potentially smaller maxk now that LUTs have been built.
        SKIkValueResid skikvr(*this, _gsparams->maxk_threshold);
        Solve<SKIkValueResid> solver(skikvr, 0.0, _maxk);
        solver.setMethod(Brent);
        _maxk = solver.root();
    }

    // This version of structureFunction explicitly integrates from kmin to infinity, which is how
    // the second kick is defined. However, I've found it's more stable to integrate from zero to
    // kmin, and then subtract this from the analytic integral from 0 to infinity.  So that's
    // what's implemented further down.
    // double SKInfo::structureFunction(double rho) const {
    //     SKISFIntegrand I(rho, _L0invsq);
    //     double result = integ::int1d(I, _kmin, integ::MOCK_INF,
    //                                  _gsparams->integration_relerr,
    //                                  _gsparams->integration_abserr);
    //     result *= magic5*_r0m53;
    //     return result;
    // }

    double SKInfo::vkStructureFunction(double rho) const {
        // rho in meters

        // 2 gamma(11/6) / (2^(5/6) pi^(8/3)) * (24/5 gamma(6/5))^(5/6)
        static const double magic1 = 0.1716613621245708932;
        // gamma(5/6) / 2^(1/6)
        static const double magic2 = 1.005634917998590172;
        // magic1 * gamma(-5/6) / 2^(11/6)
        static const double magic3 = -0.3217609479366896341;

        double rhoL0 = rho/_L0;
        if (rhoL0 < 1e-10) {
            return -magic3*fast_pow(2*M_PI*rho/_r0, 5./3);
        } else {
            double x = 2*M_PI*rhoL0;
            return magic1*_r0L0m53*(magic2-fast_pow(x, 5./6)*boost::math::cyl_bessel_k(5./6, x));
        }
    }

    class SKISFIntegrand : public std::unary_function<double,double>
    {
    public:
        SKISFIntegrand(double rho, double L0invsq) : _2pirho(2*M_PI*rho), _L0invsq(L0invsq) {}
        double operator()(double kappa) const {
            return fast_pow(kappa*kappa+_L0invsq, -11./6)*kappa*(1-j0(_2pirho*kappa));
        }
    private:
        const double _2pirho; // 2*pi*rho
        const double _L0invsq;  // inverse meters squared
    };

    double SKInfo::structureFunction(double rho) const {
        // 2 gamma(11/6)^2 / pi^(8/3) (24/5 gamma(6/5))^(5/6)
        const static double magic5 = 0.2877144330394485472;

        SKISFIntegrand I(rho, _L0invsq);
        integ::IntRegion<double> reg(0., _kmin);
        for (int s=1; s<10; s++) {
            double zero = bessel::getBesselRoot0(s)/(2*M_PI*rho);
            if (zero >= _kmin) break;
            reg.addSplit(zero);
        }

        double complement = integ::int1d(I, reg,
                                         _gsparams->integration_relerr,
                                         _gsparams->integration_abserr);

        return vkStructureFunction(rho) - magic5*complement*_r0m53;
    }

    void SKInfo::_buildKVLUT() {
        _kvLUT.addEntry(0, 1.0);

        double dlogk = _gsparams->table_spacing * sqrt(sqrt(_gsparams->kvalue_accuracy / 80.0));
        dbg<<"Using dlogk = "<<dlogk<<'\n';

        SKIkValueResid skikvr(*this, 1.0-_gsparams->kvalue_accuracy);
        Solve<SKIkValueResid> solver(skikvr, 0.0, 0.1);
        solver.bracketUpper();
        solver.setMethod(Brent);
        double k = solver.root();
        dbg<<"Initial k = "<<k<<'\n';

        for (double logk=std::log(k); logk < std::log(_maxk)+dlogk; logk += dlogk) {
            xdbg<<"logk = "<<logk<<'\n';
            k = fmath::expd(logk);
            xdbg<<"k = "<<k<<'\n';
            double val = kValueRaw(k);
            xdbg<<"val = "<<val<<'\n';
            _kvLUT.addEntry(k, val);
        }
        dbg<<"kvLUT.size() = "<<_kvLUT.size()<<'\n';
    }

    double SKInfo::kValue(double k) const {
        return k < _kvLUT.argMax() ? _kvLUT(k) : 0.;
    }

    double SKInfo::kValueRaw(double k) const {
        // k in inverse arcsec
        double kp = _lam_factor*k;
        double kpkp = kp*kp;
        return fmath::expd(-0.5*structureFunction(kp))
            * _knorm*_airy_info->kValue(kpkp*_4_over_diamsq);
    }

    class SKIXIntegrand : public std::unary_function<double,double>
    {
    public:
        SKIXIntegrand(double r, const SKInfo& ski) : _r(r), _ski(ski) {}
        double operator()(double k) const { return _ski.kValue(k)*j0(k*_r)*k; }
    private:
        const double _r;  //arcsec
        const SKInfo& _ski;
    };

    double SKInfo::xValueRaw(double r) const {
        // r in arcsec
        SKIXIntegrand I(r, *this);
        integ::IntRegion<double> reg(0, _maxk);
        if (r > 0) {
            // Add BesselJ0 zeros up to _maxk
            int s=1;
            double zero=bessel::getBesselRoot0(s)/r;
            while(zero < _maxk) {
                reg.addSplit(zero);
                s++;
                zero = bessel::getBesselRoot0(s)/r;
            }
            xdbg<<s<<" zeros found for r = "<<r<<'\n';
        }
        double result = integ::int1d(I, reg,
                            _gsparams->integration_relerr,
                            _gsparams->integration_abserr)/(2.*M_PI);
        return result;
    }

    double SKInfo::xValue(double r) const {
        return r < _radial.argMax() ? _radial(r) : 0.;
    }

    class SKIExactXIntegrand : public std::unary_function<double,double>
    {
    public:
        SKIExactXIntegrand(double r, const SKInfo& ski) : _r(r), _ski(ski) {}
        double operator()(double k) const { return _ski.kValueRaw(k)*j0(k*_r)*k; }
    private:
        const double _r;  //arcsec
        const SKInfo& _ski;
    };

    double SKInfo::xValueExact(double r) const {
        // r in arcsec
        SKIExactXIntegrand I(r, *this);
        integ::IntRegion<double> reg(0, _diam/_lam_factor);
        if (r > 0) {
            // Add BesselJ0 zeros up to _diam/_lam_factor
            int s=1;
            double zero=bessel::getBesselRoot0(s)/r;
            while(zero < _diam/_lam_factor) {
                reg.addSplit(zero);
                s++;
                zero = bessel::getBesselRoot0(s)/r;
            }
            xdbg<<s<<" zeros found for r = "<<r<<'\n';
        }
        double result = integ::int1d(I, reg,
                            _gsparams->integration_relerr,
                            _gsparams->integration_abserr)/(2.*M_PI);
        return result;
    }

    // \int 2 pi r dr f(r) from a to b, where f(r) = f(a) + (f(b) - f(a))/(b-a) * (r-a)
    double volume(double a, double b, double fa, double fb) {
        return M_PI*(b-a)/3.0*(a*(2*fa+fb)+b*(fa+2*fb));
    }

    class SKIxValueResid {
    public:
        SKIxValueResid(const SKInfo& ski, double thresh) : _ski(ski), _thresh(thresh) {}
        double operator()(double x) const {
            double val = _ski.xValueRaw(x)-_thresh;
            xdbg<<"resid(x="<<x<<")="<<val<<'\n';
            return val;
        }
    private:
        const SKInfo& _ski;
        const double _thresh;
    };

    class SKIxValueVolumeResid {
    public:
        SKIxValueVolumeResid(const SKInfo & ski, double f0, double thresh) :
            _ski(ski), _f0(f0), _thresh(thresh) {}
        double operator()(double x) const {
            return volume(0, x, _f0, _ski.xValueRaw(x)) - _thresh;
        }
    private:
        const SKInfo& _ski;
        const double _f0;
        const double _thresh;
    };

    void SKInfo::_buildRadial() {
        // set_verbose(2);
        double r = 0.0;
        double val = xValueRaw(0.0);
        _radial.addEntry(r, val);
        dbg<<"f(0) = "<<val<<" arcsec^-2\n";

        double r0 = 0.0;

        // // Figure out where to start.  A good guess is where
        // // xValueRaw(0) - xValueRaw(r0) = xvalue_accuracy
        // SKIxValueResid skixvr(*this, val-_gsparams->xvalue_accuracy);
        // Solve<SKIxValueResid> solver(skixvr, 0.0, 1e-3);
        // solver.bracketUpper();
        // solver.setMethod(Brent);
        // r0 = solver.root();
        // dbg<<'\n';
        // dbg<<"r0 method(1) = " << r0 << '\n';
        // dbg<<"xValue(r0) = " << xValueRaw(r0) << '\n';

        // Another guess is where the volume between r=0 and r=r0 is xvalue_accuracy
        SKIxValueVolumeResid skixvvr(*this, val, _gsparams->xvalue_accuracy);
        Solve<SKIxValueVolumeResid> solver2(skixvvr, 0.0, 1e-3);
        solver2.bracketUpper();
        solver2.setMethod(Brent);
        r0 = solver2.root();
        dbg<<"r0 method(2) = " << r0 << '\n';
        dbg<<"xValue(r0) = " << xValueRaw(r0) << '\n';

        double logr = log(r0);
        double dr = 0;
        double dlogr = _gsparams->table_spacing * sqrt(sqrt(_gsparams->xvalue_accuracy / 20.));

        dbg<<"r0 = "<<r0<<" arcsec\n";
        dbg<<"dlogr = "<<dlogr<<"\n";

        double sum = 0.0;
        double thresh1 = (1.-_gsparams->folding_threshold);
        double thresh2 = (1.-_gsparams->folding_threshold/5);
        double R = 1e10;
        _hlr = 1e10;
        double maxR = 600.0; // hard cut at 10 arcminutes.
        double nextr;
        double nextval;
        for(nextr = exp(logr);
            (nextr < _gsparams->stepk_minimum_hlr*_hlr) || (nextr < R) || (sum < thresh2);
            logr += dlogr, nextr=exp(logr))
        {
            nextval = xValueRaw(nextr);
            xdbg<<"f("<<nextr<<") = "<<nextval<<'\n';
            _radial.addEntry(nextr, nextval);

            sum += volume(r, nextr, val, nextval);
            xdbg<<"sum = "<<sum<<'\n';

            if (_hlr == 1e10 && sum > 0.5) {
                _hlr = r;
                dbg<<"hlr = "<<_hlr<<" arcsec\n";
            }
            if (R == 1e10 && sum > thresh1) R=r;
            if (r >= maxR) {
                if (_hlr == 1e10) {
                    dbg << "sum = " << sum << '\n';
                    throw SBError("Cannot find SecondKick half-light-radius.");
                }
                R = maxR;
                break;
            }
            val = nextval;
            r = nextr;
        }
        dbg<<"Finished building radial function.\n";
        dbg<<"R = "<<R<<" arcsec\n";
        dbg<<"HLR = "<<_hlr<<" arcsec\n";
        R = std::max(R, _gsparams->stepk_minimum_hlr*_hlr);
        _stepk = M_PI / R;
        dbg<<"stepk = "<<_stepk<<" arcsec^-1\n";
        dbg<<"sum = "<<sum<<"   (should be ~= 0.997)\n";
        if (sum < 1-_gsparams->folding_threshold)
            throw SBError("Could not find folding_threshold");
        dbg<<"_radial.size() = "<<_radial.size()<<'\n';
        std::vector<double> ranges(1, 0.);
        // Copy Airy algorithm for assigning ranges that will not have >1 extremum.
        double rmin = (1.1 - 0.5*_obscuration)*_lam/_diam*ARCSEC2RAD;
        for(r=rmin; r<=_radial.argMax(); r+=0.5*_lam/_diam*ARCSEC2RAD) ranges.push_back(r);
        dbg<<"ranges.size() = "<<ranges.size()<<'\n';
        _sampler.reset(new OneDimensionalDeviate(_radial, ranges, true, _gsparams));
    }

    boost::shared_ptr<PhotonArray> SKInfo::shoot(int N, UniformDeviate ud) const
    {
        assert(_sampler.get());
        return _sampler->shoot(N,ud);
    }

    LRUCache<boost::tuple<double,double,double,double,double,double,GSParamsPtr>,SKInfo>
        SBSecondKick::SBSecondKickImpl::cache(sbp::max_SK_cache);

    //
    //
    //
    //SBSecondKickImpl
    //
    //
    //

    SBSecondKick::SBSecondKickImpl::SBSecondKickImpl(double lam, double r0, double diam,
                                                     double obscuration, double L0, double kcrit,
                                                     double flux, double scale,
                                                     const GSParamsPtr& gsparams) :
        SBProfileImpl(gsparams),
        _lam(lam),
        _r0(r0),
        _diam(diam),
        _obscuration(obscuration),
        _L0(L0),
        _kcrit(kcrit),
        _flux(flux),
        _scale(scale),
        _info(cache.get(boost::make_tuple(1e-9*lam, r0, diam, obscuration, L0, kcrit, this->gsparams.duplicate())))
    { }

    double SBSecondKick::SBSecondKickImpl::maxK() const
    { return _info->maxK()*_scale; }

    double SBSecondKick::SBSecondKickImpl::stepK() const
    { return _info->stepK()*_scale; }

    double SBSecondKick::SBSecondKickImpl::getHalfLightRadius() const
    { return _info->getHalfLightRadius()/_scale; }

    std::string SBSecondKick::SBSecondKickImpl::serialize() const
    {
        std::ostringstream oss(" ");
        oss.precision(std::numeric_limits<double>::digits10 + 4);
        oss << "galsim._galsim.SBSecondKick("
            <<getLam()<<", "
            <<getR0()<<", "
            <<getDiam()<<", "
            <<getObscuration()<<", "
            <<getL0()<<", "
            <<getKCrit()<<", "
            <<getFlux()<<", "
            <<getScale()<<", "
            <<"galsim.GSParams("<<*gsparams<<"))";
        return oss.str();
    }

    double SBSecondKick::SBSecondKickImpl::structureFunction(double rho) const
    {
        xdbg<<"rho = "<<rho<<'\n';
        return _info->structureFunction(rho);
    }

    std::complex<double> SBSecondKick::SBSecondKickImpl::kValue(const Position<double>& p) const
    {
        // k in units of 1/_scale
        return kValue(sqrt(p.x*p.x+p.y*p.y)/_scale);
    }

    double SBSecondKick::SBSecondKickImpl::kValue(double k) const
    {
        // k in inverse arcsec
        return _info->kValue(k)*_flux;
    }

    double SBSecondKick::SBSecondKickImpl::kValueRaw(double k) const
    {
        // k in inverse arcsec
        return _info->kValueRaw(k)*_flux;
    }

    double SBSecondKick::SBSecondKickImpl::xValue(const Position<double>& p) const
    {
        // r in units of _scale
        return xValue(sqrt(p.x*p.x+p.y*p.y)*_scale);
    }

    double SBSecondKick::SBSecondKickImpl::xValue(double r) const
    {
        // r in arcsec
        return _info->xValue(r)*_flux;
    }

    double SBSecondKick::SBSecondKickImpl::xValueRaw(double r) const
    {
        //r in arcsec
        return _info->xValueRaw(r);
    }

    double SBSecondKick::SBSecondKickImpl::xValueExact(double r) const
    {
        //r in arcsec
        return _info->xValueExact(r);
    }

    boost::shared_ptr<PhotonArray> SBSecondKick::SBSecondKickImpl::shoot(
        int N, UniformDeviate ud) const
    {
        dbg<<"SK shoot: N = "<<N<<std::endl;
        dbg<<"Target flux = "<<getFlux()<<std::endl;
        // Get photons from the SKInfo structure, rescale flux and size for this instance
        boost::shared_ptr<PhotonArray> result = _info->shoot(N,ud);
        result->scaleFlux(_flux);
        result->scaleXY(_scale);
        dbg<<"SK Realized flux = "<<result->getTotalFlux()<<std::endl;
        return result;
    }

}
