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

//#define DEBUGLOGGING

#include "galsim/IgnoreWarnings.h"

#define BOOST_NO_CXX11_SMART_PTR
#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/special_functions/bessel.hpp>

#include "SBSecondKick.h"
#include "SBSecondKickImpl.h"
#include "SBVonKarmanImpl.h"
#include "fmath/fmath.hpp"
#include "Solve.h"
#include "bessel/Roots.h"

#ifdef DEBUGLOGGING
#include <ctime>
#endif

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

    SBSecondKick::SBSecondKick(double lam_over_r0, double kcrit, double flux,
                               const GSParamsPtr& gsparams) :
        SBProfile(new SBSecondKickImpl(lam_over_r0, kcrit, flux, gsparams)) {}

    SBSecondKick::SBSecondKick(const SBSecondKick &rhs) : SBProfile(rhs) {}

    SBSecondKick::~SBSecondKick() {}

    double SBSecondKick::getLamOverR0() const
    {
        assert(dynamic_cast<const SBSecondKickImpl*>(_pimpl.get()));
        return static_cast<const SBSecondKickImpl&>(*_pimpl).getLamOverR0();
    }

    double SBSecondKick::getKCrit() const
    {
        assert(dynamic_cast<const SBSecondKickImpl*>(_pimpl.get()));
        return static_cast<const SBSecondKickImpl&>(*_pimpl).getKCrit();
    }

    double SBSecondKick::getDelta() const
    {
        assert(dynamic_cast<const SBSecondKickImpl*>(_pimpl.get()));
        return static_cast<const SBSecondKickImpl&>(*_pimpl).getDelta();
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

    SKInfo::SKInfo(double k0, double kcrit, const GSParamsPtr& gsparams) :
        _k0(k0), _kcrit(kcrit), _gsparams(gsparams),
        _radial(TableDD::spline),
        _kvLUT(TableDD::spline)
    {
        dbg<<"k0 = "<<_k0<<std::endl;

        // build the radial function
#ifdef DEBUGLOGGING
        std::clock_t t0 = std::clock();
        _buildKVLUT();
        std::clock_t t1 = std::clock();
        _buildRadial();
        std::clock_t t2 = std::clock();
        dbg << "buildKV time = " << (double)(t1-t0)/CLOCKS_PER_SEC << '\n';
        dbg << "buildRad time = " << (double)(t2-t1)/CLOCKS_PER_SEC << '\n';
#else
        _buildKVLUT();
        _buildRadial();
#endif
    }

    class SKISFIntegrand : public std::unary_function<double,double>
    {
    public:
        SKISFIntegrand(double rho, double kcrit) :
            _2pirho(2*M_PI*rho), _kcrit(kcrit) {}
        double operator()(double k) const {
            double ret = fast_pow(k, -8./3)*(1-j0(_2pirho*k));
            if (_kcrit > 0.) {
                //ret *= (0.5*tanh(2*log(k/_kcrit))+0.5);
                double k4 = k*k*k*k;
                double kc4 = _kcrit*_kcrit*_kcrit*_kcrit;
                ret *= k4 / (k4 + kc4);
            }
            return ret;
        }
    private:
        const double _2pirho;   // 2*pi*rho
        const double _kcrit;  // inverse meters squared
    };


    double SKInfo::structureFunction(double rho) const {
        const static double magic6 = 0.2877144330394485472;
        SKISFIntegrand I(rho, _kcrit);
        integ::IntRegion<double> reg(0., integ::MOCK_INF);
        for (int s=1; s<10; s++) {
            double zero = bessel::getBesselRoot0(s)/(2*M_PI*rho);
            reg.addSplit(zero);
        }
        double result = integ::int1d(I, reg,
                                     _gsparams->integration_relerr,
                                     _gsparams->integration_abserr);
        result *= magic6;
        return result;
    }

    void SKInfo::_buildKVLUT() {
        // Start with the regular Kolmogorov maxk
        _maxk = std::pow(-std::log(_gsparams->kvalue_accuracy),3./5.);

        if (_kcrit > 1.e10) {
            dbg<<"large kcrit = "<<_kcrit<<std::endl;
            _delta = 1.;
            _maxk = 1.;
            _kvLUT.addEntry(0, 0.);
            _kvLUT.addEntry(0.5, 0.);
            _kvLUT.addEntry(1., 0.);
            return;
        }

        const static double magic6 = 0.2877144330394485472;
        //double limit = magic6*M_PI*std::pow(_kcrit, -5./3.);
        double limit = magic6*M_PI*std::pow(_kcrit, -5./3.) / (4.*std::sin(5.*M_PI/12.));
        dbg<<"limit = "<<limit<<std::endl;
        _delta = fmath::expd(-0.5*limit);
        dbg<<"_delta = "<<_delta<<std::endl;
        double dk = _gsparams->table_spacing * sqrt(sqrt(_gsparams->kvalue_accuracy / 10.0));
        xdbg<<"Using dk = "<<dk<<'\n';

        double k=0.;
        _kvLUT.addEntry(0, 1.-_delta);
        for (k=dk; k<1.; k+=dk) {
            double val = structureFunction(k);
            xdbg<<"sf("<<k<<") "<<val<<std::endl;
            double kv = fmath::exp(-0.5*val)-_delta;
            dbg<<"kv("<<k<<") "<<kv<<std::endl;
            _kvLUT.addEntry(k, kv);
            if (val > limit) { k += dk; break; }
        }
        // Switch to logarithmic spacing.  dk -> dlogk
        double expdlogk = exp(dk);
        for (; k<_maxk; k*=expdlogk) {
            double val = structureFunction(k);
            xdbg<<"sf("<<k<<") "<<val<<std::endl;
            double kv = fmath::exp(-0.5*val)-_delta;
            dbg<<"kv("<<k<<") "<<kv<<std::endl;
            _kvLUT.addEntry(k, kv);
            if (std::abs(kv) < _gsparams->kvalue_accuracy) {
                _maxk = k;
                break;
            }
        }
        _maxk *= _k0;
        xdbg<<"kvLUT.size() = "<<_kvLUT.size()<<'\n';
        //set_verbose(1);
    }

    double SKInfo::kValue(double k) const {
        // k in inverse arcsec
        double kp = k/_k0;
        return kp < _kvLUT.argMax() ? _kvLUT(kp) : 0.;
    }

    double SKInfo::kValueRaw(double k) const {
        // k in inverse arcsec
        double kp = k/_k0;
        double f = fmath::expd(-0.5*structureFunction(kp));
        xdbg<<"kValueRaw("<<k<<") = "<<f<<std::endl;
        return f;
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
            for (int s=1; s<10; ++s) {
                double zero=bessel::getBesselRoot0(s)/r;
                if (zero >= _maxk) break;
                reg.addSplit(zero);
            }
        }
        double result = integ::int1d(I, reg,
                                     _gsparams->integration_relerr,
                                     _gsparams->integration_abserr)/(2.*M_PI);
        dbg<<"xValueRaw("<<r<<") = "<<result<<"\n";
        return result;
    }

    double SKInfo::xValue(double r) const {
        double rp = r*_k0;
        return rp < _radial.argMax() ? _radial(rp) : 0.;
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
        integ::IntRegion<double> reg(0., integ::MOCK_INF);
        if (r > 0) {
            // Add BesselJ0 zeros up to _diam/_lam_arcsec
            for (int s=1; s<10; ++s) {
                double zero=bessel::getBesselRoot0(s)/r;
                if (zero >= _maxk) break;
                reg.addSplit(zero);
            }
        }
        double result = integ::int1d(I, reg,
                            _gsparams->integration_relerr,
                            _gsparams->integration_abserr)/(2.*M_PI);
        xdbg<<"xValueExact("<<r<<") = "<<result<<"\n";
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
            xdbg<<volume(0, x, _f0, _ski.xValueRaw(x))<<"  "<< _thresh<<std::endl;
            return volume(0, x, _f0, _ski.xValueRaw(x)) - _thresh;
        }
    private:
        const SKInfo& _ski;
        const double _f0;
        const double _thresh;
    };

    void SKInfo::_buildRadial() {
        //set_verbose(2);
        if (_delta > 1.-_gsparams->folding_threshold) {
            dbg<<"large delta = "<<_delta<<std::endl;
            _radial.addEntry(0, 0.);
            _radial.addEntry(1., 0.);
            _radial.addEntry(2., 0.);
            _stepk = 1.e10;
            std::vector<double> range(2,0.);
            range[1] = _radial.argMax();
            dbg<<"range = "<<range[0]<<"  "<<range[1]<<std::endl;
            dbg<<"Make ODD\n";
            _sampler.reset(new OneDimensionalDeviate(_radial, range, true, _gsparams));
            dbg<<"Made ODD\n";
            return;
        }

        double val = xValueRaw(0.0);
        xdbg<<"f(0) = "<<val<<" arcsec^-2\n";

        double dr = _gsparams->table_spacing * sqrt(sqrt(_gsparams->xvalue_accuracy / 10.));

        // Along the way accumulate the flux integral to determine the radius
        // that encloses (1-folding_threshold) of the flux.
        double thresh0 = (0.5 - _delta) / (2.*M_PI*dr);
        double thresh1 = (1.-_delta-_gsparams->folding_threshold) / (2.*M_PI*dr);
        double thresh2 = (1.-_delta-_gsparams->folding_threshold/5.) / (2.*M_PI*dr);
        double R = 0., hlr = 0.;

        _radial.addEntry(0., val);
        double maxR = 60.;  // Fairly arbitrary, but usually irrelevant
        double r = dr;
        double sum = 0.5*r*val;

        // Continue until accumulate 0.999 of the flux
        for (; r<1.; r+=dr) {
            val = xValueRaw(r);
            xdbg<<"f("<<r<<") = "<<val<<std::endl;

            // The result should be positive, but numerical inaccuracies can mean that some
            // values go negative.  It seems that this happens once all further values are
            // basically zero, so just stop here if/when this happens.
            if (val < _gsparams->xvalue_accuracy) break;
            _radial.addEntry(r,val);

            // Accumulate int(r*f(r)) / dr  (i.e. don't include 2*pi*dr factor as part of sum)
            sum += r * val;
            dbg<<"sum = "<<sum<<"  thresh1 = "<<thresh1<<"  thesh2 = "<<thresh2<<std::endl;
            xdbg<<"sum*2*pi*dr "<<sum*2.*M_PI*dr<<std::endl;
            if (R == 0. && sum > thresh1) R = r;
            if (hlr == 0. && sum > thresh0) hlr = r;
        }
        // Switch to logarithmic binning
        double expdlogr = std::exp(dr);
        for (; r<maxR; r *= expdlogr) {
            val = xValueRaw(r);
            xdbg<<"f("<<r<<") = "<<val<<std::endl;

            // The result should be positive, but numerical inaccuracies can mean that some
            // values go negative.  It seems that this happens once all further values are
            // basically zero, so just stop here if/when this happens.
            if (val < _gsparams->xvalue_accuracy) break;
            _radial.addEntry(r,val);

            // Accumulate int(r*f(r)) / dr  (i.e. don't include 2*pi*dr factor as part of sum)
            sum += r * r * val;
            dbg<<"sum = "<<sum<<"  thresh1 = "<<thresh1<<"  thesh2 = "<<thresh2<<std::endl;
            xdbg<<"sum*2*pi*dr "<<sum*2.*M_PI*dr<<std::endl;
            if (hlr == 0. && sum > thresh0) hlr = r;
            if (R == 0. && sum > thresh1) R = r;
            if (sum > thresh2) break;
        }
        dbg<<"Finished building radial function.\n";
        dbg<<"_radial.size() = "<<_radial.size()<<'\n';
        dbg<<"sum*2*pi*dr + delta = "<<sum*2.*M_PI*dr+_delta<<"   (should >= 0.999)\n";

        dbg<<"R = "<<R<<std::endl;
        dbg<<"hlr = "<<hlr<<std::endl;
        // Make sure it is at least 5 hlr
        if (R == 0) R = _radial.argMax();
        R = std::max(R,_gsparams->stepk_minimum_hlr*hlr);
        _stepk = M_PI / R * _k0;
        dbg<<"stepk = "<<_stepk<<std::endl;

        std::vector<double> range(2,0.);
        range[1] = _radial.argMax();
        _sampler.reset(new OneDimensionalDeviate(_radial, range, true, _gsparams));
        dbg<<"made sampler\n";
        //set_verbose(1);
    }

    boost::shared_ptr<PhotonArray> SKInfo::shoot(int N, UniformDeviate ud) const
    {
        assert(_sampler.get());
        boost::shared_ptr<PhotonArray> result = _sampler->shoot(N,ud);
        //result->scaleXY(_lam_over_r0);
        return result;
    }

    LRUCache<boost::tuple<double,double,GSParamsPtr>,SKInfo>
        SBSecondKick::SBSecondKickImpl::cache(sbp::max_SK_cache);

    //
    //
    //
    //SBSecondKickImpl
    //
    //
    //

    SBSecondKick::SBSecondKickImpl::SBSecondKickImpl(double lam_over_r0, double kcrit, double flux,
                                                     const GSParamsPtr& gsparams) :
        SBProfileImpl(gsparams),
        _lam_over_r0(lam_over_r0), _k0(2.*M_PI/lam_over_r0),
        _kcrit(kcrit),
        _flux(flux),
        _info(cache.get(boost::make_tuple(_k0, kcrit, this->gsparams.duplicate())))
    { }

    double SBSecondKick::SBSecondKickImpl::maxK() const
    { return _info->maxK(); }

    double SBSecondKick::SBSecondKickImpl::stepK() const
    { return _info->stepK(); }

    double SBSecondKick::SBSecondKickImpl::getDelta() const
    { return _info->getDelta() * _flux; }

    std::string SBSecondKick::SBSecondKickImpl::serialize() const
    {
        std::ostringstream oss(" ");
        oss.precision(std::numeric_limits<double>::digits10 + 4);
        oss << "galsim._galsim.SBSecondKick("
            <<getLamOverR0()<<", "
            <<getKCrit()<<", "
            <<getFlux()<<", "
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
        xdbg<<"p = "<<p<<std::endl;
        xdbg<<"Call kValue with k = "<<sqrt(p.x*p.x+p.y*p.y)<<std::endl;
        return kValue(sqrt(p.x*p.x+p.y*p.y));
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
        xdbg<<"p = "<<p<<std::endl;
        xdbg<<"Call xValue with r = "<<sqrt(p.x*p.x+p.y*p.y)<<std::endl;
        return xValue(sqrt(p.x*p.x+p.y*p.y));
    }

    double SBSecondKick::SBSecondKickImpl::xValue(double r) const
    {
        // r in arcsec
        return _info->xValue(r)*_flux;
    }

    double SBSecondKick::SBSecondKickImpl::xValueRaw(double r) const
    {
        //r in arcsec
        return _info->xValueRaw(r)*_flux;
    }

    double SBSecondKick::SBSecondKickImpl::xValueExact(double r) const
    {
        //r in arcsec
        return _info->xValueExact(r)*_flux;
    }

    boost::shared_ptr<PhotonArray> SBSecondKick::SBSecondKickImpl::shoot(
        int N, UniformDeviate ud) const
    {
        dbg<<"SK shoot: N = "<<N<<std::endl;
        dbg<<"Target flux = "<<getFlux()<<std::endl;
        // Get photons from the SKInfo structure, rescale flux and size for this instance
        boost::shared_ptr<PhotonArray> result = _info->shoot(N,ud);
        dbg<<"SK shoot returned flux = "<<result->getTotalFlux()<<std::endl;
        result->setTotalFlux(getFlux());
        dbg<<"SK Realized flux = "<<result->getTotalFlux()<<std::endl;
        return result;
    }

}
