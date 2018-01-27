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
// #define COUNTFEVAL

#include "galsim/IgnoreWarnings.h"

#define BOOST_NO_CXX11_SMART_PTR
#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/special_functions/bessel.hpp>

#include "SBSK.h"
#include "SBSKImpl.h"
#include "fmath/fmath.hpp"
#include "Solve.h"
#include "bessel/Roots.h"

namespace galsim {

    const double ARCSEC2RAD = 180.*60*60/M_PI;  // ~206265
    const double MOCK_INF = 1.e300;

    inline double fast_pow(double x, double y)
    { return fmath::expd(y * std::log(x)); }

    //
    //
    //
    //SBSK
    //
    //
    //

    SBSK::SBSK(double lam, double r0, double diam, double obscuration, double L0, double kcrit,
               double flux, double scale, const GSParamsPtr& gsparams) :
        SBProfile(new SBSKImpl(lam, r0, diam, obscuration, L0, kcrit, flux, scale, gsparams)) {}

    SBSK::SBSK(const SBSK &rhs) : SBProfile(rhs) {}

    SBSK::~SBSK() {}

    double SBSK::getLam() const
    {
        assert(dynamic_cast<const SBSKImpl*>(_pimpl.get()));
        return static_cast<const SBSKImpl&>(*_pimpl).getLam();
    }

    double SBSK::getR0() const
    {
        assert(dynamic_cast<const SBSKImpl*>(_pimpl.get()));
        return static_cast<const SBSKImpl&>(*_pimpl).getR0();
    }

    double SBSK::getDiam() const
    {
        assert(dynamic_cast<const SBSKImpl*>(_pimpl.get()));
        return static_cast<const SBSKImpl&>(*_pimpl).getDiam();
    }

    double SBSK::getObscuration() const
    {
        assert(dynamic_cast<const SBSKImpl*>(_pimpl.get()));
        return static_cast<const SBSKImpl&>(*_pimpl).getObscuration();
    }

    double SBSK::getL0() const
    {
        assert(dynamic_cast<const SBSKImpl*>(_pimpl.get()));
        return static_cast<const SBSKImpl&>(*_pimpl).getL0();
    }

    double SBSK::getKCrit() const
    {
        assert(dynamic_cast<const SBSKImpl*>(_pimpl.get()));
        return static_cast<const SBSKImpl&>(*_pimpl).getKCrit();
    }

    double SBSK::getScale() const
    {
        assert(dynamic_cast<const SBSKImpl*>(_pimpl.get()));
        return static_cast<const SBSKImpl&>(*_pimpl).getScale();
    }

    double SBSK::getHalfLightRadius() const
    {
        assert(dynamic_cast<const SBSKImpl*>(_pimpl.get()));
        return static_cast<const SBSKImpl&>(*_pimpl).getHalfLightRadius();
    }

    double SBSK::structureFunction(double rho) const
    {
        assert(dynamic_cast<const SBSKImpl*>(_pimpl.get()));
        return static_cast<const SBSKImpl&>(*_pimpl).structureFunction(rho);
    }

    double SBSK::kValueSlow(double k) const
    {
        assert(dynamic_cast<const SBSKImpl*>(_pimpl.get()));
        return static_cast<const SBSKImpl&>(*_pimpl).kValueSlow(k);
    }

    double SBSK::xValueSlow(double k) const
    {
        assert(dynamic_cast<const SBSKImpl*>(_pimpl.get()));
        return static_cast<const SBSKImpl&>(*_pimpl).xValueSlow(k);
    }

    //
    //
    //
    //SKInfo
    //
    //
    //

    const double SKInfo::magic1 = 2*boost::math::tgamma(11./6)/(pow(2, 5./6)*pow(M_PI, 8./3))
                                    * pow(24/5.*boost::math::tgamma(6./5), 5./6);
    const double SKInfo::magic2 = boost::math::tgamma(5./6)/pow(2., 1./6);
    const double SKInfo::magic3 = SKInfo::magic1*boost::math::tgamma(-5./6)/pow(2., 11./6);
    const double SKInfo::magic5 = 2*boost::math::tgamma(11./6)*boost::math::tgamma(11./6)
                                / pow(M_PI, 8./3)
                                * pow(24/5.*boost::math::tgamma(6./5), 5./6);

    SKInfo::SKInfo(double lam, double r0, double diam, double obscuration, double L0,
                   double kcrit, const GSParamsPtr& gsparams) :
        _lam(lam), _r0(r0), _r0m53(pow(r0, -5./3)), _diam(diam), _obscuration(obscuration), _L0(L0),
        _L0invsq(1/L0/L0), _r0L0m53(pow(r0/L0, -5./3)), _kcrit(kcrit), _gsparams(gsparams),
        _airy(new SBAiry(lam/diam*ARCSEC2RAD, obscuration, 1, gsparams)),
        _sfLUT(TableDD::spline),
        _radial(TableDD::spline)
    {
        // a conservative maxK is that of the embedded Airy profile.
        _maxk = _airy->maxK();
        _stepk = 0;

        // build the radial function, and along the way, set _stepk, _hlr.
        _buildSFLUT();
        _buildRadial();
    }

    class SKISFIntegrand : public std::unary_function<double,double>
    {
    public:
        SKISFIntegrand(double rho, double L0invsq) : _rho(rho), _L0invsq(L0invsq) {}
        double operator()(double kappa) const {
            return fast_pow(kappa*kappa+_L0invsq, -11./6)*kappa*(1-j0(2*M_PI*kappa*_rho));
        }
    private:
        const double _rho;  // meters
        const double _L0invsq;  // inverse meters squared
    };

    // This version of structureFunction explicitly integrates from kcrit to infinity, which is how
    // the second kick is defined. However, I've found it's more stable to integrate from zero to
    // kcrit, and then subtract this from the analytic integral from 0 to infinity.  So that's
    // what's implemented further down.
    // double SKInfo::structureFunction(double rho) const {
    //     SKISFIntegrand I(rho, _L0invsq);
    //     double result = integ::int1d(I, _kcrit, integ::MOCK_INF,
    //                                  _gsparams->integration_relerr,
    //                                  _gsparams->integration_abserr);
    //     result *= magic5*_r0m53;
    //     return result;
    // }

    double SKInfo::vkStructureFunction(double rho) const {
    // rho in meters
        double rhoL0 = rho/_L0;
        if (rhoL0 < 1e-6) {
            return -magic3*fast_pow(2*M_PI*rho/_r0, 5./3);
        } else {
            double x = 2*M_PI*rhoL0;
            return magic1*_r0L0m53*(magic2-fast_pow(x, 5./6)*boost::math::cyl_bessel_k(5./6, x));
        }
    }

    double SKInfo::structureFunction(double rho) const {
        SKISFIntegrand I(rho, _L0invsq);
        double complement = integ::int1d(I, 0, _kcrit,
                                         _gsparams->integration_relerr,
                                         _gsparams->integration_abserr);
        return vkStructureFunction(rho) - magic5*complement*_r0m53;
    }

    void SKInfo::_buildSFLUT() {
        double dlogrho = _gsparams->table_spacing * sqrt(sqrt(_gsparams->kvalue_accuracy / 10.));
        dbg<<"Using dlogrho = "<<dlogrho<<std::endl;
        double rhomin = 1e-6; // micron, we can't possibly care about things this small...
        // maximum rho is the telescope diameter.
        for (double logrho = std::log(rhomin)-0.001; logrho < std::log(_diam); logrho += dlogrho){
            double rho = std::exp(logrho);
            double val = structureFunction(rho);
            xdbg<<"rho = "<<rho<<", I("<<rho<<") = "<<val<<std::endl;
            _sfLUT.addEntry(rho,val);
        }
    }

    double SKInfo::kValue(double k) const {
        // k in inverse arcsec
        return fmath::expd(-0.5*_sfLUT(_lam*k*ARCSEC2RAD/(2*M_PI)))
            * _airy->kValue(Position<double>(0, k)).real();
    }

    //  This version doesn't use the lookup table.  Used for testing.
    double SKInfo::kValueSlow(double k) const {
        // k in inverse arcsec
        return fmath::expd(-0.5*structureFunction(_lam*k*ARCSEC2RAD/(2*M_PI)))
            * _airy->kValue(Position<double>(0, k)).real();
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

    double SKInfo::xValue(double r) const {
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
#ifdef COUNTFEVAL
        int nfevinit = integ::nfeval;
#endif
        double result = integ::int1d(I, reg,
                            _gsparams->integration_relerr,
                            _gsparams->integration_abserr)/(2.*M_PI);
#ifdef COUNTFEVAL
        xdbg<<"NFEVAL = "<<integ::nfeval-nfevinit<<'\n';
#endif
        return result;
    }

    class SKIXSlowIntegrand : public std::unary_function<double,double>
    {
    public:
        SKIXSlowIntegrand(double r, const SKInfo& ski) : _r(r), _ski(ski) {}
        double operator()(double k) const { return _ski.kValueSlow(k)*j0(k*_r)*k; }
    private:
        const double _r;  //arcsec
        const SKInfo& _ski;
    };

    double SKInfo::xValueSlow(double r) const {
    // r in arcsec
        SKIXSlowIntegrand I(r, *this);
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
        }
        return integ::int1d(I, reg,
                            _gsparams->integration_relerr,
                            _gsparams->integration_abserr)/(2.*M_PI);
    }

    // \int 2 pi r dr f(r) from a to b, where f(r) = f(a) + (f(b) - f(a))/(b-a) * (r-a)
    double volume(double a, double b, double fa, double fb) {
        return M_PI*(b-a)/3.0*(a*(2*fa+fb)+b*(fa+2*fb));
    }

    void SKInfo::_buildRadial() {
        // set_verbose(2);
        double r = 0.0;
        double val = xValue(0.0);
        _radial.addEntry(r, val);
        dbg<<"f(0) = "<<val<<" arcsec^-2\n";

        // double r0 = 0.05*_gsparams->table_spacing * sqrt(sqrt(_gsparams->xvalue_accuracy / 10.));
        double r0 = 0.05*_gsparams->table_spacing * sqrt(sqrt(_gsparams->xvalue_accuracy / 10.));
        double logr = log(r0);
        double dr = 0;
        // double dlogr = 0.1*_gsparams->table_spacing * sqrt(sqrt(_gsparams->xvalue_accuracy / 10.));
        double dlogr = _gsparams->table_spacing * sqrt(sqrt(_gsparams->xvalue_accuracy / 10.));
        dbg<<"r0 = "<<r0<<" arcsec\n";
        dbg<<"dlogr = "<<dlogr<<"\n";

        double sum = 0.0;
        xdbg<<"sum = "<<sum<<'\n';

        double thresh1 = (1.-_gsparams->folding_threshold);
        double thresh2 = (1.-_gsparams->folding_threshold/2);
        double R = 1e10;
        _hlr = 1e10;
        double maxR = 600.0; // hard cut at 10 arcminutes.
        double nextr;
        double nextval;
        for(nextr = exp(logr);
            (nextr < _gsparams->stepk_minimum_hlr*_hlr) || (nextr < R) || (sum < thresh2);
            // logr += dlogr, nextr=exp(logr), dr=nextr*(1-exp(-dlogr)))
            logr += dlogr, nextr=exp(logr))
        {
            nextval = xValue(nextr);
            xdbg<<"f("<<nextr<<") = "<<nextval<<'\n';
            _radial.addEntry(nextr, nextval);

            // sum += 2*M_PI*val*r*dr;
            sum += volume(r, nextr, val, nextval);
            // xdbg<<"dr = "<<dr<<'\n';
            xdbg<<"sum = "<<sum<<'\n';

            if (_hlr == 1e10 && sum > 0.5) {
                _hlr = r;
                dbg<<"hlr = "<<_hlr<<" arcsec\n";
            }
            if (R == 1e10 && sum > thresh1) R=r;
            if (r >= maxR) {
                if (_hlr == 1e10) {
                    dbg << "sum = " << sum << '\n';
                    throw SBError("Cannot find SK half-light-radius.");
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
        if (sum < 0.997)
            throw SBError("Could not find folding_threshold");

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
        SBSK::SBSKImpl::cache(sbp::max_SK_cache);

    //
    //
    //
    //SBSKImpl
    //
    //
    //

    SBSK::SBSKImpl::SBSKImpl(double lam, double r0, double diam, double obscuration, double L0,
                             double kcrit, double flux, double scale, const GSParamsPtr& gsparams) :
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

    double SBSK::SBSKImpl::maxK() const
    { return _info->maxK()*_scale; }

    double SBSK::SBSKImpl::stepK() const
    { return _info->stepK()*_scale; }

    double SBSK::SBSKImpl::getHalfLightRadius() const
    { return _info->getHalfLightRadius()/_scale; }

    std::string SBSK::SBSKImpl::serialize() const
    {
        std::ostringstream oss(" ");
        oss.precision(std::numeric_limits<double>::digits10 + 4);
        oss << "galsim._galsim.SBSK("
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

    double SBSK::SBSKImpl::structureFunction(double rho) const
    {
        xdbg<<"rho = "<<rho<<'\n';
        return _info->structureFunction(rho);
    }

    double SBSK::SBSKImpl::kValue(double k) const
    // this kValue assumes k is in inverse arcsec
    {
        return _info->kValue(k)*_flux;
    }

    std::complex<double> SBSK::SBSKImpl::kValue(const Position<double>& p) const
    // k in units of _scale.
    {
        return kValue(sqrt(p.x*p.x+p.y*p.y)/_scale);
    }

    double SBSK::SBSKImpl::kValueSlow(double k) const
    // k in units of _scale.
    {
        return _info->kValueSlow(k/_scale)*_flux;
    }

    class SKXIntegrand : public std::unary_function<double,double>
    {
    public:
        SKXIntegrand(double r, const SBSK::SBSKImpl& sbski) :
            _r(r), _sbski(sbski)
        {}

        double operator()(double k) const { return _sbski.kValue(k)*k*j0(k*_r); }
    private:
        double _r;
        const SBSK::SBSKImpl& _sbski;
    };

    double SBSK::SBSKImpl::xValue(double r) const {
    // r in arcsec.
        // SKXIntegrand I(r, *this);
        // return integ::int1d(I, 0.0, integ::MOCK_INF,
        //                     gsparams->integration_relerr, gsparams->integration_abserr)/(2*M_PI);
        return _info->xValue(r)*_flux;
    }

    double SBSK::SBSKImpl::xValue(const Position<double>& p) const
    // r in units of _scale
    {
        return xValue(sqrt(p.x*p.x+p.y*p.y)*_scale);
    }

    double SBSK::SBSKImpl::xValueSlow(double r) const
    {
    //r in units of _scale
        return _info->xValueSlow(r*_scale);
    }

    boost::shared_ptr<PhotonArray> SBSK::SBSKImpl::shoot(
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
