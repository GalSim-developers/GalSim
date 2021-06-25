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

#include "SBSecondKick.h"
#include "SBSecondKickImpl.h"
#include "SBVonKarmanImpl.h"
#include "fmath/fmath.hpp"
#include "math/Bessel.h"
#include "math/Gamma.h"
#include "math/Hankel.h"

#ifdef DEBUGLOGGING
#include <ctime>
#endif

namespace galsim {

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

    SKInfo::SKInfo(double kcrit, const GSParamsPtr& gsparams) :
        _kcrit(kcrit), _gsparams(gsparams),
        _radial(Table::spline),
        _kvLUT(Table::spline)
    {
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

    inline double pow4(double x) { double x2 = x*x; return x2*x2; }

    class SKISFIntegrand : public std::unary_function<double,double>
    {
    public:
        SKISFIntegrand(double rho, double kcrit) :
            _2pirho(2*M_PI*rho), _kc4(pow4(kcrit)) {}
        double operator()(double k) const {
            double ret = fast_pow(k, -8./3)*(1-j0(_2pirho*k));
            if (_kc4 > 0.) {
                //ret *= (0.5*tanh(2*log(k/_kcrit))+0.5);
                double k4 = pow4(k);
                ret *= k4 / (k4 + _kc4);
            }
            return ret;
        }
    private:
        const double _2pirho;   // 2*pi*rho
        const double _kc4;      // kcrit^4
    };

    double SKInfo::structureFunction(double rho) const {
        const static double magic6 = 0.2877144330394485472;
        SKISFIntegrand I(rho, _kcrit);
        integ::IntRegion<double> reg(0., integ::MOCK_INF);
        for (int s=1; s<10; s++) {
            double zero = math::getBesselRoot0(s)/(2*M_PI*rho);
            reg.addSplit(zero);
        }
        double result = integ::int1d(I, reg,
                                     _gsparams->integration_relerr,
                                     _gsparams->integration_abserr);
        result *= magic6;
        return result;
    }

    void SKInfo::_buildKVLUT() {
        // Start with 10x the regular Kolmogorov maxk (fairly arbitrarily)
        _maxk = 10*std::pow(-std::log(_gsparams->kvalue_accuracy),3./5.);

        if (_kcrit > 1.e10) {
            dbg<<"large kcrit = "<<_kcrit<<std::endl;
            _delta = 1.;
            _maxk = 1.;
            _kvLUT.addEntry(0, 0.);
            _kvLUT.addEntry(0.5, 0.);
            _kvLUT.addEntry(1., 0.);
            _kvLUT.finalize();
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
            double kv = fmath::expd(-0.5*val)-_delta;
            dbg<<"kv("<<k<<") "<<kv<<std::endl;
            _kvLUT.addEntry(k, kv);
            if (val > limit) { k += dk; break; }
        }
        // Switch to logarithmic spacing.  dk -> dlogk
        double expdlogk = exp(dk);
        int nsmall=0;
        for (; k<_maxk; k*=expdlogk) {
            double val = structureFunction(k);
            xdbg<<"sf("<<k<<") "<<val<<std::endl;
            double kv = fmath::expd(-0.5*val)-_delta;
            dbg<<"kv("<<k<<") "<<kv<<std::endl;
            _kvLUT.addEntry(k, kv);
            if (std::abs(kv) < _gsparams->kvalue_accuracy) {
                nsmall++;
            } else {
                nsmall=0;
            }
            if (nsmall == 5) {
                _maxk = k;
                break;
            }
        }
        _kvLUT.finalize();
        xdbg<<"kvLUT.size() = "<<_kvLUT.size()<<'\n';
        //set_verbose(1);
    }

    double SKInfo::kValue(double k) const {
        // k in units of k0 = 2pi r0/lambda
        return k < _kvLUT.argMax() ? _kvLUT(k) : 0.;
    }

    double SKInfo::kValueRaw(double k) const {
        // k in units of k0 = 2pi r0/lambda
        if (k == 0)
            return 1-_delta;
        return fmath::expd(-0.5*structureFunction(k))-_delta;
    }

    class SKIXIntegrand : public std::function<double(double)>
    {
    public:
        SKIXIntegrand(const SKInfo& ski) : _ski(ski) {}
        double operator()(double k) const { return _ski.kValue(k); }
    private:
        const SKInfo& _ski;
    };

    double SKInfo::xValueRaw(double r) const {
        // r in units of 1/k0 = lambda/(2pi r0)
        SKIXIntegrand I(*this);
        double result = math::hankel_inf(I, r, 0.,
                                         _gsparams->integration_relerr,
                                         _gsparams->integration_abserr)/(2.*M_PI);
        dbg<<"xValueRaw("<<r<<") = "<<result<<"\n";
        return result;
    }

    double SKInfo::xValue(double r) const {
        // r in units of 1/k0 = lambda/(2pi r0)
        return r < _radial.argMax() ? _radial(r) : 0.;
    }

    class SKIExactXIntegrand : public std::function<double(double)>
    {
    public:
        SKIExactXIntegrand(const SKInfo& ski) : _ski(ski) {}
        double operator()(double k) const { return _ski.kValueRaw(k); }
    private:
        const SKInfo& _ski;
    };

    double SKInfo::xValueExact(double r) const {
        // r in units of 1/k0 = lambda/(2pi r0)
        SKIExactXIntegrand I(*this);
        double result = math::hankel_inf(I, r, 0.,
                                         _gsparams->integration_relerr,
                                         _gsparams->integration_abserr)/(2.*M_PI);
        xdbg<<"xValueExact("<<r<<") = "<<result<<"\n";
        return result;
    }

    void SKInfo::_buildRadial() {
        //set_verbose(2);
        if (_delta > 1.-_gsparams->folding_threshold) {
            dbg<<"large delta = "<<_delta<<std::endl;
            _radial.addEntry(0, 0.);
            _radial.addEntry(1., 0.);
            _radial.addEntry(2., 0.);
            _radial.finalize();
            _stepk = 1.e10;
            std::vector<double> range(2,0.);
            range[1] = _radial.argMax();
            dbg<<"range = "<<range[0]<<"  "<<range[1]<<std::endl;
            dbg<<"Make ODD\n";
            _sampler.reset(new OneDimensionalDeviate(_radial, range, true, 1.0, *_gsparams));
            dbg<<"Made ODD\n";
            return;
        }

        double val = xValueRaw(0.0);
        xdbg<<"f(0) = "<<val<<std::endl;

        double dr = _gsparams->table_spacing * sqrt(sqrt(_gsparams->xvalue_accuracy / 10.));

        // Along the way accumulate the flux integral to determine the radius
        // that encloses (1-folding_threshold) of the flux.
        double thresh0 = (0.5 - _delta) / (2.*M_PI*dr);
        double thresh1 = (1.-_delta-_gsparams->folding_threshold) / (2.*M_PI*dr);
        double thresh2 = (1.-_delta-_gsparams->shoot_accuracy) / (2.*M_PI*dr);
        double R = 0., hlr = 0.;

        _radial.addEntry(0., val);
        // Smallest reasonable 1/k0 is about 0.06 arcsec, so this maxR corresponds to about
        // 60 arcsec in that case.
        double maxR = 1000.;
        double r = dr;
        double sum = 0.5*r*val;

        // Continue until accumulate 0.999 of the flux
        int nsmall=0;
        for (; r<1.; r+=dr) {
            val = xValueRaw(r);
            xdbg<<"f("<<r<<") = "<<val<<std::endl;

            // The result should be positive, but numerical inaccuracies can mean that some
            // values go negative.  It seems that this happens once all further values are
            // basically zero, so just stop here if/when this happens.
            if (val < _gsparams->xvalue_accuracy)
                nsmall++;
            else
                nsmall=0;
            if (nsmall==5) break;
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
        nsmall=0;
        for (; r<maxR; r *= expdlogr) {
            val = xValueRaw(r);
            xdbg<<"f("<<r<<") = "<<val<<std::endl;

            // The result should be positive, but numerical inaccuracies can mean that some
            // values go negative.  It seems that this happens once all further values are
            // basically zero, so just stop here if/when this happens.
            if (val < _gsparams->xvalue_accuracy)
                nsmall++;
            else
                nsmall=0;
            if (nsmall==5) break;
            _radial.addEntry(r,val);

            // Accumulate int(r*f(r)) / dr  (i.e. don't include 2*pi*dr factor as part of sum)
            sum += r * r * val;
            dbg<<"sum = "<<sum<<"  thresh1 = "<<thresh1<<"  thesh2 = "<<thresh2<<std::endl;
            xdbg<<"sum*2*pi*dr "<<sum*2.*M_PI*dr<<std::endl;
            if (hlr == 0. && sum > thresh0) hlr = r;
            if (R == 0. && sum > thresh1) R = r;
            if (sum > thresh2) break;
        }
        _radial.finalize();
        dbg<<"Finished building radial function.\n";
        dbg<<"_radial.size() = "<<_radial.size()<<'\n';
        dbg<<"sum*2*pi*dr + delta = "<<sum*2.*M_PI*dr+_delta<<"   (should >= 0.999)\n";

        dbg<<"R = "<<R<<std::endl;
        dbg<<"hlr = "<<hlr<<std::endl;
        // Make sure it is at least 5 hlr
        if (R == 0) R = _radial.argMax();
        R = std::max(R,_gsparams->stepk_minimum_hlr*hlr);
        dbg<<"final R = "<<R<<std::endl;
        _stepk = M_PI / R;
        dbg<<"stepk = "<<_stepk<<std::endl;

        std::vector<double> range(2,0.);
        range[1] = _radial.argMax();
        _sampler.reset(new OneDimensionalDeviate(_radial, range, true, 1.0, *_gsparams));
        dbg<<"made sampler\n";
        //set_verbose(1);
    }

    void SKInfo::shoot(PhotonArray& photons, UniformDeviate ud) const
    {
        assert(_sampler.get());
        _sampler->shoot(photons,ud);
    }

    LRUCache<Tuple<double,GSParamsPtr>,SKInfo>
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
        SBProfileImpl(*gsparams),
        _lam_over_r0(lam_over_r0), _k0(2.*M_PI/lam_over_r0), _inv_k0(1./_k0),
        _kcrit(kcrit), _flux(flux), _xnorm(flux * _k0*_k0),
        _info(cache.get(MakeTuple(kcrit, GSParamsPtr(gsparams))))
    {}

    double SBSecondKick::SBSecondKickImpl::maxK() const
    { return _info->maxK()*_k0; }

    double SBSecondKick::SBSecondKickImpl::stepK() const
    { return _info->stepK()*_k0; }

    double SBSecondKick::SBSecondKickImpl::getDelta() const
    { return _info->getDelta() * _flux; }

    double SBSecondKick::SBSecondKickImpl::structureFunction(double rho) const
    {
        return _info->structureFunction(rho);
    }

    std::complex<double> SBSecondKick::SBSecondKickImpl::kValue(const Position<double>& p) const
    {
        return kValue(sqrt(p.x*p.x+p.y*p.y));
    }

    double SBSecondKick::SBSecondKickImpl::kValue(double k) const
    {
        // k in inverse arcsec
        return _info->kValue(k*_inv_k0)*_flux;
    }

    double SBSecondKick::SBSecondKickImpl::kValueRaw(double k) const
    {
        // k in inverse arcsec
        return _info->kValueRaw(k*_inv_k0)*_flux;
    }

    double SBSecondKick::SBSecondKickImpl::xValue(const Position<double>& p) const
    {
        return xValue(sqrt(p.x*p.x+p.y*p.y));
    }

    double SBSecondKick::SBSecondKickImpl::xValue(double r) const
    {
        // r in arcsec
        return _info->xValue(r*_k0)*_xnorm;
    }

    double SBSecondKick::SBSecondKickImpl::xValueRaw(double r) const
    {
        // r in arcsec
        return _info->xValueRaw(r*_k0)*_xnorm;
    }

    double SBSecondKick::SBSecondKickImpl::xValueExact(double r) const
    {
        // r in arcsec
        return _info->xValueExact(r*_k0)*_xnorm;
    }

    void SBSecondKick::SBSecondKickImpl::shoot(PhotonArray& photons, UniformDeviate ud) const
    {
        dbg<<"SK shoot: N = "<<photons.size()<<std::endl;
        dbg<<"Target flux = "<<getFlux()<<std::endl;
        // Get photons from the SKInfo structure, rescale flux and size for this instance
        _info->shoot(photons,ud);
        photons.setTotalFlux(getFlux());
        photons.scaleXY(_inv_k0);
        dbg<<"SK Realized flux = "<<photons.getTotalFlux()<<std::endl;
    }

}
