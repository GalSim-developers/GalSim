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

#include "SBVonKarman.h"
#include "SBVonKarmanImpl.h"
#include "Solve.h"
#include "math/Bessel.h"
#include "math/Gamma.h"
#include "math/Hankel.h"
#include "fmath/fmath.hpp"

namespace galsim {

    const double ARCSEC2RAD = 180.*60*60/M_PI;  // ~206265
    const double MOCK_INF = 1.e300;

    inline double fast_pow(double x, double y)
    { return fmath::expd(y * std::log(x)); }

    //
    //
    //
    //SBVonKarman
    //
    //
    //

    SBVonKarman::SBVonKarman(double lam, double r0, double L0, double flux,
                             double scale, bool doDelta, const GSParams& gsparams,
                             double force_stepk) :
        SBProfile(new SBVonKarmanImpl(lam, r0, L0, flux, scale, doDelta, gsparams, force_stepk)) {}

    SBVonKarman::SBVonKarman(const SBVonKarman &rhs) : SBProfile(rhs) {}

    SBVonKarman::~SBVonKarman() {}

    double SBVonKarman::getLam() const
    {
        assert(dynamic_cast<const SBVonKarmanImpl*>(_pimpl.get()));
        return static_cast<const SBVonKarmanImpl&>(*_pimpl).getLam();
    }

    double SBVonKarman::getR0() const
    {
        assert(dynamic_cast<const SBVonKarmanImpl*>(_pimpl.get()));
        return static_cast<const SBVonKarmanImpl&>(*_pimpl).getR0();
    }

    double SBVonKarman::getL0() const
    {
        assert(dynamic_cast<const SBVonKarmanImpl*>(_pimpl.get()));
        return static_cast<const SBVonKarmanImpl&>(*_pimpl).getL0();
    }

    double SBVonKarman::getScale() const
    {
        assert(dynamic_cast<const SBVonKarmanImpl*>(_pimpl.get()));
        return static_cast<const SBVonKarmanImpl&>(*_pimpl).getScale();
    }

    bool SBVonKarman::getDoDelta() const
    {
        assert(dynamic_cast<const SBVonKarmanImpl*>(_pimpl.get()));
        return static_cast<const SBVonKarmanImpl&>(*_pimpl).getDoDelta();
    }

    double SBVonKarman::getDelta() const
    {
        assert(dynamic_cast<const SBVonKarmanImpl*>(_pimpl.get()));
        return static_cast<const SBVonKarmanImpl&>(*_pimpl).getDelta();
    }

    double SBVonKarman::getHalfLightRadius() const
    {
        assert(dynamic_cast<const SBVonKarmanImpl*>(_pimpl.get()));
        return static_cast<const SBVonKarmanImpl&>(*_pimpl).getHalfLightRadius();
    }

    double SBVonKarman::structureFunction(double rho) const
    {
        assert(dynamic_cast<const SBVonKarmanImpl*>(_pimpl.get()));
        return static_cast<const SBVonKarmanImpl&>(*_pimpl).structureFunction(rho);
    }

    //
    //
    //
    //VonKarmanInfo
    //
    //
    //

    class VKIkValueResid {
    public:
        VKIkValueResid(const VonKarmanInfo& vki, double mkt) : _vki(vki), _mkt(mkt) {}
        double operator()(double k) const {
            double val = _vki.kValue(k)-_mkt;
            xdbg<<"resid(k="<<k<<")="<<val<<'\n';
            return val;
        }
    private:
        const VonKarmanInfo& _vki;
        const double _mkt;
    };

    // gamma(11/6) gamma(5/6) / pi^(8/3) * (24/5 gamma(6/5))^(5/6)
    const double magic1 = 0.1726286598236691505;

    // Note: lam and L0 are both in units of r0, so are dimensionless within VKInfo.
    VonKarmanInfo::VonKarmanInfo(double lam, double L0, bool doDelta,
                                 const GSParamsPtr& gsparams, double force_stepk) :
        _lam(lam), _L0(L0),
        _L0_invcuberoot(fast_pow(_L0, -1./3)), _L053(fast_pow(L0, 5./3)),
        _stepk(force_stepk), _maxk(0.0),
        _delta(exp(-0.5*magic1*_L053)),
        _deltaScale(1./(1.-_delta)),
        _lam_arcsec(_lam * ARCSEC2RAD / (2.*M_PI)),
        _doDelta(doDelta), _gsparams(gsparams),
        _radial(Table::spline)
    {
        // determine maxK
        // want kValue(maxK)/kValue(0.0) = _gsparams->maxk_threshold;
        // note that kValue(0.0) = 1.
        double mkt = _gsparams->maxk_threshold;
        if (_doDelta) {
            if (mkt < _delta) {
                // If the delta function amplitude is too large, then no matter how far out in k we
                // go, kValue never drops below that amplitude.
                // _maxk = std::numeric_limits<double>::infinity();
                _maxk = MOCK_INF;
            } else {
                mkt = mkt*(1.-_delta)+_delta;
            }
        }
        if (_maxk != MOCK_INF) {
            VKIkValueResid vkikvr(*this, mkt);
            Solve<VKIkValueResid> solver(vkikvr, 0.1, 1);
            solver.bracket();
            solver.setMethod(Brent);
            _maxk = solver.root();
        }
        dbg<<"_maxk = "<<_maxk<<" arcsec^-1\n";
        dbg<<"SB(maxk) = "<<kValue(_maxk)<<'\n';
        dbg<<"_delta = "<<_delta<<'\n';
    }

    double vkStructureFunction(double rho, double L0, double L0_invcuberoot, double L053) {
        // rho in units of r0

        // 2 gamma(11/6) / (2^(5/6) pi^(8/3)) * (24/5 gamma(6/5))^(5/6)
        static const double magic2 = 0.1716613621245708932;
        // gamma(5/6) / 2^(1/6)
        static const double magic3 = 1.005634917998590172;
        // 8 sqrt(2) (3/5 gamma(6/5))^(5/6)
        // Note: This is the mysterious 6.8839 from Racine (1996).
        //       K0_FACTOR in SBKolmogorov.cpp is 2Pi (magic4/2)^(-3/5)
        static const double magic4 = 6.883877182293811615;
        // 24 (sqrt(2) gamma(5/6)(3/5 gamma(6/5))^(5/6) gamma(11/6)) / pi^(2/3)
        static const double magic5 = 10.222659484499054723;

        double rhoL0 = rho/L0;
        if (rhoL0 < 1e-6) {
            return magic4*fast_pow(rho, 5./3.)-magic5*L0_invcuberoot*rho*rho;
        } else {
            double x = 2.*M_PI*rhoL0;
            return magic2*L053*(magic3-fast_pow(x, 5./6.)*math::cyl_bessel_k(5./6., x));
        }
    }

    double VonKarmanInfo::kValueNoTrunc(double k) const {
        // k in inverse arcsec
        return fmath::expd(-0.5*vkStructureFunction(_lam_arcsec*k, _L0, _L0_invcuberoot, _L053));
    }

    double VonKarmanInfo::kValue(double k) const {
        // k in inverse arcsec
        // We're subtracting the asymptotic kValue limit here so that kValue->0 as k->inf.
        // This means we should also rescale by (1-_delta) though, so we still retain
        // kValue(0)=1.d
        xxdbg<<"kValue(k="<<k<<") = ";
        double val = (kValueNoTrunc(k) - _delta) * _deltaScale;
        xxdbg<<val<<'\n';
        if (std::abs(val) < std::numeric_limits<double>::epsilon())
            return 0.0;
        else
            return val;
    }

    class VKXIntegrand : public std::function<double(double)>
    {
    public:
        VKXIntegrand(const VonKarmanInfo& vki) : _vki(vki) {}
        double operator()(double k) const { return _vki.kValue(k); }
    private:
        const VonKarmanInfo& _vki;
    };

    // This version does the integral.  But it's slow, so for regular xValue calls, once the
    // _radial lookup table is built, use that.
    double VonKarmanInfo::rawXValue(double r) const
    {
        xdbg<<"rawXValue at r = "<<r<<std::endl;
        // r in arcsec
        VKXIntegrand I(*this);
        integ::IntRegion<double> reg(0, integ::MOCK_INF);
        double relerr = _gsparams->integration_relerr;
        double abserr = _gsparams->integration_abserr;
        return math::hankel_inf(I, r, 0., relerr, abserr) / (2.*M_PI);
    }

    double VonKarmanInfo::xValue(double r) const {
        if (!_radial.finalized()) _buildRadialFunc();
        return r < _radial.argMax() ? _radial(r) : 0.;
    }

    void VonKarmanInfo::_buildRadialFunc() const {
        dbg<<"Start buildRadialFunc:\n";
        dbg<<"lam = "<<_lam<<std::endl;
        dbg<<"L0 = "<<_L0<<std::endl;
        dbg<<"doDelta = "<<_doDelta<<"  "<<_delta<<"  "<<_deltaScale<<std::endl;
        //set_verbose(2);
        double val = rawXValue(0.0); // This is the value without the delta function (clearly).
        _radial.addEntry(0., val);
        dbg<<"L0^5/3 = "<<_L053<<std::endl;
        dbg<<"f(0) = "<<val<<" arcsec^-2\n";

        // For small values of r, the function goes as
        // f(r) = f0 (1 - C r^2)
        // The following formula for C is completely empirical, but it's close enough for
        // estimating a good value of r0 to start at, which is all we use this for.
        double C = (1.4 * pow(_L0,-2./3.) + 0.0767417) / (_lam_arcsec * _lam_arcsec);
#ifdef DEBUGLOGGING
        double f0 = val;
        double f1 = rawXValue(1.e-2);
        double f2 = rawXValue(2.e-2);
        // For very small values of L0, this value of C is a bit too small, so there is probably
        // another term in the Taylor expansion starting to come into play. Maybe an L0^-4/3 term.
        dbg<<"C = "<<C<<std::endl;
        dbg<<"f(1.e-2) = "<<f1<<"  "<<f0 * (1.-C*1.e-4)<<std::endl;
        dbg<<"f(2.e-2) = "<<f2<<"  "<<f0 * (1.-C*4.e-4)<<std::endl;
#endif
        // Start at r0 where f(r0) - f(0) ~= xvalue_accuracy.
        double r0 = sqrt(_gsparams->xvalue_accuracy / (val * C));

        double dlogr = _gsparams->table_spacing * sqrt(sqrt(_gsparams->xvalue_accuracy / 10.));

        dbg<<"r0 = "<<r0<<" arcsec\n";
        dbg<<"dlogr = "<<dlogr<<"\n";

        double sum = 0.0;
        if (_doDelta) sum += _delta;

        xdbg<<"sum = "<<sum<<'\n';

        // We accumulate the sum without the 2 pi dlogr factors for efficiency.
        // So the relevant thresholds we want are:
        double thresh0 = 0.5 / (2.*M_PI*dlogr);
        double thresh2 = (1.-_gsparams->shoot_accuracy) / (2.*M_PI*dlogr);
        dbg<<"thresh = "<<thresh0<<"  "<<thresh2<<std::endl;
        _hlr = 0.;
        const double maxR = 60.0; // hard cut at 1 arcminute.
        for(double logr=log(r0); logr<log(maxR) && sum < thresh2; logr+=dlogr) {
            double r = exp(logr);
            val = rawXValue(r);
            dbg<<"f("<<r<<") = "<<val<<std::endl;
            _radial.addEntry(r, val);

            // Accumulate integral int(r f(r) dr) = int(r^2 f(r) dlogr), but without dlogr factor,
            // since it is constant for all terms.  (Also not including 2pi which would be in
            // the normal integral for the enclosed flux.)
            sum += val*r*r;
            dbg<<"sum = "<<sum<<'\n';

            if (_hlr == 0. && sum > thresh0) _hlr = r;
        }
        _radial.finalize();
        if (_hlr == 0.)
            throw SBError("Cannot find von Karman half-light-radius.");
        dbg<<"Finished building radial function.\n";
        dbg<<"HLR = "<<_hlr<<" arcsec\n";

        // The large r behavior of F(r) is well approximated by a power law, F ~ r^-n
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
        // Emprically n is very close to 11/3, independent of L0.  This is probably exact
        // for L0 -> infinity (i.e. -> Kolmogorov), and there is probably some good reason
        // that L0 doesn't affect this for reasonable values of L0/r0.  But I (MJ) haven't
        // actually proven this to be true.
        // Regardless, let's just always use this for the purpose of estimating stepk,
        // since any deviations from the exactly correct answer don't matter much.
        double n = 11./3.;
        double R = fast_pow(2.*M_PI*F1*fast_pow(r1,n)/((n-2)*_gsparams->folding_threshold),
                            1./(n-2));
        dbg<<"R = "<<R<<" arcsec\n";
        if (R > maxR) R = maxR;

        // Make sure it is at least 5 hlr
        R = std::max(R, _gsparams->stepk_minimum_hlr*_hlr);
        if (_stepk == 0.0)
            _stepk = M_PI / R;

        std::vector<double> range(2, 0.);
        range[1] = _radial.argMax();
        _sampler.reset(new OneDimensionalDeviate(_radial, range, true, 1.0, *_gsparams));
    }

    void VonKarmanInfo::shoot(PhotonArray& photons, UniformDeviate ud) const
    {
        if (!_sampler)
            _buildRadialFunc();

        _sampler->shoot(photons,ud);
    }

    LRUCache<Tuple<double,double,bool,GSParamsPtr,double>,VonKarmanInfo>
        SBVonKarman::SBVonKarmanImpl::cache(sbp::max_vonKarman_cache);

    //
    //
    //
    //SBVonKarmanImpl
    //
    //
    //

    SBVonKarman::SBVonKarmanImpl::SBVonKarmanImpl(double lam, double r0, double L0, double flux,
                                                  double scale, bool doDelta,
                                                  const GSParams& gsparams, double force_stepk) :
        SBProfileImpl(gsparams),
        _lam(lam),
        _r0(r0),
        _L0(L0),
        _flux(flux),
        _scale(scale),
        _doDelta(doDelta),
        _info(cache.get(MakeTuple(1e-9*lam/r0, L0/r0, doDelta, GSParamsPtr(gsparams), force_stepk/_scale)))
    {}

    double SBVonKarman::SBVonKarmanImpl::maxK() const
    { return _info->maxK()*_scale; }

    double SBVonKarman::SBVonKarmanImpl::stepK() const
    { return _info->stepK()*_scale; }

    double SBVonKarman::SBVonKarmanImpl::getDelta() const
    { return _info->getDelta()*_flux; }

    double SBVonKarman::SBVonKarmanImpl::getHalfLightRadius() const
    { return _info->getHalfLightRadius()/_scale; }

    double SBVonKarman::SBVonKarmanImpl::structureFunction(double rho) const
    {
        xdbg<<"rho = "<<rho<<'\n';
        return vkStructureFunction(rho/_r0, _L0/_r0, fast_pow(_r0/_L0, 1./3),
                                   fast_pow(_L0/_r0, 5./3));
    }

    std::complex<double> SBVonKarman::SBVonKarmanImpl::kValue(const Position<double>& p) const
        // k in units of _scale.
    {
        return _flux * _info->kValue(sqrt(p.x*p.x+p.y*p.y)/_scale);
    }

    double SBVonKarman::SBVonKarmanImpl::xValue(const Position<double>& p) const
        // r in units of _scale
    {
        return _flux * _info->xValue(sqrt(p.x*p.x+p.y*p.y)*_scale);
    }

    void SBVonKarman::SBVonKarmanImpl::shoot(PhotonArray& photons, UniformDeviate ud) const
    {
        dbg<<"VonKarman shoot: N = "<<photons.size()<<std::endl;
        dbg<<"Target flux = "<<getFlux()<<std::endl;
        // Get photons from the VonKarmanInfo structure, rescale flux and size for this instance
         _info->shoot(photons,ud);
        photons.scaleFlux(_flux);
        photons.scaleXY(_scale);
        dbg<<"VonKarman Realized flux = "<<photons.getTotalFlux()<<std::endl;
    }

    template <typename T>
    void SBVonKarman::SBVonKarmanImpl::fillXImage(ImageView<T> im,
                                                  double x0, double dx, int izero,
                                                  double y0, double dy, int jzero) const
    {
        dbg<<"SBVonKarman fillXImage\n";
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

            x0 *= _scale;
            dx *= _scale;
            y0 *= _scale;
            dy *= _scale;

            for (int j=0; j<n; ++j,y0+=dy,ptr+=skip) {
                double x = x0;
                double ysq = y0*y0;
                for (int i=0; i<m; ++i,x+=dx)
                    *ptr++ = _flux * _info->xValue(sqrt(x*x + ysq));
            }
        }
    }

    template <typename T>
    void SBVonKarman::SBVonKarmanImpl::fillXImage(ImageView<T> im,
                                                  double x0, double dx, double dxy,
                                                  double y0, double dy, double dyx) const
    {
        dbg<<"SBVonKarman fillXImage\n";
        dbg<<"x = "<<x0<<" + i * "<<dx<<" + j * "<<dxy<<std::endl;
        dbg<<"y = "<<y0<<" + i * "<<dyx<<" + j * "<<dy<<std::endl;
        const int m = im.getNCol();
        const int n = im.getNRow();
        T* ptr = im.getData();
        const int skip = im.getNSkip();
        assert(im.getStep() == 1);

        x0 *= _scale;
        dx *= _scale;
        dxy *= _scale;
        y0 *= _scale;
        dy *= _scale;
        dyx *= _scale;

        for (int j=0; j<n; ++j,x0+=dxy,y0+=dy,ptr+=skip) {
            double x = x0;
            double y = y0;
            for (int i=0; i<m; ++i,x+=dx,y+=dyx)
                *ptr++ = _flux * _info->xValue(sqrt(x*x + y*y));
        }
    }

    template <typename T>
    void SBVonKarman::SBVonKarmanImpl::fillKImage(ImageView<std::complex<T> > im,
                                                  double kx0, double dkx, int izero,
                                                  double ky0, double dky, int jzero) const
    {
        dbg<<"SBVonKarman fillKImage\n";
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

            kx0 /= _scale;
            dkx /= _scale;
            ky0 /= _scale;
            dky /= _scale;

            for (int j=0; j<n; ++j,ky0+=dky,ptr+=skip) {
                double kx = kx0;
                double kysq = ky0*ky0;
                for (int i=0;i<m;++i,kx+=dkx)
                    *ptr++ = _flux * _info->kValue(kx*kx+kysq);
            }
        }
    }

    template <typename T>
    void SBVonKarman::SBVonKarmanImpl::fillKImage(ImageView<std::complex<T> > im,
                                                  double kx0, double dkx, double dkxy,
                                                  double ky0, double dky, double dkyx) const
    {
        dbg<<"SBVonKarman fillKImage\n";
        dbg<<"kx = "<<kx0<<" + i * "<<dkx<<" + j * "<<dkxy<<std::endl;
        dbg<<"ky = "<<ky0<<" + i * "<<dkyx<<" + j * "<<dky<<std::endl;
        const int m = im.getNCol();
        const int n = im.getNRow();
        std::complex<T>* ptr = im.getData();
        int skip = im.getNSkip();
        assert(im.getStep() == 1);

        kx0 /= _scale;
        dkx /= _scale;
        dkxy /= _scale;
        ky0 /= _scale;
        dky /= _scale;
        dkyx /= _scale;

        for (int j=0; j<n; ++j,kx0+=dkxy,ky0+=dky,ptr+=skip) {
            double kx = kx0;
            double ky = ky0;
            for (int i=0; i<m; ++i,kx+=dkx,ky+=dkyx)
                *ptr++ = _flux * _info->kValue(kx*kx+ky*ky);
        }
    }

}
