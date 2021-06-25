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

#include "SBSersic.h"
#include "SBSersicImpl.h"
#include "integ/Int.h"
#include "Solve.h"
#include "math/Bessel.h"
#include "math/Gamma.h"
#include "math/Hankel.h"
#include "fmath/fmath.hpp"

namespace galsim {

    inline double fast_pow(double x, double y)
    { return fmath::expd(y * std::log(x)); }

    SBSersic::SBSersic(double n, double scale_radius, double flux,
                       double trunc, const GSParams& gsparams) :
        SBProfile(new SBSersicImpl(n, scale_radius, flux, trunc, gsparams)) {}

    SBSersic::SBSersic(const SBSersic& rhs) : SBProfile(rhs) {}

    SBSersic::~SBSersic() {}

    double SBSersic::getN() const
    {
        assert(dynamic_cast<const SBSersicImpl*>(_pimpl.get()));
        return static_cast<const SBSersicImpl&>(*_pimpl).getN();
    }

    double SBSersic::getHalfLightRadius() const
    {
        assert(dynamic_cast<const SBSersicImpl*>(_pimpl.get()));
        return static_cast<const SBSersicImpl&>(*_pimpl).getHalfLightRadius();
    }

    double SBSersic::getScaleRadius() const
    {
        assert(dynamic_cast<const SBSersicImpl*>(_pimpl.get()));
        return static_cast<const SBSersicImpl&>(*_pimpl).getScaleRadius();
    }

    double SBSersic::getTrunc() const
    {
        assert(dynamic_cast<const SBSersicImpl*>(_pimpl.get()));
        return static_cast<const SBSersicImpl&>(*_pimpl).getTrunc();
    }

    LRUCache<Tuple<double, double, GSParamsPtr>, SersicInfo>
        SBSersic::SBSersicImpl::cache(sbp::max_sersic_cache);

    SBSersic::SBSersicImpl::SBSersicImpl(double n,  double scale_radius, double flux,
                                         double trunc, const GSParams& gsparams) :
        SBProfileImpl(gsparams),
        _n(n), _flux(flux), _r0(scale_radius), _trunc(trunc),
        _r0_sq(_r0*_r0), _inv_r0(1./_r0), _inv_r0_sq(_inv_r0*_inv_r0), _trunc_sq(trunc*trunc),
        _info(cache.get(MakeTuple(_n, _trunc/_r0, GSParamsPtr(this->gsparams))))
    {
        dbg<<"Start SBSersic constructor:\n";
        dbg<<"n = "<<_n<<std::endl;
        dbg<<"r0 = "<<_r0<<std::endl;
        dbg<<"flux = "<<_flux<<std::endl;
        dbg<<"trunc = "<<_trunc<<std::endl;

        _re = _r0 * _info->getHLR();
        dbg<<"hlr = "<<_re<<std::endl;

        _shootnorm = _flux * _info->getXNorm(); // For shooting, we don't need the 1/r0^2 factor.
        _xnorm = _shootnorm * _inv_r0_sq;
        dbg<<"norms = "<<_xnorm<<", "<<_shootnorm<<std::endl;
    }

    double SBSersic::SBSersicImpl::xValue(const Position<double>& p) const
    {
        double rsq = (p.x*p.x+p.y*p.y)*_inv_r0_sq;
        return _xnorm * _info->xValue(rsq);
    }

    std::complex<double> SBSersic::SBSersicImpl::kValue(const Position<double>& k) const
    {
        double ksq = (k.x*k.x + k.y*k.y)*_r0_sq;
        return _flux * _info->kValue(ksq);
    }

    template <typename T>
    void SBSersic::SBSersicImpl::fillXImage(ImageView<T> im,
                                            double x0, double dx, int izero,
                                            double y0, double dy, int jzero) const
    {
        dbg<<"SBSersic fillXImage\n";
        dbg<<"x = "<<x0<<" + i * "<<dx<<", izero = "<<izero<<std::endl;
        dbg<<"y = "<<y0<<" + j * "<<dy<<", jzero = "<<jzero<<std::endl;
        if (izero != 0 || jzero != 0) {
            xdbg<<"Use Quadrant\n";
            fillXImageQuadrant(im,x0,dx,izero,y0,dy,jzero);

#if 0
            // Note: This bit isn't necessary anymore, because I changed how the quadrant is
            // filled, so it always does 0,0 first, which means it is already exact.  But leave
            // this code snippet here in case we make any changes that would necessitate
            // bringing it back.

            // Sersics tend to be super peaky at the center, so if we are including
            // (0,0) in the image, then it is helpful to do (0,0) explicitly rather
            // than treating it as 0 ~= x0 + n*dx, which has rounding errors and doesn't
            // quite come out to 0, and high-n Sersics vary a lot between r = 0 and 1.e-16!
            // By a lot, I mean ~0.5%, which is enough to care about.
            if (izero != 0 && jzero != 0) {
                // NB: _info->xValue(0) = 1
                T* ptr = im.getData() + jzero*im.getStride() + izero;
                *ptr = _xnorm;
            }
#endif
        } else {
            xdbg<<"Non-Quadrant\n";
            const int m = im.getNCol();
            const int n = im.getNRow();
            T* ptr = im.getData();
            const int skip = im.getNSkip();
            assert(im.getStep() == 1);

            x0 *= _inv_r0;
            dx *= _inv_r0;
            y0 *= _inv_r0;
            dy *= _inv_r0;

            for (int j=0; j<n; ++j,y0+=dy,ptr+=skip) {
                double x = x0;
                double ysq = y0*y0;
                for (int i=0; i<m; ++i,x+=dx)
                    *ptr++ = _xnorm * _info->xValue(x*x + ysq);
            }
        }
    }

    template <typename T>
    void SBSersic::SBSersicImpl::fillXImage(ImageView<T> im,
                                            double x0, double dx, double dxy,
                                            double y0, double dy, double dyx) const
    {
        dbg<<"SBSersic fillXImage\n";
        dbg<<"x = "<<x0<<" + i * "<<dx<<" + j * "<<dxy<<std::endl;
        dbg<<"y = "<<y0<<" + i * "<<dyx<<" + j * "<<dy<<std::endl;
        const int m = im.getNCol();
        const int n = im.getNRow();
        T* ptr = im.getData();
        const int skip = im.getNSkip();
        assert(im.getStep() == 1);

        x0 *= _inv_r0;
        dx *= _inv_r0;
        dxy *= _inv_r0;
        y0 *= _inv_r0;
        dy *= _inv_r0;
        dyx *= _inv_r0;

        double x00 = x0; // Preserve the originals for below.
        double y00 = y0;
        for (int j=0; j<n; ++j,x0+=dxy,y0+=dy,ptr+=skip) {
            double x = x0;
            double y = y0;
            for (int i=0; i<m; ++i,x+=dx,y+=dyx)
                *ptr++ = _xnorm * _info->xValue(x*x + y*y);
        }

        // Check if one of these points is really (0,0) in disguise and fix it up
        // with a call to xValue(0.0), rather than using xValue(epsilon != 0), which
        // for Sersics can be rather wrong due to their super steep central peak.
        // 0 = x0 + dx i + dxy j
        // 0 = y0 + dyx i + dy j
        // ( i ) = ( dx  dxy )^-1 ( -x0 )
        // ( j )   ( dyx  dy )    ( -y0 )
        //       = 1/(dx dy - dxy dyx) (  dy  -dxy ) ( -x0 )
        //                             ( -dyx  dx  ) ( -y0 )
        double det = dx * dy - dxy * dyx;
        double i0 = (-dy * x00 + dxy * y00) / det;
        double j0 = (dyx * x00 - dx * y00) / det;
        dbg<<"i0, j0 = "<<i0<<','<<j0<<std::endl;
        dbg<<"x0 + dx i + dxy j = "<<x00+dx*i0+dxy*j0<<std::endl;
        dbg<<"y0 + dyx i + dy j = "<<y00+dyx*i0+dy*j0<<std::endl;
        int inti0 = int(floor(i0+0.5));
        int intj0 = int(floor(j0+0.5));

        if ( std::abs(i0 - inti0) < 1.e-12 && std::abs(j0 - intj0) < 1.e-12 &&
             inti0 >= 0 && inti0 < m && intj0 >= 0 && intj0 < n)  {
            ptr = im.getData() + intj0*im.getStride() + inti0;
            dbg<<"Fixing central value from "<<*ptr;
            // NB: _info->xValue(0) = 1
            *ptr = _xnorm;
            dbg<<" to "<<*ptr<<std::endl;
#ifdef DEBUGLOGGING
            double x = x00;
            double y = y00;
            for (int j=0;j<intj0;++j) { x += dxy; y += dy; }
            for (int i=0;i<inti0;++i) { x += dx; y += dyx; }
            double rsq = x*x+y*y;
            dbg<<"Note: the original rsq value for this pixel had been "<<rsq<<std::endl;
            dbg<<"xValue(rsq) = "<<_info->xValue(rsq)<<std::endl;
            dbg<<"xValue(0) = "<<_info->xValue(0.)<<std::endl;
#endif
        }
    }

    template <typename T>
    void SBSersic::SBSersicImpl::fillKImage(ImageView<std::complex<T> > im,
                                                double kx0, double dkx, int izero,
                                                double ky0, double dky, int jzero) const
    {
        dbg<<"SBSersic fillKImage\n";
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

            kx0 *= _r0;
            dkx *= _r0;
            ky0 *= _r0;
            dky *= _r0;

            for (int j=0; j<n; ++j,ky0+=dky,ptr+=skip) {
                double kx = kx0;
                double kysq = ky0*ky0;
                for (int i=0;i<m;++i,kx+=dkx)
                    *ptr++ = _flux * _info->kValue(kx*kx + kysq);
            }
        }
    }

    template <typename T>
    void SBSersic::SBSersicImpl::fillKImage(ImageView<std::complex<T> > im,
                                                double kx0, double dkx, double dkxy,
                                                double ky0, double dky, double dkyx) const
    {
        dbg<<"SBSersic fillKImage\n";
        dbg<<"kx = "<<kx0<<" + i * "<<dkx<<" + j * "<<dkxy<<std::endl;
        dbg<<"ky = "<<ky0<<" + i * "<<dkyx<<" + j * "<<dky<<std::endl;
        const int m = im.getNCol();
        const int n = im.getNRow();
        std::complex<T>* ptr = im.getData();
        int skip = im.getNSkip();
        assert(im.getStep() == 1);

        kx0 *= _r0;
        dkx *= _r0;
        dkxy *= _r0;
        ky0 *= _r0;
        dky *= _r0;
        dkyx *= _r0;

        for (int j=0; j<n; ++j,kx0+=dkxy,ky0+=dky,ptr+=skip) {
            double kx = kx0;
            double ky = ky0;
            for (int i=0; i<m; ++i,kx+=dkx,ky+=dkyx)
                *ptr++ = _flux * _info->kValue(kx*kx + ky*ky);
        }
    }

    double SBSersic::SBSersicImpl::maxK() const { return _info->maxK() * _inv_r0; }
    double SBSersic::SBSersicImpl::stepK() const { return _info->stepK() * _inv_r0; }

    SersicInfo::SersicInfo(double n, double trunc, const GSParamsPtr& gsparams) :
        _n(n), _trunc(trunc), _gsparams(gsparams),
        _invn(1./_n), _inv2n(0.5*_invn),
        _trunc_sq(_trunc*_trunc), _truncated(_trunc > 0.),
        _gamma2n(math::tgamma(2.*_n)),
        _maxk(0.), _stepk(0.), _re(0.), _flux(0.),
        _ft(Table::spline),
        _kderiv2(0.), _kderiv4(0.)
    {
        dbg<<"Start SersicInfo constructor for n = "<<_n<<std::endl;
        dbg<<"trunc = "<<_trunc<<std::endl;

        if (_n < sbp::minimum_sersic_n || _n > sbp::maximum_sersic_n)
            throw SBError("Requested Sersic index out of range");
    }

    double SersicInfo::stepK() const
    {
        if (_stepk == 0.) {
            // How far should the profile extend, if not truncated?
            // Estimate number of effective radii needed to enclose (1-folding_threshold) of flux
            double R = calculateMissingFluxRadius(_gsparams->folding_threshold);
            if (_truncated && _trunc < R)  R = _trunc;
            // Go to at least 5*re
            R = std::max(R,_gsparams->stepk_minimum_hlr);
            dbg<<"R => "<<R<<std::endl;
            _stepk = M_PI / R;
            dbg<<"stepk = "<<_stepk<<std::endl;
        }
        return _stepk;
    }

    double SersicInfo::maxK() const
    {
        if (_maxk == 0.) buildFT();
        return _maxk;
    }

    double SersicInfo::getHLR() const
    {
        if (_re == 0.) calculateHLR();
        return _re;
    }

    double SersicIntegratedFlux(double n, double r)
    {
        double z = fast_pow(r, 1./n);
        return math::gamma_p(2.*n, z);
    }

    double SersicInfo::getFluxFraction() const
    {
        if (_flux == 0.) {
            // Calculate the flux of a truncated profile (relative to the integral for
            // an untruncated profile).
            if (_truncated) {
                // integrate from 0. to _trunc
                _flux = SersicIntegratedFlux(_n, _trunc);
                dbg << "Flux fraction = " << _flux << std::endl;
            } else {
                _flux = 1.;
            }
        }
        return _flux;
    }

    double SersicInfo::getXNorm() const
    { return 1. / (2.*M_PI*_n*_gamma2n * getFluxFraction()); }

    double SersicInfo::xValue(double rsq) const
    {
        if (_truncated && rsq > _trunc_sq) return 0.;
        else return fmath::expd(-fast_pow(rsq,_inv2n));
    }

    double SersicInfo::kValue(double ksq) const
    {
        assert(ksq >= 0.);
        if (!_ft.finalized()) buildFT();

        if (ksq>=_ksq_max)
            return (_highk_a + _highk_b/sqrt(ksq))/ksq; // high-k asymptote
        else if (ksq<_ksq_min)
            return 1. + ksq*(_kderiv2 + ksq*_kderiv4); // Use quartic approx at low k
        else {
            double lk=0.5*std::log(ksq); // Lookup table is logarithmic
            return _ft(lk)/ksq;
        }
    }

    class SersicRadialFunction: public FluxDensity
    {
    public:
        SersicRadialFunction(double invn): _invn(invn) {}
        double operator()(double r) const { return fmath::expd(-fast_pow(r,_invn)); }
    private:
        double _invn;
    };

    void SersicInfo::buildFT() const
    {
        // The small-k expansion of the Hankel transform is (normalized to have flux=1):
        // 1 - Gamma(4n) / 4 Gamma(2n) + Gamma(6n) / 64 Gamma(2n) - Gamma(8n) / 2304 Gamma(2n)
        // from the series summation J_0(x) = Sum^inf_{m=0} (-1)^m (m!)^-2 (x/2)^2m
        double gamma4n;
        double gamma6n;
        double gamma8n;
        if (!_truncated) {
            gamma4n = math::tgamma(4.*_n);
            gamma6n = math::tgamma(6.*_n);
            gamma8n = math::tgamma(8.*_n);
        } else {
            double z = std::pow(_trunc, 1./_n);
            gamma4n = math::gamma_p(4.*_n, z) * math::tgamma(4.*_n);
            gamma6n = math::gamma_p(6.*_n, z) * math::tgamma(6.*_n);
            gamma8n = math::gamma_p(8.*_n, z) * math::tgamma(8.*_n);
        }
        // The quadratic term of small-k expansion:
        _kderiv2 = -gamma4n / (4.*_gamma2n) / getFluxFraction();
        // And a quartic term:
        _kderiv4 = gamma6n / (64.*_gamma2n) / getFluxFraction();
        dbg<<"kderiv2,4 = "<<_kderiv2<<"  "<<_kderiv4<<std::endl;

        // When is it safe to use low-k approximation?
        // See when next term past quartic is at accuracy threshold
        double kderiv6 = gamma8n / (2304.*_gamma2n) / getFluxFraction();
        dbg<<"kderiv6 = "<<kderiv6<<std::endl;
        double kmin = std::pow(_gsparams->kvalue_accuracy / kderiv6, 1./6.);
        dbg<<"kmin = "<<kmin<<std::endl;
        _ksq_min = kmin * kmin;
        dbg<<"ksq_min = "<<_ksq_min<<std::endl;

        // Normalization for integral at k=0:
        double hankel_norm = getFluxFraction()*_n*_gamma2n;
        dbg<<"hankel_norm = "<<hankel_norm<<std::endl;

        // We use a cubic spline for the interpolation, which has an error of O(h^4) max(f'''').
        // The fourth derivative is a bit tough to estimate of course, but doing it numerically
        // for a few different values of n, we find 10 to be a reasonably conservative estimate.
        // 10 h^4 <= kvalue_accuracy
        // h = (kvalue_accuracy/10)^0.25
        double dlogk = _gsparams->table_spacing * sqrt(sqrt(_gsparams->kvalue_accuracy / 10.));
        dbg<<"n = "<<_n<<std::endl;
        dbg<<"Using dlogk = "<<dlogk<<std::endl;

        // As we go, build up the high k approximation f(k) = a/k^2 + b/k^3, based on the last
        // 10 items. Keep going until the predicted value is accurate enough for 5 items in a row.
        // Once we're past the maxk value, we try to stop if the approximation is within
        // kvalue_accuracy of the correct value.
        int n_correct = 0;
        const int n_fit = 10;
        std::deque<double> fit_vals;
        double sf=0., skf=0., sk=0., sk2=0.;

        // Don't go past k = 500
        _ksq_max = -1.;
        _maxk = kmin; // Just in case we break on the first iteration.
        SersicRadialFunction I(_invn);
        bool found_maxk = false;
        for (double logk = std::log(kmin)-0.001; logk < std::log(500.); logk += dlogk) {
            double k = fmath::expd(logk);
            double ksq = k*k;

            double val;
            if (_truncated) {
                val = math::hankel_trunc(I, k, 0., _trunc,
                                         _gsparams->integration_relerr,
                                         _gsparams->integration_abserr*hankel_norm);
            } else {
                val = math::hankel_inf(I, k, 0.,
                                       _gsparams->integration_relerr,
                                       _gsparams->integration_abserr*hankel_norm);
            }
            val /= hankel_norm;
            xdbg<<"logk = "<<logk<<", ft("<<exp(logk)<<") = "<<val<<"   "<<val*ksq<<std::endl;

            double f0 = val * ksq;
            _ft.addEntry(logk,f0);

            // Keep track of whether we are below the maxk_threshold yet:
            if (std::abs(val) > _gsparams->maxk_threshold) { _maxk = k; n_correct = 0; }
            else {
                found_maxk = true;
                // Once we are past the last maxk_threshold value,  figure out if the
                // high-k approximation is good enough.
                _highk_a = (sf*sk2 - sk*skf) / (n_fit*sk2 - sk*sk);
                _highk_b = (n_fit*skf - sk*sf) / (n_fit*sk2 - sk*sk);
                double f0_pred = _highk_a + _highk_b/k;
                xdbg<<"f0 = "<<f0<<", f0_pred = "<<f0_pred;
                xdbg<<"   a,b = "<<_highk_a<<','<<_highk_b<<std::endl;
                if (std::abs(f0-f0_pred)/ksq < _gsparams->kvalue_accuracy) ++n_correct;
                else n_correct = 0;
                if (n_correct >= 5) {
                    _ksq_max = ksq;
                    break;
                }
            }

            // Update the terms needed for the high-k approximation
            if (int(fit_vals.size()) == n_fit) {
                double k_back = fmath::expd(logk - n_fit*dlogk);
                double f_back = fit_vals.back();
                fit_vals.pop_back();
                double inv_k = 1./k;
                double inv_k_back = 1./k_back;
                sf += f0 - f_back;
                skf += f0*inv_k - f_back*inv_k_back;
                sk += inv_k - inv_k_back;
                sk2 += inv_k*inv_k - inv_k_back*inv_k_back;
            } else {
                assert(int(fit_vals.size()) < n_fit);
                double inv_k = 1./k;
                sf += f0;
                skf += f0*inv_k;
                sk += inv_k;
                sk2 += inv_k*inv_k;
            }
            fit_vals.push_front(f0);
        }
        _ft.finalize();
        // If didn't find a good approximation for large k, just use the largest k we put in
        // in the table.  (Need to use some approximation after this anyway!)
        if (_ksq_max <= 0.) _ksq_max = fmath::expd(2. * _ft.argMax());
        xdbg<<"ft.argMax = "<<_ft.argMax()<<std::endl;
        xdbg<<"ksq_max = "<<_ksq_max<<std::endl;

        if (found_maxk) {
            // This is the last value that didn't satisfy the requirement, so just go to
            // the next value.
            xdbg<<"maxk with val > "<<_gsparams->maxk_threshold<<" = "<<_maxk<<std::endl;
            _maxk *= fmath::expd(dlogk);
            xdbg<<"maxk -> "<<_maxk<<std::endl;
        } else {
            // Then we never did find a value of k such that f(k) < maxk_threshold
            // This means that maxk needs to be larger.  Use the high-k approximation.
            xdbg<<"Never found f(k) < maxk_threshold.\n";
            _highk_a = (sf*sk2 - sk*skf) / (n_fit*sk2 - sk*sk);
            _highk_b = (n_fit*skf - sk*sf) / (n_fit*sk2 - sk*sk);
            xdbg<<"Use current best guess for high-k approximation.\n";
            xdbg<<"a,b = "<<_highk_a<<", "<<_highk_b<<std::endl;
            // Use that approximation to determine maxk
            // f(maxk) = (a + b/k)/k^2 = maxk_threshold
            _maxk = sqrt(_highk_a / _gsparams->maxk_threshold);
            xdbg<<"initial maxk = "<<_maxk<<std::endl;
            for (int i=0; i<3; ++i) {
                _maxk = sqrt( (_highk_a - _highk_b/_maxk) / _gsparams->maxk_threshold );
                xdbg<<"maxk => "<<_maxk<<std::endl;
            }
        }
    }

    // Function object for finding the r that encloses all except a particular flux fraction.
    class SersicMissingFlux
    {
    public:
        SersicMissingFlux(double n, double missing_flux) : _2n(2.*n), _target(missing_flux) {}

        // Provide z = r^1/n, rather than r.
        double operator()(double z) const
        {
            double f = (1.-math::gamma_p(_2n, z)) * math::tgamma(_2n);
            xdbg<<"func("<<z<<") = "<<f<<"-"<<_target<<" = "<< f-_target<<std::endl;
            return f - _target;
        }
    private:
        double _2n;
        double _target;
    };

    // Find what radius encloses (1-missing_flux_frac) of the total flux in a Sersic profile.
    double SersicInfo::calculateMissingFluxRadius(double missing_flux_frac) const
    {
        // int(exp(-r^1/n) r, r=R..inf) = x * int(exp(-r^1/n) r, r=0..inf)
        //                                = x n Gamma(2n)    [x == missing_flux_frac]
        //
        // (1) First, find approximate solution to x * Gamma(2n) = Gamma(2n, r^1/n)
        // Change variables: u = r^1/n,
        // du = 1/n r^(1-n)/n dr
        //    = 1/n r^1/n dr/r
        //    = u/n dr/r
        // r dr = n du r^2 / u
        //      = n du u^2n / u
        // n int(u^(2n-1) exp(-u), u=R^1/n..inf) = x n Gamma(2n)
        // Let z = R^1/n
        //
        // int(u^(2n-1) exp(-u), u=z..inf) = x Gamma(2n)
        //
        // The lhs is an incomplete gamma function: Gamma(2n,z), which according to
        // Abramowitz & Stegun (6.5.32) has a high-z asymptotic form of:
        // Gamma(2n,z) ~= z^(2n-1) exp(-z) (1 + (2n-1)/z + (2n-1)(2n-2)/z^2 + ... )
        // ln(x Gamma(2n)) = (2n-1) ln(z) - z + (2n-1)/z + (2n-1)(2n-3)/(2*z^2) + O(z^3)
        // z = -ln(x Gamma(2n) + (2n-1) ln(z) + (2n-1)/z + (2n-1)(2n-3)/(2*z^2) + O(z^3)
        // Use this as a starting point.  Then switch to a Brent method solver.
        dbg<<"Find maxr for missing_flux_frac = "<<missing_flux_frac<<std::endl;
        double missing_flux = missing_flux_frac * _gamma2n;
        // Just do one round of update here.
        double z1 = -std::log(missing_flux);
        dbg<<"z1 = "<<z1<<std::endl;

        double z;
        if (_n == 0.5) {
            // If n==0.5, then the formula is exact:
            z = z1;
        } else {
            // Otherwise, continue on...
            z = 4.*(_n+1.);  // A decent starting guess for a range of n.
            double twonm1 = 2.*_n-1.;
            double z2 = z1 + twonm1 * std::log(z) + twonm1/z + twonm1*(2.*_n-3.)/(2.*z*z);
            // If n is very, very close to 0.5, this might not be very different.
            // Make sure the gap is not super tiny.
            if (z2 > z1 && z2-z1 < 0.01) z2 = z1 + 0.01;
            else if (z2 < z1 && z2-z1 > -0.01) z2 = z1 - 0.01;
            dbg<<"Initial z from asymptotic expansion: z => "<<z2<<std::endl;

            // For larger n, z1 can be negative, which is bad.
            // So use the HLR _b value instead:
            if (z1 < 0.) {
                assert(missing_flux_frac < 0.5);
                getHLR(); // Make sure _b is set.
                z1 = _b;
            }

            // (2) Now find a more exact solution using an iterative solver
            SersicMissingFlux func(_n, missing_flux);
            // We need one bracket to be an exact bound, and we can let the other side change.
            // The value for n = 0.5 is z = -log(missing_flux).  As n increases, z increases.
            dbg<<"z1 = "<<z1<<" ... z2 = "<<z2<<std::endl;
            Solve<SersicMissingFlux> solver(func,z1,z2);
            solver.setMethod(Brent);
            solver.bracketUpper();    // expand upper bracket if necessary
            z = solver.root();
            dbg<<"From Brent solver: z => "<<z<<std::endl;
        }

        double R = std::pow(z,_n);
        dbg<<"R is "<<R<<std::endl;
        return R;
    }

    static double CalculateB(double n, double invn, double gamma2n, double flux_fraction)
    {
        dbg<<"Find HLR for (n,gamma2n) = ("<<n<<","<<gamma2n<<")"<<std::endl;
        // Find solution to gamma(2n,re^(1/n)) = gamma2n / 2
        // where gamma2n is the truncated gamma function Gamma(2n,trunc^(1/n))
        // We initially solve for b = re^1/n, and then calculate re from that.
        // Start with the approximation from Ciotti & Bertin, 1999:
        // b ~= 2n - 1/3 + 4/(405n) + 46/(25515n^2) + 131/(1148175n^3) - ...
        // Then we use a non-linear solver to tweak it up.
        double invnsq = invn*invn;
        double b1 = 2.*n-1./3.;
        double b2 = b1 + (8./405.)*invn + (46./25515.)*invnsq + (131./1148175.)*invn*invnsq;
        // Note: This is the value if the profile is untruncated.  It will be smaller if
        // the profile is actually truncated and gamma2n < Gamma(2n)

        SersicMissingFlux func(n, (1. - 0.5*flux_fraction)*gamma2n);
        Solve<SersicMissingFlux> solver(func,b1,b2);
        xdbg<<"Initial range is "<<b1<<" .. "<<b2<<std::endl;
        solver.setMethod(Brent);
        solver.bracketLowerWithLimit(0.);    // expand lower bracket if necessary
        xdbg<<"After bracket, range is "<<solver.getLowerBound()<<" .. "<<
            solver.getUpperBound()<<std::endl;
        // We store b in case we need it again for calculateScaleForTruncatedHLR(), so we
        // can save a pow call.
        double b = solver.root();
        dbg<<"Root is "<<b<<std::endl;
        return b;
    }

    void SersicInfo::calculateHLR() const
    {
        _b = CalculateB(_n, _invn, _gamma2n, getFluxFraction());

        // re = b^n
        _re = std::pow(_b,_n);
        dbg<<"re is "<<_re<<std::endl;
    }

    double SersicHLR(double n, double flux_fraction)
    {
        double b = CalculateB(n, 1./n, math::tgamma(2*n), flux_fraction);
        return std::pow(b,n);
    }

    // Function object for finding the r that encloses all except a particular flux fraction.
    class SersicTruncatedHLR
    {
    public:
        // x = (trunc/re)^1/n
        SersicTruncatedHLR(double n, double x) : _2n(2.*n), _x(x) {}

        double operator()(double b) const
        {
            double f1 = math::gamma_p(_2n, b);
            double f2 = math::gamma_p(_2n, _x*b);
            // Solve for f1 = f2/2
            xdbg<<"func("<<b<<") = 2*"<<f1<<" - "<<f2<<" = "<< 2.*f1-f2<<std::endl;
            return (2.*f1-f2) * math::tgamma(_2n);
        }
    private:
        double _2n;
        double _x;
    };

    // This helper function does the dimensionless version of the problem.
    // trunc is given in units of re, and the returned scale radius is also in units of re.
    double CalculateTruncatedScale(double n, double invn, double b, double trunc)
    {
        // This is the limit for profiles that round off in the center, since you can locally
        // approximate the profile as flat within the truncation radius.  This isn't true for
        // Sersics, so the real limit is larger than this (since more flux is inside re than in
        // the annulus between re and sqrt(2) re), but I don't know of an analytic formula for
        // the correct limit.  So we check for this here, and then if we encounter problems
        // later on, we throw a different error.
        if (trunc <= sqrt(2.)) {
            throw SBError("Sersic truncation must be larger than sqrt(2)*half_light_radius.");
        }

        // Given re and trunc, find the scale radius, r0, that makes these work.
        // f(re) = gamma(2n,(re/r0)^(1/n))
        // f(trunc) = gamma(2n,(trunc/r0)^(1/n))
        // Solve for the r0 that leads to f(re) = 1/2 f(trunc)
        // Equivalently, if b = (re/r0)^(1/n) and z = (trunc/r0)^(1/n) = b * (trunc/re)^(1/n)
        // then solve for b.
        // gamma(2n,b) = 1/2 gamma(2n,x*b), where x = (trunc/re)^(1/n)
        double x = std::pow(trunc,invn);
        dbg<<"x = "<<x<<std::endl;

        // For an initial guess, we start with the asymptotic expansing from A&S (6.5.32):
        // Gamma(2n,b) ~= b^(2n-1) exp(-b) (1 + (2n-1)/b + (2n-1)(2n-2)/b^2 + ... )
        // Gamma(2n,xb) ~= (xb)^(2n-1) exp(-xb) (1 + (2n-1)/(xb) + (2n-1)(2n-2)/(xb)^2 + ... )
        // Just take the first terms:
        // b^(2n-1) exp(-b) = 1/2 (xb)^(2n-1) exp(-xb)
        // exp( (x-1) b ) = 1/2 x^(2n-1)
        // (x-1) b = log(1/2) + (2n-1) log(x)
        double b1 = (std::log(0.5) + (2.*n-1) * std::log(x)) / (x-1.);
        dbg<<"Initial guess b = "<<b1<<std::endl;
        // Note: This isn't a very good initial guess, but the solver tends to converge pretty
        // rapidly anyway.

        // If trunc = sqrt(2) * re, then x = 2^(1/2n), and the initial guess for b is:
        // b = log( 0.5 * 2^(1/2n)^(2n-1) ) / (sqrt(2)-1)
        //   = -(1/2n) * log(2) / (sqrt(2)-1)
        // Negative b is obviously problematic.  I don't know for sure what the real limit
        // on trunc/re is.  It's possible that the full formulae can give a positive solution
        // even if the initial estimate is negative.  But unless someone complains (and proposes
        // a better prescription for this), we'll take this as a de facto limit.
        if (b1 < 1.e-3 * b) {
            //throw SBError("Sersic truncation is too small for the given half_light_radius.");
            // Update: Ricardo Herbonnet (rightly) complained.
            // He pointed out that this formula for b1 is always == 0 for n = 0.5.
            // So we can't just be throwing an exception here.
            // Since we expand the bracket below anyway, switch to just using b/2 and
            // letting the expansion happen.
            // I also updated the above check from b1 <= 0 to b1 < 1.e-3 * b.
            // Probably if we are getting really close to zero, it is better to start with
            // b/2 instead and expand it down.
            b1 = b/2;
        }

        // The upper limit to b corresponds to the half-light radius of the untruncated profile.
        double b2 = b;
        SersicTruncatedHLR func(n, x);
        Solve<SersicTruncatedHLR> solver(func,b1,b2);
        solver.setMethod(Brent);
        solver.bracketLowerWithLimit(0.);    // expand lower bracket if necessary
        xdbg<<"After bracket, range is "<<solver.getLowerBound()<<" .. "<<
            solver.getUpperBound()<<std::endl;
        b = solver.root();
        dbg<<"Root is "<<b<<std::endl;

        // r0 = re / b^n
        return 1. / std::pow(b,n);
    }

    double SersicInfo::calculateScaleForTruncatedHLR(double re, double trunc) const
    {
        // We need _b, so call getHLR(), since it may not be calculated yet.
        // We don't care about the return value, but it also stores the b value in _b.
        getHLR();
        return re * CalculateTruncatedScale(_n, _invn, _b, trunc/re);
    }

    double SersicTruncatedScale(double n, double hlr, double trunc)
    {
        double invn = 1./n;
        double b = CalculateB(n, invn, math::tgamma(2*n), 1.);
        return hlr * CalculateTruncatedScale(n, invn, b, trunc/hlr);
    }

    void SersicInfo::shoot(PhotonArray& photons, UniformDeviate ud) const
    {
        dbg<<"Target flux = 1.0\n";

        if (!_sampler) {
            // Set up the classes for photon shooting
            _radial.reset(new SersicRadialFunction(_invn));
            std::vector<double> range(2,0.);
            double shoot_maxr = calculateMissingFluxRadius(_gsparams->shoot_accuracy);
            if (_truncated && _trunc < shoot_maxr) shoot_maxr = _trunc;
            range[1] = shoot_maxr;
            double nominal_flux = 2.*M_PI*_n*_gamma2n * _flux;
            _sampler.reset(new OneDimensionalDeviate(*_radial, range, true, nominal_flux,
                                                     *_gsparams));
        }

        assert(_sampler.get());
        _sampler->shoot(photons,ud);
        dbg<<"SersicInfo Realized flux = "<<photons.getTotalFlux()<<std::endl;
    }

    void SBSersic::SBSersicImpl::shoot(PhotonArray& photons, UniformDeviate ud) const
    {
        dbg<<"Sersic shoot: N = "<<photons.size()<<std::endl;
        dbg<<"Target flux = "<<getFlux()<<std::endl;
        // Get photons from the SersicInfo structure, rescale flux and size for this instance
        _info->shoot(photons,ud);
        photons.scaleFlux(_shootnorm);
        photons.scaleXY(_r0);
        dbg<<"Sersic Realized flux = "<<photons.getTotalFlux()<<std::endl;
    }
}
