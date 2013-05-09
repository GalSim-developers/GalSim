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

#include <boost/math/special_functions/gamma.hpp>

#include "SBSersic.h"
#include "SBSersicImpl.h"
#include "integ/Int.h"
#include "Solve.h"

#ifdef DEBUGLOGGING
#include <fstream>
//std::ostream* dbgout = &std::cout;
//int verbose_level = 1;
#endif

namespace galsim {

    SBSersic::SBSersic(double n, double re, double flux,
                       double trunc, bool flux_untruncated,
                       boost::shared_ptr<GSParams> gsparams) :
        SBProfile(new SBSersicImpl(n, re, flux, trunc, flux_untruncated, gsparams)) {}

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

    const int MAX_SERSIC_INFO = 100;

    LRUCache<std::pair<SersicKey, const GSParams*>, SersicInfo> 
        SBSersic::SBSersicImpl::cache(MAX_SERSIC_INFO);

    SBSersic::SBSersicImpl::SBSersicImpl(double n,  double re, double flux,
                                         double trunc, bool flux_untruncated,
                                         boost::shared_ptr<GSParams> gsparams) :
        SBProfileImpl(gsparams),
        _n(n), _flux(flux), _re(re), _re_sq(_re*_re),
        _inv_re(1./_re), _inv_re_sq(_inv_re*_inv_re),
        _norm(_flux*_inv_re_sq), 
        _trunc(trunc), _flux_untruncated(flux_untruncated),
        _maxRre((int)(_trunc/_re * 100 + 0.5) / 100.0),  // round to two decimal places
        _info(cache.get(std::make_pair(SersicKey(_n,_maxRre,_flux_untruncated),
                                       this->gsparams.get())))
    {
        _truncated = (_trunc > 0.);
        if (!_truncated) _flux_untruncated = true;  // set unused parameter to a single value
        _actual_flux = _info->getTrueFluxFraction() * _flux;
        _actual_re = _info->getTrueReFraction() * _re;
        _maxRre_sq = _maxRre*_maxRre;
        _maxR = _maxRre * _re;
        _maxR_sq = _maxR*_maxR;
        _ksq_max = _info->getKsqMax();
        dbg<<"_ksq_max for n = "<<n<<" = "<<_ksq_max<<std::endl;
    }

    double SBSersic::SBSersicImpl::xValue(const Position<double>& p) const
    {
        double rsq = (p.x*p.x+p.y*p.y)*_inv_re_sq;
        if (_truncated && rsq > _maxRre_sq) return 0.;
        else return _norm * _info->xValue(rsq);
    }

    std::complex<double> SBSersic::SBSersicImpl::kValue(const Position<double>& k) const
    {
        double ksq = (k.x*k.x + k.y*k.y)*_re_sq;
        if (ksq > _ksq_max) return 0.;
        else return _flux * _info->kValue(ksq);
    }

    void SBSersic::SBSersicImpl::fillXValue(tmv::MatrixView<double> val,
                                            double x0, double dx, int ix_zero,
                                            double y0, double dy, int iy_zero) const
    {
        dbg<<"SBSersic fillXValue\n";
        dbg<<"x = "<<x0<<" + ix * "<<dx<<", ix_zero = "<<ix_zero<<std::endl;
        dbg<<"y = "<<y0<<" + iy * "<<dy<<", iy_zero = "<<iy_zero<<std::endl;
        if (ix_zero != 0 || iy_zero != 0) {
            xdbg<<"Use Quadrant\n";
            fillXValueQuadrant(val,x0,dx,ix_zero,y0,dy,iy_zero);
        } else {
            xdbg<<"Non-Quadrant\n";
            assert(val.stepi() == 1);
            const int m = val.colsize();
            const int n = val.rowsize();
            typedef tmv::VIt<double,1,tmv::NonConj> It;

            x0 *= _inv_re;
            dx *= _inv_re;
            y0 *= _inv_re;
            dy *= _inv_re;

            for (int j=0;j<n;++j,y0+=dy) {
                double x = x0;
                double ysq = y0*y0;
                It valit = val.col(j).begin();
                for (int i=0;i<m;++i,x+=dx) {
                    double rsq = x*x + ysq;
                    if (_truncated && rsq > _maxRre_sq) *valit++ = 0.;
                    else *valit++ = _norm * _info->xValue(rsq);
                }
            }
        }
    }

    void SBSersic::SBSersicImpl::fillKValue(tmv::MatrixView<std::complex<double> > val,
                                            double x0, double dx, int ix_zero,
                                            double y0, double dy, int iy_zero) const
    {
        dbg<<"SBSersic fillKValue\n";
        dbg<<"x = "<<x0<<" + ix * "<<dx<<", ix_zero = "<<ix_zero<<std::endl;
        dbg<<"y = "<<y0<<" + iy * "<<dy<<", iy_zero = "<<iy_zero<<std::endl;
        if (ix_zero != 0 || iy_zero != 0) {
            xdbg<<"Use Quadrant\n";
            fillKValueQuadrant(val,x0,dx,ix_zero,y0,dy,iy_zero);
        } else {
            xdbg<<"Non-Quadrant\n";
            assert(val.stepi() == 1);
            const int m = val.colsize();
            const int n = val.rowsize();
            typedef tmv::VIt<std::complex<double>,1,tmv::NonConj> It;

            x0 *= _re;
            dx *= _re;
            y0 *= _re;
            dy *= _re;

            for (int j=0;j<n;++j,y0+=dy) {
                double x = x0;
                double ysq = y0*y0;
                It valit(val.col(j).begin().getP(),1);
                for (int i=0;i<m;++i,x+=dx) {
                    double ksq = x*x + ysq;
                    if (ksq > _ksq_max) *valit++ = 0.;
                    else *valit++ = _flux * _info->kValue(ksq);
                }
            }
        }
    }

    void SBSersic::SBSersicImpl::fillXValue(tmv::MatrixView<double> val,
                                            double x0, double dx, double dxy,
                                            double y0, double dy, double dyx) const
    {
        dbg<<"SBSersic fillXValue\n";
        dbg<<"x = "<<x0<<" + ix * "<<dx<<" + iy * "<<dxy<<std::endl;
        dbg<<"y = "<<y0<<" + ix * "<<dyx<<" + iy * "<<dy<<std::endl;
        assert(val.stepi() == 1);
        assert(val.canLinearize());
        const int m = val.colsize();
        const int n = val.rowsize();
        typedef tmv::VIt<double,1,tmv::NonConj> It;

        x0 *= _inv_re;
        dx *= _inv_re;
        dxy *= _inv_re;
        y0 *= _inv_re;
        dy *= _inv_re;
        dyx *= _inv_re;

        It valit = val.linearView().begin();
        for (int j=0;j<n;++j,x0+=dxy,y0+=dy) {
            double x = x0;
            double y = y0;
            It valit = val.col(j).begin();
            for (int i=0;i<m;++i,x+=dx,y+=dyx) {
                double rsq = x*x + y*y;
                if (_truncated && rsq > _maxRre_sq) *valit++ = 0.;
                else *valit++ = _norm * _info->xValue(rsq);
            }
        }
    }

    void SBSersic::SBSersicImpl::fillKValue(tmv::MatrixView<std::complex<double> > val,
                                            double x0, double dx, double dxy,
                                            double y0, double dy, double dyx) const
    {
        dbg<<"SBSersic fillKValue\n";
        dbg<<"x = "<<x0<<" + ix * "<<dx<<" + iy * "<<dxy<<std::endl;
        dbg<<"y = "<<y0<<" + ix * "<<dyx<<" + iy * "<<dy<<std::endl;
        assert(val.stepi() == 1);
        assert(val.canLinearize());
        const int m = val.colsize();
        const int n = val.rowsize();
        typedef tmv::VIt<std::complex<double>,1,tmv::NonConj> It;

        x0 *= _re;
        dx *= _re;
        dxy *= _re;
        y0 *= _re;
        dy *= _re;
        dyx *= _re;

        It valit(val.linearView().begin().getP(),1);
        for (int j=0;j<n;++j,x0+=dxy,y0+=dy) {
            double x = x0;
            double y = y0;
            It valit(val.col(j).begin().getP(),1);
            for (int i=0;i<m;++i,x+=dx,y+=dyx) {
                double ksq = x*x + y*y;
                if (ksq > _ksq_max) *valit++ = 0.;
                else *valit++ = _flux * _info->kValue(ksq);
            }
        }
    }

    double SBSersic::SBSersicImpl::maxK() const { return _info->maxK() / _re; }
    double SBSersic::SBSersicImpl::stepK() const { return _info->stepK() / _re; }

    double SersicInfo::xValue(double xsq) const 
    {
        if (_truncated && xsq > _maxRre_sq) return 0.;
        else return _norm * std::exp(-_b*std::pow(xsq,_inv2n));
    }

    double SersicInfo::kValue(double ksq) const 
    {
        // TODO: Use asymptotic formula for high-k?

        assert(ksq >= 0.);

        if (ksq>=_ksq_max)
            return 0.; // truncate the Fourier transform
        else if (ksq<_ksq_min)
            return 1. + ksq*(_kderiv2 + ksq*_kderiv4); // Use quartic approx at low k
        else {
            double lk=0.5*std::log(ksq); // Lookup table is logarithmic
            return _ft(lk);
        }
    }

    // Integrand class for the Hankel transform of Sersic
    class SersicIntegrand : public std::unary_function<double,double>
    {
    public:
        SersicIntegrand(double n, double b, double k):
            _invn(1./n), _b(b), _k(k) {}
        double operator()(double r) const 
        { return r*std::exp(-_b*std::pow(r, _invn))*j0(_k*r); }

    private:
        double _invn;
        double _b;
        double _k;
    };

    class SersicMissingFluxFunc
    {
    public:
        SersicMissingFluxFunc(double n, double b, double missing_flux) :
            _2n(2*n), _invn(1./n), _b(b), _missing_flux(missing_flux) {}
        double operator()(double Rre) const
        {
            double z = _b * std::pow(Rre, _invn);
            double fre = boost::math::tgamma(_2n, z);  // integrates the tail from z to inf
            xdbg<<"func("<<_b<<") = "<<fre<<" - "<<_missing_flux<<" = "<<fre-_missing_flux
                <<std::endl;
            return fre - _missing_flux;
        }
    private:
        double _2n;
        double _invn;
        double _b;
        double _missing_flux;
    };

    // Find what radius encloses (1-missing_flux_frac) of the total flux in a Sersic profile,
    // in units of half-light radius re.
    double SersicInfo::findMaxRre(double missing_flux_frac, double gamma2n)
    { 
        // int(exp(-b r^1/n) r, r=R..inf) = x * int(exp(-b r^1/n) r, r=0..inf)
        //                                = x n b^-2n Gamma(2n)    [x == missing_flux_frac]
        // (1) First, find approximate solution to x * Gamma(2n) = Gamma(2n, b*(R/re)^(1/n))
        // Change variables: u = b r^1/n,
        // du = b/n r^(1-n)/n dr
        //    = b/n r^1/n dr/r
        //    = u/n dr/r
        // r dr = n du r^2 / u
        //      = n du (u/b)^2n / u
        // n b^-2n int(u^(2n-1) exp(-u), u=bR^1/n..inf) = x n b^-2n Gamma(2n)
        // Let z = b R^1/n
        //
        // int(u^(2n-1) exp(-u), u=z..inf) = x Gamma(2n)
        //
        // The lhs is an incomplete gamma function: Gamma(2n,z), which according to
        // Abramowitz & Stegun (6.5.32) has a high-z asymptotic form of:
        // Gamma(2n,z) ~= z^(2n-1) exp(-z) (1 + (2n-1)/z + (2n-1)(2n-2)/z^2 + ... )
        // ln(x Gamma(2n)) = (2n-1) ln(z) - z + (2n-1)/z + (2n-1)(2n-3)/(2*z^2) + O(z^3)
        // z = -ln(x Gamma(2n) + (2n-1) ln(z) + (2n-1)/z + (2n-1)(2n-3)/(2*z^2) + O(z^3)
        // Iterate this until it converges.  Should be quick.
        dbg<<"Find maxR for missing_flux_frac = "<<missing_flux_frac<<std::endl;
        double missing_flux = missing_flux_frac * gamma2n;
        double z0 = -std::log(missing_flux);
        // Successive approximation method:
        double z = 4.*(_n+1.);  // A decent starting guess for a range of n.
        double oldz = 0.;
        const int MAXIT = 15;
        xdbg<<"Start with z = "<<z<<std::endl;
        for(int niter=0; niter < MAXIT; ++niter) {
            oldz = z;
            z = z0 + (2.*_n-1.) * std::log(z) + (2.*_n-1.)/z + (2.*_n-1.)*(2.*_n-3.)/(2.*z*z);
            xxdbg<<"z = "<<z<<", dz = "<<z-oldz<<std::endl;
            if (std::abs(z-oldz) < 0.01) break;
        }
        xdbg<<"Converged at z = "<<z<<std::endl;
        double Rre_estimate = std::pow(z/_b, _n);
        xdbg<<"Rre_estimate = (z/b)^n = "<<Rre_estimate<<std::endl;

        // (2) Now find a more exact solution using an iterative solver
        SersicMissingFluxFunc func(_n, _b, missing_flux);
        // Start with the approximate solution Rre_estimate, and create bounds at its vicinities
        double b1 = Rre_estimate * 1.1;  // solution usually within 1% for small n
        xdbg<<"b1 = "<<b1<<" ..";
        double b2 = Rre_estimate * 0.9;
        xdbg<<".. b2 = "<<b2<<std::endl;
        Solve<SersicMissingFluxFunc> solver(func,b1,b2);
        solver.setMethod(Brent);
        double Rre = solver.root();
        dbg<<"maxRre is "<<Rre<<std::endl;

        return Rre;
    }

    class SersicScaleRadiusFunc
    {
    public:
        SersicScaleRadiusFunc(double n, double maxRre) :
            _2n(2.*n), _rinvn(std::pow(maxRre, 1./n)) {}
        double operator()(double b) const
        {
            double fre = boost::math::tgamma_lower(_2n, b);
            double frm;
            if (_rinvn == 0.)  frm = boost::math::tgamma(_2n);
            else {
                double z = b * _rinvn;
                frm = boost::math::tgamma_lower(_2n, z);
            }
            xxdbg<<"func("<<b<<") = 2*"<<fre<<" - "<<frm<<" = "<<2.*fre-frm<<std::endl;
            return 2.*fre - frm;
        }
    private:
        double _2n;
        double _rinvn;
    };

    // Calculate Sersic scale b, given half-light radius re and truncation radius trunc
    // Sersic scale in Sersic profile is defined from exp(-b*r^(1/n))
    double SersicCalculateScaleBFromHLR(double n, double maxRre)
    {
        // Find Sersic scale b.
        // Total flux in a Sersic profile is:  F = I0*re^2 (2pi*n) b^(-2n) Gamma(2n)
        // Flux within radius R is:  F(<R) = I0*re^2 (2pi*n) b^(-2n) gamma(2n, b*(R/re)^(1/n))
        // where Gamma(a) == int_0^inf exp(-t) t^(a-1) dt
        //       gamma(a,x) == int_0^x exp(-t) t^(a-1) dt
        // Find solution to gamma(2n,b) = Gamma(2n)/2 for the untruncated case;
        //                  gamma(2n,b) = Gamma(2n,b*(trunc/re)^(1/n)) / 2, for the truncated case.
        if (maxRre!=0. && maxRre <= 1.)   // TODO: find a better minimum truncation check criteria.
            throw SBError("Sersic truncation radius must be > half_light_radius.");
        dbg<<"Find Sersic scale b for maxR/HLR = "<<maxRre<<std::endl;
        SersicScaleRadiusFunc func(n, maxRre);
        // For the upper bound of b, use 2*n, based on the Ciotti & Bertin (1999) approximate 
        // solution for untruncated Sersic; for the lower bound, we don't really have a good choice,
        // so start with half the upper bound, and we'll expand it if necessary.
        double b1 = n;
        xdbg<<"b1 = "<<b1<<std::endl;
        double b2 = 2*n;    // approximate soluton is ~2n-1/3, so 2n is a fair upper limit
        xdbg<<"b2 = "<<b2<<std::endl;
        Solve<SersicScaleRadiusFunc> solver(func,b1,b2);
        solver.setMethod(Brent);
        solver.bracketLower();    // expand lower bracket if necessary
        xdbg<<"After bracket, range is "<<solver.getLowerBound()<<" .. "<<
            solver.getUpperBound()<<std::endl;
        double b = solver.root();
        dbg<<"Root is "<<b<<std::endl;

        return b;
    }

    class SersicHalfLightRadiusFunc
    {
    public:
        SersicHalfLightRadiusFunc(double n, double b, double gamma2n) :
            _2n(2.*n), _invn(1./n), _b(b), _half_gamma2n(gamma2n/2.) {}
        double operator()(double hlr) const
        {
            double fre = boost::math::tgamma_lower(_2n, _b*std::pow(hlr,_invn));
            dbg<<"func("<<hlr<<") = "<<fre<<"-"<<_half_gamma2n<<" = "<<fre-_half_gamma2n<<std::endl;
            return fre - _half_gamma2n;
        }
    private:
        double _2n;
        double _invn;
        double _b;
        double _half_gamma2n;
    };

    // Calculate half-light radius scale, given Sersic `n`, `b` and total flux equivalent `gamma2n`
    double SersicCalculateHLRScale(double n, double b, double gamma2n)
    {
        // Find solution to gamma(2n,b*hlr^(1/n)) = gamma2n / 2
        // where gamma2n is the truncated gamma function Gamma(2n,b*(maxRre)^(1/n))
        //
        dbg<<"Find HLR scale for (n,b,gamma2n) = ("<<n<<","<<b<<","<<gamma2n<<")"<<std::endl;
        SersicHalfLightRadiusFunc func(n, b, gamma2n);
        // For the upper bound of b, use 1 (i.e., the specified half-light radius); the true half-
        // light radius will always be smaller for a truncated Sersic.  For the lower bound, we
        // don't really have a good choice, so start with half the upper bound, and we'll expand it
        // if necessary.
        double b1 = 1.;
        dbg<<"b1 = "<<b1<<std::endl;
        double b2 = 0.5;
        dbg<<"b2 = "<<b2<<std::endl;
        Solve<SersicHalfLightRadiusFunc> solver(func,b1,b2);
        solver.setMethod(Brent);
        solver.bracketLower();    // expand lower bracket if necessary
        dbg<<"After bracket, range is "<<solver.getLowerBound()<<" .. "<<
            solver.getUpperBound()<<std::endl;
        double hlr = solver.root();
        dbg<<"Root is "<<hlr<<std::endl;

        return hlr;
    }

    // Constructor to initialize Sersic constants and k lookup table
    SersicInfo::SersicInfo(const SersicKey& key, const GSParams* gsparams) :
        _n(key.n), _maxRre(key.maxRre), _maxRre_sq(_maxRre*_maxRre), _inv2n(1./(2.*_n)),
        _flux_untruncated(key.flux_untruncated), _flux_fraction(1.), _re_fraction(1.)
    {
        // Going to constrain range of allowed n to those for which testing was done
        // (Lower bounds has hard limit at ~0.29)
        if (_n<0.3 || _n>4.2) throw SBError("Requested Sersic index out of range");

        _truncated = (_maxRre > 0.);

        if ( _truncated && _flux_untruncated ) {
            dbg << "Calculating b with maxR/re => 0 (inf)" << std::endl;
            _b = SersicCalculateScaleBFromHLR(_n, 0.);
        }
        else {
            dbg << "Calculating b with maxR/re => " << _maxRre << std::endl;
            _b = SersicCalculateScaleBFromHLR(_n, _maxRre);
        }

        // set-up frequently used numbers (for flux normalization and FT small-k approximations)
        double b2n = std::pow(_b,2.*_n);
        double b4n = b2n*b2n;
        double gamma2n;
        double gamma4n;
        double gamma6n;
        double gamma8n;
        if (!_truncated) {
            gamma2n = boost::math::tgamma(2.*_n);  // integrate r/re from 0. to inf
            gamma4n = boost::math::tgamma(4.*_n);
            gamma6n = boost::math::tgamma(6.*_n);
            gamma8n = boost::math::tgamma(8.*_n);
        } else {
            double z = _b * std::pow(_maxRre, 1./_n);
            gamma2n = boost::math::tgamma_lower(2.*_n, z);  // integrate r/re from 0. to _maxRre
            gamma4n = boost::math::tgamma_lower(4.*_n, z);
            gamma6n = boost::math::tgamma_lower(6.*_n, z);
            gamma8n = boost::math::tgamma_lower(8.*_n, z);
        }

        // Find the ratio of actual (truncated) flux to specified flux
        if (_truncated && _flux_untruncated) {
            _flux_fraction = gamma2n / boost::math::tgamma(2.*_n);       // _flux_fraction < 1
            _re_fraction = SersicCalculateHLRScale(_n,_b,gamma2n);
        }
        dbg << "Flux fraction: " << _flux_fraction << std::endl;
        dbg << "HLR fraction: " << _re_fraction << std::endl;
        dbg << "gamma2n: " << gamma2n << std::endl;

        // The normalization factor to give unity flux integral:
        _norm = b2n / (2.*M_PI*_n*gamma2n) * _flux_fraction;

        // The small-k expansion of the Hankel transform is (normalized to have flux=1):
        // 1 - Gamma(4n) / 4 b^2n Gamma(2n) + Gamma(6n) / 64 b^4n Gamma(2n)
        //   - Gamma(8n) / 2304 b^6n Gamma(2n)
        // from the series summation J_0(x) = Sum^inf_{m=0} (-1)^m (m!)^-2 (x/2)^2m
        // The quadratic term of small-k expansion:
        _kderiv2 = -gamma4n / (4.*b2n*gamma2n);
        // And a quartic term:
        _kderiv4 = gamma6n / (64.*b4n*gamma2n);

        dbg << "Building for n=" << _n << " b=" << _b << " norm=" << _norm
            << " maxRre=" << _maxRre << std::endl;
        dbg << "Deriv terms: " << _kderiv2 << " " << _kderiv4 << std::endl;

        // When is it safe to use low-k approximation?  
        // See when next term past quartic is at accuracy threshold
        double kderiv6 = gamma8n / (2304.*b4n*b2n*gamma2n);
        dbg<<"kderiv6 = "<<kderiv6<<std::endl;
        double kmin = std::pow(gsparams->kvalue_accuracy / kderiv6, 1./6.);
        dbg<<"kmin = "<<kmin<<std::endl;
        _ksq_min = kmin * kmin;

        // How far should the profile extend, if not truncated?
        // Estimate number of effective radii needed to enclose (1-alias_threshold) of flux
        double Rre = findMaxRre(gsparams->alias_threshold,gamma2n);
        if (_truncated && _maxRre < Rre)  Rre = _maxRre;
        // Go to at least 5*re
        if (Rre < 5.) Rre = 5.;
        dbg<<"maxR/re => "<<_maxRre<<std::endl;
        _stepK = M_PI / Rre;
        dbg<<"stepK = "<<_stepK<<std::endl;

        // Now start building the lookup table for FT of the profile.

        // Normalization for integral at k=0:
        double hankel_norm = _n*gamma2n/b2n / _flux_fraction;
        dbg<<"hankel_norm = "<<hankel_norm<<std::endl;

        // Keep going until at least 5 in a row have kvalues below kvalue_accuracy.
        int n_below_thresh = 0;

        double integ_maxRre;
        if (!_truncated)
            integ_maxRre = findMaxRre(gsparams->kvalue_accuracy * hankel_norm,gamma2n);
        else
            integ_maxRre = _maxRre;

        // There are two "max k" values that we care about.
        // 1) _maxK is where |f| <= maxk_threshold
        // 2) _ksq_max is where |f| <= kvalue_accuracy
        // The two thresholds are typically different, since they are used in different ways.
        // We keep track of maxlogk_1 and maxlogk_2 to keep track of each of these.
        double maxlogk_1 = 0.;
        double maxlogk_2 = 0.;

        double dlogk = 0.1;
        // Don't go past k = 500
        for (double logk = std::log(kmin)-0.001; logk < std::log(500.); logk += dlogk) {
            SersicIntegrand I(_n, _b, std::exp(logk));
            double val = integ::int1d(
                I, 0., integ_maxRre, 
                gsparams->integration_relerr, gsparams->integration_abserr*hankel_norm);
            val /= hankel_norm;
            xdbg<<"logk = "<<logk<<", ft("<<exp(logk)<<") = "<<val<<std::endl;
            _ft.addEntry(logk,val);

            if (std::abs(val) > gsparams->maxk_threshold) maxlogk_1 = logk;
            if (std::abs(val) > gsparams->kvalue_accuracy) maxlogk_2 = logk;

            if (std::abs(val) > gsparams->kvalue_accuracy) n_below_thresh = 0;
            else ++n_below_thresh;
            if (n_below_thresh == 5) break;
        }
        // These marked the last value that didn't satisfy our requirement, so just go to 
        // the next value.
        maxlogk_1 += dlogk;
        maxlogk_2 += dlogk;
        _maxK = exp(maxlogk_1);
        xdbg<<"maxlogk_1 = "<<maxlogk_1<<std::endl;
        xdbg<<"maxK with val >= "<<gsparams->maxk_threshold<<" = "<<_maxK<<std::endl;
        _ksq_max = exp(2.*maxlogk_2);
        xdbg<<"ft.argMax = "<<_ft.argMax()<<std::endl;
        xdbg<<"maxlogk_2 = "<<maxlogk_2<<std::endl;
        xdbg<<"ksq_max = "<<_ksq_max<<std::endl;

        // Next, set up the classes for photon shooting
        _radial.reset(new SersicRadialFunction(_n, _b));
        std::vector<double> range(2,0.);
        if (!_truncated)
            range[1] = findMaxRre(gsparams->shoot_accuracy,gamma2n);
        else
            range[1] = _maxRre;
        _sampler.reset(new OneDimensionalDeviate( *_radial, range, true));
    }

    boost::shared_ptr<PhotonArray> SersicInfo::shoot(int N, UniformDeviate ud) const
    {
        dbg<<"SersicInfo shoot: N = "<<N<<std::endl;
        dbg<<"Target flux = 1.0\n";
        assert(_sampler.get());
        boost::shared_ptr<PhotonArray> result = _sampler->shoot(N,ud);
        result->scaleFlux(_norm);
        dbg<<"SersicInfo Realized flux = "<<result->getTotalFlux()<<std::endl;
        return result;
    }

    boost::shared_ptr<PhotonArray> SBSersic::SBSersicImpl::shoot(int N, UniformDeviate ud) const
    {
        dbg<<"Sersic shoot: N = "<<N<<std::endl;
        dbg<<"Target flux = "<<getFlux()<<std::endl;
        // Get photons from the SersicInfo structure, rescale flux and size for this instance
        boost::shared_ptr<PhotonArray> result = _info->shoot(N,ud);
        result->scaleFlux(_flux);
        result->scaleXY(_re);
        dbg<<"Sersic Realized flux = "<<result->getTotalFlux()<<std::endl;
        return result;
    }
}
