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

// clang doesn't like some of the code in boost files included by gamma.hpp.
#ifdef __clang__
#if __has_warning("-Wlogical-op-parentheses")
#pragma GCC diagnostic ignored "-Wlogical-op-parentheses"
#endif
#endif

#ifndef __INTEL_COMPILER
#if defined(__GNUC__) && __GNUC__ >= 4 && (__GNUC__ >= 5 || __GNUC_MINOR__ >= 8)
#pragma GCC diagnostic ignored "-Wunused-local-typedefs"
#endif
#endif

#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/special_functions/bessel.hpp>

#include "SBSersic.h"
#include "SBSersicImpl.h"
#include "integ/Int.h"
#include "Solve.h"
#include "bessel/Roots.h"

#ifdef DEBUGLOGGING
#include <fstream>
//std::ostream* dbgout = new std::ofstream("debug.out");
//std::ostream* dbgout = &std::cout;
//int verbose_level = 1;
#endif

namespace galsim {

    SBSersic::SBSersic(double n, double size, RadiusType rType, double flux,
                       double trunc, bool flux_untruncated, const GSParamsPtr& gsparams) :
        SBProfile(new SBSersicImpl(n, size, rType, flux, trunc, flux_untruncated, gsparams)) {}

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

    LRUCache< boost::tuple<double, double, GSParamsPtr >, SersicInfo > 
        SBSersic::SBSersicImpl::cache(sbp::max_sersic_cache);

    SBSersic::SBSersicImpl::SBSersicImpl(double n,  double size, RadiusType rType, double flux,
                                         double trunc, bool flux_untruncated,
                                         const GSParamsPtr& gsparams) :
        SBProfileImpl(gsparams),
        _n(n), _flux(flux), _trunc(trunc), 
        // Start with untruncated SersicInfo regardless of value of trunc
        _info(cache.get(boost::make_tuple(_n, 0., this->gsparams)))
    {
        dbg<<"Start SBSersic constructor:\n";
        dbg<<"n = "<<_n<<std::endl;
        dbg<<"size = "<<size<<"  rType = "<<rType<<std::endl;
        dbg<<"flux = "<<_flux<<std::endl;
        dbg<<"trunc = "<<_trunc<<"  flux_untruncated = "<<flux_untruncated<<std::endl;

        _truncated = (_trunc > 0.);

        // Set size of this instance according to type of size given in constructor
        switch (rType) {
          case HALF_LIGHT_RADIUS:
               {
                   _re = size;
                   if (_truncated) {
                       if (flux_untruncated) {
                           // Then given HLR and flux are the values for the untruncated profile.
                           _r0 = _re / _info->getHLR(); // getHLR() is in units of r0.
                       } else {
                           // This is the one case that is a bit complicated, since the 
                           // half-light radius and trunc are both given in physical units, 
                           // so we need to solve for what scale radius this corresponds to.
                           _r0 = _info->calculateScaleForTruncatedHLR(_re, _trunc);
                       }

                       // Update _info with the correct truncated version.
                       _info = cache.get(boost::make_tuple(_n,_trunc/_r0,this->gsparams));

                       if (flux_untruncated) {
                           // Update the stored _flux and _re with the correct values
                           _flux *= _info->getFluxFraction();
                           _re = _r0 * _info->getHLR();
                       }
                   } else {
                       // Then given HLR and flux are the values for the untruncated profile.
                       _r0 = _re / _info->getHLR();
                   }
               }
               break;
          case SCALE_RADIUS:
               {
                   _r0 = size;
                   if (_truncated) {
                       // Update _info with the correct truncated version.
                       _info = cache.get(boost::make_tuple(_n,_trunc/_r0,this->gsparams));

                       if (flux_untruncated) {
                           // Update the stored _flux with the correct value
                           _flux *= _info->getFluxFraction();
                       }
                   }
                   // In all cases, _re is the real HLR
                   _re = _r0 * _info->getHLR();
               }
               break;
          default:
               throw SBError("Unknown SBSersic::RadiusType");
        }
        dbg<<"hlr = "<<_re<<std::endl;
        dbg<<"r0 = "<<_r0<<std::endl;

        _r0_sq = _r0*_r0;
        _inv_r0 = 1./_r0;
        _inv_r0_sq = _inv_r0*_inv_r0;

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

            x0 *= _inv_r0;
            dx *= _inv_r0;
            y0 *= _inv_r0;
            dy *= _inv_r0;

            for (int j=0;j<n;++j,y0+=dy) {
                double x = x0;
                double ysq = y0*y0;
                It valit = val.col(j).begin();
                for (int i=0;i<m;++i,x+=dx) {
                    double rsq = x*x + ysq;
                    *valit++ = _xnorm * _info->xValue(rsq);
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

            x0 *= _r0;
            dx *= _r0;
            y0 *= _r0;
            dy *= _r0;

            for (int j=0;j<n;++j,y0+=dy) {
                double x = x0;
                double ysq = y0*y0;
                It valit(val.col(j).begin().getP(),1);
                for (int i=0;i<m;++i,x+=dx) {
                    double ksq = x*x + ysq;
                    *valit++ = _flux * _info->kValue(ksq);
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

        x0 *= _inv_r0;
        dx *= _inv_r0;
        dxy *= _inv_r0;
        y0 *= _inv_r0;
        dy *= _inv_r0;
        dyx *= _inv_r0;

        It valit = val.linearView().begin();
        for (int j=0;j<n;++j,x0+=dxy,y0+=dy) {
            double x = x0;
            double y = y0;
            It valit = val.col(j).begin();
            for (int i=0;i<m;++i,x+=dx,y+=dyx) {
                double rsq = x*x + y*y;
                *valit++ = _xnorm * _info->xValue(rsq);
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

        x0 *= _r0;
        dx *= _r0;
        dxy *= _r0;
        y0 *= _r0;
        dy *= _r0;
        dyx *= _r0;

        It valit(val.linearView().begin().getP(),1);
        for (int j=0;j<n;++j,x0+=dxy,y0+=dy) {
            double x = x0;
            double y = y0;
            It valit(val.col(j).begin().getP(),1);
            for (int i=0;i<m;++i,x+=dx,y+=dyx) {
                double ksq = x*x + y*y;
                *valit++ = _flux * _info->kValue(ksq);
            }
        }
    }

    double SBSersic::SBSersicImpl::maxK() const { return _info->maxK() * _inv_r0; }
    double SBSersic::SBSersicImpl::stepK() const { return _info->stepK() * _inv_r0; }

    SersicInfo::SersicInfo(double n, double trunc, const GSParamsPtr& gsparams) :
        _n(n), _trunc(trunc), _gsparams(gsparams),
        _invn(1./_n), _inv2n(0.5*_invn), 
        _trunc_sq(_trunc*_trunc), _truncated(_trunc > 0.),
        _gamma2n(boost::math::tgamma(2.*_n)),
        _maxk(0.), _stepk(0.), _re(0.), _flux(0.), 
        _ft(Table<double,double>::spline)
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
            // Estimate number of effective radii needed to enclose (1-alias_threshold) of flux
            double R = calculateMissingFluxRadius(_gsparams->alias_threshold);
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
 
    double SersicInfo::getFluxFraction() const
    {
        if (_flux == 0.) {
            // Calculate the flux of a truncated profile (relative to the integral for 
            // an untruncated profile).
            if (_truncated) {
                double z = std::pow(_trunc, 1./_n);
                // integrate from 0. to _trunc
                double gamma2nz = boost::math::tgamma_lower(2.*_n, z);
                _flux = gamma2nz / _gamma2n;  // _flux < 1
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
        else return std::exp(-std::pow(rsq,_inv2n));
    }

    double SersicInfo::kValue(double ksq) const 
    {
        assert(ksq >= 0.);
        if (_ft.size() == 0) buildFT();

        if (ksq>=_ksq_max)
            return (_highk_a + _highk_b/sqrt(ksq))/ksq; // high-k asymptote
        else if (ksq<_ksq_min)
            return 1. + ksq*(_kderiv2 + ksq*_kderiv4); // Use quartic approx at low k
        else {
            double lk=0.5*std::log(ksq); // Lookup table is logarithmic
            return _ft(lk)/ksq;
        }
    }

    // Integrand class for the Hankel transform of Sersic
    class SersicHankel : public std::unary_function<double,double>
    {
    public:
        SersicHankel(double invn, double k): _invn(invn), _k(k) {}

        double operator()(double r) const 
        { return r*std::exp(-std::pow(r, _invn))*j0(_k*r); }

    private:
        double _invn;
        double _k;
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
            gamma4n = boost::math::tgamma(4.*_n);
            gamma6n = boost::math::tgamma(6.*_n);
            gamma8n = boost::math::tgamma(8.*_n);
        } else {
            double z = std::pow(_trunc, 1./_n);
            gamma4n = boost::math::tgamma_lower(4.*_n, z);
            gamma6n = boost::math::tgamma_lower(6.*_n, z);
            gamma8n = boost::math::tgamma_lower(8.*_n, z);
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

        double integ_maxr;
        if (!_truncated) {
            //integ_maxr = calculateMissingFluxRadius(_gsparams->kvalue_accuracy);
            integ_maxr = integ::MOCK_INF;
        } else {
            //integ_maxr = calculateMissingFluxRadius(_gsparams->kvalue_accuracy);
            //if (_trunc < integ_maxr) integ_maxr = _trunc;
            integ_maxr = _trunc;
        }
        dbg<<"integ_maxr = "<<integ_maxr<<std::endl;

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
        bool found_maxk = false;
        for (double logk = std::log(kmin)-0.001; logk < std::log(500.); logk += dlogk) {
            double k = std::exp(logk);
            double ksq = k*k;
            SersicHankel I(_invn, k);

#ifdef DEBUGLOGGING
            std::ostream* integ_dbgout = verbose_level >= 3 ? dbgout : 0;
            integ::IntRegion<double> reg(0, integ_maxr, integ_dbgout);
#else
            integ::IntRegion<double> reg(0, integ_maxr);
#endif

            // Add explicit splits at first several roots of J0.
            // This tends to make the integral more accurate.
            for (int s=1; s<=10; ++s) {
                double root = bessel::getBesselRoot0(s);
                if (root > k * integ_maxr) break;
                reg.addSplit(root/k);
            }

            double val = integ::int1d(I, reg,
                                      _gsparams->integration_relerr,
                                      _gsparams->integration_abserr*hankel_norm);
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
                double k_back = std::exp(logk - n_fit*dlogk);
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
        // If didn't find a good approximation for large k, just use the largest k we put in
        // in the table.  (Need to use some approximation after this anyway!)
        if (_ksq_max <= 0.) _ksq_max = std::exp(2. * _ft.argMax());
        xdbg<<"ft.argMax = "<<_ft.argMax()<<std::endl;
        xdbg<<"ksq_max = "<<_ksq_max<<std::endl;

        if (found_maxk) {
            // This is the last value that didn't satisfy the requirement, so just go to 
            // the next value.
            xdbg<<"maxk with val > "<<_gsparams->maxk_threshold<<" = "<<_maxk<<std::endl;
            _maxk *= exp(dlogk);
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
            double f = boost::math::tgamma(_2n, z);  // integrates the tail from z to inf
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

    void SersicInfo::calculateHLR() const
    {
        dbg<<"Find HLR for (n,gamma2n) = ("<<_n<<","<<_gamma2n<<")"<<std::endl;
        // Find solution to gamma(2n,re^(1/n)) = gamma2n / 2
        // where gamma2n is the truncated gamma function Gamma(2n,trunc^(1/n))
        // We initially solve for b = re^1/n, and then calculate re from that.
        // Start with the approximation from Ciotti & Bertin, 1999:
        // b ~= 2n - 1/3 + 4/(405n) + 46/(25515n^2) + 131/(1148175n^3) - ...
        // Then we use a non-linear solver to tweak it up.
        double invnsq = _invn*_invn;
        double b1 = 2.*_n-1./3.;
        double b2 = b1 + (8./405.)*_invn + (46./25515.)*invnsq + (131./1148175.)*_invn*invnsq;
        // Note: This is the value if the profile is untruncated.  It will be smaller if 
        // the profile is actually truncated and _gamma2n < Gamma(2n)

        SersicMissingFlux func(_n, (1. - 0.5*getFluxFraction())*_gamma2n);
        Solve<SersicMissingFlux> solver(func,b1,b2);
        xdbg<<"Initial range is "<<b1<<" .. "<<b2<<std::endl;
        solver.setMethod(Brent);
        solver.bracketLowerWithLimit(0.);    // expand lower bracket if necessary
        xdbg<<"After bracket, range is "<<solver.getLowerBound()<<" .. "<<
            solver.getUpperBound()<<std::endl;
        // We store b in case we need it again for calculateScaleForTruncatedHLR(), so we 
        // can save a pow call.
        _b = solver.root();
        dbg<<"Root is "<<_b<<std::endl;

        // re = b^n
        _re = std::pow(_b,_n);
        dbg<<"re is "<<_re<<std::endl;
    }

    // Function object for finding the r that encloses all except a particular flux fraction.
    class SersicTruncatedHLR
    {
    public:
        // x = (trunc/re)^1/n
        SersicTruncatedHLR(double n, double x) : _2n(2.*n), _x(x) {}

        double operator()(double b) const 
        {
            double f1 = boost::math::tgamma_lower(_2n, b);
            double f2 = boost::math::tgamma_lower(_2n, _x*b);
            // Solve for f1 = f2/2
            xdbg<<"func("<<b<<") = 2*"<<f1<<" - "<<f2<<" = "<< 2.*f1-f2<<std::endl;
            return 2.*f1-f2;
        }
    private:
        double _2n;
        double _x;
    };

    double SersicInfo::calculateScaleForTruncatedHLR(double re, double trunc) const
    {
        // This is the limit for profiles that round off in the center, since you can locally
        // approximate the profile as flat within the truncation radius.  This isn't true for
        // Sersics, so the real limit is larger than this (since more flux is inside re than in
        // the annulus between re and sqrt(2) re), but I don't know of an analytic formula for
        // the correct limit.  So we check for this here, and then if we encounter problems
        // later on, we throw a different error.
        if (trunc <= sqrt(2.) * re) {
            throw SBError("Sersic truncation must be larger than sqrt(2)*half_light_radius.");
        }

        // Given re and trunc, find the scale radius, r0, that makes these work.
        // f(re) = gamma(2n,(re/r0)^(1/n))
        // f(trunc) = gamma(2n,(trunc/r0)^(1/n))
        // Solve for the r0 that leads to f(re) = 1/2 f(trunc)
        // Equivalently, if b = (re/r0)^(1/n) and z = (trunc/r0)^(1/n) = b * (trunc/re)^(1/n)
        // then solve for b.
        // gamma(2n,b) = 1/2 gamma(2n,x*b), where x = (trunc/re)^(1/n)
        double x = std::pow(trunc/re,_invn);
        dbg<<"x = "<<x<<std::endl;

        // For an initial guess, we start with the asymptotic expansing from A&S (6.5.32):
        // Gamma(2n,b) ~= b^(2n-1) exp(-b) (1 + (2n-1)/b + (2n-1)(2n-2)/b^2 + ... )
        // Gamma(2n,xb) ~= (xb)^(2n-1) exp(-xb) (1 + (2n-1)/(xb) + (2n-1)(2n-2)/(xb)^2 + ... )
        // Just take the first terms:
        // b^(2n-1) exp(-b) = 1/2 (xb)^(2n-1) exp(-xb)
        // exp( (x-1) b ) = 1/2 x^(2n-1)
        // (x-1) b = log(1/2) + (2n-1) log(x)
        double b1 = (std::log(0.5) + (2.*_n-1) * std::log(x)) / (x-1.);
        dbg<<"Initial guess b = "<<b1<<std::endl;
        // Note: This isn't a very good initial guess, but the solver tends to converge pretty
        // rapidly anyway.

        // We need _b below, so call getHLR(), since it may not be calculated yet. 
        // We don't care about the return value, but it also stores the b value in _b.
        getHLR(); 

        // If trunc = sqrt(2) * re, then x = 2^(1/2n), and the initial guess for b is:
        // b = log( 0.5 * 2^(1/2n)^(2n-1) ) / (sqrt(2)-1)
        //   = -(1/2n) * log(2) / (sqrt(2)-1)
        // Negative b is obviously problematic.  I don't know for sure what the real limit
        // on trunc/re is.  It's possible that the full formulae can give a positive solution
        // even if the initial estimate is negative.  But unless someone complains (and proposes
        // a better prescription for this), we'll take this as a de facto limit.
        if (b1 < 1.e-3 * _b) {
            //throw SBError("Sersic truncation is too small for the given half_light_radius.");
            // Update: Ricardo Herbonnet (rightly) complained.
            // He pointed out that this formula for b1 is always == 0 for n = 0.5.
            // So we can't just be throwing an exception here.
            // Since we expand the bracket below anyway, switch to just using _b/2 and 
            // letting the expansion happen.
            // I also updated the above check from b1 <= 0 to b1 < 1.e-3 * _b.
            // Probably if we are getting really close to zero, it is better to start with 
            // _b/2 instead and expand it down.
            b1 = _b/2;
        }

        // The upper limit to b corresponds to the half-light radius of the untruncated profile.
        double b2 = _b;
        SersicTruncatedHLR func(_n, x);
        Solve<SersicTruncatedHLR> solver(func,b1,b2);
        solver.setMethod(Brent);
        solver.bracketLowerWithLimit(0.);    // expand lower bracket if necessary
        xdbg<<"After bracket, range is "<<solver.getLowerBound()<<" .. "<<
            solver.getUpperBound()<<std::endl;
        double b = solver.root();
        dbg<<"Root is "<<b<<std::endl;

        // r0 = re / b^n
        return re / std::pow(b,_n);
    }

    class SersicRadialFunction: public FluxDensity
    {
    public:
        SersicRadialFunction(double invn): _invn(invn) {}
        double operator()(double r) const { return std::exp(-std::pow(r,_invn)); }
    private:
        double _invn; 
    };

    boost::shared_ptr<PhotonArray> SersicInfo::shoot(int N, UniformDeviate ud) const
    {
        dbg<<"SersicInfo shoot: N = "<<N<<std::endl;
        dbg<<"Target flux = 1.0\n";

        if (!_sampler) {
            // Set up the classes for photon shooting
            _radial.reset(new SersicRadialFunction(_invn));
            std::vector<double> range(2,0.);
            double shoot_maxr = calculateMissingFluxRadius(_gsparams->shoot_accuracy);
            if (_truncated && _trunc < shoot_maxr) shoot_maxr = _trunc;
            range[1] = shoot_maxr;
            _sampler.reset(new OneDimensionalDeviate( *_radial, range, true, _gsparams));
        }
 
        assert(_sampler.get());
        boost::shared_ptr<PhotonArray> result = _sampler->shoot(N,ud);
        dbg<<"SersicInfo Realized flux = "<<result->getTotalFlux()<<std::endl;
        return result;
    }

    boost::shared_ptr<PhotonArray> SBSersic::SBSersicImpl::shoot(int N, UniformDeviate ud) const
    {
        dbg<<"Sersic shoot: N = "<<N<<std::endl;
        dbg<<"Target flux = "<<getFlux()<<std::endl;
        // Get photons from the SersicInfo structure, rescale flux and size for this instance
        boost::shared_ptr<PhotonArray> result = _info->shoot(N,ud);
        result->scaleFlux(_shootnorm);
        result->scaleXY(_r0);
        dbg<<"Sersic Realized flux = "<<result->getTotalFlux()<<std::endl;
        return result;
    }
}
