/* -*- c++ -*-
 * Copyright (c) 2012-2014 by the GalSim developers team on GitHub
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

#include "SBSpergel.h"
#include "SBSpergelImpl.h"
#include <boost/math/special_functions/bessel.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include "Solve.h"
#include "bessel/Roots.h"

// Define this variable to find azimuth (and sometimes radius within a unit disc) of 2d photons by
// drawing a uniform deviate for theta, instead of drawing 2 deviates for a point on the unit
// circle and rejecting corner photons.
// The relative speed of the two methods was tested as part of issue #163, and the results
// are collated in devutils/external/time_photon_shooting.
// The conclusion was that using sin/cos was faster for icpc, but not g++ or clang++.
#ifdef _INTEL_COMPILER
#define USE_COS_SIN
#endif

#ifdef DEBUGLOGGING
#include <fstream>
//std::ostream* dbgout = new std::ofstream("debug.out");
//std::ostream* dbgout = &std::cout;
//int verbose_level = 1;
#endif

namespace galsim {

    SBSpergel::SBSpergel(double nu, double size, RadiusType rType, double flux,
                         double trunc, bool flux_untruncated, const GSParamsPtr& gsparams) :
        SBProfile(new SBSpergelImpl(nu, size, rType, flux, trunc, flux_untruncated, gsparams)) {}

    SBSpergel::SBSpergel(const SBSpergel& rhs) : SBProfile(rhs) {}

    SBSpergel::~SBSpergel() {}

    double SBSpergel::getNu() const
    {
        assert(dynamic_cast<const SBSpergelImpl*>(_pimpl.get()));
        return static_cast<const SBSpergelImpl&>(*_pimpl).getNu();
    }

    double SBSpergel::getScaleRadius() const
    {
        assert(dynamic_cast<const SBSpergelImpl*>(_pimpl.get()));
        return static_cast<const SBSpergelImpl&>(*_pimpl).getScaleRadius();
    }

    double SBSpergel::getHalfLightRadius() const
    {
        assert(dynamic_cast<const SBSpergelImpl*>(_pimpl.get()));
        return static_cast<const SBSpergelImpl&>(*_pimpl).getHalfLightRadius();
    }

    LRUCache<boost::tuple<double,double,GSParamsPtr>,SpergelInfo> SBSpergel::SBSpergelImpl::cache(
        sbp::max_spergel_cache);

    SBSpergel::SBSpergelImpl::SBSpergelImpl(double nu, double size, RadiusType rType,
                                            double flux, double trunc, bool flux_untruncated,
                                            const GSParamsPtr& gsparams) :
        SBProfileImpl(gsparams),
        _nu(nu), _flux(flux), _trunc(trunc), _trunc_sq(trunc*trunc),
        // Start with untruncated SpergelInfo regardless of value of trunc
        _info(cache.get(boost::make_tuple(_nu, 0., this->gsparams.duplicate())))
    {
        dbg<<"Start SBSpergel constructor:\n";
        dbg<<"nu = "<<_nu<<std::endl;
        dbg<<"size = "<<size<<"  rType = "<<rType<<std::endl;
        dbg<<"flux = "<<_flux<<std::endl;
        dbg<<"trunc = "<<_trunc<<"  flux_untruncated = "<<flux_untruncated<<std::endl;

        _truncated = (_trunc > 0.);

        // Set size of this instance according to type of size given in constructor
        switch(rType) {
          case HALF_LIGHT_RADIUS:
              {
                  _re = size;
                  if (_truncated) {
                      if (flux_untruncated) {
                          // The given HLR and flux are the values for the untruncated profile.
                          _r0 = _re / _info->getHLR(); // getHLR() is in units of r0.
                      } else {
                           // This is the one case that is a bit complicated, since the
                           // half-light radius and trunc are both given in physical units,
                           // so we need to solve for what scale radius this corresponds to.
                          _r0 = _info->calculateScaleForTruncatedHLR(_re, _trunc);
                      }

                      // Update _info with the correct truncated version.
                      _info = cache.get(boost::make_tuple(_nu, _trunc/_r0,
                                                          this->gsparams.duplicate()));

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
                      _info = cache.get(boost::make_tuple(_nu, _trunc/_r0,
                                                          this->gsparams.duplicate()));
                      if (flux_untruncated) {
                          // Update the stored _flux with the correct value
                          _flux *= _info->getFluxFraction();
                      }
                  }
                  // In all cases, _re is the real HLR
                  _re = _r0 * _info->getHLR();
              }
              break;
        }

        _r0_sq = _r0 * _r0;
        _inv_r0 = 1. / _r0;
        _shootnorm = _flux * _info->getXNorm();
        _xnorm = _shootnorm / _r0_sq;

        dbg<<"scale radius = "<<_r0<<std::endl;
        dbg<<"HLR = "<<_re<<std::endl;
    }

    double SBSpergel::SBSpergelImpl::maxK() const { return _info->maxK() * _inv_r0; }
    double SBSpergel::SBSpergelImpl::stepK() const { return _info->stepK() * _inv_r0; }

    // Equations (3, 4) of Spergel (2010)
    double SBSpergel::SBSpergelImpl::xValue(const Position<double>& p) const
    {
        double r = sqrt(p.x * p.x + p.y * p.y) * _inv_r0;
        return _xnorm * _info->xValue(r);
    }

    // Equation (2) of Spergel (2010)
    std::complex<double> SBSpergel::SBSpergelImpl::kValue(const Position<double>& k) const
    {
        double ksq = (k.x*k.x + k.y*k.y) * _r0_sq;
        return _flux * _info->kValue(ksq);
    }

    void SBSpergel::SBSpergelImpl::fillXValue(tmv::MatrixView<double> val,
                                              double x0, double dx, int ix_zero,
                                              double y0, double dy, int iy_zero) const
    {
        dbg<<"SBSpergel fillXValue\n";
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
                    double r = sqrt(x*x + ysq);
                    *valit++ = _xnorm * _info->xValue(r);
                }
            }
        }
    }

    void SBSpergel::SBSpergelImpl::fillKValue(tmv::MatrixView<std::complex<double> > val,
                                            double x0, double dx, int ix_zero,
                                            double y0, double dy, int iy_zero) const
    {
        dbg<<"SBSpergel fillKValue\n";
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
                It valit = val.col(j).begin();
                for (int i=0;i<m;++i,x+=dx) {
                    double ksq = x*x + ysq;
                    *valit++ = _flux * _info->kValue(ksq);
                }
            }
        }
    }

    void SBSpergel::SBSpergelImpl::fillXValue(tmv::MatrixView<double> val,
                                            double x0, double dx, double dxy,
                                            double y0, double dy, double dyx) const
    {
        dbg<<"SBSpergel fillXValue\n";
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
            for (int i=0;i<m;++i,x+=dx,y+=dyx) {
                double r = sqrt(x*x + y*y);
                *valit++ = _xnorm * _info->xValue(r);
            }
        }
    }

    void SBSpergel::SBSpergelImpl::fillKValue(tmv::MatrixView<std::complex<double> > val,
                                            double x0, double dx, double dxy,
                                            double y0, double dy, double dyx) const
    {
        dbg<<"SBSpergel fillKValue\n";
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

        It valit = val.linearView().begin();
        for (int j=0;j<n;++j,x0+=dxy,y0+=dy) {
            double x = x0;
            double y = y0;
            for (int i=0;i<m;++i,x+=dx,y+=dyx) {
                double ksq = x*x + y*y;
                *valit++ = _flux * _info->kValue(ksq);
            }
        }
    }

    SpergelInfo::SpergelInfo(double nu, double trunc, const GSParamsPtr& gsparams) :
        _nu(nu), _trunc(trunc), _gsparams(gsparams),
        _gamma_nup1(boost::math::tgamma(_nu+1.0)),
        _gamma_nup2(_gamma_nup1 * (_nu+1)),
        _xnorm0((_nu > 0.) ? _gamma_nup1 / (2. * _nu) * std::pow(2., _nu) : INFINITY),
        _truncated(_trunc > 0.),
        _maxk(0.), _stepk(0.), _re(0.), _flux(0.),
        _ft(Table<double,double>::spline)
    {
        dbg<<"Start SpergelInfo constructor for nu = "<<_nu<<std::endl;
        dbg<<"trunc = "<<_trunc<<std::endl;

        if (_nu < sbp::minimum_spergel_nu || _nu > sbp::maximum_spergel_nu)
            throw SBError("Requested Spergel index out of range");
    }

    class SpergelIntegratedFlux
    {
    public:
        SpergelIntegratedFlux(double nu, double gamma_nup2, double flux_frac=0.0)
            : _nu(nu), _gamma_nup2(gamma_nup2),  _target(flux_frac) {}

        double operator()(double u) const
        {
        // Return flux integrated up to radius `u` in units of r0, minus `flux_frac`
        // (i.e., make a residual so this can be used to search for a target flux.
        // This result is derived in Spergel (2010) eqn. 8 by going to Fourier space
        // and integrating by parts.
        // The key Bessel identities:
        // int(r J0(k r), r=0..R) = R J1(k R) / k
        // d[-J0(k R)]/dk = R J1(k R)
        // The definition of the radial surface brightness profile and Fourier transform:
        // Sigma_nu(r) = (r/2)^nu K_nu(r)/Gamma(nu+1)
        //             = int(k J0(k r) / (1+k^2)^(1+nu), k=0..inf)
        // and the main result:
        // F(R) = int(2 pi r Sigma(r), r=0..R)
        //      = int(r int(k J0(k r) / (1+k^2)^(1+nu), k=0..inf), r=0..R) // Do the r-integral
        //      = int(R J1(k R)/(1+k^2)^(1+nu), k=0..inf)
        // Now integrate by parts with
        //      u = 1/(1+k^2)^(1+nu)                 dv = R J1(k R) dk
        // =>  du = -2 k (1+nu)/(1+k^2)^(2+nu) dk     v = -J0(k R)
        // => F(R) = u v | k=0,inf - int(v du, k=0..inf)
        //         = (0 + 1) - 2 (1+nu) int(k J0(k R) / (1+k^2)^2+nu, k=0..inf)
        //         = 1 - 2 (1+nu) (R/2)^(nu+1) K_{nu+1}(R) / Gamma(nu+2)
            double fnup1 = std::pow(u / 2., _nu+1.)
                * boost::math::cyl_bessel_k(_nu+1., u)
                / _gamma_nup2;
            double f = 1.0 - 2.0 * (1.+_nu)*fnup1;
            return f - _target;
        }
    private:
        double _nu;
        double _gamma_nup2;
        double _target;
    };

    double SpergelInfo::calculateFluxRadius(const double& flux_frac) const
    {
        // Calcute r such that L(r/r0) / L_tot == flux_frac

        // These seem to bracket pretty much every reasonable possibility
        // that I checked in Mathematica, though I'm not very happy about
        // the lower bound....
        double z1=1.e-16;
        double z2=20.0;
        SpergelIntegratedFlux func(_nu, _gamma_nup2, flux_frac);
        Solve<SpergelIntegratedFlux> solver(func, z1, z2);
        solver.setMethod(Brent);
        solver.bracketLowerWithLimit(0.0); // Just in case...
        double R = solver.root();
        dbg<<"flux_frac = "<<flux_frac<<std::endl;
        dbg<<"r/r0 = "<<R<<std::endl;
        return R;
    }

    double SpergelInfo::stepK() const
    {
        if (_stepk == 0.) {
            double R = calculateFluxRadius(1.0 - _gsparams->folding_threshold);
            if (_truncated && _trunc < R)  R = _trunc;
            // Go to at least 5*re
            R = std::max(R,_gsparams->stepk_minimum_hlr);
            dbg<<"R => "<<R<<std::endl;
            _stepk = M_PI / R;
            dbg<<"stepk = "<<_stepk<<std::endl;
        }
        return _stepk;
    }

    double SpergelInfo::maxK() const
    {
        if(_maxk == 0.) {
            if (_truncated) buildFT();
            else {
                // Solving (1+k^2)^(-1-nu) = maxk_threshold for k
                // exact:
                //_maxk = std::sqrt(std::pow(gsparams->maxk_threshold, -1./(1+_nu))-1.0);
                // approximate 1+k^2 ~ k^2 => good enough:
                _maxk = std::pow(_gsparams->maxk_threshold, -1./(2*(1+_nu)));
            }
        }
        return _maxk;
    }

    double SpergelInfo::getHLR() const
    {
        if (_re == 0.0) calculateHLR();
        return _re;
    }

    double SpergelInfo::getFluxFraction() const
    {
        if (_flux == 0.) {
            // Calculate the flux of a truncated profile (relative to the integral for
            // an untruncated profile).
            if (_truncated) {
                SpergelIntegratedFlux func(_nu, _gamma_nup2, 0.0);
                _flux = func(_trunc);
                dbg << "Flux fraction = " << _flux << std::endl;
            } else {
                _flux = 1.;
            }
        }
        return _flux;
    }

    void SpergelInfo::calculateHLR() const
    {
        dbg<<"Find HLR for (nu,trunc) = ("<<_nu<<","<<_trunc<<")"<<std::endl;
        SpergelIntegratedFlux func(_nu, _gamma_nup2, 0.5*getFluxFraction());
        double b1 = 0.1; // These are sufficient for -0.85 < nu < 100
        double b2 = 17.0;
        Solve<SpergelIntegratedFlux> solver(func, b1, b2);
        xdbg<<"Initial range is "<<b1<<" .. "<<b2<<std::endl;
        solver.setMethod(Brent);
        _re = solver.root();
        dbg<<"re is "<<_re<<std::endl;
    }

    // Function object for finding scale radius given a HLR and truncation radius.
    class SpergelTruncatedHLR
    {
    public:
        SpergelTruncatedHLR(double nu, double gamma_nup1, double re, double trunc) :
            _nu(nu), _gamma_nup1(gamma_nup1), _re(re), _trunc(trunc) {}

        double operator()(double r0) const
        {
            double term1 = 2. * std::pow(_re, _nu+1.)*boost::math::cyl_bessel_k(_nu+1, _re/r0);
            double term2 = std::pow(_trunc, _nu+1.)*boost::math::cyl_bessel_k(_nu+1, _trunc/r0);
            double term3 = 0.5 * _gamma_nup1 * std::pow(2*r0, _nu+1.);
            return term1 - term2 - term3;
        }
    private:
        double _nu;
        double _gamma_nup1;
        double _re;
        double _trunc;
    };

    double SpergelInfo::calculateScaleForTruncatedHLR(double re, double trunc) const
    {
        // This is the limit for profiles that round off in the center, since you can locally
        // approximate the profile as flat within the truncation radius.  This isn't true for
        // Spergels, so the real limit is larger than this (since more flux is inside re than in
        // the annulus between re and sqrt(2) re), but I don't know of an analytic formula for
        // the correct limit.  So we check for this here, and then if we encounter problems
        // later on, we throw a different error.
        if (trunc <= sqrt(2.) * re) {
            throw SBError("Spergel truncation must be larger than sqrt(2)*half_light_radius.");
        }

        // Given re and trunc, find the scale radius, r0, that makes these work.
        // f(re) = 1 - 2(1+nu)(re/2r0)^(nu+1) K_{nu+1}(re/r0)/Gamma(nu+2)
        // f(trunc) = 1 - 2(1+nu)(trunc/2r0)^(nu+1) K_{nu+1}(trunc/r0)/Gamma(nu+2)
        // Solve for the r0 that leads to f(re) = 1/2 f(trunc)
        // Algebra:
        // 0 = 2 K_{nu+1}(re/r0) re^(nu+1) - K_{nu+1}(trunc/r0) trunc^(nu+1)
        //     - Gamma(nu+1) (2 r0)^(nu+1)/2

        // The scale radius given the untruncated HLR is always a lower bound:
        double b1 = re / getHLR();
        // I'm not sure what a reasonable upper bound could be, so start at factor of 10 and expand.
        double b2 = b1 * 10.0;
        SpergelTruncatedHLR func(_nu, _gamma_nup1, re, trunc);
        Solve<SpergelTruncatedHLR> solver(func,b1,b2);
        solver.bracketUpper();
        dbg<<"Initial range is "<<solver.getLowerBound()<<" .. "
            <<solver.getUpperBound()<<std::endl;
        dbg<<"which evaluates to "<<func(solver.getLowerBound())<<" .. "
           <<func(solver.getUpperBound())<<std::endl;
        solver.setMethod(Brent);
        double r0 = solver.root();
        dbg<<"Root is "<<r0<<std::endl;
        return r0;
    }

    double SpergelInfo::getXNorm() const
    { return std::pow(2., -_nu) / _gamma_nup1 / (2.0 * M_PI) / getFluxFraction(); }

    double SpergelInfo::xValue(double r) const
    {
        if (_truncated && r > _trunc) return 0.;
        else if (r == 0.) return _xnorm0;
        else return boost::math::cyl_bessel_k(_nu, r) * std::pow(r, _nu);
    }

    double SpergelInfo::kValue(double ksq) const
    {
        if (_truncated) {
            if (_ft.size() == 0) buildFT();
            double lk=0.5*std::log(ksq);
            if (lk < _a1) {
                //linearly interpolate the first bin (in ksq)
                return (1. - ksq/_a1ksq * (1. - _fta1/_a1ksq));
            }
            else if (lk > _ft.argMax()) return 0.;
            else return _ft(lk)/ksq;
        } else {
            return std::pow(1. + ksq, -1. - _nu);
        }
    }

    class SpergelIntegrand : public std::unary_function<double, double>
    {
    public:
        SpergelIntegrand(double nu, double k) :
            _nu(nu), _k(k) {}
        double operator()(double r) const
        { return std::pow(r, _nu)*boost::math::cyl_bessel_k(_nu, r) * r*j0(_k*r); }

    private:
        double _nu;
        double _k;
    };

    void SpergelInfo::buildFT() const
    {
        assert(_trunc > 0.);
        if (_ft.size() > 0) return;
        dbg<<"Building truncated Spergel Hankel transform"<<std::endl;
        dbg<<"nu = "<<_nu<<std::endl;
        dbg<<"trunc = "<<_trunc<<std::endl;
        // Do a Hankel transform and store the results in a lookup table.
        double prefactor = std::pow(2., -_nu) / _gamma_nup1 / _flux;
        dbg<<"prefactor = "<<prefactor<<std::endl;

        // Along the way, find the last k that has a kValue > 1.e-3
        double maxk_val = this->_gsparams->maxk_threshold;
        dbg<<"Looking for maxk_val = "<<maxk_val<<std::endl;
        // Keep going until at least 5 in a row have kvalues below kvalue_accuracy.
        // (It's oscillatory, so want to make sure not to stop at a zero crossing.)

        // We use a cubic spline for the interpolation, which has an error of O(h^4) max(f'''').
        // I have no idea what range the fourth derivative can take for the hankel transform,
        // so let's take the completely arbitrary value of 10.  (This value was found to be
        // conservative for Sersic, but I haven't investigated here.)
        // 10 h^4 <= kvalue_accuracy
        // h = (kvalue_accuracy/10)^0.25
        double dlogk = _gsparams->table_spacing * sqrt(sqrt(_gsparams->kvalue_accuracy / 10.));
        dbg<<"Using dlogk = "<<dlogk<<std::endl;
        int n_below_thresh = 0;

        // Don't go past k = 500
        double kmin = dlogk; // have to begin somewhere...
        for (double logk = std::log(kmin)-0.001; logk < std::log(500.); logk += dlogk) {
            double k = std::exp(logk);
            double ksq = k*k;

            SpergelIntegrand I(_nu, k);

#ifdef DEBUGLOGGING
            std::ostream* integ_dbgout = verbose_level >= 3 ? dbgout : 0;
            integ::IntRegion<double> reg(0, _trunc, integ_dbgout);
#else
            integ::IntRegion<double> reg(0, _trunc);
#endif

            // Add explicit splits at first several roots of J0.
            // This tends to make the integral more accurate.
            for (int s=1; s<=10; ++s) {
                double root = bessel::getBesselRoot0(s);
                if (root > k * _trunc) break;
                reg.addSplit(root/k);
            }

            double val = integ::int1d(
                I, reg,
                this->_gsparams->integration_relerr,
                this->_gsparams->integration_abserr);
            val *= prefactor;

            xdbg<<"logk = "<<logk<<", ft("<<exp(logk)<<") = "<<val<<"   "<<val*ksq<<std::endl;

            double f0 = val * ksq;
            _ft.addEntry(logk, f0);

            if (std::abs(val) > maxk_val) _maxk = k;

            if (std::abs(val) > this->_gsparams->kvalue_accuracy) n_below_thresh = 0;
            else ++n_below_thresh;
            if (n_below_thresh == 5) break;
        }
        dbg<<"maxk = "<<_maxk<<std::endl;
        _a1 = _ft.argMin();
        _a1ksq = std::exp(2. * _a1);
        _fta1 = _ft(_a1);
    }

    class SpergelNuPositiveRadialFunction: public FluxDensity
    {
    public:
        SpergelNuPositiveRadialFunction(double nu, double xnorm0):
            _nu(nu), _xnorm0(xnorm0) {}
        double operator()(double r) const {
            if (r == 0.) return _xnorm0;
            else return boost::math::cyl_bessel_k(_nu, r) * std::pow(r,_nu);
        }
    private:
        double _nu;
        double _xnorm0;
    };

    class SpergelNuNegativeRadialFunction: public FluxDensity
    {
    public:
        SpergelNuNegativeRadialFunction(double nu, double rmin, double a, double b):
            _nu(nu), _rmin(rmin), _a(a), _b(b) {}
        double operator()(double r) const {
            if (r <= _rmin) {
                return _a + _b*r;
            } else return boost::math::cyl_bessel_k(_nu, r) * std::pow(r,_nu);
        }
    private:
        double _nu;
        double _rmin;
        double _a;
        double _b;
    };

    boost::shared_ptr<PhotonArray> SpergelInfo::shoot(int N, UniformDeviate ud) const
    {
        dbg<<"SpergelInfo shoot: N = "<<N<<std::endl;
        dbg<<"Target flux = 1.0\n";

        if (!_sampler) {
            // Set up the classes for photon shooting
            double shoot_rmax = calculateFluxRadius(1. - _gsparams->shoot_accuracy);
            if (_truncated && _trunc < shoot_rmax) shoot_rmax = _trunc;
            if (_nu > 0.) {
                std::vector<double> range(2,0.);
                range[1] = shoot_rmax;
                _radial.reset(new SpergelNuPositiveRadialFunction(_nu, _xnorm0));
                _sampler.reset(new OneDimensionalDeviate( *_radial, range, true, _gsparams));
            } else {
                // exact s.b. profile diverges at origin, so replace the inner most circle
                // (defined such that enclosed flux is shoot_acccuracy*_flux) with a linear function
                // that contains the same flux and has the right value at r = rmin.
                // So need to solve the following for a and b:
                // int_0^rmin 2 pi r (a + b r) 0..rmin = shoot_accuracy
                // a + b rmin = K_nu(rmin) * rmin^nu
                double flux_target = _gsparams->shoot_accuracy*_flux;
                double shoot_rmin = calculateFluxRadius(flux_target);
                double knur = boost::math::cyl_bessel_k(_nu, shoot_rmin)*std::pow(shoot_rmin, _nu);
                double b = 3./shoot_rmin*(knur - flux_target/(M_PI*shoot_rmin*shoot_rmin));
                double a = knur - shoot_rmin*b;
                dbg<<"flux target: "<<flux_target<<std::endl;
                dbg<<"shoot rmin: "<<shoot_rmin<<std::endl;
                dbg<<"shoot rmax: "<<shoot_rmax<<std::endl;
                dbg<<"knur: "<<knur<<std::endl;
                dbg<<"b: "<<b<<std::endl;
                dbg<<"a: "<<a<<std::endl;
                dbg<<"a+b*rmin:"<<a+b*shoot_rmin<<std::endl;
                std::vector<double> range(3,0.);
                range[1] = shoot_rmin;
                range[2] = shoot_rmax;
                _radial.reset(new SpergelNuNegativeRadialFunction(_nu, shoot_rmin, a, b));
                _sampler.reset(new OneDimensionalDeviate( *_radial, range, true, _gsparams));
            }
        }

        assert(_sampler.get());
        boost::shared_ptr<PhotonArray> result = _sampler->shoot(N,ud);
        dbg<<"SpergelInfo Realized flux = "<<result->getTotalFlux()<<std::endl;
        return result;
    }

    boost::shared_ptr<PhotonArray> SBSpergel::SBSpergelImpl::shoot(int N, UniformDeviate ud) const
    {
        dbg<<"Spergel shoot: N = "<<N<<std::endl;
        dbg<<"Target flux = "<<getFlux()<<std::endl;
        // Get photons from the SpergelInfo structure, rescale flux and size for this instance
        boost::shared_ptr<PhotonArray> result = _info->shoot(N,ud);
        result->scaleFlux(_shootnorm);
        result->scaleXY(_r0);
        dbg<<"Spergel Realized flux = "<<result->getTotalFlux()<<std::endl;
        return result;
    }
}
