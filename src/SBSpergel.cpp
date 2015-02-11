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
                         const GSParamsPtr& gsparams) :
        SBProfile(new SBSpergelImpl(nu, size, rType, flux, gsparams)) {}

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

    double SBSpergel::calculateIntegratedFlux(const double& r) const
    {
        assert(dynamic_cast<const SBSpergelImpl*>(_pimpl.get()));
        return static_cast<const SBSpergelImpl&>(*_pimpl).calculateIntegratedFlux(r);
    }

    double SBSpergel::calculateFluxRadius(const double& f) const
    {
        assert(dynamic_cast<const SBSpergelImpl*>(_pimpl.get()));
        return static_cast<const SBSpergelImpl&>(*_pimpl).calculateFluxRadius(f);
    }

    LRUCache<boost::tuple<double,GSParamsPtr>,SpergelInfo> SBSpergel::SBSpergelImpl::cache(
        sbp::max_spergel_cache);

    SBSpergel::SBSpergelImpl::SBSpergelImpl(double nu, double size, RadiusType rType,
                                            double flux, const GSParamsPtr& gsparams) :
        SBProfileImpl(gsparams),
        _nu(nu), _flux(flux), _info(cache.get(boost::make_tuple(_nu, this->gsparams.duplicate())))
    {
        dbg<<"Start SBSpergel constructor:\n";
        dbg<<"nu = "<<_nu<<std::endl;
        dbg<<"size = "<<size<<"  rType = "<<rType<<std::endl;
        dbg<<"flux = "<<_flux<<std::endl;

        // Set size of this instance according to type of size given in constructor
        switch(rType) {
          case HALF_LIGHT_RADIUS:
              {
                  _re = size;
                  _r0 = _re / _info->getHLR();
              }
              break;
          case SCALE_RADIUS:
              {
                  _r0 = size;
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

    double SBSpergel::SBSpergelImpl::calculateIntegratedFlux(const double& r) const
    { return _info->calculateIntegratedFlux(r*_inv_r0);}

    double SBSpergel::SBSpergelImpl::calculateFluxRadius(const double& f) const
    { return _info->calculateFluxRadius(f) * _r0; }
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

    SpergelInfo::SpergelInfo(double nu, const GSParamsPtr& gsparams) :
        _nu(nu), _gsparams(gsparams),
        _gamma_nup1(boost::math::tgamma(_nu+1.0)),
        _gamma_nup2(_gamma_nup1 * (_nu+1)),
        _gamma_nup3(_gamma_nup2 * (_nu+2)),
        _gamma_mnum1(_nu < 0.0 ? boost::math::tgamma(-_nu-1.0) : 0.0), // only use this if nu<0
        _xnorm0((_nu > 0.) ? _gamma_nup1 / (2. * _nu) * std::pow(2., _nu) : INFINITY),
        _maxk(0.), _stepk(0.), _re(0.)
    {
        dbg<<"Start SpergelInfo constructor for nu = "<<_nu<<std::endl;

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

    class SpergelTaylorIntegratedFlux
    {
    public:
        SpergelTaylorIntegratedFlux(double nu, double gamma_nup2, double gamma_nup3,
                                    double gamma_mnum1, double flux_frac=0.0)
            : _nu(nu), _gamma_nup2(gamma_nup2), _gamma_nup3(gamma_nup3),
              _gamma_mnum1(gamma_mnum1), _target(flux_frac) {}

        double operator()(double u) const
        {
        // Use the small radius Taylor series approximation to the Spergel surface brightness
        // profile to compute the flux enclosed within a small radius.
        // Only use this for profiles with negative nu.
        // Here's the approximation to second order in u:
        // f_nu(u) = (u/2)^nu K_nu(u)/Gamma(nu+1)
        //         ~  u^(2 nu) [2^(-1 - 2 nu) Gamma(-nu)/Gamma(nu+1)
        //                      + 2^(-3-2*nu) Gamma(-nu) u^2 / Gamma(nu+2)]
        //            + (1/(2 nu) + u^2 / (8 nu - 8 nu^2)
        // Recall from above that F(u) = 1 - 2 (1 + nu) f_{nu+1}(u)
            double nup1 = _nu + 1.0;
            double term1 =
                std::pow(u, 2*nup1)*(std::pow(2., -1.-2*nup1) * _gamma_mnum1 / _gamma_nup2
                                     + std::pow(2., -3.-2*nup1) * _gamma_mnum1 *u*u / _gamma_nup3);
            double term2 = 1./(2*nup1) + u*u / (8.*nup1*(1.-nup1));
            return 1.0-2.0*(nup1)*(term1 + term2) - _target;
        }
    private:
        double _nu;
        double _gamma_nup2;
        double _gamma_nup3;
        double _gamma_mnum1;
        double _target;
    };

    double SpergelInfo::calculateFluxRadius(const double& flux_frac) const
    {
        // Calcute r such that L(r/r0) / L_tot == flux_frac

        // These bracket the range of calculateFluxRadius(0.5) for -0.85 < nu < 4.0.
        double z1=0.1;
        double z2=3.5;
        SpergelIntegratedFlux func(_nu, _gamma_nup2, flux_frac);
        Solve<SpergelIntegratedFlux> solver(func, z1, z2);
        solver.setXTolerance(1.e-25); // Spergels can be super peaky, so need a tight tolerance.
        solver.setMethod(Brent);
        if (flux_frac < 0.5)
            solver.bracketLowerWithLimit(0.0);
        else
            solver.bracketUpper();
        double R = solver.root();
        dbg<<"flux_frac = "<<flux_frac<<std::endl;
        dbg<<"r/r0 = "<<R<<std::endl;
        return R;
    }

    double SpergelInfo::calculateIntegratedFlux(const double& r) const
    {
        SpergelIntegratedFlux func(_nu, _gamma_nup2);
        return func(r);
    }

    double SpergelInfo::stepK() const
    {
        if (_stepk == 0.) {
            double R = calculateFluxRadius(1.0 - _gsparams->folding_threshold);
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
            // Solving (1+k^2)^(-1-nu) = maxk_threshold for k
            // exact:
            //_maxk = std::sqrt(std::pow(gsparams->maxk_threshold, -1./(1+_nu))-1.0);
            // approximate 1+k^2 ~ k^2 => good enough:
            _maxk = std::pow(_gsparams->maxk_threshold, -1./(2*(1+_nu)));
        }
        return _maxk;
    }

    double SpergelInfo::getHLR() const
    {
        if (_re == 0.0) _re = calculateFluxRadius(0.5);
        return _re;
    }

    // void SpergelInfo::calculateHLR() const
    // {
    //     dbg<<"Find HLR for nu = "<<_nu<<std::endl;
    //     SpergelIntegratedFlux func(_nu, _gamma_nup2, 0.5);
    //     double b1 = 0.1; // These are sufficient for -0.85 < nu < 100
    //     double b2 = 17.0;
    //     Solve<SpergelIntegratedFlux> solver(func, b1, b2);
    //     xdbg<<"Initial range is "<<b1<<" .. "<<b2<<std::endl;
    //     solver.setMethod(Brent);
    //     _re = solver.root();
    //     dbg<<"re is "<<_re<<std::endl;
    // }

    double SpergelInfo::getXNorm() const
    { return std::pow(2., -_nu) / _gamma_nup1 / (2.0 * M_PI); }

    double SpergelInfo::xValue(double r) const
    {
        if (r == 0.) return _xnorm0;
        else return boost::math::cyl_bessel_k(_nu, r) * std::pow(r, _nu);
    }

    double SpergelInfo::kValue(double ksq) const
    {
        return std::pow(1. + ksq, -1. - _nu);
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
            if (_nu > 0.) {
                std::vector<double> range(2,0.);
                range[1] = shoot_rmax;
                _radial.reset(new SpergelNuPositiveRadialFunction(_nu, _xnorm0));
                _sampler.reset(new OneDimensionalDeviate( *_radial, range, true, _gsparams));
            } else {
                // exact s.b. profile diverges at origin, so replace the inner most circle
                // (defined such that enclosed flux is shoot_acccuracy) with a linear function
                // that contains the same flux and has the right value at r = rmin.
                // So need to solve the following for a and b:
                // int(2 pi r (a + b r) dr, 0..rmin) = shoot_accuracy
                // a + b rmin = K_nu(rmin) * rmin^nu
                double flux_target = _gsparams->shoot_accuracy;
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
        // Get photons from the SpergelInfo structure, rescale flux and size for this instance
        boost::shared_ptr<PhotonArray> result = _info->shoot(N,ud);
        result->scaleFlux(_shootnorm);
        result->scaleXY(_r0);
        dbg<<"Spergel Realized flux = "<<result->getTotalFlux()<<std::endl;
        return result;
    }
}
