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

#define DEBUGLOGGING

#include "SBSpergelet.h"
#include "SBSpergeletImpl.h"
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
std::ostream* dbgout = &std::cout;
int verbose_level = 1;
#endif

namespace galsim {

    SBSpergelet::SBSpergelet(double nu, double r0, int j, int q, const GSParamsPtr& gsparams) :
        SBProfile(new SBSpergeletImpl(nu, r0, j, q, gsparams)) {}

    SBSpergelet::SBSpergelet(const SBSpergelet& rhs) : SBProfile(rhs) {}

    SBSpergelet::~SBSpergelet() {}

    double SBSpergelet::getNu() const
    {
        assert(dynamic_cast<const SBSpergeletImpl*>(_pimpl.get()));
        return static_cast<const SBSpergeletImpl&>(*_pimpl).getNu();
    }

    double SBSpergelet::getScaleRadius() const
    {
        assert(dynamic_cast<const SBSpergeletImpl*>(_pimpl.get()));
        return static_cast<const SBSpergeletImpl&>(*_pimpl).getScaleRadius();
    }

    int SBSpergelet::getJ() const
    {
        assert(dynamic_cast<const SBSpergeletImpl*>(_pimpl.get()));
        return static_cast<const SBSpergeletImpl&>(*_pimpl).getJ();
    }

    int SBSpergelet::getQ() const
    {
        assert(dynamic_cast<const SBSpergeletImpl*>(_pimpl.get()));
        return static_cast<const SBSpergeletImpl&>(*_pimpl).getQ();
    }

    LRUCache<boost::tuple<double,int,int,GSParamsPtr>,SpergeletInfo>
        SBSpergelet::SBSpergeletImpl::cache(sbp::max_spergelet_cache);

    SBSpergelet::SBSpergeletImpl::SBSpergeletImpl(double nu, double r0,
                                                  int j, int q,
                                                  const GSParamsPtr& gsparams) :
        SBProfileImpl(gsparams),
        _nu(nu), _r0(r0), _j(j), _q(q),
        _info(cache.get(boost::make_tuple(_nu, _j, _q, this->gsparams.duplicate())))
    {
        if ((j < 0) or (q < 0) or (j < q))
            throw SBError("Requested Spegelet indices out of range");

        dbg<<"Start SBSpergelet constructor:\n";
        dbg<<"nu = "<<_nu<<std::endl;
        dbg<<"r0 = "<<_r0<<std::endl;
        dbg<<"(j,q) = ("<<_j<<","<<_q<<")"<<std::endl;

        _r0_sq = _r0 * _r0;
        _inv_r0 = 1. / _r0;
        _xnorm = _info->getXNorm() / _r0_sq;
    }

    double SBSpergelet::SBSpergeletImpl::maxK() const { return _info->maxK() * _inv_r0; }
    double SBSpergelet::SBSpergeletImpl::stepK() const { return _info->stepK() * _inv_r0; }

    // double SBSpergelet::SBSpergeletImpl::xValue(const Position<double>& p) const
    // {
    //     double r = sqrt(p.x * p.x + p.y * p.y) * _inv_r0;
    //     double phi = atan2(p.y, p.x);
    //     return _xnorm * _info->xValue(r, phi);
    // }

    // Modified equation (47) of Spergel (2010)
    std::complex<double> SBSpergelet::SBSpergeletImpl::kValue(const Position<double>& k) const
    {
        double ksq = (k.x*k.x + k.y*k.y) * _r0_sq;
        double phi = atan2(k.y, k.x);
        return _info->kValue(ksq, phi);
    }

    // TODO: Figure this out
    double SBSpergelet::SBSpergeletImpl::getFlux() const
    {
        return 1.;
    }

    // void SBSpergelet::SBSpergeletImpl::fillXValue(tmv::MatrixView<double> val,
    //                                               double x0, double dx, int ix_zero,
    //                                               double y0, double dy, int iy_zero) const
    // {
    //     dbg<<"SBSpergelet fillXValue\n";
    //     dbg<<"x = "<<x0<<" + ix * "<<dx<<", ix_zero = "<<ix_zero<<std::endl;
    //     dbg<<"y = "<<y0<<" + iy * "<<dy<<", iy_zero = "<<iy_zero<<std::endl;
    //     // Not sure about quadrant.  Spergelets are sometimes even and sometimes odd.
    //     // if (ix_zero != 0 || iy_zero != 0) {
    //     //     xdbg<<"Use Quadrant\n";
    //     //     fillXValueQuadrant(val,x0,dx,ix_zero,y0,dy,iy_zero);
    //     //     // Spergels can be super peaky at the center, so handle explicitly like Sersics
    //     //     if (ix_zero != 0 && iy_zero != 0)
    //     //         val(ix_zero, iy_zero) = _xnorm * _info->xValue(0.);
    //     // } else {
    //         xdbg<<"Non-Quadrant\n";
    //         assert(val.stepi() == 1);
    //         const int m = val.colsize();
    //         const int n = val.rowsize();
    //         typedef tmv::VIt<double,1,tmv::NonConj> It;

    //         x0 *= _inv_r0;
    //         dx *= _inv_r0;
    //         y0 *= _inv_r0;
    //         dy *= _inv_r0;

    //         for (int j=0;j<n;++j,y0+=dy) {
    //             double x = x0;
    //             double ysq = y0*y0;
    //             It valit = val.col(j).begin();
    //             for (int i=0;i<m;++i,x+=dx) {
    //                 double r = sqrt(x*x + ysq);
    //                 double phi = atan2(y0, x);
    //                 *valit++ = _xnorm * _info->xValue(r, phi);
    //             }
    //         }
    //     // }
    // }

    void SBSpergelet::SBSpergeletImpl::fillKValue(tmv::MatrixView<std::complex<double> > val,
                                                  double x0, double dx, int ix_zero,
                                                  double y0, double dy, int iy_zero) const
    {
        dbg<<"SBSpergelet fillKValue\n";
        dbg<<"x = "<<x0<<" + ix * "<<dx<<", ix_zero = "<<ix_zero<<std::endl;
        dbg<<"y = "<<y0<<" + iy * "<<dy<<", iy_zero = "<<iy_zero<<std::endl;
        // Not sure about quadrant.  Spergelets are sometimes even, sometimes odd.
        // if (ix_zero != 0 || iy_zero != 0) {
        //     xdbg<<"Use Quadrant\n";
        //     fillKValueQuadrant(val,x0,dx,ix_zero,y0,dy,iy_zero);
        // } else {
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
                    double phi = atan2(y0, x);
                    *valit++ = _info->kValue(ksq, phi);
                }
            }
        // }
    }

    // void SBSpergelet::SBSpergeletImpl::fillXValue(tmv::MatrixView<double> val,
    //                                               double x0, double dx, double dxy,
    //                                               double y0, double dy, double dyx) const
    // {
    //     dbg<<"SBSpergelet fillXValue\n";
    //     dbg<<"x = "<<x0<<" + ix * "<<dx<<" + iy * "<<dxy<<std::endl;
    //     dbg<<"y = "<<y0<<" + ix * "<<dyx<<" + iy * "<<dy<<std::endl;
    //     assert(val.stepi() == 1);
    //     assert(val.canLinearize());
    //     const int m = val.colsize();
    //     const int n = val.rowsize();
    //     typedef tmv::VIt<double,1,tmv::NonConj> It;

    //     x0 *= _inv_r0;
    //     dx *= _inv_r0;
    //     dxy *= _inv_r0;
    //     y0 *= _inv_r0;
    //     dy *= _inv_r0;
    //     dyx *= _inv_r0;

    //     It valit = val.linearView().begin();
    //     for (int j=0;j<n;++j,x0+=dxy,y0+=dy) {
    //         double x = x0;
    //         double y = y0;
    //         It valit = val.col(j).begin();
    //         for (int i=0;i<m;++i,x+=dx,y+=dyx) {
    //             double r = sqrt(x*x + y*y);
    //             double phi = atan2(y,x);
    //             *valit++ = _xnorm * _info->xValue(r, phi);
    //         }
    //     }
    // }

    void SBSpergelet::SBSpergeletImpl::fillKValue(tmv::MatrixView<std::complex<double> > val,
                                                  double x0, double dx, double dxy,
                                                  double y0, double dy, double dyx) const
    {
        dbg<<"SBSpergelet fillKValue\n";
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

        It valit(val.linearView().begin().getP());
        for (int j=0;j<n;++j,x0+=dxy,y0+=dy) {
            double x = x0;
            double y = y0;
            It valit(val.col(j).begin().getP(),1);
            for (int i=0;i<m;++i,x+=dx,y+=dyx) {
                double ksq = x*x + y*y;
                double phi = atan2(y,x);
                *valit++ = _info->kValue(ksq, phi);
            }
        }
    }

    SpergeletInfo::SpergeletInfo(double nu, int j, int q, const GSParamsPtr& gsparams) :
        _nu(nu), _j(j), _q(q), _gsparams(gsparams),
        _gamma_nup1(boost::math::tgamma(_nu+1.0)),
        _gamma_nup2(_gamma_nup1 * (_nu+1.0)),
        _gamma_nupjp1(boost::math::tgamma(_nu+_j+1.0)),
        _knorm(_gamma_nupjp1/_gamma_nup1),
        _maxk(0.), _stepk(0.),
        _ft(Table<double,double>::spline)
    {
        dbg<<"Start SpergeletInfo constructor for nu = "<<_nu<<std::endl;

        if (_nu < sbp::minimum_spergel_nu || _nu > sbp::maximum_spergel_nu)
            throw SBError("Requested Spergel index out of range");
    }

    // Use Spergel properties to set Spergelet properties (maxK, stepK)
    class SpergelIntegratedFlux
    {
    public:
        SpergelIntegratedFlux(double nu, double gamma_nup2, double flux_frac=0.0)
            : _nu(nu), _gamma_nup2(gamma_nup2),  _target(flux_frac) {}

        double operator()(double u) const
        // Return flux integrated up to radius `u` in units of r0, minus `flux_frac`
        // (i.e., make a residual so this can be used to search for a target flux.
        {
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

    double SpergeletInfo::calculateFluxRadius(const double& flux_frac) const
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

    double SpergeletInfo::stepK() const
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

    double SpergeletInfo::maxK() const
    {
        if(_maxk == 0.) {
            // Setting maxK based on Spergel profile associate with this Spergelet.
            // Solving (1+k^2)^(-1-nu) = maxk_threshold for k
            // exact:
            //_maxk = std::sqrt(std::pow(gsparams->maxk_threshold, -1./(1+_nu))-1.0);
            // approximate 1+k^2 ~ k^2 => good enough:
            _maxk = std::pow(_gsparams->maxk_threshold, -1./(2*(1+_nu)));
        }
        return _maxk;
    }

    // TODO: Figure this out!
    double SpergeletInfo::getXNorm() const
    { return 1.0; }

    double SpergeletInfo::kValue(double ksq, double phi) const
    {
        return _knorm * std::pow(ksq, _j) * std::pow(1.+ksq, -1.-_nu-_j) * cos(2*_q*phi);
    }

//     double SpergeletInfo::xValue(double r, double phi) const
//     {
//         if (_ft.size() == 0) buildFT();
//         return _ft(r) * std::cos(2*_q*phi);
//     }

//     class SpergeletIntegrand : public std::unary_function<double, double>
//     {
//     public:
//         SpergeletIntegrand(double nu, int j, int q, double r) :
//             _nu(nu), _j(j), _q(q), _r(r) {}
//         double operator()(double k) const
//         { return std::pow(k, 2.*_j)*std::pow(1+k*k, -1.-_nu-_j)
//                  *k*boost::math::cyl_bessel_j(2*_q, k*_r);}

//     private:
//         double _nu;
//         int _j;
//         int _q;
//         double _r;
//     };

//     void SpergeletInfo::buildFT() const
//     {
//         if (_ft.size() > 0) return;
//         dbg<<"Building Spergelet Hankel transform"<<std::endl;
//         dbg<<"nu = "<<_nu<<std::endl;
//         // Do a Hankel transform and store the results in a lookup table.
//         double prefactor = _knorm;
//         dbg<<"prefactor = "<<prefactor<<std::endl;

//         // // Along the way, find the last k that has a kValue > 1.e-3
//         // double maxk_val = this->_gsparams->maxk_threshold;
//         // dbg<<"Looking for maxk_val = "<<maxk_val<<std::endl;
//         // // Keep going until at least 5 in a row have kvalues below kvalue_accuracy.
//         // // (It's oscillatory, so want to make sure not to stop at a zero crossing.)

//         // We use a cubic spline for the interpolation, which has an error of O(h^4) max(f'''').
//         // I have no idea what range the fourth derivative can take for the Hankel transform,
//         // so let's take the completely arbitrary value of 10.  (This value was found to be
//         // conservative for Sersic, but I haven't investigated here.)
//         // 10 h^4 <= xvalue_accuracy
//         // h = (xvalue_accuracy/10)^0.25
//         double dr = _gsparams->table_spacing * sqrt(sqrt(_gsparams->xvalue_accuracy / 10.));
//         dbg<<"Using dr = "<<dr<<std::endl;

//         double rmin = dr; // have to begin somewhere...
//         for (double r = rmin; r < M_PI/_stepk; r += dr) {
//             SpergeletIntegrand I(_nu, _j, _q, r);

// #ifdef DEBUGLOGGING
//             std::ostream* integ_dbgout = verbose_level >= 3 ? dbgout : 0;
//             integ::IntRegion<double> reg(0, 30.0, integ_dbgout);
// #else
//             integ::IntRegion<double> reg(0, 30.0);
// #endif

//             // Add explicit splits at first several roots of Jn.
//             // This tends to make the integral more accurate.
//             for (int s=1; s<=30; ++s) {
//                 double root = boost::math::cyl_bessel_j_zero(double(2*_q), s);
//                 if (root > r*30.0) break;
//                 xxdbg<<"root="<<root/r<<std::endl;
//                 reg.addSplit(root/r);
//             }
//             xxdbg<<"int reg = ("<<0<<","<<30<<")"<<std::endl;

//             double val = integ::int1d(
//                 I, reg,
//                 this->_gsparams->integration_relerr,
//                 this->_gsparams->integration_abserr);
//             val *= prefactor;

//             xdbg<<"ft("<<r<<") = "<<val<<"   "<<val<<std::endl;

//             _ft.addEntry(r, val);
//         }
//     }
}
