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

#include "SBKolmogorovlet.h"
#include "SBKolmogorovletImpl.h"
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

    SBKolmogorovlet::SBKolmogorovlet(double r0, int j, int q, const GSParamsPtr& gsparams) :
        SBProfile(new SBKolmogorovletImpl(r0, j, q, gsparams)) {}

    SBKolmogorovlet::SBKolmogorovlet(const SBKolmogorovlet& rhs) : SBProfile(rhs) {}

    SBKolmogorovlet::~SBKolmogorovlet() {}

    double SBKolmogorovlet::getScaleRadius() const
    {
        assert(dynamic_cast<const SBKolmogorovletImpl*>(_pimpl.get()));
        return static_cast<const SBKolmogorovletImpl&>(*_pimpl).getScaleRadius();
    }

    int SBKolmogorovlet::getJ() const
    {
        assert(dynamic_cast<const SBKolmogorovletImpl*>(_pimpl.get()));
        return static_cast<const SBKolmogorovletImpl&>(*_pimpl).getJ();
    }

    int SBKolmogorovlet::getQ() const
    {
        assert(dynamic_cast<const SBKolmogorovletImpl*>(_pimpl.get()));
        return static_cast<const SBKolmogorovletImpl&>(*_pimpl).getQ();
    }

    LRUCache<boost::tuple<int,int,GSParamsPtr>,KolmogorovletInfo>
        SBKolmogorovlet::SBKolmogorovletImpl::cache(sbp::max_kolmogorovlet_cache);

    SBKolmogorovlet::SBKolmogorovletImpl::SBKolmogorovletImpl(double r0, int j, int q,
                                                              const GSParamsPtr& gsparams) :
        SBProfileImpl(gsparams), _r0(r0), _j(j), _q(q),
        _info(cache.get(boost::make_tuple(_j, _q, this->gsparams.duplicate())))
    {
        if ((j < 0) or (q < -j) or (q > j))
            throw SBError("Requested Kolmogorovlet indices out of range");

        dbg<<"Start SBKolmogorovlet constructor:\n";
        dbg<<"r0 = "<<_r0<<std::endl;
        dbg<<"(j,q) = ("<<_j<<","<<_q<<")"<<std::endl;

        _r0_sq = _r0 * _r0;
        _inv_r0 = 1. / _r0;
        _xnorm = _info->getXNorm() / _r0_sq;
    }

    double SBKolmogorovlet::SBKolmogorovletImpl::maxK() const { return _info->maxK() * _inv_r0; }
    double SBKolmogorovlet::SBKolmogorovletImpl::stepK() const { return _info->stepK() * _inv_r0; }

    // double SBKolmogorovlet::SBKolmogorovletImpl::xValue(const Position<double>& p) const
    // {
    //     double r = sqrt(p.x * p.x + p.y * p.y) * _inv_r0;
    //     double phi = atan2(p.y, p.x);
    //     return _xnorm * _info->xValue(r, phi);
    // }

    std::complex<double> SBKolmogorovlet::SBKolmogorovletImpl::kValue(const Position<double>& k) const
    {
        double ksq = (k.x*k.x + k.y*k.y) * _r0_sq;
        double phi = atan2(k.y, k.x);
        return _info->kValue(ksq, phi);
    }

    // TODO: Figure this out
    double SBKolmogorovlet::SBKolmogorovletImpl::getFlux() const
    {
        return 1.;
    }

    void SBKolmogorovlet::SBKolmogorovletImpl::fillKValue(
             tmv::MatrixView<std::complex<double> > val,
             double x0, double dx, int ix_zero,
             double y0, double dy, int iy_zero) const
    {
        dbg<<"SBKolmogorovlet fillKValue\n";
        dbg<<"x = "<<x0<<" + ix * "<<dx<<", ix_zero = "<<ix_zero<<std::endl;
        dbg<<"y = "<<y0<<" + iy * "<<dy<<", iy_zero = "<<iy_zero<<std::endl;
        // Not sure about quadrant.  Kolmogorovlets are sometimes even, sometimes odd.
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

    void SBKolmogorovlet::SBKolmogorovletImpl::fillKValue(
              tmv::MatrixView<std::complex<double> > val,
              double x0, double dx, double dxy,
              double y0, double dy, double dyx) const
    {
        dbg<<"SBKolmogorovlet fillKValue\n";
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

    KolmogorovletInfo::KolmogorovletInfo(int j, int q, const GSParamsPtr& gsparams) :
        _j(j), _q(q), _gsparams(gsparams),
        _maxk(0.), _stepk(0.);

    // Use Kolmogorov properties to set Kolmogorovlet properties (maxK, stepK)
    // This needs to be updated!
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

    // This needs to be updated!
    double KolmogorovletInfo::calculateFluxRadius(const double& flux_frac) const
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

    // This needs to be updated!
    double KolmogorovletInfo::stepK() const
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

    // This needs to be updated!
    double KolmogorovletInfo::maxK() const
    {
        if(_maxk == 0.) {
            // Setting maxK based on Spergel profile associate with this Kolmogorovlet.
            // Solving (1+k^2)^(-1-nu) = maxk_threshold for k
            // exact:
            //_maxk = std::sqrt(std::pow(gsparams->maxk_threshold, -1./(1+_nu))-1.0);
            // approximate 1+k^2 ~ k^2 => good enough:
            _maxk = std::pow(_gsparams->maxk_threshold, -1./(2*(1+_nu)));
        }
        return _maxk;
    }

    // TODO: Figure this out!
    double KolmogorovletInfo::getXNorm() const
    { return 1.0; }

    // This needs to be updated!
    double KolmogorovletInfo::kValue(double ksq, double phi) const
    {
        double amplitude = _knorm * std::pow(ksq, _j) * std::pow(1.+ksq, -1.-_nu-_j);
        if (_q > 0)
            return amplitude * cos(2*_q*phi);
        else if (_q < 0)
            return amplitude * sin(2*_q*phi);
        else
            return amplitude;
    }
}
