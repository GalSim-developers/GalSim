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
//std::ostream* dbgout = &std::cout;
//int verbose_level = 1;
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

    std::string SBSpergelet::SBSpergeletImpl::repr() const
    {
        std::ostringstream oss(" ");
        oss.precision(std::numeric_limits<double>::digits10 + 4);
        oss << "galsim._galsim.SBSpergelet("<<getNu()<<", "<<getScaleRadius();
        oss << ", " << getJ() << ", " << getQ() << ", galsim.GSParams("<<*gsparams<<"))";
        return oss.str();
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
        if ((j < 0) or (q < -j) or (q > j))
            throw SBError("Requested Spergelet indices out of range");

        dbg<<"Start SBSpergelet constructor:\n";
        dbg<<"nu = "<<_nu<<std::endl;
        dbg<<"r0 = "<<_r0<<std::endl;
        dbg<<"(j,q) = ("<<_j<<","<<_q<<")"<<std::endl;

        _r0_sq = _r0 * _r0;
        _inv_r0 = 1. / _r0;
    }

    double SBSpergelet::SBSpergeletImpl::maxK() const { return _info->maxK() * _inv_r0; }
    double SBSpergelet::SBSpergeletImpl::stepK() const { return _info->stepK() * _inv_r0; }

    // Modified equation (47) of Spergel (2010)
    std::complex<double> SBSpergelet::SBSpergeletImpl::kValue(const Position<double>& k) const
    {
        double ksq = (k.x*k.x + k.y*k.y) * _r0_sq;
        double phi = atan2(k.y, k.x);
        return _info->kValue(ksq, phi);
    }

    // // TODO: Figure this out
    // double SBSpergelet::SBSpergeletImpl::getFlux() const
    // {
    //     return 1.;
    // }

    void SBSpergelet::SBSpergeletImpl::fillKValue(tmv::MatrixView<std::complex<double> > val,
                                                  double kx0, double dkx, int izero,
                                                  double ky0, double dky, int jzero) const
    {
        dbg<<"SBSpergelet fillKValue\n";
        dbg<<"kx = "<<kx0<<" + i * "<<dkx<<", izero = "<<izero<<std::endl;
        dbg<<"ky = "<<ky0<<" + i * "<<dky<<", jzero = "<<jzero<<std::endl;
        // Not sure about quadrant.  Spergelets have different symmetries than most other profiles.
        // if (izero != 0 || jzero != 0) {
        //     xdbg<<"Use Quadrant\n";
        //     fillKValueQuadrant(val,kx0,dkx,izero,ky0,dky,jzero);
        // } else {
            xdbg<<"Non-Quadrant\n";
            assert(val.stepi() == 1);
            const int m = val.colsize();
            const int n = val.rowsize();
            typedef tmv::VIt<std::complex<double>,1,tmv::NonConj> It;

            kx0 *= _r0;
            dkx *= _r0;
            ky0 *= _r0;
            dky *= _r0;

            for (int j=0;j<n;++j,ky0+=dky) {
                double kx = kx0;
                double kysq = ky0*ky0;
                //It valit(val.col(j).begin().getP(),1);
                It valit = val.col(j).begin();
                for (int i=0;i<m;++i,kx+=dkx) {
                    double ksq = kx*kx + kysq;
                    double phi = atan2(ky0, kx);
                    *valit++ = _info->kValue(ksq, phi);
                }
            }
        // }
    }

    void SBSpergelet::SBSpergeletImpl::fillKValue(tmv::MatrixView<std::complex<double> > val,
                                                  double kx0, double dkx, double dkxy,
                                                  double ky0, double dky, double dkyx) const
    {
        dbg<<"SBSpergelet fillKValue\n";
        dbg<<"x = "<<kx0<<" + i * "<<dkx<<" + j * "<<dkxy<<std::endl;
        dbg<<"y = "<<ky0<<" + i * "<<dkyx<<" + j * "<<dky<<std::endl;
        assert(val.stepi() == 1);
        assert(val.canLinearize());
        const int m = val.colsize();
        const int n = val.rowsize();
        typedef tmv::VIt<std::complex<double>,1,tmv::NonConj> It;

        kx0 *= _r0;
        dkx *= _r0;
        dkxy *= _r0;
        ky0 *= _r0;
        dky *= _r0;
        dkyx *= _r0;

        It valit = val.linearView().begin();
        for (int j=0;j<n;++j,kx0+=dkxy,ky0+=dky) {
            double kx = kx0;
            double ky = ky0;
            for (int i=0;i<m;++i,kx+=dkx,ky+=dkyx) {
                double ksq = kx*kx + ky*ky;
                double phi = atan2(ky,kx);
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
        _maxk(0.), _stepk(0.)
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

    double SpergeletInfo::kValue(double ksq, double phi) const
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
