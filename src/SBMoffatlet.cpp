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

#include "SBMoffatlet.h"
#include "SBMoffatletImpl.h"
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

    SBMoffatlet::SBMoffatlet(double beta, double r0, int j, int q, const GSParamsPtr& gsparams) :
        SBProfile(new SBMoffatletImpl(beta, r0, j, q, gsparams)) {}

    SBMoffatlet::SBMoffatlet(const SBMoffatlet& rhs) : SBProfile(rhs) {}

    SBMoffatlet::~SBMoffatlet() {}

    double SBMoffatlet::getBeta() const
    {
        assert(dynamic_cast<const SBMoffatletImpl*>(_pimpl.get()));
        return static_cast<const SBMoffatletImpl&>(*_pimpl).getBeta();
    }

    double SBMoffatlet::getScaleRadius() const
    {
        assert(dynamic_cast<const SBMoffatletImpl*>(_pimpl.get()));
        return static_cast<const SBMoffatletImpl&>(*_pimpl).getScaleRadius();
    }

    int SBMoffatlet::getJ() const
    {
        assert(dynamic_cast<const SBMoffatletImpl*>(_pimpl.get()));
        return static_cast<const SBMoffatletImpl&>(*_pimpl).getJ();
    }

    int SBMoffatlet::getQ() const
    {
        assert(dynamic_cast<const SBMoffatletImpl*>(_pimpl.get()));
        return static_cast<const SBMoffatletImpl&>(*_pimpl).getQ();
    }

    LRUCache<boost::tuple<double,int,int,GSParamsPtr>,MoffatletInfo>
        SBMoffatlet::SBMoffatletImpl::cache(sbp::max_moffatlet_cache);

    SBMoffatlet::SBMoffatletImpl::SBMoffatletImpl(double beta, double r0,
                                                  int j, int q,
                                                  const GSParamsPtr& gsparams) :
        SBProfileImpl(gsparams),
        _beta(beta), _r0(r0), _j(j), _q(q),
        _info(cache.get(boost::make_tuple(_beta, _j, _q, this->gsparams.duplicate())))
    {
        if ((j < 0) or (q < -j) or (q > j))
            throw SBError("Requested Moffatlet indices out of range");

        dbg<<"Start SBMoffatlet constructor:\n";
        dbg<<"beta = "<<_beta<<std::endl;
        dbg<<"r0 = "<<_r0<<std::endl;
        dbg<<"(j,q) = ("<<_j<<","<<_q<<")"<<std::endl;

        _r0_sq = _r0 * _r0;
        _inv_r0 = 1. / _r0;
        _inv_r0_sq = _inv_r0 * _inv_r0;
        _xnorm = _info->getXNorm() / _r0_sq;
        //_xnorm = boost::math::tgamma(_beta+_j) / boost::math::tgamma(_beta) / _r0_sq;
    }

    double SBMoffatlet::SBMoffatletImpl::maxK() const { return _info->maxK() * _inv_r0; }
    double SBMoffatlet::SBMoffatletImpl::stepK() const { return _info->stepK() * _inv_r0; }

    double SBMoffatlet::SBMoffatletImpl::xValue(const Position<double>& p) const
    {
        double rsq = (p.x * p.x + p.y * p.y) * _inv_r0_sq;
        double phi = atan2(p.y, p.x);
        return _xnorm * _info->xValue(rsq, phi);
    }

    std::complex<double> SBMoffatlet::SBMoffatletImpl::kValue(const Position<double>& k) const
    {
        double ksq = (k.x*k.x + k.y*k.y) * _r0_sq;
        double phi = atan2(k.y, k.x);
        return _info->kValue(ksq, phi);
    }

    void SBMoffatlet::SBMoffatletImpl::fillXValue(tmv::MatrixView<double> val,
                                                  double x0, double dx, int izero,
                                                  double y0, double dy, int jzero) const
    {
        dbg<<"SBMoffatlet fillXValue\n";
        dbg<<"x = "<<x0<<" + i * "<<dx<<", izero = "<<izero<<std::endl;
        dbg<<"y = "<<y0<<" + j * "<<dy<<", jzero = "<<jzero<<std::endl;
        // Not sure about quadrant.  Moffatlets are sometimes even and sometimes odd.
        // if (izero != 0 || jzero != 0) {
        //     xdbg<<"Use Quadrant\n";
        //     fillXValueQuadrant(val,x0,dx,izero,y0,dy,jzero);
        //     // Spergels can be super peaky at the center, so handle explicitly like Sersics
        //     if (izero != 0 && jzero != 0)
        //         val(izero, jzero) = _xnorm * _info->xValue(0.);
        // } else {
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
                    double phi = atan2(y0, x);
                    *valit++ = _xnorm * _info->xValue(rsq, phi);
                }
            }
        // }
    }

    void SBMoffatlet::SBMoffatletImpl::fillXValue(tmv::MatrixView<double> val,
                                                  double x0, double dx, double dxy,
                                                  double y0, double dy, double dyx) const
    {
        dbg<<"SBMoffatlet fillXValue\n";
        dbg<<"x = "<<x0<<" + i * "<<dx<<" + j * "<<dxy<<std::endl;
        dbg<<"y = "<<y0<<" + i * "<<dyx<<" + j * "<<dy<<std::endl;
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
                double phi = atan2(y,x);
                *valit++ = _xnorm * _info->xValue(rsq, phi);
            }
        }
    }

    void SBMoffatlet::SBMoffatletImpl::fillKValue(tmv::MatrixView<std::complex<double> > val,
                                                  double kx0, double dkx, int izero,
                                                  double ky0, double dky, int jzero) const
    {
        dbg<<"SBMoffatlet fillKValue\n";
        dbg<<"kx = "<<kx0<<" + i * "<<dkx<<", izero = "<<izero<<std::endl;
        dbg<<"ky = "<<ky0<<" + i * "<<dky<<", jzero = "<<jzero<<std::endl;
        // Not sure about quadrant.  Moffatlets have different symmetries than most other profiles.
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

    void SBMoffatlet::SBMoffatletImpl::fillKValue(tmv::MatrixView<std::complex<double> > val,
                                                  double kx0, double dkx, double dkxy,
                                                  double ky0, double dky, double dkyx) const
    {
        dbg<<"SBMoffatlet fillKValue\n";
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

    MoffatletInfo::MoffatletInfo(double beta, int j, int q, const GSParamsPtr& gsparams) :
        _beta(beta), _j(j), _q(q), _gsparams(gsparams),
        _maxk(0.), _stepk(0.),
        _xnorm(0.),
        _ft(Table<double,double>::spline)
    {
        dbg<<"Start MoffatletInfo constructor for beta = "<<_beta<<std::endl;

        if (_beta < sbp::minimum_moffat_beta || _beta > sbp::maximum_moffat_beta)
            throw SBError("Requested Moffat index out of range");
    }

    double MoffatletInfo::getXNorm() const
    {
        if (_xnorm == 0.0) {
            dbg<<"_beta = "<<_beta<<std::endl;
            dbg<<"_j = "<<_j<<std::endl;
            _xnorm = (_beta-1.0)/M_PI * boost::math::tgamma(_beta+_j)/boost::math::tgamma(_beta);
        }
        return _xnorm;
    }

    // Use Moffat properties to set Moffatlet properties (maxK, stepK)
    // TODO: update this
    // class MoffatIntegratedFlux
    // {
    // public:
    //     MoffatIntegratedFlux(double beta, double flux_frac=0.0)
    //         : _beta(beta), _target(flux_frac) {}

    //     double operator()(double u) const
    //     // Return flux integrated up to radius `u` in units of r0, minus `flux_frac`
    //     // (i.e., make a residual so this can be used to search for a target flux.
    //     {
    //         // double fnup1 = std::pow(u / 2., _nu+1.)
    //         //     * boost::math::cyl_bessel_k(_nu+1., u)
    //         //     / _gamma_nup2;
    //         // double f = 1.0 - 2.0 * (1.+_nu)*fnup1;
    //         // return f - _target;
    //         return 1.0;
    //     }
    // private:
    //     double _beta;
    //     double _target;
    // };

    // TODO: update this!
    // double MoffatletInfo::calculateFluxRadius(const double& flux_frac) const
    // {
    //     // Calcute r such that L(r/r0) / L_tot == flux_frac

    //     // These seem to bracket pretty much every reasonable possibility
    //     // that I checked in Mathematica, though I'm not very happy about
    //     // the lower bound....
    //     double z1=1.e-16;
    //     double z2=20.0;
    //     MoffatIntegratedFlux func(_beta, flux_frac);
    //     Solve<MoffatIntegratedFlux> solver(func, z1, z2);
    //     solver.setMethod(Brent);
    //     solver.bracketLowerWithLimit(0.0); // Just in case...
    //     double R = solver.root();
    //     dbg<<"flux_frac = "<<flux_frac<<std::endl;
    //     dbg<<"r/r0 = "<<R<<std::endl;
    //     return R;
    // }

    double MoffatletInfo::getHalfLightRadius() const
    {
        return std::sqrt(std::pow(0.5, 1./(1.-_beta)) - 1.);
    }

    // Use code from SBMoffat, but ignore truncation.
    double MoffatletInfo::stepK() const
    {
        if (_stepk == 0.) {
            // Ignore the 1 in (1+R^2), so approximately:
            double R = std::pow(_gsparams->folding_threshold, 0.5/(1.-_beta));
            dbg<<"R => "<<R<<std::endl;
            dbg<<"stepk = "<<(M_PI/R)<<std::endl;
            // Make sure it is at least 5 hlr
            R = std::max(R,_gsparams->stepk_minimum_hlr*getHalfLightRadius());
            _stepk = M_PI / R;
        }
        return _stepk;
    }

    // Use code from SBMoffat, but ignore truncation.
    double MoffatletInfo::maxK() const
    {
        if (_maxk == 0.) {
            // f(k) = 4 K(beta-1,k) (k/2)^beta / Gamma(beta-1)
            //
            // The asymptotic formula for K(beta-1,k) is
            //     K(beta-1,k) ~= sqrt(pi/(2k)) exp(-k)
            //
            // So f(k) becomes
            //
            // f(k) ~= 2 sqrt(pi) (k/2)^(beta-1/2) exp(-k) / Gamma(beta-1)
            //
            // Solve for f(k) = maxk_threshold
            //
            double temp = (_gsparams->maxk_threshold
                           * boost::math::tgamma(_beta-1.)
                           * std::pow(2.,_beta-0.5)
                           / (2. * sqrt(M_PI)));
            // Solve k^(beta-1/2) exp(-k) = temp
            // (beta-1/2) log(k) - k = log(temp)
            // k = (beta-1/2) log(k) - log(temp)
            temp = std::log(temp);
            _maxk = -temp;
            dbg<<"temp = "<<temp<<std::endl;
            for (int i=0;i<10;++i) {
                _maxk = (_beta-0.5) * std::log(_maxk) - temp;
                dbg<<"_maxk = "<<_maxk<<std::endl;
            }
        }
        return _maxk;
    }

    double MoffatletInfo::xValue(double rsq, double phi) const
    {
        double amplitude = std::pow(rsq, _j) * std::pow(1.+rsq, -_beta-_j);
        if (_q > 0)
            return amplitude * cos(2*_q*phi);
        else if (_q < 0)
            return amplitude * sin(2*_q*phi);
        else
            return amplitude;
    }

    double MoffatletInfo::kValue(double ksq, double phi) const
    {
        if (_ft.size() == 0) buildFT();
        double amplitude = _ft(std::sqrt(ksq));
        if (_q > 0) {
            if (_q & 1) // if q is odd
                return -amplitude * cos(2*_q*phi);
            else // q even
                return amplitude * cos(2*_q*phi);
        } else if (_q < 0) {
            if (_q & 1) // if q is odd
                return -amplitude * sin(2*_q*phi);
            else // q even
                return amplitude * sin(2*_q*phi);
        } else
            return amplitude;
    }

    class MoffatletIntegrand : public std::unary_function<double, double>
    {
    public:
        MoffatletIntegrand(double beta, int j, int q, double k) :
            _beta(beta), _j(j), _q(q), _k(k) {}
        double operator()(double r) const
        {
            return std::pow(r, 2.*_j)*std::pow(1+r*r, -_beta-_j)
                *r*boost::math::cyl_bessel_j(2*_q, _k*r);
        }
    private:
        double _beta;
        int _j;
        int _q;
        double _k;
    };

    void MoffatletInfo::buildFT() const
    {
        if (_ft.size() > 0) return;
        dbg<<"Building Moffatlet Hankel transform"<<std::endl;
        dbg<<"beta = "<<_beta<<std::endl;
        // Do a Hankel transform and store the results in a lookup table.
        //double prefactor = 2*M_PI*_xnorm/boost::math::tgamma(_j+1);
        double prefactor = 2.0*(_beta-1.0)*boost::math::tgamma(_beta+_j)/boost::math::tgamma(_beta);
        dbg<<"prefactor = "<<prefactor<<std::endl;

        // // Along the way, find the last k that has a kValue > 1.e-3
        // double maxk_val = this->_gsparams->maxk_threshold;
        // dbg<<"Looking for maxk_val = "<<maxk_val<<std::endl;
        // // Keep going until at least 5 in a row have kvalues below kvalue_accuracy.
        // // (It's oscillatory, so want to make sure not to stop at a zero crossing.)

        // We use a cubic spline for the interpolation, which has an error of O(h^4) max(f'''').
        // I have no idea what range the fourth derivative can take for the Hankel transform,
        // so let's take the completely arbitrary value of 10.  (This value was found to be
        // conservative for Sersic, but I haven't investigated here.)
        // 10 h^4 <= xvalue_accuracy
        // h = (xvalue_accuracy/10)^0.25
        double dk = _gsparams->table_spacing * sqrt(sqrt(_gsparams->kvalue_accuracy / 10.));
        double kmax = maxK();
        dbg<<"Using dk = "<<dk<<std::endl;
        dbg<<"Max k = "<<kmax<<std::endl;

        double kmin = 0.0;
        for (double k = kmin; k < kmax; k += dk) {
            MoffatletIntegrand I(_beta, _j, _q, k);

#ifdef DEBUGLOGGING
            std::ostream* integ_dbgout = verbose_level >= 3 ? dbgout : 0;
            integ::IntRegion<double> reg(0, integ::MOCK_INF, integ_dbgout);
#else
            integ::IntRegion<double> reg(0, integ::MOCK_INF);
#endif

            // Add explicit splits at first several roots of Jn.
            // This tends to make the integral more accurate.
            if (k != 0.0) {
                for (int s=1; s<=30; ++s) {
                    double root = boost::math::cyl_bessel_j_zero(double(2*_q), s);
                    //if (root > r*30.0) break;
                    xxdbg<<"root="<<root/k<<std::endl;
                    reg.addSplit(root/k);
                }
            }
            xxdbg<<"int reg = ("<<0<<","<<30<<")"<<std::endl;

            double val = integ::int1d(
                I, reg,
                this->_gsparams->integration_relerr,
                this->_gsparams->integration_abserr);
            val *= prefactor;

            xdbg<<"ft("<<k<<") = "<<val<<"   "<<val<<std::endl;

            _ft.addEntry(k, val);
        }
    }
}
