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

#include "SBLinearOpticalet.h"
#include "SBLinearOpticaletImpl.h"
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

    SBLinearOpticalet::SBLinearOpticalet(double r0, int n1, int m1, int n2, int m2,
                                         const GSParamsPtr& gsparams) :
        SBProfile(new SBLinearOpticaletImpl(r0, n1, m1, n2, m2, gsparams)) {}

    SBLinearOpticalet::SBLinearOpticalet(const SBLinearOpticalet& rhs) : SBProfile(rhs) {}

    SBLinearOpticalet::~SBLinearOpticalet() {}

    double SBLinearOpticalet::getScaleRadius() const
    {
        assert(dynamic_cast<const SBLinearOpticaletImpl*>(_pimpl.get()));
        return static_cast<const SBLinearOpticaletImpl&>(*_pimpl).getScaleRadius();
    }

    int SBLinearOpticalet::getN1() const
    {
        assert(dynamic_cast<const SBLinearOpticaletImpl*>(_pimpl.get()));
        return static_cast<const SBLinearOpticaletImpl&>(*_pimpl).getN1();
    }

    int SBLinearOpticalet::getM1() const
    {
        assert(dynamic_cast<const SBLinearOpticaletImpl*>(_pimpl.get()));
        return static_cast<const SBLinearOpticaletImpl&>(*_pimpl).getM1();
    }

    int SBLinearOpticalet::getN2() const
    {
        assert(dynamic_cast<const SBLinearOpticaletImpl*>(_pimpl.get()));
        return static_cast<const SBLinearOpticaletImpl&>(*_pimpl).getN2();
    }

    int SBLinearOpticalet::getM2() const
    {
        assert(dynamic_cast<const SBLinearOpticaletImpl*>(_pimpl.get()));
        return static_cast<const SBLinearOpticaletImpl&>(*_pimpl).getM2();
    }

    LRUCache<boost::tuple<int,int,int,int,GSParamsPtr>,LinearOpticaletInfo>
        SBLinearOpticalet::SBLinearOpticaletImpl::cache(sbp::max_linearopticalet_cache);

    SBLinearOpticalet::SBLinearOpticaletImpl::SBLinearOpticaletImpl(double r0,
                                                                    int n1, int m1,
                                                                    int n2, int m2,
                                                                    const GSParamsPtr& gsparams) :
        SBProfileImpl(gsparams),
        _r0(r0), _n1(n1), _m1(m1), _n2(n2), _m2(m2),
        _info(cache.get(boost::make_tuple(_n1, _m1, _n2, _m2, this->gsparams.duplicate())))
    {
        if ((n1 < 0) or (n2 < 0) or (m1 < -n1) or (m1 > n1) or (m2 < -n2) or (m2 > n2))
            throw SBError("Requested LinearOpticalet indices out of range");

        dbg<<"Start LinearOpticalet constructor:\n";
        dbg<<"r0 = "<<_r0<<std::endl;
        dbg<<"(n1,m1) = ("<<_n1<<","<<_m1<<")"<<std::endl;
        dbg<<"(n2,m2) = ("<<_n2<<","<<_m2<<")"<<std::endl;

        _r0_sq = _r0 * _r0;
        _inv_r0 = 1. / _r0;
        _inv_r0_sq = _inv_r0 * _inv_r0;
        _xnorm = _info->getXNorm() / _r0_sq;
    }

    double SBLinearOpticalet::SBLinearOpticaletImpl::maxK() const
    { return _info->maxK() * _inv_r0; }
    double SBLinearOpticalet::SBLinearOpticaletImpl::stepK() const
    { return _info->stepK() * _inv_r0; }

    double SBLinearOpticalet::SBLinearOpticaletImpl::xValue(const Position<double>& p) const
    {
        double r = sqrt(p.x * p.x + p.y * p.y) * _inv_r0;
        double phi = atan2(p.y, p.x);
        return _xnorm * _info->xValue(r, phi);
    }

    std::complex<double> SBLinearOpticalet::SBLinearOpticaletImpl::kValue(const Position<double>& k) const
    {
        double ksq = (k.x*k.x + k.y*k.y) * _r0_sq;
        double phi = atan2(k.y, k.x);
        return _info->kValue(ksq, phi);
    }

    void SBLinearOpticalet::SBLinearOpticaletImpl::fillXValue(tmv::MatrixView<double> val,
                                                              double x0, double dx, int izero,
                                                              double y0, double dy, int jzero) const
    {
        dbg<<"SBLinearOpticalet fillXValue\n";
        dbg<<"x = "<<x0<<" + i * "<<dx<<", izero = "<<izero<<std::endl;
        dbg<<"y = "<<y0<<" + j * "<<dy<<", jzero = "<<jzero<<std::endl;
        // Not sure about quadrant.  LinearOpticalets are sometimes even and sometimes odd.
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
                    double r = sqrt(x*x + ysq);
                    double phi = atan2(y0, x);
                    *valit++ = _xnorm * _info->xValue(r, phi);
                }
            }
        // }
    }

    // void SBLinearOpticalet::SBLinearOpticaletImpl::fillKValue(tmv::MatrixView<std::complex<double> > val,
    //                                               double kx0, double dkx, int izero,
    //                                               double ky0, double dky, int jzero) const
    // {
    //     dbg<<"SBLinearOpticalet fillKValue\n";
    //     dbg<<"kx = "<<kx0<<" + i * "<<dkx<<", izero = "<<izero<<std::endl;
    //     dbg<<"ky = "<<ky0<<" + i * "<<dky<<", jzero = "<<jzero<<std::endl;
    //     // Not sure about quadrant.  LinearOpticalets are sometimes even, sometimes odd.
    //     // if (izero != 0 || jzero != 0) {
    //     //     xdbg<<"Use Quadrant\n";
    //     //     fillKValueQuadrant(val,kx0,dkx,izero,ky0,dky,jzero);
    //     // } else {
    //         xdbg<<"Non-Quadrant\n";
    //         assert(val.stepi() == 1);
    //         const int m = val.colsize();
    //         const int n = val.rowsize();
    //         typedef tmv::VIt<std::complex<double>,1,tmv::NonConj> It;

    //         kx0 *= _r0;
    //         dkx *= _r0;
    //         ky0 *= _r0;
    //         dky *= _r0;

    //         for (int j=0;j<n;++j,ky0+=dky) {
    //             double kx = kx0;
    //             double kysq = ky0*ky0;
    //             It valit = val.col(j).begin();
    //             for (int i=0;i<m;++i,kx+=dkx) {
    //                 double ksq = kx*kx + kysq;
    //                 double phi = atan2(ky0, kx);
    //                 *valit++ = _info->kValue(ksq, phi);
    //             }
    //         }
    //     // }
    // }

    void SBLinearOpticalet::SBLinearOpticaletImpl::fillXValue(tmv::MatrixView<double> val,
                                                              double x0, double dx, double dxy,
                                                              double y0, double dy, double dyx) const
    {
        dbg<<"SBLinearOpticalet fillXValue\n";
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
                double r = sqrt(x*x + y*y);
                double phi = atan2(y,x);
                *valit++ = _xnorm * _info->xValue(r, phi);
            }
        }
    }

    // void SBLinearOpticalet::SBLinearOpticaletImpl::fillKValue(tmv::MatrixView<std::complex<double> > val,
    //                                               double kx0, double dkx, double dkxy,
    //                                               double ky0, double dky, double dkyx) const
    // {
    //     dbg<<"SBLinearOpticalet fillKValue\n";
    //     dbg<<"x = "<<kx0<<" + i * "<<dkx<<" + j * "<<dkxy<<std::endl;
    //     dbg<<"y = "<<ky0<<" + i * "<<dkyx<<" + j * "<<dky<<std::endl;
    //     assert(val.stepi() == 1);
    //     assert(val.canLinearize());
    //     const int m = val.colsize();
    //     const int n = val.rowsize();
    //     typedef tmv::VIt<std::complex<double>,1,tmv::NonConj> It;

    //     kx0 *= _r0;
    //     dkx *= _r0;
    //     dkxy *= _r0;
    //     ky0 *= _r0;
    //     dky *= _r0;
    //     dkyx *= _r0;

    //     It valit = val.linearView().begin();
    //     for (int j=0;j<n;++j,kx0+=dkxy,ky0+=dky) {
    //         double kx = kx0;
    //         double ky = ky0;
    //         for (int i=0;i<m;++i,kx+=dkx,ky+=dkyx) {
    //             double ksq = kx*kx + ky*ky;
    //             double phi = atan2(ky,kx);
    //             *valit++ = _info->kValue(ksq, phi);
    //         }
    //     }
    // }

    LinearOpticaletInfo::LinearOpticaletInfo(int n1, int m1, int n2, int m2,
                                             const GSParamsPtr& gsparams) :
        _n1(n1), _m1(m1), _n2(n2), _m2(m2), _gsparams(gsparams),
        _maxk(0.), _stepk(0.),
        _xnorm(0.),
        _ft(Table<double,double>::spline)
    {
        dbg<<"Start LinearOpticaletInfo constructor for (n1,m1,n2,m2) = ("
           <<_n1<<","<<_m1<<","<<_n2<<","<<_m2<<")"<<std::endl;
    }

    double LinearOpticaletInfo::getXNorm() const
    {
        if (_xnorm == 0.0) {
            dbg<<"(n1,m1,n2,m2) = ("<<_n1<<","<<_m1<<","<<_n2<<","<<_m2<<")"<<std::endl;
            //_xnorm = (_beta-1.0)/M_PI * boost::math::tgamma(_beta+_j)/boost::math::tgamma(_beta);
        }
        return _xnorm;
    }

    // work with respect to Airy
    double LinearOpticaletInfo::getHalfLightRadius() const
    {
        return 1.0;
        //return std::sqrt(std::pow(0.5, 1./(1.-_beta)) - 1.);
    }

    // Use code from SBAiry, but ignore obscuration.
    double LinearOpticaletInfo::stepK() const
    {
        // stole this very roughly from SBAiry
        if (_stepk == 0.) {
            double R = 1./ ( _gsparams->folding_threshold * 0.5 * M_PI * M_PI);
            _stepk = M_PI / R;
        }
        return _stepk;
    }

    // Use code from SBAiry, but ignore obscuration.
    double LinearOpticaletInfo::maxK() const
    {
        if (_maxk == 0.) {
            _maxk = 2.*M_PI;
        }
        return _maxk;
    }

    // The workhorse routines...
    double LinearOpticaletInfo::xValue(double r, double phi) const
    {
        double ret = boost::math::cyl_bessel_j(_n1+1, r)*boost::math::cyl_bessel_j(_n2+1, r)/r/r;
        if ((_n1-_m1+_n2-_m2) & 2) // if (n1-m1 + n2-m2) = 2 mod 4
            ret *= -1;
        if (_m1 > 0)
            ret *= cos(_m1 * phi);
        else if (_m1 < 0)
            ret *= sin(_m1 * phi);
        if (_m2 > 0)
            ret *= cos(_m2 * phi);
        else if (_m2 < 0)
            ret *= sin(_m2 * phi);
        return ret
    }

    std::complex<double> LinearOpticaletInfo::kValue(double ksq, double phi) const
    {
        if (_ftsum.size() == 0) buildFT();
        int msum = _m1+_m2;
        int mdiff = _m1-_m2;
        double ampsum = _ftsum(std::sqrt(ksq));
        double ampdiff = _ftdiff(std::sqrt(ksq));
        double sumcoeff = 0.0;
        double diffcoeff = 0.0;
        if (_m1 >= 0) { // cos term
            if (_m2 >= 0) { // cos term
                sumcoeff = cos(msum*phi);
                diffcoeff = cos(mdiff*phi);
            } else { // sin term
                sumcoeff = sin(msum*phi);
                diffcoeff = -sin(mdiff*phi);
            }
        } else { // sin term
            if (_m2 >= 0) { // cos term
                sumcoeff = cos(msum*phi);
                diffcoeff = cos(mdiff*phi);
            } else { // sin term
                sumcoeff = -sin(msum*phi);
                diffcoeff = sin(mdiff*phi);
            }
        }
        ret = std::complex<double>(0.0, 0.0);
        // contribution from m1+m2 order Hankel transform
        if (msum & 1) { // imag sum output
            if (msum & 2) // neg
                ret += std::complex<double>(0.0, -0.5*ampsum*sumcoeff);
            else // pos
                ret += std::complex<double>(0.0, 0.5*ampsum*sumcoeff);
        } else { // real sum output
            if (msum & 2) // neg
                ret += -0.5*ampsum*sumcoeff;
            else
                ret += 0.5*ampsum*sumcoeff;
        }
        // contribution from m1-m2 order Hankel transform
        if (mdiff & 1) { // imag diff output
            if (mdiff & 2) // neg
                ret += std::complex<double>(0.0, -0.5*ampdiff*diffcoeff);
            else // pos
                ret += std::complex<double>(0.0, 0.5*ampdiff*diffcoeff);
        } else { // real diff output
            if (mdiff & 2) // neg
                ret += -0.5*ampdiff*diffcoeff;
            else
                ret += 0.5*ampdiff*diffcoeff;
        }
        return ret;
    }

    class LinearOpticaletIntegrandSum : public std::unary_function<double, double>
    {
    public:
        LinearOpticaletIntegrandSum(int n1, int m1, int n2, int m2, double k) :
            _n1(n1), _m1(m1), _n2(n2), _m2(m2), _k(k) {}
        double operator()(double r) const
        {
            return boost::math::cyl_bessel_j(_n1+1, r) * boost::math::cyl_bessel_j(_n2+1, r)
                * boost::math::cyl_bessel_j(_m1+_m2, r*_k);
        }
    private:
        int _n1, _m1, _n2, _m2;
        double _k;
    };

    class LinearOpticaletIntegrandDiff : public std::unary_function<double, double>
    {
    public:
        LinearOpticaletIntegrandDiff(int n1, int m1, int n2, int m2, double k) :
            _n1(n1), _m1(m1), _n2(n2), _m2(m2), _k(k) {}
        double operator()(double r) const
        {
            return boost::math::cyl_bessel_j(_n1+1, r) * boost::math::cyl_bessel_j(_n2+1, r)
                * boost::math::cyl_bessel_j(_m1-_m2, r*_k);
        }
    private:
        int _n1, _m1, _n2, _m2;
        double _k;
    };

    void LinearOpticaletInfo::buildFT() const
    {
        if (_ft.size() > 0) return;
        dbg<<"Building LinearOpticalet Hankel transform"<<std::endl;
        dbg<<"(n1,m1,n2,m2) = ("<<_n1<<","<<_m1<<","<<_n2<<","<<_m2<<")"<<std::endl;
        // Do a Hankel transform and store the results in a lookup table.
        double prefactor = 1;
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
            LinearOpticaletIntegrandSum Isum(_n1, _m1, _n2, _m2, k);
            LinearOpticaletIntegrandDiff Idiff(_n1, _m1, _n2, _m2, k);

#ifdef DEBUGLOGGING
            std::ostream* integ_dbgout = verbose_level >= 3 ? dbgout : 0;
            integ::IntRegion<double> regsum(0, integ::MOCK_INF, integ_dbgout);
            integ::IntRegion<double> regdiff(0, integ::MOCK_INF, integ_dbgout);
#else
            integ::IntRegion<double> regsum(0, integ::MOCK_INF);
            integ::IntRegion<double> regdiff(0, integ::MOCK_INF);
#endif

            // Add explicit splits at first several roots of Jn.
            // This tends to make the integral more accurate.
            if (k != 0.0) {
                for (int s=1; s<=30; ++s) {
                    double rootsum = boost::math::cyl_bessel_j_zero(double(_m1+_m2), s);
                    double rootdiff = boost::math::cyl_bessel_j_zero(double(_m1-_m2), s);
                    //if (root > r*30.0) break;
                    xxdbg<<"rootsum="<<rootsum/k<<std::endl;
                    xxdbg<<"rootdiff="<<rootdiff/k<<std::endl;
                    regsum.addSplit(rootsum/k);
                    regdiff.addSplit(rootdiff/k);
                }
            }
            xxdbg<<"int reg = ("<<0<<",inf)"<<std::endl;

            double valsum = integ::int1d(
                Isum, regsum,
                this->_gsparams->integration_relerr,
                this->_gsparams->integration_abserr);
            valsum *= prefactor;

            double valdiff = integ::int1d(
                Idiff, regdiff,
                this->_gsparams->integration_relerr,
                this->_gsparams->integration_abserr);
            valdiff *= prefactor;

            xdbg<<"ftsum("<<k<<") = "<<valsum<<"   "<<val<<std::endl;
            xdbg<<"ftdiff("<<k<<") = "<<valdiff<<"   "<<val<<std::endl;

            _ftsum.addEntry(k, valsum);
            _ftdiff.addEntry(k, valdiff);
        }
    }
}
