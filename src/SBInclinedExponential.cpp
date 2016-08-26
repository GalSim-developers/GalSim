/* -*- c++ -*-
 * Copyright (c) 2012-2016 by the GalSim developers team on GitHub
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

#include "galsim/IgnoreWarnings.h"

#define BOOST_NO_CXX11_SMART_PTR

#include "SBInclinedExponential.h"
#include "SBInclinedExponentialImpl.h"
#include "integ/Int.h"
#include "Solve.h"

#ifdef DEBUGLOGGING
#include <fstream>
std::ostream* dbgout = new std::ofstream("debug.out");
// std::ostream* dbgout = &std::cout;
int verbose_level = 3;
#endif

namespace galsim {

    SBInclinedExponential::SBInclinedExponential(double i, double scale_radius, double scale_height, double flux,
            const GSParamsPtr& gsparams) :
        SBProfile(new SBInclinedExponentialImpl(i, scale_radius, scale_height, flux, gsparams)) {}

    SBInclinedExponential::SBInclinedExponential(const SBInclinedExponential& rhs) : SBProfile(rhs) {}

    SBInclinedExponential::~SBInclinedExponential() {}

    double SBInclinedExponential::getInclination() const
    {
        assert(dynamic_cast<const SBInclinedExponentialImpl*>(_pimpl.get()));
        return static_cast<const SBInclinedExponentialImpl&>(*_pimpl).getInclination();
    }

    double SBInclinedExponential::getScaleRadius() const
    {
        assert(dynamic_cast<const SBInclinedExponentialImpl*>(_pimpl.get()));
        return static_cast<const SBInclinedExponentialImpl&>(*_pimpl).getScaleRadius();
    }

    double SBInclinedExponential::getScaleHeight() const
    {
        assert(dynamic_cast<const SBInclinedExponentialImpl*>(_pimpl.get()));
        return static_cast<const SBInclinedExponentialImpl&>(*_pimpl).getScaleHeight();
    }

    // NB.  This function is virtually wrapped by repr() in SBProfile.cpp
    std::string SBInclinedExponential::SBInclinedExponentialImpl::serialize() const
    {
        std::ostringstream oss(" ");
        // NB. The choice of digits10 + 4 is because the normal general output
        // scheme for double uses fixed notation if >= 0.0001, but then switches
        // to scientific for smaller numbers.  So those first 4 digits in 0.0001 don't
        // count for the number of required digits, which is nominally given by digits10.
        // cf. http://stackoverflow.com/questions/4738768/printing-double-without-losing-precision
        // Unfortunately, there doesn't seem to be an easy equivalent of python's %r for
        // printing the repr of a double that always works and doesn't print more digits than
        // it needs.  This is the reason why we reimplement the __repr__ methods in python
        // for all the SB classes except SBProfile.  Only the last one can't be done properly
        // in python, so it will use the C++ virtual function to get the right thing for
        // any subclass.  But possibly with ugly extra digits.
        oss.precision(std::numeric_limits<double>::digits10 + 4);
        oss << "galsim._galsim.SBInclinedExponential("<<getInclination()<<", "<<getScaleRadius()<<", "<<getScaleHeight();
        oss <<", "<<getFlux()<<", False";
        oss << ", galsim.GSParams("<<*gsparams<<"))";
        return oss.str();
    }

    LRUCache< boost::tuple<double, GSParamsPtr >, InclinedExponentialInfo >
        SBInclinedExponential::SBInclinedExponentialImpl::cache(sbp::max_inclined_exponential_cache);

    SBInclinedExponential::SBInclinedExponentialImpl::SBInclinedExponentialImpl(double inclination, double scale_radius,
            double scale_height, double flux, const GSParamsPtr& gsparams) :
        SBProfileImpl(gsparams),
        _inclination(inclination),
        _r0(scale_radius),
        _h0(scale_height),
        _flux(flux),
        _inv_r0(1./_r0),
        _info(cache.get(boost::make_tuple(_h_tani_over_r, this->gsparams.duplicate())))
    {
        dbg<<"Start SBInclinedExponential constructor:\n";
        dbg<<"inclination = "<<_inclination<<std::endl;
        dbg<<"scale radius = "<<_r0<<std::endl;
        dbg<<"scale height = "<<_h0<<std::endl;
        dbg<<"flux = "<<_flux<<std::endl;

        // Check if cos(inclination) is within allowed limits, and institute special handling if it isn't
        
        double cosi = std::abs(std::cos(_inclination));
        
        if(cosi<sbp::minimum_cosi)
        {
            // Perfectly edge-on isn't analytic, so we truncate at the minimum cos(inclination) value
            cosi = sbp::minimum_cosi;
        }
        
        // Now set up, using this value of cosi
        
        _r0_cosi = _r0*cosi;
        _inv_r0_cosi = 1./_r0_cosi;
        
        _h_tani_over_r = scale_height*std::abs(std::sin(inclination))*_inv_r0_cosi; // A tiny bit more accurate than using tan of
            // the truncated value

        _info = boost::shared_ptr<InclinedExponentialInfo>(cache.get(boost::make_tuple(_h_tani_over_r, this->gsparams.duplicate())));

        xdbg<<"_h_tani_over_r = "<<_h_tani_over_r<<std::endl;
        xdbg<<"_r0_cosi = "<<_r0_cosi<<std::endl;

        /* Shooting NYI
        _shootnorm = _flux * _info->getXNorm(); // For shooting, we don't need the 1/r0^2 factor.
        _xnorm = _shootnorm * _inv_r0 * _inv_r0 ;
        dbg<<"norms = "<<_xnorm<<", "<<_shootnorm<<std::endl;
        */
    }

    double SBInclinedExponential::SBInclinedExponentialImpl::xValue(const Position<double>& p) const
    {
        throw std::runtime_error("Real-space expression of SBInclinedExponential NYI.");
        return 0;
    }

    std::complex<double> SBInclinedExponential::SBInclinedExponentialImpl::kValue(const Position<double>& k) const
    {
        double kx = k.x*_r0;
        double ky = k.y*_r0_cosi;
        return _flux * _info->kValue(kx,ky);
    }

    void SBInclinedExponential::SBInclinedExponentialImpl::fillKValue(tmv::MatrixView<std::complex<double> > val,
                                            double kx0, double dkx, int izero,
                                            double ky0, double dky, int jzero) const
    {
        dbg<<"SBInclinedExponential fillKValue\n";
        dbg<<"kx = "<<kx0<<" + i * "<<dkx<<", izero = "<<izero<<std::endl;
        dbg<<"ky = "<<ky0<<" + j * "<<dky<<", jzero = "<<jzero<<std::endl;
        if (izero != 0 || jzero != 0) {
            xdbg<<"Use Quadrant\n";
            fillKValueQuadrant(val,kx0,dkx,izero,ky0,dky,jzero);
        } else {
            xdbg<<"Non-Quadrant\n";
            assert(val.stepi() == 1);
            const int m = val.colsize();
            const int n = val.rowsize();
            typedef tmv::VIt<std::complex<double>,1,tmv::NonConj> It;

            kx0 *= _r0;
            dkx *= _r0;
            ky0 *= _r0_cosi;
            dky *= _r0_cosi;

            for (int j=0;j<n;++j,ky0+=dky) {
                double kx = kx0;
                It valit = val.col(j).begin();
                for (int i=0;i<m;++i,kx+=dkx) {
                    double new_val = _flux * _info->kValue(kx,ky0);
                    xxdbg << "kx = " << kx << "\tky = " << ky0 << "\tval = " << new_val << std::endl;
                    *valit++ = _flux * _info->kValue(kx,ky0);
                }
            }
        }
    }

    void SBInclinedExponential::SBInclinedExponentialImpl::fillKValue(tmv::MatrixView<std::complex<double> > val,
                                            double kx0, double dkx, double dkxy,
                                            double ky0, double dky, double dkyx) const
    {
        dbg<<"SBInclinedExponential fillKValue\n";
        dbg<<"kx = "<<kx0<<" + i * "<<dkx<<" + j * "<<dkxy<<std::endl;
        dbg<<"ky = "<<ky0<<" + i * "<<dkyx<<" + j * "<<dky<<std::endl;
        assert(val.stepi() == 1);
        assert(val.canLinearize());
        const int m = val.colsize();
        const int n = val.rowsize();
        typedef tmv::VIt<std::complex<double>,1,tmv::NonConj> It;

        kx0 *= _r0;
        dkx *= _r0;
        dkxy *= _r0;
        ky0 *= _r0_cosi;
        dky *= _r0_cosi;
        dkyx *= _r0_cosi;

        It valit = val.linearView().begin();
        for (int j=0;j<n;++j,kx0+=dkxy,ky0+=dky) {
            double kx = kx0;
            double ky = ky0;
            for (int i=0;i<m;++i,kx+=dkx,ky+=dkyx) {
                *valit++ = _flux * _info->kValue(kx,ky);
            }
        }
    }

    double SBInclinedExponential::SBInclinedExponentialImpl::maxK() const { return _info->maxK() * _inv_r0; }
    double SBInclinedExponential::SBInclinedExponentialImpl::stepK() const { return _info->stepK() * _inv_r0; }

    InclinedExponentialInfo::InclinedExponentialInfo(double h_tani_over_r, const GSParamsPtr& gsparams) :
        _h_tani_over_r(h_tani_over_r),
        _half_pi_h_tani_over_r(0.5*M_PI*h_tani_over_r),
        _gsparams(gsparams),
        _maxk(0.), _stepk(0.)
    {
        dbg<<"Start InclinedExponentialInfo constructor for h_tani_over_r = "<<h_tani_over_r<<std::endl;

        assert(h_tani_over_r >= 0); // Should only ever have non-negative

        // For large k, we clip the result of kValue to 0.
        // We do this when the correct answer is less than kvalue_accuracy.
        // (1+k^2 r0^2)^-1.5 = kvalue_accuracy
        _ksq_max = (std::pow(gsparams->kvalue_accuracy,-1./1.5)-1.);

        // For small k, we can use up to quartic in the taylor expansion to avoid the sqrt.
        // This is acceptable when the next term is less than kvalue_accuracy.
        // 35/16 (k^2 r0^2)^3 = kvalue_accuracy
        _ksq_min = std::pow(gsparams->kvalue_accuracy * 16./35., 1./3.);

        // Calculate stepk (assuming that this is similar enough to an exponential disk):
        // int( exp(-r) r, r=0..R) = (1 - exp(-R) - Rexp(-R))
        // Fraction excluded is thus (1+R) exp(-R)
        // A fast solution to (1+R)exp(-R) = x:
        // log(1+R) - R = log(x)
        // R = log(1+R) - log(x)
        double logx = std::log(gsparams->folding_threshold);
        double R = -logx;
        for (int i=0; i<3; i++) R = std::log(1.+R) - logx;
        // Make sure it is at least 5 hlr
        // half-light radius = 1.6783469900166605 * r0
        const double hlr = 1.6783469900166605;
        R = std::max(R,gsparams->stepk_minimum_hlr*hlr);
        _stepk = M_PI / R;
        dbg<<"stepk = "<<_stepk<<std::endl;

        _maxk = std::sqrt(_ksq_max);
    }

    double InclinedExponentialInfo::stepK() const
    { return _stepk; }

    double InclinedExponentialInfo::maxK() const
    {
        return _maxk;
    }

    /* NYI
    double InclinedExponentialInfo::getXNorm() const
    {
        return 1.;
    }
    */

    /* NYI
    double InclinedExponentialInfo::xValue(double rx, double ry) const
    {
        if (_ift.getNx() == 0) buildIFT();

        double rsq = rx*rx + ry*ry;
        double theta;
        if(rx==0)
        {
            theta = 0;
        }
        else
        {
            theta = std::atan(std::abs(ry/rx));
        }

        double lr=0.5*std::log(rsq); // Lookup table is logarithmic
        return rsq*_ift(lr,theta);

    }
    */

    double InclinedExponentialInfo::kValue(double kx, double ky) const
    {
        // Calculate the base value for an exponential profile

        double kysq = ky*ky;
        double ksq = kx*kx + kysq;
        double res_base;
        if (ksq > _ksq_max)
        {
            return 0.;
        }
        else if (ksq < _ksq_min)
        {
            res_base = (1. - 1.5*ksq*(1. - 1.25*ksq));
        }
        else
        {
            double temp = 1. + ksq;
            res_base =  1./(temp*sqrt(temp));
        }

        // Calculate the convolution factor
        double res_conv;

        double scaled_ky = _half_pi_h_tani_over_r*ky;
        double scaled_ky_squared = scaled_ky*scaled_ky;

        if (scaled_ky_squared < _ksq_min)
        {
            // Use Taylor expansion to speed up calculation
            res_conv = (1. - 0.16666666667*scaled_ky_squared*(1. - 0.116666666667*scaled_ky_squared));
        }
        else
        {
            res_conv = _half_pi_h_tani_over_r*ky / std::sinh(_half_pi_h_tani_over_r*ky);
        }

        double res = res_base*res_conv;

        return res;
    }


    /* NYI
    void InclinedExponentialInfo::buildIFT() const
    {
        // To make the table, we'll have to fill in k values in an array, then perform an inverse transform
        assert(false);
    }
    */

    // NYI, but needs to be defined
    boost::shared_ptr<PhotonArray> InclinedExponentialInfo::shoot(int N, UniformDeviate ud) const
    {
        throw std::runtime_error("Photon shooting NYI for InclinedExponential profile.");
    }

    // NYI, but needs to be defined
    boost::shared_ptr<PhotonArray> SBInclinedExponential::SBInclinedExponentialImpl::shoot(int N, UniformDeviate ud) const
    {
        throw std::runtime_error("Photon shooting NYI for InclinedExponential profile.");
    }
}
