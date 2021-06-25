/* -*- c++ -*-
 * Copyright (c) 2012-2021 by the GalSim developers team on GitHub
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

// See https://www.dropbox.com/s/z6h14bgd199czsi/Inclined_Exponential.pdf?dl=0
// for a write-up of much of the math involved in this file.

// #define DEBUGLOGGING

#include "galsim/IgnoreWarnings.h"

#define BOOST_NO_CXX11_SMART_PTR

#include "SBInclinedExponential.h"
#include "SBInclinedExponentialImpl.h"
#include "integ/Int.h"
#include "Solve.h"

namespace galsim {

    SBInclinedExponential::SBInclinedExponential(
            double inclination, double scale_radius, double scale_height,
            double flux, const GSParams& gsparams) :
        SBProfile(new SBInclinedExponentialImpl(
                inclination, scale_radius, scale_height, flux, gsparams))
    {}

    SBInclinedExponential::SBInclinedExponential(const SBInclinedExponential& rhs) :
        SBProfile(rhs)
    {}

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

    SBInclinedExponential::SBInclinedExponentialImpl::SBInclinedExponentialImpl(
            double inclination, double scale_radius,
            double scale_height, double flux, const GSParams& gsparams) :
        SBProfileImpl(gsparams),
        _inclination(inclination),
        _r0(scale_radius),
        _h0(scale_height),
        _flux(flux),
        _inv_r0(1./scale_radius),
        _half_pi_h_sini_over_r(0.5*M_PI*scale_height*std::abs(std::sin(inclination))/scale_radius),
        _cosi(std::abs(std::cos(inclination))),
        _ksq_max(integ::MOCK_INF) // Start with infinite _ksq_max so we can use kValueHelper to
                                  // get a better value
    {
        dbg<<"Start SBInclinedExponential constructor:\n";
        dbg<<"inclination = "<<_inclination<<std::endl;
        dbg<<"scale radius = "<<_r0<<std::endl;
        dbg<<"scale height = "<<_h0<<std::endl;
        dbg<<"flux = "<<_flux<<std::endl;

        // Now set up, using this value of cosi

        double cosi_squared = _cosi*_cosi;

        xdbg<<"_half_pi_h_sini_over_r = "<<_half_pi_h_sini_over_r<<std::endl;
        xdbg<<"_cosi = "<<_cosi<<std::endl;

        // Calculate stepk, based on a conservative comparison to an exponential disk. The
        // half-light radius of this will be smaller, so if we use an exponential's hlr, it
        // will be at least large enough.

        // int( exp(-r) r, r=0..R) = (1 - exp(-R) - Rexp(-R))
        // Fraction excluded is thus (1+R) exp(-R)
        // A fast solution to (1+R)exp(-R) = x:
        // log(1+R) - R = log(x)
        // R = log(1+R) - log(x)
        double logx = std::log(this->gsparams.folding_threshold);
        double R = -logx;
        for (int i=0; i<3; i++) R = std::log(1.+R) - logx;
        // Make sure it is at least 5 hlr of corresponding exponential
        // half-light radius = 1.6783469900166605 * r0
        const double exp_hlr = 1.6783469900166605;
        R = std::max(R,this->gsparams.stepk_minimum_hlr*exp_hlr);
        _stepk = M_PI / R;
        dbg<<"stepk = "<<_stepk<<std::endl;

        // For small k, we can use up to quartic in the taylor expansion of both terms
        // in the calculation.
        // This is acceptable when the next term is less than kvalue_accuracy.
        // (35/16 + 31/15120 pi/2*h*sin(i)/r) * (k^2*r^2)^3 = kvalue_accuracy
        // This is a bit conservative, note, assuming kx = 0
        _ksq_min = std::pow(this->gsparams.kvalue_accuracy /
                            (35./16. + 31./15120.*_half_pi_h_sini_over_r), 1./3.);

        // Solve for the proper _maxk and _ksq_max

        double maxk_min = std::pow(this->gsparams.maxk_threshold, -1./3.);
        double clipk_min = std::pow(this->gsparams.kvalue_accuracy, -1./3.);

        // Check for face-on case, which doesn't need the solver
        if(_cosi==1)
        {
            _maxk = maxk_min;
            _ksq_max = clipk_min*clipk_min;
        }
        else // Use the solver
        {
            // Bracket it appropriately, starting with guesses based on the 1/cosi scaling
            double maxk_max, clipk_max;
            // Check bounds on _cosi to make sure initial guess range isn't too big or small
            if(_cosi>0.01)
            {
                if(_cosi<0.96)
                {
                    maxk_max = maxk_min/_cosi;
                    clipk_max = clipk_min/_cosi;
                }
                else
                {
                    maxk_max = 1.05*maxk_min;
                    clipk_max = 1.05*clipk_min;
                }
            }
            else
            {
                maxk_max = 100*maxk_min;
                clipk_max = 100*clipk_min;
            }

            xdbg << "maxk_threshold = " << this->gsparams.maxk_threshold << std::endl;
            xdbg << "F(" << maxk_min << ") = " << std::max(kValueHelper(maxk_min,0.),kValueHelper(0.,maxk_min)) << std::endl;
            xdbg << "F(" << maxk_max << ") = " << std::max(kValueHelper(maxk_max,0.),kValueHelper(0.,maxk_max)) << std::endl;

            SBInclinedExponentialKValueFunctor maxk_func(this,this->gsparams.maxk_threshold);
            Solve<SBInclinedExponentialKValueFunctor> maxk_solver(maxk_func, maxk_min, maxk_max);
            maxk_solver.setMethod(Brent);

            if(maxk_func(maxk_min)<=0)
                maxk_solver.bracketLowerWithLimit(0.);
            else
                maxk_solver.bracketUpper();

            // Get the _maxk from the solver here. We add back on the tolerance to the result to
            // ensure that the k-value will be below the threshold.
            _maxk = maxk_solver.root() + maxk_solver.getXTolerance();

            xdbg << "_maxk = " << _maxk << std::endl;
            xdbg << "F(" << _maxk << ") = " << kValueHelper(0.,_maxk) << std::endl;

            xdbg << "kvalue_accuracy = " << this->gsparams.kvalue_accuracy << std::endl;
            xdbg << "F(" << clipk_min << ") = " << kValueHelper(0.,clipk_min) << std::endl;
            xdbg << "F(" << clipk_max << ") = " << kValueHelper(0.,clipk_max) << std::endl;

            SBInclinedExponentialKValueFunctor clipk_func(this,this->gsparams.kvalue_accuracy);
            Solve<SBInclinedExponentialKValueFunctor> clipk_solver(clipk_func, clipk_min, clipk_max);

            if(clipk_func(clipk_min)<=0)
                clipk_solver.bracketLowerWithLimit(0.);
            else
                clipk_solver.bracketUpper();

            // Get the clipk from the solver here. We add back on the tolerance to the result to
            // ensure that the k-value will be below the threshold.
            double clipk = clipk_solver.root() + clipk_solver.getXTolerance();
            _ksq_max = clipk*clipk;

            xdbg << "clipk = " << clipk << std::endl;
            xdbg << "F(" << clipk << ") = " << kValueHelper(0.,clipk) << std::endl;
        }
    }

    double SBInclinedExponential::SBInclinedExponentialImpl::maxSB() const
    {
        // When the disk is face on, the max SB is flux / 2 pi r0^2
        // When the disk is edge on, the max SB is flux / 2 pi r0^2 * (r0/h0)
        double maxsb = _flux * _inv_r0 * _inv_r0 / (2. * M_PI);
        // The relationship for inclinations in between these is not linear.
        // Empirically, it is vaguely linearish in ln(maxsb) vs. sqrt(cosi), so we use that for
        // the interpolation.
        double sc = sqrt(std::abs(_cosi));
        maxsb *= std::exp(std::log(_r0/_h0) * (1.-sc));

        // Err on the side of overestimating by multiplying by conservative_factor,
        // which was found to work for the worst-case scenario
        return std::abs(maxsb);
    }

    double SBInclinedExponential::SBInclinedExponentialImpl::xValue(const Position<double>& p) const
    {
        throw std::runtime_error(
            "Real-space expression of SBInclinedExponential is not yet implemented.");
        return 0;
    }

    std::complex<double> SBInclinedExponential::SBInclinedExponentialImpl::kValue(
        const Position<double>& k) const
    {
        double kx = k.x*_r0;
        double ky = k.y*_r0;
        return _flux * kValueHelper(kx,ky);
    }

    template <typename T>
    void SBInclinedExponential::SBInclinedExponentialImpl::fillKImage(
        ImageView<std::complex<T> > im,
        double kx0, double dkx, int izero,
        double ky0, double dky, int jzero) const
    {
        dbg<<"SBInclinedExponential fillKImage\n";
        dbg<<"kx = "<<kx0<<" + i * "<<dkx<<", izero = "<<izero<<std::endl;
        dbg<<"ky = "<<ky0<<" + j * "<<dky<<", jzero = "<<jzero<<std::endl;
        if (izero != 0 || jzero != 0) {
            xdbg<<"Use Quadrant\n";
            fillKImageQuadrant(im,kx0,dkx,izero,ky0,dky,jzero);
        } else {
            xdbg<<"Non-Quadrant\n";
            const int m = im.getNCol();
            const int n = im.getNRow();
            std::complex<T>* ptr = im.getData();
            int skip = im.getNSkip();
            assert(im.getStep() == 1);

            kx0 *= _r0;
            dkx *= _r0;
            ky0 *= _r0;
            dky *= _r0;

            for (int j=0; j<n; ++j,ky0+=dky,ptr+=skip) {
                double kx = kx0;
                for (int i=0; i<m; ++i,kx+=dkx)
                    *ptr++ = _flux * kValueHelper(kx,ky0);
            }
        }
    }

    template <typename T>
    void SBInclinedExponential::SBInclinedExponentialImpl::fillKImage(
        ImageView<std::complex<T> > im,
        double kx0, double dkx, double dkxy,
        double ky0, double dky, double dkyx) const
    {
        dbg<<"SBInclinedExponential fillKImage\n";
        dbg<<"kx = "<<kx0<<" + i * "<<dkx<<" + j * "<<dkxy<<std::endl;
        dbg<<"ky = "<<ky0<<" + i * "<<dkyx<<" + j * "<<dky<<std::endl;
        const int m = im.getNCol();
        const int n = im.getNRow();
        std::complex<T>* ptr = im.getData();
        int skip = im.getNSkip();
        assert(im.getStep() == 1);

        kx0 *= _r0;
        dkx *= _r0;
        dkxy *= _r0;
        ky0 *= _r0;
        dky *= _r0;
        dkyx *= _r0;

        for (int j=0; j<n; ++j,kx0+=dkxy,ky0+=dky,ptr+=skip) {
            double kx = kx0;
            double ky = ky0;
            for (int i=0; i<m; ++i,kx+=dkx,ky+=dkyx)
                *ptr++ = _flux * kValueHelper(kx,ky);
        }
    }

    double SBInclinedExponential::SBInclinedExponentialImpl::maxK() const
    { return _maxk * _inv_r0; }
    double SBInclinedExponential::SBInclinedExponentialImpl::stepK() const
    { return _stepk * _inv_r0; }

    double SBInclinedExponential::SBInclinedExponentialImpl::kValueHelper(
        double kx, double ky) const
    {
        // Calculate the base value for an exponential profile

        double ky_cosi = ky*_cosi;

        double ky_cosi_sq = ky_cosi*ky_cosi;
        double ksq = kx*kx + ky_cosi_sq;
        double res_base;
        if (ksq > _ksq_max)
        {
            return 0.;
        }
        else if (ksq < _ksq_min)
        {
            res_base = (1. - 1.5*ksq*(1. - 1.25*ksq));

            xxdbg << "res_base (upper limit) = " << res_base << std::endl;
        }
        else
        {
            double temp = 1. + ksq;
            res_base =  1./(temp*sqrt(temp));

            xxdbg << "res_base (normal) = " << res_base << std::endl;
        }

        // Calculate the convolution factor
        double res_conv;

        double scaled_ky = _half_pi_h_sini_over_r*ky;
        double scaled_ky_squared = scaled_ky*scaled_ky;

        if (scaled_ky_squared < _ksq_min)
        {
            // Use Taylor expansion to speed up calculation
            res_conv = (1. - 0.16666666667*scaled_ky_squared *
                          (1. - 0.116666666667*scaled_ky_squared));
            xxdbg << "res_conv (lower limit) = " << res_conv << std::endl;
        }
        else
        {
            res_conv = scaled_ky / std::sinh(scaled_ky);
            xxdbg << "res_conv (normal) = " << res_conv << std::endl;
        }


        double res = res_base*res_conv;

        return res;
    }

    // Not yet implemented, but needs to be defined
    void SBInclinedExponential::SBInclinedExponentialImpl::shoot(
        PhotonArray& photons, UniformDeviate ud) const
    {
        throw std::runtime_error(
            "Photon shooting not yet implemented for SBInclinedExponential profile.");
    }

    SBInclinedExponential::SBInclinedExponentialImpl::
        SBInclinedExponentialKValueFunctor::SBInclinedExponentialKValueFunctor(
            const SBInclinedExponential::SBInclinedExponentialImpl * p_owner,
            double target_k_value) :
        _p_owner(p_owner), _target_k_value(target_k_value)
    {}

    double SBInclinedExponential::SBInclinedExponentialImpl::
        SBInclinedExponentialKValueFunctor::operator()(double k) const
    {
        assert(_p_owner);
        double k_value = std::max(_p_owner->kValueHelper(0.,k),_p_owner->kValueHelper(k,0.));
        return k_value - _target_k_value;
    }


}
