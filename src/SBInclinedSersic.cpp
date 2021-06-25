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

// #define DEBUGLOGGING
// #define VERBOSITY_LEVEL 1

#include "galsim/IgnoreWarnings.h"

#include "SBInclinedSersic.h"
#include "SBInclinedSersicImpl.h"
#include "SBSersic.h"
#include "SBSersicImpl.h"
#include "integ/Int.h"
#include "Solve.h"
#include <math/Gamma.h>

namespace galsim {

    // Helper functor to solve for the proper _maxk
    class SBInclinedSersic::SBInclinedSersicImpl::SBInclinedSersicKValueFunctor
    {
    public:
        SBInclinedSersicKValueFunctor(const SBInclinedSersic::SBInclinedSersicImpl * p_owner,
                                      double target_k_value);
        double operator() (double k) const;
    private:
        const SBInclinedSersic::SBInclinedSersicImpl * _p_owner;
        double _target_k_value;
    };

    SBInclinedSersic::SBInclinedSersic(double n, double inclination, double scale_radius,
                                       double height, double flux, double trunc,
                                       const GSParams& gsparams) :
        SBProfile(new SBInclinedSersicImpl(n, inclination, scale_radius, height, flux, trunc,
                                           gsparams)) {}

    SBInclinedSersic::SBInclinedSersic(const SBInclinedSersic& rhs) : SBProfile(rhs) {}

    SBInclinedSersic::~SBInclinedSersic() {}

    double SBInclinedSersic::getN() const
    {
        assert(dynamic_cast<const SBInclinedSersicImpl*>(_pimpl.get()));
        return static_cast<const SBInclinedSersicImpl&>(*_pimpl).getN();
    }

    double SBInclinedSersic::getInclination() const
    {
        assert(dynamic_cast<const SBInclinedSersicImpl*>(_pimpl.get()));
        return static_cast<const SBInclinedSersicImpl&>(*_pimpl).getInclination();
    }

    double SBInclinedSersic::getHalfLightRadius() const
    {
        assert(dynamic_cast<const SBInclinedSersicImpl*>(_pimpl.get()));
        return static_cast<const SBInclinedSersicImpl&>(*_pimpl).getHalfLightRadius();
    }

    double SBInclinedSersic::getScaleRadius() const
    {
        assert(dynamic_cast<const SBInclinedSersicImpl*>(_pimpl.get()));
        return static_cast<const SBInclinedSersicImpl&>(*_pimpl).getScaleRadius();
    }

    double SBInclinedSersic::getScaleHeight() const
    {
        assert(dynamic_cast<const SBInclinedSersicImpl*>(_pimpl.get()));
        return static_cast<const SBInclinedSersicImpl&>(*_pimpl).getScaleHeight();
    }

    double SBInclinedSersic::getTrunc() const
    {
        assert(dynamic_cast<const SBInclinedSersicImpl*>(_pimpl.get()));
        return static_cast<const SBInclinedSersicImpl&>(*_pimpl).getTrunc();
    }

    SBInclinedSersic::SBInclinedSersicImpl::SBInclinedSersicImpl(
        double n, double inclination, double scale_radius,
        double height, double flux, double trunc, const GSParams& gsparams) :
        SBProfileImpl(gsparams),
        _n(n),
        _inclination(inclination),
        _flux(flux),
        _r0(scale_radius),
        _h0(height),
        _trunc(trunc),
        _cosi(std::abs(std::cos(inclination))),
        _trunc_sq(trunc*trunc),
        _ksq_max(integ::MOCK_INF), // Start with infinite _ksq_max so we can use kValueHelper to
                                  // get a better value
        // Start with untruncated SersicInfo regardless of value of trunc
        _info(SBSersic::SBSersicImpl::cache.get(MakeTuple(_n, _trunc/_r0,
                                                          GSParamsPtr(this->gsparams))))
    {
        dbg<<"Start SBInclinedSersic constructor:\n";
        dbg<<"n = "<<_n<<std::endl;
        dbg<<"inclination = "<<_inclination<<std::endl;
        dbg<<"scale_radius = "<<scale_radius<<std::endl;
        dbg<<"height = "<<height<<std::endl;
        dbg<<"flux = "<<_flux<<std::endl;
        dbg<<"trunc = "<<_trunc<<std::endl;

        _re = _r0 * _info->getHLR();
        dbg << "hlr = " <<_re << std::endl;
        dbg << "r0 = " <<_r0 << std::endl;

        _inv_r0 = 1./_r0;
        dbg << "inv_r0 = " << _inv_r0 << std::endl;

        dbg << "scale height = "<<_h0<<std::endl;

        _half_pi_h_sini_over_r = 0.5*M_PI*_h0*std::abs(std::sin(_inclination))/_r0;

        dbg << "half_pi_h_sini_over_r = " << _half_pi_h_sini_over_r << std::endl;

        _r0_sq = _r0*_r0;
        _inv_r0 = 1./_r0;
        _inv_r0_sq = _inv_r0*_inv_r0;

        _xnorm = _flux * _info->getXNorm() * _inv_r0_sq;
        dbg<<"xnorm = "<<_xnorm<<std::endl;

        // For small k, we can use up to quartic in the taylor expansion of both terms
        // in the calculation.
        // This is acceptable when the next term is less than kvalue_accuracy.
        // (35/16 + 31/15120 pi/2*h*sin(i)/r) * (k^2*r^2)^3 = kvalue_accuracy
        // This is a bit conservative, note, assuming kx = 0
        double kderiv6 = 31./15120.*_half_pi_h_sini_over_r;
        _ksq_min = std::pow(this->gsparams.kvalue_accuracy / kderiv6, 1./3.);

        dbg << "ksq_min = " << _ksq_min << std::endl;

        // Solve for the proper _maxk and _ksq_max

        double maxk_min = std::pow(this->gsparams.maxk_threshold, -1./3.);
        double clipk_min = std::pow(this->gsparams.kvalue_accuracy, -1./3.);

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

        SBInclinedSersicKValueFunctor maxk_func(this,this->gsparams.maxk_threshold);
        Solve<SBInclinedSersicKValueFunctor> maxk_solver(maxk_func, maxk_min, maxk_max);

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

        SBInclinedSersicKValueFunctor clipk_func(this,this->gsparams.kvalue_accuracy);
        Solve<SBInclinedSersicKValueFunctor> clipk_solver(clipk_func, clipk_min, clipk_max);

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

        dbg << "info maxk = " << _info->maxK() << std::endl;
        dbg << "maxk = " << _maxk << std::endl;
        dbg << "ksq_max = " << _ksq_max << std::endl;
        dbg << "info stepk = " << _info->stepK() << std::endl;
    }

    double SBInclinedSersic::SBInclinedSersicImpl::maxSB() const
    {
        // When the disk is face on, the max SB is _xnorm
        // When the disk is edge on, the max SB is _xnorm * _h0/_r0 * gamma(_n)/_n
        double maxsb = _xnorm;
        // The relationship for inclinations in between these is not linear.
        // Empirically, it is vaguely linearish in ln(maxsb) vs. sqrt(cosi), so we use that for
        // the interpolation.
        double sc = sqrt(std::abs(_cosi));
        maxsb *= std::exp((1.-sc)*std::log((_r0 * math::tgamma(_n) ) / _h0*_n));

        // Err on the side of overestimating by multiplying by conservative_factor,
        // which was found to work for the worst-case scenario
        return std::abs(maxsb);
    }

    double SBInclinedSersic::SBInclinedSersicImpl::xValue(const Position<double>& p) const
    {
        throw std::runtime_error(
            "Real-space expression of SBInclinedSersic is not yet implemented.");
        return 0;
    }

    std::complex<double> SBInclinedSersic::SBInclinedSersicImpl::kValue(const Position<double>& k) const
    {
        double kx = k.x*_r0;
        double ky = k.y*_r0;
        return _flux * kValueHelper(kx,ky);
    }

    template <typename T>
    void SBInclinedSersic::SBInclinedSersicImpl::fillKImage(ImageView<std::complex<T> > im,
                                                double kx0, double dkx, int izero,
                                                double ky0, double dky, int jzero) const
    {
        dbg<<"SBInclinedSersic fillKImage\n";
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
                for (int i=0;i<m;++i,kx+=dkx)
                    *ptr++ = _flux * kValueHelper(kx,ky0);
            }
        }
    }

    template <typename T>
    void SBInclinedSersic::SBInclinedSersicImpl::fillKImage(ImageView<std::complex<T> > im,
                                                double kx0, double dkx, double dkxy,
                                                double ky0, double dky, double dkyx) const
    {
        dbg<<"SBInclinedSersic fillKImage\n";
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

    double SBInclinedSersic::SBInclinedSersicImpl::maxK() const
    {
        return _maxk * _inv_r0;
    }
    double SBInclinedSersic::SBInclinedSersicImpl::stepK() const
    {
        double stepk = _info->stepK() * _inv_r0;
        return stepk;
    }

    double SBInclinedSersic::SBInclinedSersicImpl::kValueHelper(
        double kx, double ky) const
    {
        // Get the base value for a Sersic profile

        xxdbg << "Calling SBInclinedSersic::SBInclinedSersicImpl::kValueHelper on " << kx << ", " << ky << "." << std::endl;

        double ky_cosi = ky*_cosi;

        double ky_cosi_sq = ky_cosi*ky_cosi;
        double ksq = kx*kx + ky_cosi_sq;
        double res_base;
        if (ksq > _ksq_max)
        {
            return 0.;
        }
        else
        {
            res_base =  _info->kValue(ksq);

            xxdbg << "res_base = " << res_base << std::endl;
        }

        // Calculate the convolution factor
        double res_conv;

        double scaled_ky = _half_pi_h_sini_over_r*ky;
        double scaled_ky_squared = scaled_ky*scaled_ky;

        xxdbg << "scaled_ky = " << scaled_ky << std::endl;

        if (scaled_ky_squared < _ksq_min)
        {
            // Use Taylor expansion to speed up calculation
            res_conv = (1. - 0.16666666667*scaled_ky_squared *
                          (1. - 0.116666666667*scaled_ky_squared));
            xxdbg << "res_conv (lower limit) = " << res_conv << "; ksq_min = " << _ksq_min << std::endl;
        }
        else
        {
            res_conv = scaled_ky / std::sinh(scaled_ky);
            xxdbg << "res_conv (normal) = " << res_conv << "; ksq_min = " << _ksq_min << std::endl;
        }


        double res = res_base*res_conv;

        return res;
    }

    void SBInclinedSersic::SBInclinedSersicImpl::shoot(
        PhotonArray& photons, UniformDeviate ud) const
    {
        throw std::runtime_error(
            "Photon shooting not yet implemented for SBInclinedSersic profile.");
    }

    SBInclinedSersic::SBInclinedSersicImpl::
        SBInclinedSersicKValueFunctor::SBInclinedSersicKValueFunctor(
            const SBInclinedSersic::SBInclinedSersicImpl * p_owner,
            double target_k_value) :
        _p_owner(p_owner), _target_k_value(target_k_value)
    {}

    double SBInclinedSersic::SBInclinedSersicImpl::
        SBInclinedSersicKValueFunctor::operator()(double k) const
    {
        assert(_p_owner);
        double kx_value = _p_owner->kValueHelper(k,0.);
        double ky_value = _p_owner->kValueHelper(0.,k);
        double k_value = std::max(kx_value,ky_value);

        xdbg << "k = " << k << "; k_value = " << k_value << "; target_k_value = " << _target_k_value << std::endl;
        xdbg << "kx_value = " << kx_value << "; ky_value = " << ky_value << std::endl;

        return k_value - _target_k_value;
    }
}
