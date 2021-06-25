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

//#define DEBUGLOGGING

// To enable some extra debugging statements
//#define AIRY_DEBUG

#include "SBAiry.h"
#include "SBAiryImpl.h"
#include "math/Bessel.h"

namespace galsim {

    // Specialize the NewValue function used by LRUCache:
    template <>
    struct LRUCacheHelper<AiryInfo, Tuple<double, GSParamsPtr> >
    {
        static AiryInfo* NewValue(const Tuple<double, GSParamsPtr>& key)
        {
            const double obscuration = key.first;
            GSParamsPtr gsparams = key.second;
            if (obscuration == 0.0)
                return new AiryInfoNoObs(gsparams);
            else
                return new AiryInfoObs(obscuration, gsparams);
        }
    };

    SBAiry::SBAiry(double lam_over_D, double obscuration, double flux,
                   const GSParams& gsparams) :
        SBProfile(new SBAiryImpl(lam_over_D, obscuration, flux, gsparams)) {}

    SBAiry::SBAiry(const SBAiry& rhs) : SBProfile(rhs) {}

    SBAiry::~SBAiry() {}

    double SBAiry::getLamOverD() const
    {
        assert(dynamic_cast<const SBAiryImpl*>(_pimpl.get()));
        return static_cast<const SBAiryImpl&>(*_pimpl).getLamOverD();
    }

    double SBAiry::getObscuration() const
    {
        assert(dynamic_cast<const SBAiryImpl*>(_pimpl.get()));
        return static_cast<const SBAiryImpl&>(*_pimpl).getObscuration();
    }

    SBAiry::SBAiryImpl::SBAiryImpl(double lam_over_D, double obscuration, double flux,
                                   const GSParams& gsparams) :
        SBProfileImpl(gsparams),
        _lam_over_D(lam_over_D),
        _D(1. / lam_over_D),
        _obscuration(obscuration),
        _flux(flux),
        _Dsq(_D*_D), _obssq(_obscuration*_obscuration),
        _inv_D_pi(1. / (_D * M_PI)),
        _inv_Dsq_pisq(_inv_D_pi * _inv_D_pi),
        _xnorm(flux * _Dsq),
        _knorm(flux / (M_PI * (1.-_obssq))),
        _info(cache.get(MakeTuple(_obscuration, GSParamsPtr(gsparams))))
    {
        xdbg<<"SBAiryImpl constructor: gsparams = "<<gsparams<<std::endl;
    }

    LRUCache<Tuple<double, GSParamsPtr>, AiryInfo> SBAiry::SBAiryImpl::cache(sbp::max_airy_cache);

    // This is a scale-free version of the Airy radial function.
    // Input radius is in units of lambda/D.  Output normalized
    // to integrate to unity over input units.
    double AiryInfoObs::RadialFunction::operator()(double radius) const
    {
        double nu = radius*M_PI;
        // Taylor expansion of j1(u)/u = 1/2 - 1/16 x^2 + ...
        // We can truncate this to 1/2 when neglected term is less than xvalue_accuracy
        // (relative error, so divide by 1/2)
        // xvalue_accuracy = 1/8 x^2
        const double thresh = sqrt(8.*_gsparams->xvalue_accuracy);
        double xval;
        if (nu < thresh) {
            // lim j1(u)/u = 1/2
            xval =  0.5 * (1.-_obssq);
        } else {
            // See Schroeder eq (10.1.10)
            xval = ( math::j1(nu) - _obscuration * math::j1(_obscuration*nu) ) / nu ;
        }
        xval *= xval;
        // Normalize to give unit flux integrated over area.
        xval *= _norm;
        return xval;
    }

    double SBAiry::SBAiryImpl::xValue(const Position<double>& p) const
    {
        double r = sqrt(p.x*p.x+p.y*p.y) * _D;
        return _xnorm * _info->xValue(r);
    }

    double AiryInfoObs::xValue(double r) const
    { return _radial(r); }

    std::complex<double> SBAiry::SBAiryImpl::kValue(const Position<double>& k) const
    {
        double ksq_over_pisq = (k.x*k.x+k.y*k.y) * _inv_Dsq_pisq;
        // calculate circular FT(PSF) on p'=(x',y')
        return _knorm * _info->kValue(ksq_over_pisq);
    }

    template <typename T>
    void SBAiry::SBAiryImpl::fillXImage(ImageView<T> im,
                                        double x0, double dx, int izero,
                                        double y0, double dy, int jzero) const
    {
        dbg<<"SBAiry fillXImage\n";
        dbg<<"x = "<<x0<<" + i * "<<dx<<", izero = "<<izero<<std::endl;
        dbg<<"y = "<<y0<<" + j * "<<dy<<", jzero = "<<jzero<<std::endl;
        if (izero != 0 || jzero != 0) {
            xdbg<<"Use Quadrant\n";
            fillXImageQuadrant(im,x0,dx,izero,y0,dy,jzero);
        } else {
            xdbg<<"Non-Quadrant\n";
            const int m = im.getNCol();
            const int n = im.getNRow();
            T* ptr = im.getData();
            const int skip = im.getNSkip();
            assert(im.getStep() == 1);

            x0 *= _D;
            dx *= _D;
            y0 *= _D;
            dy *= _D;

            for (int j=0; j<n; ++j,y0+=dy,ptr+=skip) {
                double x = x0;
                double ysq = y0*y0;
                for (int i=0;i<m;++i,x+=dx)
                    *ptr++ = _xnorm * _info->xValue(sqrt(x*x + ysq));
            }
        }
    }

    template <typename T>
    void SBAiry::SBAiryImpl::fillXImage(ImageView<T> im,
                                        double x0, double dx, double dxy,
                                        double y0, double dy, double dyx) const
    {
        dbg<<"SBAiry fillXImage\n";
        dbg<<"x = "<<x0<<" + i * "<<dx<<" + j * "<<dxy<<std::endl;
        dbg<<"y = "<<y0<<" + i * "<<dyx<<" + j * "<<dy<<std::endl;
        const int m = im.getNCol();
        const int n = im.getNRow();
        T* ptr = im.getData();
        const int skip = im.getNSkip();
        assert(im.getStep() == 1);

        x0 *= _D;
        dx *= _D;
        dxy *= _D;
        y0 *= _D;
        dy *= _D;
        dyx *= _D;

        for (int j=0; j<n; ++j,x0+=dxy,y0+=dy,ptr+=skip) {
            double x = x0;
            double y = y0;
            for (int i=0; i<m; ++i,x+=dx,y+=dyx)
                *ptr++ = _xnorm * _info->xValue(sqrt(x*x + y*y));
        }
    }

    template <typename T>
    void SBAiry::SBAiryImpl::fillKImage(ImageView<std::complex<T> > im,
                                        double kx0, double dkx, int izero,
                                        double ky0, double dky, int jzero) const
    {
        dbg<<"SBAiry fillKImage\n";
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

            kx0 *= _inv_D_pi;
            dkx *= _inv_D_pi;
            ky0 *= _inv_D_pi;
            dky *= _inv_D_pi;

            for (int j=0; j<n; ++j,ky0+=dky,ptr+=skip) {
                double kx = kx0;
                double kysq = ky0*ky0;
                for (int i=0; i<m; ++i,kx+=dkx)
                    *ptr++ = _knorm * _info->kValue(kx*kx + kysq);
            }
        }
    }

    template <typename T>
    void SBAiry::SBAiryImpl::fillKImage(ImageView<std::complex<T> > im,
                                        double kx0, double dkx, double dkxy,
                                        double ky0, double dky, double dkyx) const
    {
        dbg<<"SBAiry fillKImage\n";
        dbg<<"kx = "<<kx0<<" + i * "<<dkx<<" + j * "<<dkxy<<std::endl;
        dbg<<"ky = "<<ky0<<" + i * "<<dkyx<<" + j * "<<dky<<std::endl;
        const int m = im.getNCol();
        const int n = im.getNRow();
        std::complex<T>* ptr = im.getData();
        int skip = im.getNSkip();
        assert(im.getStep() == 1);

        kx0 *= _inv_D_pi;
        dkx *= _inv_D_pi;
        dkxy *= _inv_D_pi;
        ky0 *= _inv_D_pi;
        dky *= _inv_D_pi;
        dkyx *= _inv_D_pi;

        for (int j=0; j<n; ++j,kx0+=dkxy,ky0+=dky,ptr+=skip) {
            double kx = kx0;
            double ky = ky0;
            for (int i=0; i<m; ++i,kx+=dkx,ky+=dkyx)
                *ptr++ = _knorm * _info->kValue(kx*kx + ky*ky);
        }
    }

    // Set maxK to hard limit for Airy disk.
    double SBAiry::SBAiryImpl::maxK() const
    { return 2.*M_PI*_D; }

    // The amount of flux missed in a circle of radius pi/stepk should be at
    // most folding_threshold of the flux.
    double SBAiry::SBAiryImpl::stepK() const
    { return _info->stepK() * _D; }

    double AiryInfoObs::chord(double r, double h, double rsq, double hsq) const
    {
        if (r==0.)
            return 0.;
#ifdef AIRY_DEBUG
        else if (r<h)
            throw SBError("Airy calculation r<h");
        else if (h < 0.)
            throw SBError("Airy calculation h<0");
#endif
        else
            return rsq*std::asin(h/r) - h*sqrt(rsq-hsq);
    }

    /* area inside intersection of 2 circles radii r & s, seperated by t*/
    double AiryInfoObs::circle_intersection(
        double r, double s, double rsq, double ssq, double tsq) const
    {
        assert(r >= s);
        assert(s >= 0.);
        double rps_sq = (r+s)*(r+s);
        if (tsq >= rps_sq) return 0.;
        double rms_sq = (r-s)*(r-s);
        if (tsq <= rms_sq) return M_PI*ssq;

        /* in between we calculate half-height at intersection */
        double hsq = 0.5*(rsq + ssq) - (tsq*tsq + rps_sq*rms_sq)/(4.*tsq);
#ifdef AIRY_DEBUG
        if (hsq<0.) throw SBError("Airy calculation half-height invalid");
#endif
        double h = sqrt(hsq);

        if (tsq < rsq - ssq)
            return M_PI*ssq - chord(s,h,ssq,hsq) + chord(r,h,rsq,hsq);
        else
            return chord(s,h,ssq,hsq) + chord(r,h,rsq,hsq);
    }

    /* area inside intersection of 2 circles both with radius r, seperated by t*/
    double AiryInfoObs::circle_intersection(
        double r, double rsq, double tsq) const
    {
        assert(r >= 0.);
        if (tsq >= 4.*rsq) return 0.;
        if (tsq == 0.) return M_PI*rsq;

        /* in between we calculate half-height at intersection */
        double hsq = rsq - tsq/4.;
#ifdef AIRY_DEBUG
        if (hsq<0.) throw SBError("Airy calculation half-height invalid");
#endif
        double h = sqrt(hsq);

        return 2.*chord(r,h,rsq,hsq);
    }

    /* area of two intersecting identical annuli */
    double AiryInfoObs::annuli_intersect(
        double r1, double r2, double r1sq, double r2sq, double tsq) const
    {
        assert(r1 >= r2);
        return circle_intersection(r1,r1sq,tsq)
            - 2. * circle_intersection(r1,r2,r1sq,r2sq,tsq)
            +  circle_intersection(r2,r2sq,tsq);
    }

    // Beam pattern of annular aperture, in k space, which is just the
    // autocorrelation of two annuli.
    // Unnormalized -- value at k=0 is Pi * (1-obs^2)
    double AiryInfoObs::kValue(double ksq_over_pisq) const
    { return annuli_intersect(1.,_obscuration,1.,_obssq,ksq_over_pisq); }

    // Constructor to initialize Airy constants and k lookup table
    AiryInfoObs::AiryInfoObs(double obscuration, const GSParamsPtr& gsparams) :
        _obscuration(obscuration),
        _obssq(obscuration * obscuration),
        _radial(_obscuration, _obssq, gsparams),
        _gsparams(gsparams)
    {
        dbg<<"Initializing AiryInfo for obs = "<<obscuration<<std::endl;
        xdbg<<"gsparams = "<<*_gsparams<<std::endl;
        // Calculate stepK:
        // Schroeder (10.1.18) gives limit of EE at large radius.
        // This stepK could probably be relaxed, it makes overly accurate FFTs.
        double R = 1. / (_gsparams->folding_threshold * 0.5 * M_PI * M_PI * (1.-_obscuration));
        // Make sure it is at least 5 hlr
        // The half-light radius of an obscured Airy disk is not so easy to calculate.
        // So we just use the value for the unobscured Airy disk.
        // TODO: We could do this numerically if we decide that it's important to get this right.
        const double hlr = 0.5348321477;
        R = std::max(R,_gsparams->stepk_minimum_hlr*hlr);
        this->_stepk = M_PI / R;
    }

    void SBAiry::SBAiryImpl::shoot(PhotonArray& photons, UniformDeviate ud) const
    {
        dbg<<"Airy shoot: N = "<<photons.size()<<std::endl;
        dbg<<"Target flux = "<<getFlux()<<std::endl;
        _info->shoot(photons, ud);
        // Then rescale for this flux & size
        photons.scaleFlux(_flux);
        photons.scaleXY(1./_D);
        dbg<<"Airy Realized flux = "<<photons.getTotalFlux()<<std::endl;
    }

    void AiryInfo::shoot(PhotonArray& photons, UniformDeviate ud) const
    {
        // Use the OneDimensionalDeviate to sample from scale-free distribution
        checkSampler();
        assert(_sampler.get());
        _sampler->shoot(photons, ud);
    }

    void AiryInfoObs::checkSampler() const
    {
        if (this->_sampler.get()) return;
        dbg<<"Airy sampler\n";
        dbg<<"obsc = "<<_obscuration<<std::endl;
        std::vector<double> ranges(1,0.);
        // Break Airy function into ranges that will not have >1 extremum:
        double rmin = 1.1 - 0.5*_obscuration;
        // Use Schroeder (10.1.18) limit of EE at large radius.
        // to stop sampler at radius with EE>(1-shoot_accuracy)
        double rmax = 2./(_gsparams->shoot_accuracy * M_PI*M_PI * (1.-_obscuration));
        dbg<<"rmin = "<<rmin<<std::endl;
        dbg<<"rmax = "<<rmax<<std::endl;
        // NB: don't need floor, since rhs is positive, so floor is superfluous.
        ranges.reserve(int((rmax-rmin+2)/0.5+0.5));
        for(double r=rmin; r<=rmax; r+=0.5) ranges.push_back(r);
        this->_sampler.reset(new OneDimensionalDeviate(_radial, ranges, true, 1.0, *_gsparams));
    }

    // Now the specializations for when obs = 0
    double AiryInfoNoObs::xValue(double r) const
    { return _radial(r); }

    double AiryInfoNoObs::kValue(double ksq_over_pisq) const
    {
        if (ksq_over_pisq >= 4.) return 0.;
        if (ksq_over_pisq == 0.) return M_PI;

        /* in between we calculate half-height at intersection */
        double hsq = 1. - ksq_over_pisq/4.;
#ifdef AIRY_DEBUG
        if (hsq<0.) throw SBError("Airy calculation half-height invalid");
#endif
        double h = sqrt(hsq);

        return 2. * (std::asin(h) - h*sqrt(1.-hsq));
    }

    double AiryInfoNoObs::RadialFunction::operator()(double radius) const
    {
        double nu = radius*M_PI;
        // Taylor expansion of j1(u)/u = 1/2 - 1/16 x^2 + ...
        // We can truncate this to 1/2 when neglected term is less than xvalue_accuracy
        // (relative error, so divide by 1/2)
        // xvalue_accuracy = 1/8 x^2
        const double thresh = sqrt(8.*_gsparams->xvalue_accuracy);
        double xval;
        if (nu < thresh) {
            // lim j1(u)/u = 1/2
            xval = 0.5;
        } else {
            xval = math::j1(nu) / nu;
        }
        xval *= xval;
        // Normalize to give unit flux integrated over area.
        xval *= M_PI;
        return xval;
    }

    // Constructor to initialize Airy constants and k lookup table
    AiryInfoNoObs::AiryInfoNoObs(const GSParamsPtr& gsparams) :
        _radial(gsparams), _gsparams(gsparams)
    {
        dbg<<"Initializing AiryInfoNoObs\n";
        xdbg<<"gsparams = "<<*_gsparams<<std::endl;
        // Calculate stepK:
        double R = 1. / (_gsparams->folding_threshold * 0.5 * M_PI * M_PI);
        // Make sure it is at least 5 hlr
        // The half-light radius of an Airy disk is 0.5348321477 * lam/D
        const double hlr = 0.5348321477;
        R = std::max(R,_gsparams->stepk_minimum_hlr*hlr);
        this->_stepk = M_PI / R;
    }

    void AiryInfoNoObs::checkSampler() const
    {
        if (this->_sampler.get()) return;
        dbg<<"AiryNoObs sampler\n";
        std::vector<double> ranges(1,0.);
        double rmin = 1.1;
        double rmax = 2./(_gsparams->shoot_accuracy * M_PI*M_PI);
        dbg<<"rmin = "<<rmin<<std::endl;
        dbg<<"rmax = "<<rmax<<std::endl;
        // NB: don't need floor, since rhs is positive, so floor is superfluous.
        ranges.reserve(int((rmax-rmin+2)/0.5+0.5));
        for(double r=rmin; r<=rmax; r+=0.5) ranges.push_back(r);
        this->_sampler.reset(new OneDimensionalDeviate(_radial, ranges, true, 1.0, *_gsparams));
    }
}
