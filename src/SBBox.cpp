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

#include "SBBox.h"
#include "SBBoxImpl.h"
#include "FFT.h"
#include "Interpolant.h"  // For sinc(x)

// cf. comments about USE_COS_SIN in SBGaussian.cpp
#ifdef _INTEL_COMPILER
#define USE_COS_SIN
#endif

#ifdef DEBUGLOGGING
#include <fstream>
//std::ostream* dbgout = &std::cerr;
//std::ostream* dbgout = new std::ofstream("debug.out");
//int verbose_level = 2;
#endif

namespace galsim {


    SBBox::SBBox(double width, double height, double flux, const GSParamsPtr& gsparams) :
        SBProfile(new SBBoxImpl(width,height,flux,gsparams)) {}

    SBBox::SBBox(const SBBox& rhs) : SBProfile(rhs) {}

    SBBox::~SBBox() {}

    double SBBox::getWidth() const 
    {
        assert(dynamic_cast<const SBBoxImpl*>(_pimpl.get()));
        return static_cast<const SBBoxImpl&>(*_pimpl).getWidth(); 
    }

    double SBBox::getHeight() const 
    {
        assert(dynamic_cast<const SBBoxImpl*>(_pimpl.get()));
        return static_cast<const SBBoxImpl&>(*_pimpl).getHeight(); 
    }

    SBBox::SBBoxImpl::SBBoxImpl(double width, double height, double flux,
                                const GSParamsPtr& gsparams) :
        SBProfileImpl(gsparams), _width(width), _height(height), _flux(flux)
    {
        if (_height==0.) _height=_width;
        _norm = _flux / (_width * _height);
    }


    double SBBox::SBBoxImpl::xValue(const Position<double>& p) const 
    {
        if (fabs(p.x) < 0.5*_width && fabs(p.y) < 0.5*_height) return _norm;
        else return 0.;  // do not use this function for filling image!
    }

    std::complex<double> SBBox::SBBoxImpl::kValue(const Position<double>& k) const
    {
        return _flux * sinc(k.x*_width/(2.*M_PI))*sinc(k.y*_height/(2.*M_PI));
    }

    void SBBox::SBBoxImpl::fillXValue(tmv::MatrixView<double> val,
                                      double x0, double dx, int ix_zero,
                                      double y0, double dy, int iy_zero) const
    {
        dbg<<"SBBox fillXValue\n";
        dbg<<"x = "<<x0<<" + ix * "<<dx<<", ix_zero = "<<ix_zero<<std::endl;
        dbg<<"y = "<<y0<<" + iy * "<<dy<<", iy_zero = "<<iy_zero<<std::endl;
        assert(val.stepi() == 1);
        const int m = val.colsize();
        const int n = val.rowsize();
        typedef tmv::VIt<double,1,tmv::NonConj> It;

        // We need to make sure the pixels where the edges of the box fall only get
        // a fraction of the flux.
        //
        // We divide up the range into 3 sections in x: 
        //    left of the box where val = 0
        //    in the box where val = _norm
        //    right of the box where val = 0 again
        //
        // ... and 3 sections in y:
        //    below the box where val = 0
        //    in the box where val = _norm
        //    above the box where val = 0 again
        //
        // Furthermore, we have to calculate the correct values for the pixels on the border.
        
        // It will be useful to do everything in units of dx,dy
        x0 /= dx;
        double width = _width / dx;
        y0 /= dy;
        double height = _height / dy;
        xdbg<<"x0,y0 -> "<<x0<<','<<y0<<std::endl;
        xdbg<<"width,height -> "<<width<<','<<height<<std::endl;

        int ix_left, ix_right, iy_bottom, iy_top;
        double x_left, x_right, y_bottom, y_top;

        // Find the x edges:
        double tmp = 0.5*width + 0.5;
        ix_left = int(-tmp-x0+1);
        ix_right = int(tmp-x0);

        // If the box goes off the image, it's ok, but it will cause us problems
        // later on if we don't change it.  Just use ix_left = 0.
        if (ix_left < 0) { ix_left = 0; x_left = 1.; } 

        // If the whole box is off the image, just zero and return.
        else if (ix_left >= m) { val.setZero(); return; } 

        // Normal case: calculate the fractional flux in the edge
        else x_left = tmp+x0+ix_left;

        // Now the right side.
        if (ix_right >= m) { ix_right = m-1; x_right = 1.; } 
        else if (ix_right < 0) { val.setZero(); return; } 
        else x_right = tmp-x0-ix_right;
        xdbg<<"ix_left = "<<ix_left<<" with partial flux "<<x_left<<std::endl;
        xdbg<<"ix_right = "<<ix_right<<" with partial flux "<<x_right<<std::endl;
        
        // Repeat for y values
        tmp = 0.5*height + 0.5;
        iy_bottom = int(-tmp-y0+1);
        iy_top = int(tmp-y0);

        if (iy_bottom < 0) { iy_bottom = 0; y_bottom = 1.; } 
        else if (iy_bottom >= n) { val.setZero(); return; } 
        else y_bottom = tmp+y0+iy_bottom;

        if (iy_top >= n) { iy_top = n-1; y_top = 1.; } 
        else if (iy_top < 0) { val.setZero(); return; } 
        else y_top = tmp-y0-iy_top;
        xdbg<<"iy_bottom = "<<iy_bottom<<" with partial flux "<<y_bottom<<std::endl;
        xdbg<<"iy_top = "<<iy_top<<" with partial flux "<<y_top<<std::endl;
        xdbg<<"m,n = "<<m<<','<<n<<std::endl;

        // Now we need to fill the matrix with the appropriate values in each section.
        // Start with the zeros:
        if (0 < ix_left)
            val.subMatrix(0,ix_left,iy_bottom,iy_top+1).setZero();
        if (ix_right+1 < m)
            val.subMatrix(ix_right+1,m,iy_bottom,iy_top+1).setZero();
        if (0 < iy_bottom)
            val.colRange(0,iy_bottom).setZero();
        if (iy_top+1 < n)
            val.colRange(iy_top+1,n).setZero();
        // Then the interior:
        if (ix_left+1 < ix_right && iy_bottom+1 < iy_top)
            val.subMatrix(ix_left+1,ix_right,iy_bottom+1,iy_top).setAllTo(_norm);
        // And now the edges:
        if (ix_left+1 < ix_right) {
            val.col(iy_bottom,ix_left+1,ix_right).setAllTo(y_bottom * _norm);
            val.col(iy_top,ix_left+1,ix_right).setAllTo(y_top * _norm);
        }
        if (iy_bottom+1 < iy_top) {
            val.row(ix_left,iy_bottom+1,iy_top).setAllTo(x_left * _norm);
            val.row(ix_right,iy_bottom+1,iy_top).setAllTo(x_right * _norm);
        }
        // Finally the corners
        val(ix_left,iy_bottom) = x_left * y_bottom * _norm;
        val(ix_right,iy_bottom) = x_right * y_bottom * _norm;
        val(ix_left,iy_top) = x_left * y_top * _norm;
        val(ix_right,iy_top) = x_right * y_top * _norm;
    }

    void SBBox::SBBoxImpl::fillKValue(tmv::MatrixView<std::complex<double> > val,
                                      double x0, double dx, int ix_zero,
                                      double y0, double dy, int iy_zero) const
    {
        dbg<<"SBBox fillKValue\n";
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
            typedef tmv::VIt<double,1,tmv::NonConj> It;

            x0 *= _width/(2.*M_PI);
            dx *= _width/(2.*M_PI);
            y0 *= _height/(2.*M_PI);
            dy *= _height/(2.*M_PI);

            // The Box profile in Fourier space is separable:
            //    val(x,y) = _flux * sinc(x * _width/2pi) * sinc(y * _height/2pi) 
            tmv::Vector<double> sinc_x(m);
            It xit = sinc_x.begin();
            for (int i=0;i<m;++i,x0+=dx) *xit++ = sinc(x0);
            tmv::Vector<double> sinc_y(n);
            It yit = sinc_y.begin();
            for (int j=0;j<n;++j,y0+=dy) *yit++ = sinc(y0);

            val = _flux * sinc_x ^ sinc_y;
        }
    }

    void SBBox::SBBoxImpl::fillXValue(tmv::MatrixView<double> val,
                                      double x0, double dx, double dxy,
                                      double y0, double dy, double dyx) const
    {
        // This is complicated to get right, since the edges cut through the image grid
        // at angles.  Fortunately, we also don't really have any need for it.
        // It would only get called if you draw a sheared or rotated Pixel without convolving 
        // it by anything else, which we don't really do.  If we ever decide we need this,
        // someone can try to figure out all the math involved here.
        if (dxy == 0. && dyx == 0.)
            fillXValue(val,x0,dx,0,y0,dy,0);
        else
            throw std::runtime_error(
                "fillXValue for a sheared or rotated SBBox is not implemented.");
    }

    void SBBox::SBBoxImpl::fillKValue(tmv::MatrixView<std::complex<double> > val,
                                      double x0, double dx, double dxy,
                                      double y0, double dy, double dyx) const
    {
        dbg<<"SBBox fillKValue\n";
        dbg<<"x = "<<x0<<" + ix * "<<dx<<" + iy * "<<dxy<<std::endl;
        dbg<<"y = "<<y0<<" + ix * "<<dyx<<" + iy * "<<dy<<std::endl;
        assert(val.stepi() == 1);
        assert(val.canLinearize());
        const int m = val.colsize();
        const int n = val.rowsize();
        typedef tmv::VIt<std::complex<double>,1,tmv::NonConj> It;

        x0 *= _width/(2.*M_PI);
        dx *= _width/(2.*M_PI);
        dxy *= _width/(2.*M_PI);
        y0 *= _height/(2.*M_PI);
        dy *= _height/(2.*M_PI);
        dyx *= _height/(2.*M_PI);

        It valit = val.linearView().begin();
        for (int j=0;j<n;++j,x0+=dxy,y0+=dy) {
            double x = x0;
            double y = y0;
            for (int i=0;i<m;++i,x+=dx,y+=dyx) 
                *valit++ = _flux * sinc(x) * sinc(y);
        }
    }

    // Set maxK to the value where the FT is down to maxk_threshold
    double SBBox::SBBoxImpl::maxK() const 
    { 
        return 2. / (this->gsparams->maxk_threshold * std::min(_width,_height));
    }

    // The amount of flux missed in a circle of radius pi/stepk should be at 
    // most folding_threshold of the flux.
    double SBBox::SBBoxImpl::stepK() const
    {
        // In this case max(width,height) encloses all the flux, so use that.
        return M_PI / std::max(_width,_height);
    }

    boost::shared_ptr<PhotonArray> SBBox::SBBoxImpl::shoot(int N, UniformDeviate u) const
    {
        dbg<<"Box shoot: N = "<<N<<std::endl;
        dbg<<"Target flux = "<<getFlux()<<std::endl;
        boost::shared_ptr<PhotonArray> result(new PhotonArray(N));
        for (int i=0; i<result->size(); i++)
            result->setPhoton(i, _width*(u()-0.5), _height*(u()-0.5), _flux/N);
        dbg<<"Box Realized flux = "<<result->getTotalFlux()<<std::endl;
        return result;
    }



    SBTopHat::SBTopHat(double radius, double flux, const GSParamsPtr& gsparams) :
        SBProfile(new SBTopHatImpl(radius,flux,gsparams)) {}

    SBTopHat::SBTopHat(const SBTopHat& rhs) : SBProfile(rhs) {}

    SBTopHat::~SBTopHat() {}

    double SBTopHat::getRadius() const 
    {
        assert(dynamic_cast<const SBTopHatImpl*>(_pimpl.get()));
        return static_cast<const SBTopHatImpl&>(*_pimpl).getRadius(); 
    }

    SBTopHat::SBTopHatImpl::SBTopHatImpl(double radius, double flux,
                                         const GSParamsPtr& gsparams) :
        SBProfileImpl(gsparams),
        _r0(radius), _r0sq(_r0*_r0), _flux(flux),
        _norm(_flux / (M_PI * _r0sq))
    {
    }


    double SBTopHat::SBTopHatImpl::xValue(const Position<double>& p) const 
    {
        double rsq = p.x*p.x + p.y*p.y;
        if (rsq < _r0sq) return _norm;
        else return 0.;
    }

    std::complex<double> SBTopHat::SBTopHatImpl::kValue(const Position<double>& k) const
    {
        double kr0sq = (k.x*k.x + k.y*k.y) * _r0sq;
        return kValue2(kr0sq);
    }

    std::complex<double> SBTopHat::SBTopHatImpl::kValue2(double kr0sq) const
    {
        if (kr0sq < 1.e-4) {
            // Error from omitting next term is about 1.e-16 for kr0sq = 1.e-4
            return _flux * (1. - kr0sq * ( (1./8.) + (1./192.) * kr0sq ));
        } else {
            double kr0 = sqrt(kr0sq);
            return 2.*_flux * j1(kr0)/kr0;
        }
    }

    void SBTopHat::SBTopHatImpl::fillXValue(tmv::MatrixView<double> val,
                                            double x0, double dx, int ix_zero,
                                            double y0, double dy, int iy_zero) const
    {
        // For now just call back to the base class implementation.
        SBProfile::SBProfileImpl::fillXValue(val,x0,dx,ix_zero,y0,dy,iy_zero);
#if 0
        dbg<<"SBTopHat fillXValue\n";
        dbg<<"x = "<<x0<<" + ix * "<<dx<<", ix_zero = "<<ix_zero<<std::endl;
        dbg<<"y = "<<y0<<" + iy * "<<dy<<", iy_zero = "<<iy_zero<<std::endl;
        assert(val.stepi() == 1);
        const int m = val.colsize();
        const int n = val.rowsize();
        typedef tmv::VIt<double,1,tmv::NonConj> It;

        // We need to make sure the pixels where the edges of the box fall only get
        // a fraction of the flux.
        //
        // We divide up the range into 3 sections in x: 
        //    left of the box where val = 0
        //    in the box where val = _norm
        //    right of the box where val = 0 again
        //
        // ... and 3 sections in y:
        //    below the box where val = 0
        //    in the box where val = _norm
        //    above the box where val = 0 again
        //
        // Furthermore, we have to calculate the correct values for the pixels on the border.
        
        // It will be useful to do everything in units of dx,dy
        x0 /= dx;
        double width = _width / dx;
        y0 /= dy;
        double height = _height / dy;
        xdbg<<"x0,y0 -> "<<x0<<','<<y0<<std::endl;
        xdbg<<"width,height -> "<<width<<','<<height<<std::endl;

        int ix_left, ix_right, iy_bottom, iy_top;
        double x_left, x_right, y_bottom, y_top;

        // Find the x edges:
        double tmp = 0.5*width + 0.5;
        ix_left = int(-tmp-x0+1);
        ix_right = int(tmp-x0);

        // If the box goes off the image, it's ok, but it will cause us problems
        // later on if we don't change it.  Just use ix_left = 0.
        if (ix_left < 0) { ix_left = 0; x_left = 1.; } 

        // If the whole box is off the image, just zero and return.
        else if (ix_left >= m) { val.setZero(); return; } 

        // Normal case: calculate the fractional flux in the edge
        else x_left = tmp+x0+ix_left;

        // Now the right side.
        if (ix_right >= m) { ix_right = m-1; x_right = 1.; } 
        else if (ix_right < 0) { val.setZero(); return; } 
        else x_right = tmp-x0-ix_right;
        xdbg<<"ix_left = "<<ix_left<<" with partial flux "<<x_left<<std::endl;
        xdbg<<"ix_right = "<<ix_right<<" with partial flux "<<x_right<<std::endl;
        
        // Repeat for y values
        tmp = 0.5*height + 0.5;
        iy_bottom = int(-tmp-y0+1);
        iy_top = int(tmp-y0);

        if (iy_bottom < 0) { iy_bottom = 0; y_bottom = 1.; } 
        else if (iy_bottom >= n) { val.setZero(); return; } 
        else y_bottom = tmp+y0+iy_bottom;

        if (iy_top >= n) { iy_top = n-1; y_top = 1.; } 
        else if (iy_top < 0) { val.setZero(); return; } 
        else y_top = tmp-y0-iy_top;
        xdbg<<"iy_bottom = "<<iy_bottom<<" with partial flux "<<y_bottom<<std::endl;
        xdbg<<"iy_top = "<<iy_top<<" with partial flux "<<y_top<<std::endl;
        xdbg<<"m,n = "<<m<<','<<n<<std::endl;

        // Now we need to fill the matrix with the appropriate values in each section.
        // Start with the zeros:
        if (0 < ix_left)
            val.subMatrix(0,ix_left,iy_bottom,iy_top+1).setZero();
        if (ix_right+1 < m)
            val.subMatrix(ix_right+1,m,iy_bottom,iy_top+1).setZero();
        if (0 < iy_bottom)
            val.colRange(0,iy_bottom).setZero();
        if (iy_top+1 < n)
            val.colRange(iy_top+1,n).setZero();
        // Then the interior:
        if (ix_left+1 < ix_right && iy_bottom+1 < iy_top)
            val.subMatrix(ix_left+1,ix_right,iy_bottom+1,iy_top).setAllTo(_norm);
        // And now the edges:
        if (ix_left+1 < ix_right) {
            val.col(iy_bottom,ix_left+1,ix_right).setAllTo(y_bottom * _norm);
            val.col(iy_top,ix_left+1,ix_right).setAllTo(y_top * _norm);
        }
        if (iy_bottom+1 < iy_top) {
            val.row(ix_left,iy_bottom+1,iy_top).setAllTo(x_left * _norm);
            val.row(ix_right,iy_bottom+1,iy_top).setAllTo(x_right * _norm);
        }
        // Finally the corners
        val(ix_left,iy_bottom) = x_left * y_bottom * _norm;
        val(ix_right,iy_bottom) = x_right * y_bottom * _norm;
        val(ix_left,iy_top) = x_left * y_top * _norm;
        val(ix_right,iy_top) = x_right * y_top * _norm;
#endif
    }

    void SBTopHat::SBTopHatImpl::fillKValue(tmv::MatrixView<std::complex<double> > val,
                                            double x0, double dx, int ix_zero,
                                            double y0, double dy, int iy_zero) const
    {
        dbg<<"SBTopHat fillKValue\n";
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
                    *valit++ = _flux * kValue2(ksq);
                }
            }
        }
    }

    void SBTopHat::SBTopHatImpl::fillXValue(tmv::MatrixView<double> val,
                                            double x0, double dx, double dxy,
                                            double y0, double dy, double dyx) const
    {
        // As with SBBox, this is hard to get right at the boundard.  The pixels are parallelograms
        // intersecting the TopHat circle.  But also as with SBBox, it's not really ever needed, 
        // so we don't implement it.
        if (dxy == 0. && dyx == 0.)
            fillXValue(val,x0,dx,0,y0,dy,0);
        else
            throw std::runtime_error(
                "fillXValue for a sheared or rotated SBTopHat is not implemented.");
    }

    void SBTopHat::SBTopHatImpl::fillKValue(tmv::MatrixView<std::complex<double> > val,
                                            double x0, double dx, double dxy,
                                            double y0, double dy, double dyx) const
    {
        dbg<<"SBTopHat fillKValue\n";
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
                *valit++ = _flux * kValue2(ksq);
            }
        }
    }

    // Set maxK to the value where the FT is down to maxk_threshold
    double SBTopHat::SBTopHatImpl::maxK() const 
    { 
        // |j1(x)| ~ sqrt(2/(Pi x)) for large x, so using this, we get
        // maxk_thresh = 2 * sqrt(2/(Pi k r0)) / (k r0) = 2 sqrt(2/Pi) (k r0)^-3/2
        return std::pow(2. * sqrt(2./M_PI) / this->gsparams->maxk_threshold, 2./3.) / _r0;
    }

    // The amount of flux missed in a circle of radius pi/stepk should be at 
    // most folding_threshold of the flux.
    double SBTopHat::SBTopHatImpl::stepK() const
    {
        // _r0 encloses all the flux, so use that.
        return M_PI / _r0;
    }

    boost::shared_ptr<PhotonArray> SBTopHat::SBTopHatImpl::shoot(int N, UniformDeviate u) const
    {
        dbg<<"TopHat shoot: N = "<<N<<std::endl;
        dbg<<"Target flux = "<<getFlux()<<std::endl;
        boost::shared_ptr<PhotonArray> result(new PhotonArray(N));
        double fluxPerPhoton = _flux/N;
        // cf. SBGaussian's shoot function
        for (int i=0; i<N; i++) {
            // First get a point uniformly distributed on unit circle
#ifdef USE_COS_SIN
            double theta = 2.*M_PI*u();
            double rsq = u(); // cumulative dist function P(<r) = r^2 for unit circle
#ifdef _GLIBCXX_HAVE_SINCOS
            // Most optimizing compilers will do this automatically, but just in case...
            double sint,cost;
            sincos(theta,&sint,&cost);
#else
            double cost = std::cos(theta);
            double sint = std::sin(theta);
#endif
            // Then map radius to the desired Gaussian with analytic transformation
            double r = sqrt(rsq) * _r0;;
            result->setPhoton(i, r*cost, r*sint, fluxPerPhoton);
#else
            double xu, yu, rsq;
            do {
                xu = 2.*u()-1.;
                yu = 2.*u()-1.;
                rsq = xu*xu+yu*yu;
            } while (rsq>=1.);
            result->setPhoton(i, xu * _r0, _r0 * yu * _r0, fluxPerPhoton);
#endif
        }
        dbg<<"TopHat Realized flux = "<<result->getTotalFlux()<<std::endl;
        return result;
    }
}
