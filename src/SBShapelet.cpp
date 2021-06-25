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

#include "SBShapelet.h"
#include "SBShapeletImpl.h"

namespace galsim {

    SBShapelet::SBShapelet(double sigma, LVector bvec, const GSParams& gsparams) :
        SBProfile(new SBShapeletImpl(sigma, bvec, gsparams)) {}

    SBShapelet::SBShapelet(const SBShapelet& rhs) : SBProfile(rhs) {}

    SBShapelet::~SBShapelet() {}

    const LVector& SBShapelet::getBVec() const
    {
        assert(dynamic_cast<const SBShapeletImpl*>(_pimpl.get()));
        return static_cast<const SBShapeletImpl&>(*_pimpl).getBVec();
    }

    double SBShapelet::getSigma() const
    {
        assert(dynamic_cast<const SBShapeletImpl*>(_pimpl.get()));
        return static_cast<const SBShapeletImpl&>(*_pimpl).getSigma();
    }

    void SBShapelet::rotate(double theta)
    {
        assert(dynamic_cast<SBShapeletImpl*>(_pimpl.get()));
        LVector& bvec = static_cast<SBShapeletImpl&>(*_pimpl).getBVec();
        bvec.rotate(theta);
    }

    SBShapelet::SBShapeletImpl::SBShapeletImpl(double sigma, const LVector& bvec,
                                               const GSParams& gsparams) :
        SBProfileImpl(gsparams),
        _sigma(sigma), _bvec(bvec.copy()) {}

    double SBShapelet::SBShapeletImpl::maxK() const
    {
        // Start with value for plain old Gaussian:
        double maxk = sqrt(-2.*std::log(this->gsparams.maxk_threshold))/_sigma;
        // Grow as sqrt of (order+1)
        // Note: this is an approximation.  The right value would require looking at
        // the actual coefficients and doing something smart with them.
        maxk *= sqrt(double(_bvec.getOrder()+1));
        return maxk;
    }

    double SBShapelet::SBShapeletImpl::stepK() const
    {
        // Start with value for plain old Gaussian:
        double R = std::max(4., sqrt(-2.*std::log(this->gsparams.folding_threshold)));
        // Grow as sqrt of (order+1)
        R *= sqrt(double(_bvec.getOrder()+1));
        return M_PI / (R*_sigma);
    }

    double SBShapelet::SBShapeletImpl::xValue(const Position<double>& p) const
    {
        LVector psi(_bvec.getOrder());
        psi.fillBasis(p.x/_sigma, p.y/_sigma, _sigma);
        double xval = _bvec.dot(psi);
        return xval;
    }

    std::complex<double> SBShapelet::SBShapeletImpl::kValue(const Position<double>& k) const
    {
        int N=_bvec.getOrder();
        LVector psi(N);
        psi.fillBasis(k.x*_sigma, k.y*_sigma);  // Fourier[Psi_pq] is unitless
        // rotate kvalues of Psi with i^(p+q)
        // dotting b_pq with psi in k-space:
        double rr=0.;
        double ii=0.;
        for (PQIndex pq(0,0); !pq.pastOrder(N); pq.nextDistinct()) {
            int j = pq.rIndex();
            double x = _bvec[j]*psi[j] + (pq.isReal() ? 0 : _bvec[j+1]*psi[j+1]);
            switch (pq.N() % 4) {
              case 0:
                   rr += x;
                   break;
              case 1:
                   ii -= x;
                   break;
              case 2:
                   rr -= x;
                   break;
              case 3:
                   ii += x;
                   break;
            }
        }
        // difference in Fourier convention with FFTW ???
        return std::complex<double>(2.*M_PI*rr, 2.*M_PI*ii);
    }

    double SBShapelet::SBShapeletImpl::getFlux() const
    {
        double flux=0.;
        for (PQIndex pp(0,0); !pp.pastOrder(_bvec.getOrder()); pp.incN())
            flux += _bvec[pp].real();  // _bvec[pp] is real, but need type conv.
        return flux;
    }

    double SBShapelet::SBShapeletImpl::maxSB() const
    {
        // Usually b0 dominates the flux, so just take the maximum SB for that Gaussian.
        return std::abs(_bvec[0]) / (2. * M_PI * _sigma * _sigma);
    }

    Position<double> SBShapelet::SBShapeletImpl::centroid() const
    {
        std::complex<double> cen(0.);
        double n = 1.;
        for (PQIndex pq(1,0); !pq.pastOrder(_bvec.getOrder()); pq.incN(), n+=2)
            cen += sqrt(n+1.) * _bvec[pq];
        cen *= sqrt(2.)*_sigma/getFlux();
        return Position<double>(real(cen),-imag(cen));
    }

    double SBShapelet::SBShapeletImpl::getSigma() const { return _sigma; }
    const LVector& SBShapelet::SBShapeletImpl::getBVec() const { return _bvec; }
    LVector& SBShapelet::SBShapeletImpl::getBVec() { return _bvec; }

    void FillXValue(const LVector& bvec, double sigma,
                    VectorXd& val, const VectorXd& x, const VectorXd& y)
    {
        dbg<<"order = "<<bvec.getOrder()<<", sigma = "<<sigma<<std::endl;
        xdbg<<"FillXValue with bvec = "<<bvec<<std::endl;
        MatrixXd psi(val.size(),bvec.size());
        LVector::basis(x,y,psi,bvec.getOrder(),sigma);
        val = psi * bvec.rVector();
    }

    void FillKValue(const LVector& bvec, double sigma,
                    VectorXcd& val, const VectorXd& kx, const VectorXd& ky)
    {
        dbg<<"order = "<<bvec.getOrder()<<", sigma = "<<sigma<<std::endl;
        xdbg<<"fillKValue with bvec = "<<bvec<<std::endl;
        MatrixXcd psi_k(val.size(),bvec.size());
        LVector::kBasis(kx,ky,psi_k,bvec.getOrder(),sigma);
#ifdef USE_TMV
        // Note: the explicit cast to Vector<complex<double> > shouldn't be necessary.
        // But not doing so fails for Apple's default BLAS library.  It should be a pretty
        // minimal efficiency difference, so we always do the explicit cast to be safe.
        val = psi_k * VectorXcd(bvec.rVector());
#else
        val = psi_k * bvec.rVector();
#endif
    }

    template <typename T>
    void SBShapelet::SBShapeletImpl::fillXImage(ImageView<T> im,
                                                double x0, double dx, int izero,
                                                double y0, double dy, int jzero) const
    {
        dbg<<"SBShapelet fillXImage\n";
        dbg<<"x = "<<x0<<" + i * "<<dx<<", izero = "<<izero<<std::endl;
        dbg<<"y = "<<y0<<" + j * "<<dy<<", jzero = "<<jzero<<std::endl;
        const int m = im.getNCol();
        const int n = im.getNRow();
        T* ptr = im.getData();
        const int skip = im.getNSkip();
        assert(im.getStep() == 1);

        x0 /= _sigma;
        dx /= _sigma;
        y0 /= _sigma;
        dy /= _sigma;

        VectorXd mx(m*n);
        for (int i=0; i<m; ++i,x0+=dx)
            for (int j=0; j<n; ++j) mx[j*m + i] = x0;
        VectorXd my(m*n);
        for (int j=0, k=0; j<n; ++j,y0+=dy)
            for (int i=0; i<m; ++i, ++k) my[k] = y0;

        VectorXd val(m*n);
        FillXValue(_bvec,_sigma,val,mx,my);

        for (int j=0,k=0; j<n; ++j,ptr+=skip)
            for (int i=0; i<m; ++i,++k)
                *ptr++ = val[k];
    }

    template <typename T>
    void SBShapelet::SBShapeletImpl::fillXImage(ImageView<T> im,
                                                double x0, double dx, double dxy,
                                                double y0, double dy, double dyx) const
    {
        dbg<<"SBShapelet fillXImage\n";
        dbg<<"x = "<<x0<<" + i * "<<dx<<" + j * "<<dxy<<std::endl;
        dbg<<"y = "<<y0<<" + i * "<<dyx<<" + j * "<<dy<<std::endl;
        const int m = im.getNCol();
        const int n = im.getNRow();
        T* ptr = im.getData();
        const int skip = im.getNSkip();
        assert(im.getStep() == 1);

        x0 /= _sigma;
        dx /= _sigma;
        dxy /= _sigma;
        y0 /= _sigma;
        dy /= _sigma;
        dyx /= _sigma;

        VectorXd mx(m*n);
        VectorXd my(m*n);
        for (int j=0,k=0; j<n; ++j,x0+=dxy,y0+=dy) {
            double x = x0;
            double y = y0;
            for (int i=0; i<m; ++i,++k,x+=dx,y+=dyx) {
                mx[k] = x; my[k] = y;
            }
        }

        VectorXd val(m*n);
        FillXValue(_bvec,_sigma,val,mx,my);

        for (int j=0,k=0; j<n; ++j,ptr+=skip)
            for (int i=0; i<m; ++i,++k)
                *ptr++ = val[k];
    }

    template <typename T>
    void SBShapelet::SBShapeletImpl::fillKImage(ImageView<std::complex<T> > im,
                                                double kx0, double dkx, int izero,
                                                double ky0, double dky, int jzero) const
    {
        dbg<<"SBShapelet fillKImage\n";
        dbg<<"kx = "<<kx0<<" + i * "<<dkx<<", izero = "<<izero<<std::endl;
        dbg<<"ky = "<<ky0<<" + j * "<<dky<<", jzero = "<<jzero<<std::endl;
        const int m = im.getNCol();
        const int n = im.getNRow();
        std::complex<T>* ptr = im.getData();
        int skip = im.getNSkip();
        assert(im.getStep() == 1);

        kx0 *= _sigma;
        dkx *= _sigma;
        ky0 *= _sigma;
        dky *= _sigma;

        VectorXd mkx(m*n);
        for (int i=0; i<m; ++i,kx0+=dkx)
            for (int j=0; j<n; ++j) mkx[j*m + i] = kx0;
        VectorXd mky(m*n);
        for (int j=0, k=0; j<n; ++j,ky0+=dky)
            for (int i=0; i<m; ++i, ++k) mky[k] = ky0;

        VectorXcd val(m*n);
        FillKValue(_bvec,_sigma,val,mkx,mky);

        for (int j=0,k=0; j<n; ++j,ptr+=skip)
            for (int i=0; i<m; ++i,++k)
                *ptr++ = val[k];
     }

    template <typename T>
    void SBShapelet::SBShapeletImpl::fillKImage(ImageView<std::complex<T> > im,
                                                double kx0, double dkx, double dkxy,
                                                double ky0, double dky, double dkyx) const
    {
        dbg<<"SBShapelet fillKImage\n";
        dbg<<"kx = "<<kx0<<" + i * "<<dkx<<" + j * "<<dkxy<<std::endl;
        dbg<<"ky = "<<ky0<<" + i * "<<dkyx<<" + j * "<<dky<<std::endl;
        const int m = im.getNCol();
        const int n = im.getNRow();
        std::complex<T>* ptr = im.getData();
        int skip = im.getNSkip();
        assert(im.getStep() == 1);

        kx0 *= _sigma;
        dkx *= _sigma;
        dkxy *= _sigma;
        ky0 *= _sigma;
        dky *= _sigma;
        dkyx *= _sigma;

        VectorXd mkx(m*n);
        VectorXd mky(m*n);
        for (int j=0,k=0; j<n; ++j,kx0+=dkxy,ky0+=dky) {
            double kx = kx0;
            double ky = ky0;
            for (int i=0; i<m; ++i,++k,kx+=dkx,ky+=dkyx) {
                mkx[k] = kx; mky[k] = ky;
            }
        }

        VectorXcd val(m*n);
        FillKValue(_bvec,_sigma,val,mkx,mky);

        for (int j=0,k=0; j<n; ++j,ptr+=skip)
            for (int i=0; i<m; ++i,++k)
                *ptr++ = val[k];
    }

    template <typename T>
    void ShapeletFitImage(double sigma, LVector& bvec, const BaseImage<T>& image,
                          double image_scale, const Position<double>& center)
    {
        // TODO: It would be nice to be able to fit this with an arbitrary WCS to fit in
        //       sky coordinates.  For now, just use the image_scale.
        dbg<<"Start ShapeletFitImage:\n";
        xdbg<<"sigma = "<<sigma<<std::endl;
        xdbg<<"bvec = "<<bvec<<std::endl;
        xdbg<<"center = "<<center<<std::endl;
        double scale = image_scale / sigma;
        xdbg<<"scale = "<<scale<<std::endl;
        const int nx = image.getXMax() - image.getXMin() + 1;
        const int ny = image.getYMax() - image.getYMin() + 1;
        xdbg<<"nx,ny = "<<nx<<','<<ny<<std::endl;
        const int npts = nx * ny;
        xdbg<<"npts = "<<npts<<std::endl;
        VectorXd x(npts);
        VectorXd y(npts);
        VectorXd I(npts);
        int i=0;
        for (int ix = image.getXMin(); ix <= image.getXMax(); ++ix) {
            for (int iy = image.getYMin(); iy <= image.getYMax(); ++iy,++i) {
                x[i] = (ix - center.x) * scale;
                y[i] = (iy - center.y) * scale;
                I[i] = image(ix,iy);
            }
        }
        xxdbg<<"x = "<<x<<std::endl;
        xxdbg<<"y = "<<y<<std::endl;
        xxdbg<<"I = "<<I<<std::endl;

        MatrixXd psi(npts,bvec.size());
        LVector::basis(x,y,psi,bvec.getOrder(),sigma);
        // I = psi * b
#ifdef USE_TMV
        // TMV solves this by writing b = I/psi.
        // We use QRP in case the psi matrix is close to singular (although it shouldn't be).
        psi.divideUsing(tmv::QRP);
        bvec.rVector() = I/psi;
#else
        bvec.rVector() = psi.colPivHouseholderQr().solve(I);
#endif
        xdbg<<"Done FitImage: bvec = "<<bvec<<std::endl;
    }

    template void ShapeletFitImage(
        double sigma, LVector& bvec, const BaseImage<double>& image, double image_scale,
        const Position<double>& center);
    template void ShapeletFitImage(
        double sigma, LVector& bvec, const BaseImage<float>& image, double image_scale,
        const Position<double>& center);
    template void ShapeletFitImage(
        double sigma, LVector& bvec, const BaseImage<int32_t>& image, double image_scale,
        const Position<double>& center);
    template void ShapeletFitImage(
        double sigma, LVector& bvec, const BaseImage<int16_t>& image, double image_scale,
        const Position<double>& center);
    template void ShapeletFitImage(
        double sigma, LVector& bvec, const BaseImage<uint32_t>& image, double image_scale,
        const Position<double>& center);
    template void ShapeletFitImage(
        double sigma, LVector& bvec, const BaseImage<uint16_t>& image, double image_scale,
        const Position<double>& center);
}
