/* -*- c++ -*-
 * Copyright (c) 2012-2017 by the GalSim developers team on GitHub
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

    SBShapelet::SBShapelet(double sigma, LVector bvec, const GSParamsPtr& gsparams) :
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

    std::string SBShapelet::SBShapeletImpl::serialize() const
    {
        std::ostringstream oss(" ");
        oss.precision(std::numeric_limits<double>::digits10 + 4);
        oss << "galsim._galsim.SBShapelet("<<getSigma()<<", "<<getBVec().repr();
        oss << ", galsim.GSParams("<<*gsparams<<"))";
        return oss.str();
    }

    SBShapelet::SBShapeletImpl::SBShapeletImpl(double sigma, const LVector& bvec,
                                               const GSParamsPtr& gsparams) :
        SBProfileImpl(gsparams),
        _sigma(sigma), _bvec(bvec.copy()) {}

    double SBShapelet::SBShapeletImpl::maxK() const
    {
        // Start with value for plain old Gaussian:
        double maxk = sqrt(-2.*std::log(this->gsparams->maxk_threshold))/_sigma;
        // Grow as sqrt of (order+1)
        // Note: this is an approximation.  The right value would require looking at
        // the actual coefficients and doing something smart with them.
        maxk *= sqrt(double(_bvec.getOrder()+1));
        return maxk;
    }

    double SBShapelet::SBShapeletImpl::stepK() const
    {
        // Start with value for plain old Gaussian:
        double R = std::max(4., sqrt(-2.*std::log(this->gsparams->folding_threshold)));
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

        tmv::Matrix<double> mx(m,n);
        for (int i=0; i<m; ++i,x0+=dx) mx.row(i).setAllTo(x0);
        tmv::Matrix<double> my(m,n);
        for (int j=0; j<n; ++j,y0+=dy) my.col(j).setAllTo(y0);

        tmv::Matrix<double> val(m,n);
        fillXValue(val.view(),mx,my);

        typedef tmv::VIt<double,1,tmv::NonConj> It;
        It valit = val.linearView().begin();
        for (int j=0; j<n; ++j,ptr+=skip) {
            for (int i=0; i<m; ++i)
                *ptr++ = *valit++;
        }
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

        tmv::Matrix<double> mx(m,n);
        tmv::Matrix<double> my(m,n);
        typedef tmv::VIt<double,1,tmv::NonConj> It;
        It xit = mx.linearView().begin();
        It yit = my.linearView().begin();
        for (int j=0; j<n; ++j,x0+=dxy,y0+=dy) {
            double x = x0;
            double y = y0;
            for (int i=0; i<m; ++i,x+=dx,y+=dyx) { *xit++ = x; *yit++ = y; }
        }

        tmv::Matrix<double> val(m,n);
        fillXValue(val.view(),mx,my);

        It valit = val.linearView().begin();
        for (int j=0; j<n; ++j,ptr+=skip) {
            for (int i=0; i<m; ++i)
                *ptr++ = *valit++;
        }
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

        tmv::Matrix<double> mkx(m,n);
        for (int i=0; i<m; ++i,kx0+=dkx) mkx.row(i).setAllTo(kx0);
        tmv::Matrix<double> mky(m,n);
        for (int j=0; j<n; ++j,ky0+=dky) mky.col(j).setAllTo(ky0);

        tmv::Matrix<std::complex<double> > val(m,n);
        fillKValue(val.view(),mkx,mky);

        typedef tmv::VIt<std::complex<double>,1,tmv::NonConj> CIt;
        CIt valit = val.linearView().begin();
        for (int j=0; j<n; ++j,ptr+=skip) {
            for (int i=0; i<m; ++i)
                *ptr++ = *valit++;
        }
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

        tmv::Matrix<double> mkx(m,n);
        tmv::Matrix<double> mky(m,n);
        typedef tmv::VIt<double,1,tmv::NonConj> It;
        It kxit = mkx.linearView().begin();
        It kyit = mky.linearView().begin();
        for (int j=0; j<n; ++j,kx0+=dkxy,ky0+=dky) {
            double kx = kx0;
            double ky = ky0;
            for (int i=0; i<m; ++i,kx+=dkx,ky+=dkyx) { *kxit++ = kx; *kyit++ = ky; }
        }

        tmv::Matrix<std::complex<double> > val(m,n);
        fillKValue(val.view(),mkx,mky);

        typedef tmv::VIt<std::complex<double>,1,tmv::NonConj> CIt;
        CIt valit = val.linearView().begin();
        for (int j=0; j<n; ++j,ptr+=skip) {
            for (int i=0; i<m; ++i)
                *ptr++ = *valit++;
        }
    }

    void SBShapelet::SBShapeletImpl::fillXValue(
        tmv::MatrixView<double> val,
        const tmv::Matrix<double>& x, const tmv::Matrix<double>& y) const
    {
        dbg<<"order = "<<_bvec.getOrder()<<", sigma = "<<_sigma<<std::endl;
        xdbg<<"fillXValue with bvec = "<<_bvec<<std::endl;
        assert(val.stepi() == 1);
        assert(val.canLinearize());
        const int m = val.colsize();
        const int n = val.rowsize();
        tmv::Matrix<double> psi(m*n,_bvec.size());
        LVector::basis(x.constLinearView(),y.constLinearView(),psi.view(),
                       _bvec.getOrder(),_sigma);
        val.linearView() = psi * _bvec.rVector();
    }

    void SBShapelet::SBShapeletImpl::fillKValue(
        tmv::MatrixView<std::complex<double> > val,
        const tmv::Matrix<double>& kx, const tmv::Matrix<double>& ky) const
    {
        dbg<<"order = "<<_bvec.getOrder()<<", sigma = "<<_sigma<<std::endl;
        xdbg<<"fillKValue with bvec = "<<_bvec<<std::endl;
        assert(val.stepi() == 1);
        assert(val.canLinearize());
        const int m = val.colsize();
        const int n = val.rowsize();
        tmv::Matrix<std::complex<double> > psi_k(m*n,_bvec.size());
        LVector::kBasis(kx.constLinearView(),ky.constLinearView(),psi_k.view(),
                        _bvec.getOrder(),_sigma);
        // Note: the explicit cast to Vector<complex<double> > shouldn't be necessary.
        // But not doing so fails for Apple's default BLAS library.  It should be a pretty
        // minimal efficiency difference, so we always do the explicit cast to be safe.
        val.linearView() = psi_k * tmv::Vector<std::complex<double> >(_bvec.rVector());
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
        tmv::Vector<double> x(npts);
        tmv::Vector<double> y(npts);
        tmv::Vector<double> I(npts);
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

        tmv::Matrix<double> psi(npts,bvec.size());
        LVector::basis(x.view(),y.view(),psi.view(),bvec.getOrder(),sigma);
        // I = psi * b
        // TMV solves this by writing b = I/psi.
        // We use QRP in case the psi matrix is close to singular (although it shouldn't be).
        psi.divideUsing(tmv::QRP);
        bvec.rVector() = I/psi;
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
