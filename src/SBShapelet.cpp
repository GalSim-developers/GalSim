// -*- c++ -*-
/*
 * Copyright 2012, 2013 The GalSim developers:
 * https://github.com/GalSim-developers
 *
 * This file is part of GalSim: The modular galaxy image simulation toolkit.
 *
 * GalSim is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * GalSim is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with GalSim.  If not, see <http://www.gnu.org/licenses/>
 */

//#define DEBUGLOGGING

#include "SBShapelet.h"
#include "SBShapeletImpl.h"

#ifdef DEBUGLOGGING
#include <fstream>
//std::ostream* dbgout = new std::ofstream("debug.out");
std::ostream* dbgout = &std::cout;
int verbose_level = 2;
#endif

namespace galsim {

    SBShapelet::SBShapelet(double sigma, LVector bvec, boost::shared_ptr<GSParams> gsparams) :
        SBProfile(new SBShapeletImpl(sigma, bvec, gsparams)) {}

    SBShapelet::SBShapelet(const SBShapelet& rhs) : SBProfile(rhs) {}

    SBShapelet::~SBShapelet() {}

    SBShapelet::SBShapeletImpl::SBShapeletImpl(double sigma, const LVector& bvec,
                                               boost::shared_ptr<GSParams> gsparams) :
        SBProfileImpl(gsparams), _sigma(sigma), _bvec(bvec.copy()) {}

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
        double R = std::max(4., sqrt(-2.*std::log(this->gsparams->alias_threshold)));
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

    void SBShapelet::SBShapeletImpl::fillXValue(tmv::MatrixView<double> val,
                                                double x0, double dx, int ix_zero,
                                                double y0, double dy, int iy_zero) const
    {
        dbg<<"SBShapelet fillXValue\n";
        dbg<<"x = "<<x0<<" + ix * "<<dx<<", ix_zero = "<<ix_zero<<std::endl;
        dbg<<"y = "<<y0<<" + iy * "<<dy<<", iy_zero = "<<iy_zero<<std::endl;
        const int m = val.colsize();
        const int n = val.rowsize();

        x0 /= _sigma;
        dx /= _sigma;
        y0 /= _sigma;
        dy /= _sigma;

        tmv::Matrix<double> mx(m,n);
        for (int i=0;i<m;++i,x0+=dx) mx.row(i).setAllTo(x0);
        tmv::Matrix<double> my(m,n);
        for (int j=0;j<n;++j,y0+=dy) my.col(j).setAllTo(y0);

        fillXValue(val,mx,my);
    }

    void SBShapelet::SBShapeletImpl::fillKValue(tmv::MatrixView<std::complex<double> > val,
                                                double x0, double dx, int ix_zero,
                                                double y0, double dy, int iy_zero) const
    {
        dbg<<"SBShapelet fillKValue\n";
        dbg<<"x = "<<x0<<" + ix * "<<dx<<", ix_zero = "<<ix_zero<<std::endl;
        dbg<<"y = "<<y0<<" + iy * "<<dy<<", iy_zero = "<<iy_zero<<std::endl;
        const int m = val.colsize();
        const int n = val.rowsize();

        x0 *= _sigma;
        dx *= _sigma;
        y0 *= _sigma;
        dy *= _sigma;

        tmv::Matrix<double> mx(m,n);
        for (int i=0;i<m;++i,x0+=dx) mx.row(i).setAllTo(x0);
        tmv::Matrix<double> my(m,n);
        for (int j=0;j<n;++j,y0+=dy) my.col(j).setAllTo(y0);

        fillKValue(val,mx,my);
    }

    void SBShapelet::SBShapeletImpl::fillXValue(tmv::MatrixView<double> val,
                                                double x0, double dx, double dxy,
                                                double y0, double dy, double dyx) const
    {
        dbg<<"SBShapelet fillXValue\n";
        dbg<<"x = "<<x0<<" + ix * "<<dx<<" + iy * "<<dxy<<std::endl;
        dbg<<"y = "<<y0<<" + ix * "<<dyx<<" + iy * "<<dy<<std::endl;
        const int m = val.colsize();
        const int n = val.rowsize();

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
        for (int j=0;j<n;++j,x0+=dxy,y0+=dy) {
            double x = x0;
            double y = y0;
            for (int i=0;i<m;++i,x+=dx,y+=dyx) { *xit++ = x; *yit++ = y; }
        }

        fillXValue(val,mx,my);
    }

    void SBShapelet::SBShapeletImpl::fillKValue(tmv::MatrixView<std::complex<double> > val,
                                                double x0, double dx, double dxy,
                                                double y0, double dy, double dyx) const
    {
        dbg<<"SBShapelet fillKValue\n";
        dbg<<"x = "<<x0<<" + ix * "<<dx<<" + iy * "<<dxy<<std::endl;
        dbg<<"y = "<<y0<<" + ix * "<<dyx<<" + iy * "<<dy<<std::endl;
        const int m = val.colsize();
        const int n = val.rowsize();

        x0 *= _sigma;
        dx *= _sigma;
        dxy *= _sigma;
        y0 *= _sigma;
        dy *= _sigma;
        dyx *= _sigma;

        tmv::Matrix<double> mx(m,n);
        tmv::Matrix<double> my(m,n);
        typedef tmv::VIt<double,1,tmv::NonConj> It;
        It xit = mx.linearView().begin();
        It yit = my.linearView().begin();
        for (int j=0;j<n;++j,x0+=dxy,y0+=dy) {
            double x = x0;
            double y = y0;
            for (int i=0;i<m;++i,x+=dx,y+=dyx) { *xit++ = x; *yit++ = y; }
        }

        fillKValue(val,mx,my);
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
        const tmv::Matrix<double>& x, const tmv::Matrix<double>& y) const
    {
        dbg<<"order = "<<_bvec.getOrder()<<", sigma = "<<_sigma<<std::endl;
        xdbg<<"fillKValue with bvec = "<<_bvec<<std::endl;
        assert(val.stepi() == 1);
        assert(val.canLinearize());
        const int m = val.colsize();
        const int n = val.rowsize();
        tmv::Matrix<std::complex<double> > psi_k(m*n,_bvec.size());
        LVector::kBasis(x.constLinearView(),y.constLinearView(),psi_k.view(),
                        _bvec.getOrder(),_sigma);
        // Note: the explicit cast to Vector<complex<double> > shouldn't be necessary.
        // But not doing so fails for Apple's default BLAS library.  It should be a pretty
        // minimal efficiency difference, so we always do the explicit cast to be safe.
        val.linearView() = psi_k * tmv::Vector<std::complex<double> >(_bvec.rVector());
    }

    template <typename T>
    void ShapeletFitImage(double sigma, LVector& bvec, const BaseImage<T>& image,
                          const Position<double>& center)
    {
        dbg<<"Start ShapeletFitImage:\n";
        xdbg<<"sigma = "<<sigma<<std::endl;
        xdbg<<"bvec = "<<bvec<<std::endl;
        xdbg<<"center = "<<center<<std::endl;
        double scale = image.getScale() / sigma;
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
        double sigma, LVector& bvec, const BaseImage<double>& image,
        const Position<double>& center);
    template void ShapeletFitImage(
        double sigma, LVector& bvec, const BaseImage<float>& image,
        const Position<double>& center);
    template void ShapeletFitImage(
        double sigma, LVector& bvec, const BaseImage<int32_t>& image,
        const Position<double>& center);
    template void ShapeletFitImage(
        double sigma, LVector& bvec, const BaseImage<int16_t>& image,
        const Position<double>& center);
}

