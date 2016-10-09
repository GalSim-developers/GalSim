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

//#define OUTPUT_FFT // Output the fft grids to files.  (Requires DEBUGLOGGING to be on as well.)

#include "SBProfile.h"
#include "SBTransform.h"
#include "SBProfileImpl.h"
#include "FFT.h"

#ifdef DEBUGLOGGING
#include <fstream>
//std::ostream* dbgout = new std::ofstream("debug.out");
std::ostream* dbgout = &std::cerr;
int verbose_level = 2;
// There are three levels of verbosity which can be helpful when debugging,
// which are written as dbg, xdbg, xxdbg (all defined in Std.h).
// It's Mike's way to have debug statements in the code that are really easy to turn
// on and off.
//
// If DEBUGLOGGING is #defined, then these write out to *dbgout, according to the value
// of verbose_level.
// dbg requires verbose_level >= 1
// xdbg requires verbose_level >= 2
// xxdbg requires verbose_level >= 3
//
// If DEBUGLOGGING is not defined, the all three becomes just `if (false) std::cerr`,
// so the compiler parses the statement fine, but trivially optimizes the code away,
// so there is no efficiency hit from leaving them in the code.
#endif

namespace galsim {

    SBProfile::SBProfile() {}

    SBProfile::SBProfile(const SBProfile& rhs) : _pimpl(rhs._pimpl) {}

    SBProfile& SBProfile::operator=(const SBProfile& rhs)
    { _pimpl = rhs._pimpl; return *this; }

    SBProfile::~SBProfile()
    {
        // Not strictly necessary, but it sets the ptr to 0, so if somehow someone
        // manages to use an SBProfile after it was deleted, the assert(_pimpl.get())
        // will trigger an exception.
        _pimpl.reset();
    }

    std::string SBProfile::serialize() const
    {
        assert(_pimpl.get());
        return _pimpl->serialize();
    }

    std::string SBProfile::repr() const
    {
        assert(_pimpl.get());
        return _pimpl->repr();
    }

    const boost::shared_ptr<GSParams> SBProfile::getGSParams() const
    {
        assert(_pimpl.get());
        return _pimpl->gsparams.getP();
    }

    double SBProfile::xValue(const Position<double>& p) const
    {
        assert(_pimpl.get());
        return _pimpl->xValue(p);
    }

    std::complex<double> SBProfile::kValue(const Position<double>& k) const
    {
        assert(_pimpl.get());
        return _pimpl->kValue(k);
    }

    void SBProfile::getXRange(double& xmin, double& xmax, std::vector<double>& splits) const
    {
        assert(_pimpl.get());
        _pimpl->getXRange(xmin,xmax,splits);
    }

    void SBProfile::getYRange(double& ymin, double& ymax, std::vector<double>& splits) const
    {
        assert(_pimpl.get());
        _pimpl->getYRange(ymin,ymax,splits);
    }

    void SBProfile::getYRangeX(
        double x, double& ymin, double& ymax, std::vector<double>& splits) const
    {
        assert(_pimpl.get());
        _pimpl->getYRangeX(x,ymin,ymax,splits);
    }

    double SBProfile::maxK() const
    {
        assert(_pimpl.get());
        return _pimpl->maxK();
    }

    double SBProfile::stepK() const
    {
        assert(_pimpl.get());
        return _pimpl->stepK();
    }

    bool SBProfile::isAxisymmetric() const
    {
        assert(_pimpl.get());
        return _pimpl->isAxisymmetric();
    }

    bool SBProfile::hasHardEdges() const
    {
        assert(_pimpl.get());
        return _pimpl->hasHardEdges();
    }

    bool SBProfile::isAnalyticX() const
    {
        assert(_pimpl.get());
        return _pimpl->isAnalyticX();
    }

    bool SBProfile::isAnalyticK() const
    {
        assert(_pimpl.get());
        return _pimpl->isAnalyticK();
    }

    Position<double> SBProfile::centroid() const
    {
        assert(_pimpl.get());
        return _pimpl->centroid();
    }

    double SBProfile::getFlux() const
    {
        assert(_pimpl.get());
        return _pimpl->getFlux();
    }

    double SBProfile::maxSB() const
    {
        assert(_pimpl.get());
        return _pimpl->maxSB();
    }

    boost::shared_ptr<PhotonArray> SBProfile::shoot(int N, UniformDeviate ud) const
    {
        assert(_pimpl.get());
        return _pimpl->shoot(N,ud);
    }

    double SBProfile::getPositiveFlux() const
    {
        assert(_pimpl.get());
        return _pimpl->getPositiveFlux();
    }

    double SBProfile::getNegativeFlux() const
    {
        assert(_pimpl.get());
        return _pimpl->getNegativeFlux();
    }

    SBProfile::SBProfile(SBProfileImpl* pimpl) : _pimpl(pimpl) {}

    SBProfile::SBProfileImpl::SBProfileImpl(const GSParamsPtr& gsparams) :
        gsparams(gsparams ? gsparams : GSParamsPtr::getDefault()) {}

    SBProfile::SBProfileImpl* SBProfile::GetImpl(const SBProfile& rhs)
    { return rhs._pimpl.get(); }

    SBTransform SBProfile::scaleFlux(double fluxRatio) const
    { return SBTransform(*this,1.,0.,0.,1.,Position<double>(0.,0.),fluxRatio); }

    SBTransform SBProfile::expand(double scale) const
    { return SBTransform(*this,scale,0.,0.,scale); }

    SBTransform SBProfile::rotate(const Angle& theta) const
    {
        double sint,cost;
        theta.sincos(sint,cost);
        return SBTransform(*this,cost,-sint,sint,cost);
    }

    SBTransform SBProfile::transform(double dudx, double dudy, double dvdx, double dvdy) const
    { return SBTransform(*this, dudx, dudy, dvdx, dvdy); }

    SBTransform SBProfile::shift(const Position<double>& delta) const
    { return SBTransform(*this,1.,0.,0.,1., delta); }

    //
    // Common methods of Base Class "SBProfile"
    //

    int SBProfile::getGoodImageSize(double dx, double wmult) const
    {
        dbg<<"Start getGoodImageSize\n";

        // Find a good size based on dx and stepK
        double Nd = 2.*M_PI/(dx*stepK());
        dbg<<"Nd = "<<Nd<<std::endl;
        Nd *= wmult; // make even bigger if desired
        dbg<<"Nd => "<<Nd<<std::endl;

        // Make it an integer
        // Some slop to keep from getting extra pixels due to roundoff errors in calculations.
        int N = int(std::ceil(Nd*(1.-1.e-12)));
        dbg<<"N = "<<N<<std::endl;

        // Round up to an even value
        N = 2*( (N+1)/2);
        dbg<<"N => "<<N<<std::endl;

        return N;
    }

    // First is a simple case wherein we have a formula for x values:
    template <typename T>
    double SBProfile::plainDraw(ImageView<T> I) const
    {
        dbg<<"Start plainDraw"<<std::endl;
        assert(_pimpl.get());
        return _pimpl->fillXImage(I);
    }

    // The derived classes pretty much all override these functions, since there are
    // almost always (at least minor) efficiency gains from doing so.  But we have
    // them here in case someone doesn't want to bother for a new class.
    void SBProfile::SBProfileImpl::fillXValue(tmv::MatrixView<double> val,
                                              double x0, double dx, int izero,
                                              double y0, double dy, int jzero) const
    {
        dbg<<"SBProfile fillXValue\n";
        dbg<<"x = "<<x0<<" + i * "<<dx<<", izero = "<<izero<<std::endl;
        dbg<<"y = "<<y0<<" + j * "<<dy<<", jzero = "<<jzero<<std::endl;
        assert(val.stepi() == 1);
        assert(val.canLinearize());
        const int m = val.colsize();
        const int n = val.rowsize();
        typedef tmv::VIt<double,1,tmv::NonConj> It;

        It valit = val.linearView().begin();
        double y = y0;
        for (int j=0;j<n;++j,y+=dy) {
            double x = x0;
            for (int i=0;i<m;++i,x+=dx) {
                *valit++ = xValue(Position<double>(x,y));
            }
        }
    }

    void SBProfile::SBProfileImpl::fillKValue(tmv::MatrixView<std::complex<double> > val,
                                              double kx0, double dkx, int izero,
                                              double ky0, double dky, int jzero) const
    {
        dbg<<"SBProfile fillKValue\n";
        dbg<<"kx = "<<kx0<<" + i * "<<dkx<<", izero = "<<izero<<std::endl;
        dbg<<"ky = "<<ky0<<" + j * "<<dky<<", jzero = "<<jzero<<std::endl;
        assert(val.stepi() == 1);
        assert(val.canLinearize());
        const int m = val.colsize();
        const int n = val.rowsize();
        typedef tmv::VIt<std::complex<double>,1,tmv::NonConj> It;

        It valit = val.linearView().begin();
        double ky = ky0;
        for (int j=0;j<n;++j,ky+=dky) {
            double kx = kx0;
            for (int i=0;i<m;++i,kx+=dkx) *valit++ = kValue(Position<double>(kx,ky));
        }
    }

    void SBProfile::SBProfileImpl::fillXValue(tmv::MatrixView<double> val,
                                              double x0, double dx, double dxy,
                                              double y0, double dy, double dyx) const
    {
        dbg<<"SBProfile fillXValue\n";
        dbg<<"x = "<<x0<<" + i * "<<dx<<" + j * "<<dxy<<std::endl;
        dbg<<"y = "<<y0<<" + i * "<<dyx<<" + j * "<<dy<<std::endl;
        assert(val.stepi() == 1);
        assert(val.canLinearize());
        const int m = val.colsize();
        const int n = val.rowsize();
        typedef tmv::VIt<double,1,tmv::NonConj> It;

        It valit = val.linearView().begin();
        for (int j=0;j<n;++j,x0+=dxy,y0+=dy) {
            double x = x0;
            double y = y0;
            for (int i=0;i<m;++i,x+=dx,y+=dyx) *valit++ = xValue(Position<double>(x,y));
        }
    }

    void SBProfile::SBProfileImpl::fillKValue(tmv::MatrixView<std::complex<double> > val,
                                              double kx0, double dkx, double dkxy,
                                              double ky0, double dky, double dkyx) const
    {
        dbg<<"SBProfile fillKValue\n";
        dbg<<"kx = "<<kx0<<" + i * "<<dkx<<" + j * "<<dkxy<<std::endl;
        dbg<<"ky = "<<ky0<<" + i * "<<dkyx<<" + j * "<<dky<<std::endl;
        assert(val.stepi() == 1);
        assert(val.canLinearize());
        const int m = val.colsize();
        const int n = val.rowsize();
        typedef tmv::VIt<std::complex<double>,1,tmv::NonConj> It;

        It valit = val.linearView().begin();
        for (int j=0;j<n;++j,kx0+=dkxy,ky0+=dky) {
            double kx = kx0;
            double ky = ky0;
            for (int i=0;i<m;++i,kx+=dkx,ky+=dkyx) *valit++ = kValue(Position<double>(kx,ky));
        }
    }

    // Note: Once we have TMV 0.90, this won't be necessary, since arithmetic between different
    // types will be allowed.
    template <typename T>
    void addMatrix(tmv::MatrixView<T> m1, const tmv::ConstMatrixView<double>& m2)
    {
        tmv::Matrix<T> m2T = m2;
        m1 += m2T;
    }

    void addMatrix(tmv::MatrixView<double> m1, const tmv::ConstMatrixView<double>& m2)
    { m1 += m2; }

    template <typename T>
    double SBProfile::SBProfileImpl::fillXImage(ImageView<T>& I) const
    {
        xdbg<<"Start fillXImage"<<std::endl;

        const int m = I.getXMax()-I.getXMin()+1;
        const int n = I.getYMax()-I.getYMin()+1;
        xdbg<<"m,n = "<<m<<','<<n<<std::endl;
        tmv::Vector<double> x(m);
        const int xmin = I.getXMin();
        for (int i=0;i<m;++i) x.ref(i) = (xmin+i);
        xdbg<<"xmin = "<<xmin<<std::endl;
        xdbg<<"x = "<<x<<std::endl;

        tmv::Vector<double> y(n);
        const int ymin = I.getYMin();
        xdbg<<"ymin = "<<ymin<<std::endl;
        for (int i=0;i<n;++i) y.ref(i) = (ymin+i);
        xdbg<<"y = "<<y<<std::endl;

        tmv::Matrix<double> val(m,n);
#ifdef DEBUGLOGGING
        val.setAllTo(999.);
#endif
        assert(xmin <= 0 && ymin <= 0 && -xmin < m && -ymin < n);
        xdbg<<"Call fillXValue with "<<xmin<<','<<1.<<','<<-xmin<<
            ','<<ymin<<','<<1.<<','<<-ymin<<std::endl;
        fillXValue(val.view(),xmin,1.,-xmin,ymin,1.,-ymin);

        tmv::MatrixView<T> mI(I.getData(),m,n,1,I.getStride(),tmv::NonConj);
        //mI += val;
        addMatrix(mI,val);
        double totalflux = val.sumElements();
        return totalflux;
    }

    // Now the more complex case: real space via FT from k space.
    // Will enforce image size is power of 2 or 3x2^n.
    // Aliasing will be handled by folding the k values before transforming
    // And enforce no image folding
    template <typename T>
    double SBProfile::fourierDraw(ImageView<T> I, double wmult) const
    {
        dbg<<"Start fourierDraw"<<std::endl;
        Bounds<int> imgBounds; // Bounds for output image
        dbg<<"  maxK() = "<<maxK()<<std::endl;
        dbg<<"  stepK() = "<<stepK()<<std::endl;
        dbg<<"  image bounds = "<<I.getBounds()<<std::endl;
        dbg<<"  wmult = "<<wmult<<std::endl;

        int Nnofold = getGoodImageSize(1.,wmult);
        dbg<<"Nnofold = "<<Nnofold<<std::endl;

        // We must make something big enough to cover the target image size:
        int xSize, ySize;
        xSize = I.getXMax()-I.getXMin()+1;
        ySize = I.getYMax()-I.getYMin()+1;
        if (xSize  > Nnofold) Nnofold = xSize;
        if (ySize  > Nnofold) Nnofold = ySize;
        dbg<<" After scale up to image size, Nnofold = "<<Nnofold<<std::endl;

        // Round up to a good size for making FFTs:
        int NFT = goodFFTSize(Nnofold);
        NFT = std::max(NFT,_pimpl->gsparams->minimum_fft_size);
        dbg << " After adjustments: Nnofold " << Nnofold << " NFT " << NFT << std::endl;

        double dk = 2.*M_PI/NFT;
        dbg <<
            " After adjustments: dk " << dk <<
            " maxK " << dk*NFT/2 << std::endl;
        xdbg<<"dk - stepK() = "<<dk-(stepK()*(1.+1.e-8))<<std::endl;
        xassert(dk <= stepK()*(1. + 1.e-8)); // Add a little slop in case of rounding errors.
        boost::shared_ptr<XTable> xt;
        if (NFT*dk/2 > maxK()) {
            dbg<<"NFT*dk/2 = "<<NFT*dk/2<<" > maxK() = "<<maxK()<<std::endl;
            dbg<<"Use NFT = "<<NFT<<std::endl;
            if (NFT > _pimpl->gsparams->maximum_fft_size)
                FormatAndThrow<SBError>() <<
                    "fourierDraw() requires an FFT that is too large, " << NFT <<
                    "\nIf you can handle the large FFT, you may update gsparams.maximum_fft_size.";
            // No aliasing: build KTable and transform
            KTable kt(NFT,dk);
            assert(_pimpl.get());
            _pimpl->fillKGrid(kt);
            xt = kt.transform();
        } else {
            dbg<<"NFT*dk/2 = "<<NFT*dk/2<<" <= maxK() = "<<maxK()<<std::endl;
            // There will be aliasing.  Construct a KTable out to maxK() and
            // then wrap it
            int Nk = int(std::ceil(maxK()/dk)) * 2;
            dbg<<"Initial Nk = "<<Nk<<std::endl;
            // Round up to the next multiple of NFT.  Otherwise, the wrapping will start/stop
            // somewhere in the middle of the image, which can lead to subtle artifacts like
            // losing symmetry that should be present in the final image.
            Nk = ((Nk-1)/NFT + 1) * NFT;
            dbg<<"Use Nk = "<<Nk<<std::endl;
            if (Nk > _pimpl->gsparams->maximum_fft_size)
                FormatAndThrow<SBError>() <<
                    "fourierDraw() requires an FFT that is too large, " << Nk <<
                    "\nIf you can handle the large FFT, you may update gsparams.maximum_fft_size.";
            KTable kt(Nk, dk);
            assert(_pimpl.get());
            _pimpl->fillKGrid(kt);
            xt = kt.wrap(NFT)->transform();
        }
        int Nxt = xt->getN();
        dbg<<"Nxt = "<<Nxt<<std::endl;

#ifdef OUTPUT_FFT
        std::ofstream fout("xt.dat");
        tmv::MatrixView<double> mxt(xt->getArray(),Nxt,Nxt,1,Nxt,tmv::NonConj);
        fout << tmv::EigenIO() << mxt << std::endl;
        fout.close();
#endif

        Bounds<int> xb(-Nxt/2, Nxt/2-1, -Nxt/2, Nxt/2-1);
        if (I.getYMin() < xb.getYMin()
            || I.getYMax() > xb.getYMax()
            || I.getXMin() < xb.getXMin()
            || I.getXMax() > xb.getXMax()) {
            dbg << "Bounds error!! target image bounds " << I.getBounds()
                << " and FFT range " << xb << std::endl;
            throw SBError("fourierDraw() FT bounds do not cover target image");
        }
        double sum=0.;
        for (int y = I.getYMin(); y <= I.getYMax(); y++) {
            for (int x = I.getXMin(); x <= I.getXMax(); x++) {
                double temp = xt->xval(x,y);
                I(x,y) += T(temp);
                sum += temp;
            }
        }

        return sum;
    }

    template <typename T>
    void SBProfile::drawK(ImageView<T> Re, ImageView<T> Im) const
    {
        dbg<<"Start drawK: \n";
        // Make sure input images match or are both null
        assert(Re.getBounds() == Im.getBounds());

        const int m = (Re.getXMax()-Re.getXMin()+1);
        const int n = (Re.getYMax()-Re.getYMin()+1);
        const int xmin = Re.getXMin();
        const int ymin = Re.getYMin();
        dbg<<"m,n = "<<m<<','<<n<<std::endl;
        dbg<<"xmin,ymin = "<<xmin<<','<<ymin<<std::endl;

        tmv::Matrix<std::complex<double> > val(m,n);
#ifdef DEBUGLOGGING
        val.setAllTo(999.);
#endif
        // Calculate all the kValues at once, since this is often faster than many calls to kValue.
        assert(xmin <= 0 && ymin <= 0 && -xmin < m && -ymin < n);
        _pimpl->fillKValue(val.view(),xmin,1.,-xmin,ymin,1.,-ymin);
        dbg<<"F(k=0) = "<<val(-xmin,-ymin)<<std::endl;

        tmv::MatrixView<T> mRe(Re.getData(),m,n,1,Re.getStride(),tmv::NonConj);
        tmv::MatrixView<T> mIm(Im.getData(),m,n,1,Im.getStride(),tmv::NonConj);
        addMatrix(mRe,val.realPart());
        addMatrix(mIm,val.imagPart());
    }

    void SBProfile::SBProfileImpl::fillXGrid(XTable& xt) const
    {
        xdbg<<"Start fillXGrid"<<std::endl;

        int N = xt.getN();
        double dx = xt.getDx();
        xt.clearCache();

        tmv::Matrix<double> val(N,N);
#ifdef DEBUGLOGGING
        val.setAllTo(999.);
#endif
        fillXValue(val.view(),-(N/2)*dx,dx,N/2,-(N/2)*dx,dx,N/2);

        tmv::MatrixView<double> mxt(xt.getArray(),N,N,1,N,tmv::NonConj);
        mxt = val;
    }

    void SBProfile::SBProfileImpl::fillKGrid(KTable& kt) const
    {
        dbg<<"Start fillKGrid\n";
        int N = kt.getN();
        double dk = kt.getDk();
        kt.clearCache();

        tmv::Matrix<std::complex<double> > val(N/2+1,N+1);
#ifdef DEBUGLOGGING
        val.setAllTo(999.);
#endif
        fillKValue(val.view(),0.,dk,0,-N/2*dk,dk,N/2);

        tmv::MatrixView<std::complex<double> > mkt(kt.getArray(),N/2+1,N,1,N/2+1,tmv::NonConj);
#ifdef DEBUGLOGGING
        mkt.setAllTo(1.e100);
#endif
        // The KTable wants the locations of the + and - ky values swapped.
        mkt.colRange(0,N/2) = val.colRange(N/2,N);
        mkt.colRange(N/2+1,N) = val.colRange(1,N/2);
        // For the N/2 column, we use the average of the ky = +N/2 and -N/2 values
        // Otherwise you can get strange effects when the profile isn't radially symmetric.
        // e.g. A shift will induce a spurious shear. (BAD!!)
        mkt.col(N/2) = 0.5*val.col(0) + 0.5*val.col(N);
        // Similarly, the N/2 row should really be the average of the kx = +N/2 and -N/2 values,
        // which again is exactly 0.  We didn't calculate the kx = -N/2 values, but we know that
        // f(-kx,-ky) = conj(f(kx,ky)), so the calculation becomes:
        mkt.row(N/2).subVector(1,N/2) += mkt.row(N/2).subVector(N-1,N/2,-1).conjugate();
        mkt.row(N/2).subVector(1,N/2) *= 0.5;
        mkt.row(N/2).subVector(N-1,N/2,-1) = mkt.row(N/2).subVector(1,N/2).conjugate();

#ifdef OUTPUT_FFT
        xdbg<<"val.row(0) = "<<val.row(0)<<std::endl;
        xdbg<<"val.row(N/2) = "<<val.row(N/2)<<std::endl;
        xdbg<<"val.col(0) = "<<val.col(0)<<std::endl;
        xdbg<<"val.col(N) = "<<val.col(N)<<std::endl;
        xdbg<<"val.col(N/2) = "<<val.col(N/2)<<std::endl;
        xdbg<<"mkt.col(N/2) = "<<mkt.col(N/2)<<std::endl;
        xdbg<<"mkt.row(N/2) = "<<mkt.row(N/2)<<std::endl;
        std::ofstream fout_re("ktr.dat");
        std::ofstream fout_im("kti.dat");
        fout_re << tmv::EigenIO() << mkt.realPart() << std::endl;
        fout_im << tmv::EigenIO() << mkt.imagPart() << std::endl;
        fout_re.close();
        fout_im.close();
#endif
    }

    // The type of T (real or complex) determines whether the call-back is to
    // fillXValue or fillKValue.
    template <typename T>
    struct QuadrantHelper
    {
        template <class Prof>
        static void fill(const Prof& prof, tmv::MatrixView<T> q,
                         double x0, double dx, double y0, double dy)
        { prof.fillXValue(q,x0,dx,0,y0,dy,0); }
    };

    template <typename T>
    struct QuadrantHelper<std::complex<T> >
    {
        typedef std::complex<T> CT;
        template <class Prof>
        static void fill(const Prof& prof, tmv::MatrixView<CT> q,
                         double kx0, double dkx, double ky0, double dky)
        { prof.fillKValue(q,kx0,dkx,0,ky0,dky,0); }
    };

    // The code is basically the same for X or K.
    template <class Prof, typename T>
    static void FillQuadrant(const Prof& prof, tmv::MatrixView<T> val,
                             double x0, double dx, int nx1, double y0, double dy, int ny1)
    {
        dbg<<"Start FillQuadrant\n";
        dbg<<x0<<" "<<dx<<" "<<nx1<<"   "<<y0<<" "<<dy<<" "<<ny1<<std::endl;
        // Figure out which quadrant is the largest.  Need to use that one.
        const int nx = val.colsize();
        const int nx2 = nx - nx1-1;
        const int ny = val.rowsize();
        const int ny2 = ny - ny1-1;
        xdbg<<"nx = "<<nx1<<" + "<<nx2<<" + 1 = "<<nx<<std::endl;
        xdbg<<"ny = "<<ny1<<" + "<<ny2<<" + 1 = "<<ny<<std::endl;
        // Keep track of which quadrant is done in the first section.
        bool ur_done = false;
        bool ul_done = false;
        bool lr_done = false;
        bool ll_done = false;
        boost::shared_ptr<tmv::MatrixView<T> > q; // The matrix to copy to each quadrant
        if (nx2 >= nx1) {
            if (ny2 >= ny1) {
                // Upper right is the big quadrant
                xdbg<<"Use Upper right (nx2,ny2)"<<std::endl;
                q.reset(new tmv::MatrixView<T>(val.subMatrix(nx1,nx,ny1,ny)));
                QuadrantHelper<T>::fill(prof,*q,nx1==0?x0:0.,dx,ny1==0?y0:0.,dy);
                ur_done = true;
                // Also do the rest of the ix=0 row and iy=0 col
                val.row(nx1,0,ny1).reverse() = q->row(0,1,ny1+1);
                val.col(ny1,0,nx1).reverse() = q->col(0,1,nx1+1);
            } else {
                // Lower right is the big quadrant
                xdbg<<"Use Lower right (nx2,ny1)"<<std::endl;
                q.reset(new tmv::MatrixView<T>(val.subMatrix(nx1,nx,ny1,-1,1,-1)));
                QuadrantHelper<T>::fill(prof,val.subMatrix(nx1,nx,0,ny1+1),nx1==0?x0:0.,dx,y0,dy);
                lr_done = true;
                val.row(nx1,ny1+1,ny) = q->row(0,1,ny2+1);
                val.col(ny1,0,nx1).reverse() = q->row(0,1,nx1+1);
            }
        } else {
            if (ny2 >= ny1) {
                // Upper left is the big quadrant
                xdbg<<"Use Upper left (nx1,ny2)"<<std::endl;
                q.reset(new tmv::MatrixView<T>(val.subMatrix(nx1,-1,ny1,ny,-1,1)));
                QuadrantHelper<T>::fill(prof,val.subMatrix(0,nx1+1,ny1,ny),x0,dx,ny1==0?y0:0.,dy);
                ul_done = true;
                val.row(nx1,0,ny1).reverse() = q->row(0,1,ny1+1);
                val.col(ny1,nx1+1,nx) = q->col(0,1,nx2+1);
            } else {
                // Lower left is the big quadrant
                xdbg<<"Use Lower left (nx1,ny1)"<<std::endl;
                q.reset(new tmv::MatrixView<T>(val.subMatrix(nx1,-1,ny1,-1,-1,-1)));
                QuadrantHelper<T>::fill(prof,val.subMatrix(0,nx1+1,0,ny1+1),x0,dx,y0,dy);
                ll_done = true;
                val.row(nx1,ny1+1,ny) = q->row(0,1,ny2+1);
                val.col(ny1,nx1+1,nx) = q->col(0,1,nx2+1);
            }
        }
        if (!ur_done && nx2 > 0 && ny2 > 0)
            val.subMatrix(nx1+1,nx,ny1+1,ny) = q->subMatrix(1,nx2+1,1,ny2+1);
        if (!lr_done && nx2 > 0 && ny1 > 0)
            val.subMatrix(nx1+1,nx,ny1-1,-1,1,-1) = q->subMatrix(1,nx2+1,1,ny1+1);
        if (!ul_done && nx1 > 0 && ny2 > 0)
            val.subMatrix(nx1-1,-1,ny1+1,ny,-1,1) = q->subMatrix(1,nx1+1,1,ny2+1);
        if (!ll_done && nx1 > 0 && ny1 > 0)
            val.subMatrix(nx1-1,-1,ny1-1,-1,-1,-1) = q->subMatrix(1,nx1+1,1,ny1+1);
        xdbg<<"Done copying quadrants"<<std::endl;
    }
    void SBProfile::SBProfileImpl::fillXValueQuadrant(tmv::MatrixView<double> val,
                                                      double x0, double dx, int nx1,
                                                      double y0, double dy, int ny1) const
    {
        // Guard against infinite loop.
        assert(nx1 != 0 || ny1 != 0);
        FillQuadrant(*this,val,x0,dx,nx1,y0,dy,ny1);
    }
    void SBProfile::SBProfileImpl::fillKValueQuadrant(tmv::MatrixView<std::complex<double> > val,
                                                      double kx0, double dkx, int nkx1,
                                                      double ky0, double dky, int nky1) const
    {
        // Guard against infinite loop.
        assert(nkx1 != 0 || nky1 != 0);
        FillQuadrant(*this,val,kx0,dkx,nkx1,ky0,dky,nky1);
    }

    // instantiate template functions for expected image types
    template double SBProfile::SBProfileImpl::fillXImage(ImageView<float>& img) const;
    template double SBProfile::SBProfileImpl::fillXImage(ImageView<double>& img) const;

    template double SBProfile::plainDraw(ImageView<float> I) const;
    template double SBProfile::plainDraw(ImageView<double> I) const;

    template double SBProfile::fourierDraw(ImageView<float> I, double wmult) const;
    template double SBProfile::fourierDraw(ImageView<double> I, double wmult) const;

    template void SBProfile::drawK(ImageView<float> Re, ImageView<float> Im) const;
    template void SBProfile::drawK(ImageView<double> Re, ImageView<double> Im) const;

}
