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

//#define OUTPUT_FFT // Output the fft grids to files.  (Requires DEBUGLOGGING to be on as well.)

#include "SBProfile.h"
#include "SBTransform.h"
#include "SBProfileImpl.h"
#include "FFT.h"

#ifdef DEBUGLOGGING
#include <fstream>
std::ostream* dbgout = new std::ofstream("debug.out");
//std::ostream* dbgout = &std::cout;
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

    boost::shared_ptr<GSParams> SBProfile::SBProfileImpl::default_gsparams(new GSParams());

    SBProfile::SBProfileImpl::SBProfileImpl(boost::shared_ptr<GSParams> gsparams) :
        gsparams(gsparams.get() ? gsparams : default_gsparams) {}

    SBProfile::SBProfileImpl* SBProfile::GetImpl(const SBProfile& rhs) 
    { return rhs._pimpl.get(); }

    void SBProfile::scaleFlux(double fluxRatio)
    { 
        SBTransform d(*this,1.,0.,0.,1.,Position<double>(0.,0.),fluxRatio); 
        _pimpl = d._pimpl;
    }

    void SBProfile::setFlux(double flux)
    { 
        SBTransform d(*this,1.,0.,0.,1.,Position<double>(0.,0.),flux/getFlux());
        _pimpl = d._pimpl;
    }

    void SBProfile::applyScale(double scale)
    {
        SBTransform d(*this,scale,0.,0.,scale);
        _pimpl = d._pimpl;
    }

    void SBProfile::applyShear(CppShear s)
    {
        double a, b, c;
        s.getMatrix(a,b,c);
        SBTransform d(*this,a,c,c,b);
        _pimpl = d._pimpl;
    }

    void SBProfile::applyRotation(const Angle& theta)
    {
#ifdef _GLIBCXX_HAVE_SINCOS
        // Most optimizing compilers will do this automatically, but just in case...
        double sint,cost;
        sincos(theta.rad(),&sint,&cost);
#else
        double cost = std::cos(theta.rad());
        double sint = std::sin(theta.rad());
#endif
        SBTransform d(*this,cost,-sint,sint,cost);
        _pimpl = d._pimpl;
    }

    void SBProfile::applyShift(double dx, double dy)
    { 
        SBTransform d(*this,1.,0.,0.,1., Position<double>(dx,dy));
        _pimpl = d._pimpl;
    }

    //
    // Common methods of Base Class "SBProfile"
    //

    // Basic draw command calls either plainDraw or fourierDraw
    template <typename T>
    double SBProfile::draw(ImageView<T> img, double gain, double wmult) const 
    {
        dbg<<"Start draw ImageView"<<std::endl;
        if (isAnalyticX())
            return plainDraw(img, gain);
        else
            return fourierDraw(img, gain, wmult);
    }

    int SBProfile::getGoodImageSize(double dx, double wmult) const
    {
        dbg<<"Start getGoodImageSize\n";

        // Find a good size based on dx and stepK
        double Nd = 2*M_PI/(dx*stepK());
        dbg<<"Nd = "<<Nd<<std::endl;
        Nd *= wmult; // make even bigger if desired
        dbg<<"Nd => "<<Nd<<std::endl;

        // Make it an integer
        // Some slop to keep from getting extra pixels due to roundoff errors in calculations.
        int N = int(std::ceil(Nd-1.e-6));
        dbg<<"N = "<<N<<std::endl;

        // Round up to an even value
        N = 2*( (N+1)/2);
        dbg<<"N => "<<N<<std::endl;

        return N;
    }

    // First is a simple case wherein we have a formula for x values:
    template <typename T>
    double SBProfile::plainDraw(ImageView<T> I, double gain) const 
    {
        dbg<<"Start plainDraw"<<std::endl;
        // recenter an existing image, to be consistent with fourierDraw:
        I.setCenter(0,0);

        assert(_pimpl.get());
        return _pimpl->fillXImage(I, gain);
    }

    // The derived classes pretty much all override these functions, since there are
    // almost always (at least minor) efficiency gains from doing so.  But we have
    // them here in case someone doesn't want to bother for a new class.
    void SBProfile::SBProfileImpl::fillXValue(tmv::MatrixView<double> val,
                                              double x0, double dx, int ix_zero,
                                              double y0, double dy, int iy_zero) const
    {
        dbg<<"SBProfile fillXValue\n";
        dbg<<"x = "<<x0<<" + ix * "<<dx<<", ix_zero = "<<ix_zero<<std::endl;
        dbg<<"y = "<<y0<<" + iy * "<<dy<<", iy_zero = "<<iy_zero<<std::endl;
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
                                              double x0, double dx, int ix_zero,
                                              double y0, double dy, int iy_zero) const
    { 
        dbg<<"SBProfile fillKValue\n";
        dbg<<"x = "<<x0<<" + ix * "<<dx<<", ix_zero = "<<ix_zero<<std::endl;
        dbg<<"y = "<<y0<<" + iy * "<<dy<<", iy_zero = "<<iy_zero<<std::endl;
        assert(val.stepi() == 1);
        assert(val.canLinearize());
        const int m = val.colsize();
        const int n = val.rowsize();
        typedef tmv::VIt<std::complex<double>,1,tmv::NonConj> It;

        //It valit = val.linearView().begin();
        // There is a bug in TMV v0.71 that the above line doesn't work.
        // The workaround is the following:
        It valit(val.linearView().begin().getP(),1);
        double y = y0;
        for (int j=0;j<n;++j,y+=dy) {
            double x = x0;
            for (int i=0;i<m;++i,x+=dx) *valit++ = kValue(Position<double>(x,y));
        }
    }

    void SBProfile::SBProfileImpl::fillXValue(tmv::MatrixView<double> val,
                                              double x0, double dx, double dxy,
                                              double y0, double dy, double dyx) const
    { 
        dbg<<"SBProfile fillXValue\n";
        dbg<<"x = "<<x0<<" + ix * "<<dx<<" + iy * "<<dxy<<std::endl;
        dbg<<"y = "<<y0<<" + ix * "<<dyx<<" + iy * "<<dy<<std::endl;
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
                                              double x0, double dx, double dxy,
                                              double y0, double dy, double dyx) const
    { 
        dbg<<"SBProfile fillKValue\n";
        dbg<<"x = "<<x0<<" + ix * "<<dx<<" + iy * "<<dxy<<std::endl;
        dbg<<"y = "<<y0<<" + ix * "<<dyx<<" + iy * "<<dy<<std::endl;
        assert(val.stepi() == 1);
        assert(val.canLinearize());
        const int m = val.colsize();
        const int n = val.rowsize();
        typedef tmv::VIt<std::complex<double>,1,tmv::NonConj> It;

        It valit(val.linearView().begin().getP(),1);
        for (int j=0;j<n;++j,x0+=dxy,y0+=dy) {
            double x = x0;
            double y = y0;
            for (int i=0;i<m;++i,x+=dx,y+=dyx) *valit++ = kValue(Position<double>(x,y));
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
    double SBProfile::SBProfileImpl::fillXImage(ImageView<T>& I, double gain) const 
    {
        xdbg<<"Start fillXImage"<<std::endl;
        double dx = I.getScale();
        xdbg<<"dx = "<<dx<<", gain = "<<gain<<std::endl;

        const int m = I.getXMax()-I.getXMin()+1;
        const int n = I.getYMax()-I.getYMin()+1;
        tmv::Vector<double> x(m);
        const int xmin = I.getXMin();
        for (int i=0;i<m;++i) x.ref(i) = (xmin+i)*dx;

        tmv::Vector<double> y(n);
        const int ymin = I.getYMin();
        for (int i=0;i<n;++i) y.ref(i) = (ymin+i)*dx;

        tmv::Matrix<double> val(m,n);
#ifdef DEBUGLOGGING
        val.setAllTo(999.);
#endif
        assert(xmin <= 0 && ymin <= 0 && -xmin < m && -ymin < n);
        fillXValue(val.view(),xmin*dx,dx,-xmin,ymin*dx,dx,-ymin);

        // Sometimes rounding errors cause the nominal (0,0) to be slightly off.
        // So redo (0,0) just to be sure.
        // TODO: This is really just to get the unit tests to pass.  It's usually the value
        // for Sersic that fails to match the central peak at 5 digits of accuracy.
        // Probaby, we should just update reference images and remove this line...
        val(-xmin,-ymin) = xValue(Position<double>(0.,0.));

        if (gain != 1.) val /= gain;

        tmv::MatrixView<T> mI(I.getData(),m,n,1,I.getStride(),tmv::NonConj);
        //mI += val;
        addMatrix(mI,val);
        double totalflux = val.sumElements();
        return totalflux * gain * (dx*dx);
    }

    // Now the more complex case: real space via FT from k space.
    // Will enforce image size is power of 2 or 3x2^n.
    // Aliasing will be handled by folding the k values before transforming
    // And enforce no image folding
    template <typename T>
    double SBProfile::fourierDraw(ImageView<T> I, double gain, double wmult) const 
    {
        dbg<<"Start fourierDraw"<<std::endl;
        double dx = I.getScale();
        Bounds<int> imgBounds; // Bounds for output image
        dbg<<"  maxK() = "<<maxK()<<" dx "<<dx<<std::endl;
        dbg<<"  stepK() = "<<stepK()<<std::endl;
        dbg<<"  image bounds = "<<I.getBounds()<<std::endl;
        dbg<<"  image scale = "<<I.getScale()<<std::endl;
        dbg<<"  wmult = "<<wmult<<std::endl;

        int Nnofold = getGoodImageSize(dx,wmult);
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

        // Move the output image to be centered near zero
        I.setCenter(0,0);
        double dk = 2.*M_PI/(NFT*dx);
        dbg << 
            " After adjustments: dx " << dx << " dk " << dk << 
            " maxK " << dk*NFT/2 << std::endl;
        assert(dk <= stepK());
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
        fout << tmv::EigenIO() << (mxt*dx*dx) << std::endl;
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
                double temp = xt->xval(x,y) / gain;
                I(x,y) += T(temp);
                sum += temp;
            }
        }

        I.setScale(dx);

        return sum * gain * (dx*dx);
    }

    template <typename T>
    void SBProfile::drawK(ImageView<T> Re, ImageView<T> Im, double gain, double wmult) const 
    {
        if (isAnalyticK()) 
            plainDrawK(Re, Im, gain);   // calculate in k space
        else               
            fourierDrawK(Re, Im, gain, wmult); // calculate via FT from real space
    }

    template <typename T>
    void SBProfile::plainDrawK(ImageView<T> Re, ImageView<T> Im, double gain) const 
    {
        // Make sure input images match or are both null
        assert(Re.getScale() == Im.getScale());
        assert(Re.getBounds() == Im.getBounds());

        double dk = Re.getScale();
        dbg<<"Start plainDrawK: dk = "<<dk<<std::endl;

        // recenter an existing image, to be consistent with fourierDrawK:
        Re.setCenter(0,0);
        Im.setCenter(0,0);

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
        _pimpl->fillKValue(val.view(),xmin*dk,dk,-xmin,ymin*dk,dk,-ymin);
        dbg<<"F(k=0) = "<<val(-xmin,-ymin)<<std::endl;

        if (gain != 1.) val /= gain;

        tmv::MatrixView<T> mRe(Re.getData(),m,n,1,Re.getStride(),tmv::NonConj);
        tmv::MatrixView<T> mIm(Im.getData(),m,n,1,Im.getStride(),tmv::NonConj);
        addMatrix(mRe,val.realPart());
        addMatrix(mIm,val.imagPart());
    }

    // Build K domain by transform from X domain.  This is likely
    // to be a rare event but what the heck.  Enforce no "aliasing"
    // by oversampling and extending x domain if needed.  Force
    // power of 2 for transform
    //
    // Note: There are no unit tests of this, since all profiles have isAnalyticK() == true.
    //       So drawK never sends anything this way.
    template <typename T>
    void SBProfile::fourierDrawK(ImageView<T> Re, ImageView<T> Im, double gain, double wmult) const 
    {
        // Make sure input images match or are both null
        assert(Re.getBounds() == Im.getBounds());
        assert(Re.getScale() == Im.getScale());

        double dk = Re.getScale();

        // Do we need to oversample in k to avoid folding from real space?
        // Note a little room for numerical slop before triggering oversampling:
        int oversamp = int( std::ceil(dk/stepK() - 0.0001));

        // Now decide how big the FT must be to avoid folding
        double kRange = 2*maxK()*wmult;
        // Some slop to keep from getting extra pixels due to roundoff errors in calculations.
        int Nnofold = int(std::ceil(oversamp*kRange / dk -0.0001));
        dbg<<"Nnofold = "<<Nnofold<<std::endl;

        // And if there is a target image size, we must make something big enough to cover
        // the target image size:
        int xSize, ySize;
        xSize = Re.getXMax()-Re.getXMin()+1;
        ySize = Re.getYMax()-Re.getYMin()+1;
        if (xSize * oversamp > Nnofold) Nnofold = xSize*oversamp;
        if (ySize * oversamp > Nnofold) Nnofold = ySize*oversamp;
        kRange = Nnofold * dk / oversamp;

        // Round up to a power of 2 to get required FFT size
        // Round up to a good size for making FFTs:
        int NFT = goodFFTSize(Nnofold);
        NFT = std::max(NFT,_pimpl->gsparams->minimum_fft_size);
        dbg << " After adjustments: Nnofold " << Nnofold << " NFT " << NFT << std::endl;
        if (NFT > _pimpl->gsparams->maximum_fft_size)
            FormatAndThrow<SBError>() << 
                "fourierDrawK() requires an FFT that is too large, " << NFT;

        // Move the output image to be centered near zero
        Re.setCenter(0,0);
        Im.setCenter(0,0);

        double dx = 2.*M_PI*oversamp/(NFT*dk);
        XTable xt(NFT,dx);
        assert(_pimpl.get());
        _pimpl->fillXGrid(xt);
        boost::shared_ptr<KTable> ktmp = xt.transform();

        int Nkt = ktmp->getN();
        Bounds<int> kb(-Nkt/2, Nkt/2-1, -Nkt/2, Nkt/2-1);
        if (Re.getYMin() < kb.getYMin()
            || Re.getYMax()*oversamp > kb.getYMax()
            || Re.getXMin()*oversamp < kb.getXMin()
            || Re.getXMax()*oversamp > kb.getXMax()) {
            dbg << "Bounds error!! oversamp is " << oversamp
                << " target image bounds " << Re.getBounds()
                << " and FFT range " << kb << std::endl;
            throw SBError("fourierDrawK() FT bounds do not cover target image");
        }

        for (int y = Re.getYMin(); y <= Re.getYMax(); y++) {
            for (int x = Re.getXMin(); x <= Re.getXMax(); x++) {
                std::complex<double> c = ktmp->kval(x*oversamp,y*oversamp) / gain;
                Re(x,y) = c.real();
                Im(x,y) = c.imag();
            }
        }

        Re.setScale(dk);
        Im.setScale(dk);
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
                         double x0, double dx, double y0, double dy)
        { prof.fillKValue(q,x0,dx,0,y0,dy,0); }
    };

    // The code is basically the same for X or K.
    template <class Prof, typename T>
    static void FillQuadrant(const Prof& prof, tmv::MatrixView<T> val,
                             double x0, double dx, int nx1, double y0, double dy, int ny1)
    {
        dbg<<"Start FillQuadrant\n";
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
                QuadrantHelper<T>::fill(prof,*q,0.,dx,0.,dy);
                ur_done = true;
                // Also do the rest of the ix=0 row and iy=0 col
                val.row(nx1,0,ny1).reverse() = q->row(0,1,ny1+1);
                val.col(ny1,0,nx1).reverse() = q->col(0,1,nx1+1);
            } else {
                // Lower right is the big quadrant
                xdbg<<"Use Lower right (nx2,ny1)"<<std::endl;
                q.reset(new tmv::MatrixView<T>(val.subMatrix(nx1,nx,ny1,-1,1,-1)));
                QuadrantHelper<T>::fill(prof,val.subMatrix(nx1,nx,0,ny1+1),0.,dx,y0,dy);
                lr_done = true;
                val.row(nx1,ny1+1,ny) = q->row(0,1,ny2+1);
                val.col(ny1,0,nx1).reverse() = q->row(0,1,nx1+1);
            }
        } else {
            if (ny2 >= ny1) {
                // Upper left is the big quadrant
                xdbg<<"Use Upper left (nx1,ny2)"<<std::endl;
                q.reset(new tmv::MatrixView<T>(val.subMatrix(nx1,-1,ny1,ny,-1,1)));
                QuadrantHelper<T>::fill(prof,val.subMatrix(0,nx1+1,ny1,ny),x0,dx,0.,dy);
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
                                                      double x0, double dx, int nx1,
                                                      double y0, double dy, int ny1) const
    {
        // Guard against infinite loop.
        assert(nx1 != 0 || ny1 != 0);
        FillQuadrant(*this,val,x0,dx,nx1,y0,dy,ny1);
    }

    template <class T>
    double SBProfile::drawShoot(
        ImageView<T> img, double N, UniformDeviate u, double gain, double max_extra_noise,
        bool poisson_flux, bool add_to_image) const 
    {
        // If N = 0, this routine will try to end up with an image with the number of real 
        // photons = flux that has the corresponding Poisson noise. For profiles that are 
        // positive definite, then N = flux. Easy.
        //
        // However, some profiles shoot some of their photons with negative flux. This means that 
        // we need a few more photons to get the right S/N = sqrt(flux). Take eta to be the 
        // fraction of shot photons that have negative flux.
        //
        // S^2 = (N+ - N-)^2 = (N+ + N- - 2N-)^2 = (Ntot - 2N-)^2 = Ntot^2(1 - 2 eta)^2
        // N^2 = Var(S) = (N+ + N-) = Ntot
        //
        // So flux = (S/N)^2 = Ntot (1-2eta)^2
        // Ntot = flux / (1-2eta)^2
        //
        // However, if each photon has a flux of 1, then S = (1-2eta) Ntot = flux / (1-2eta).
        // So in fact, each photon needs to carry a flux of g = 1-2eta to get the right 
        // total flux.
        //
        // That's all the easy case. The trickier case is when we are sky-background dominated.
        // Then we can usually get away with fewer shot photons than the above.  In particular,
        // if the noise from the photon shooting is much less than the sky noise, then we can 
        // use fewer shot photons and essentially have each photon have a flux > 1. This is ok 
        // as long as the additional noise due to this approximation is "much less than" the 
        // noise we'll be adding to the image for the sky noise.
        //
        // Let's still have Ntot photons, but now each with a flux of g. And let's look at the 
        // noise we get in the brightest pixel that has a nominal total flux of fmax.
        //
        // The number of photons hitting this pixel will be fmax/flux * Ntot.
        // The variance of this number is the same thing (Poisson counting). 
        // So the noise in that pixel is:
        //
        // N^2 = fmax/flux * Ntot * g^2
        //
        // And the signal in that pixel will be:
        //
        // S = fmax/flux * (N+ - N-) * g which has to equal fmax, so
        // g = flux / Ntot(1-2eta)
        // N^2 = fmax/Ntot * flux / (1-2eta)^2
        //
        // As expected, we see that lowering Ntot will increase the noise in that (and every 
        // other) pixel.
        // The input max_extra_noise parameter is the maximum value of spurious noise we want 
        // to allow.
        //
        // So setting N^2 = max_extra_noise, we get
        //
        // Ntot = fmax * flux / (1-2eta)^2 / max_extra_noise
        //
        // One wrinkle about this calculation is that we don't know fmax a priori.
        // So we start with a plausible number of photons to get going.  Then we keep adding 
        // more photons until we either hit N = flux / (1-2eta)^2 or the noise in the brightest
        // pixel is < max_extra_noise.
        //
        // We also make the assumption that the pixel to look at for fmax is at the centroid.
        //
        // Returns the total flux placed inside the image bounds by photon shooting.
        // 

        dbg<<"Start drawShoot.\n";
        dbg<<"N = "<<N<<std::endl;
        dbg<<"gain = "<<gain<<std::endl;
        dbg<<"max_extra_noise = "<<max_extra_noise<<std::endl;
        dbg<<"poisson = "<<poisson_flux<<std::endl;

        // Don't do more than this at a time to keep the  memory usage reasonable.
        const int maxN = 100000; 

        double flux = getFlux();
        dbg<<"flux = "<<flux<<std::endl;
        double posflux = getPositiveFlux();
        double negflux = getNegativeFlux();
        double eta = negflux / (posflux + negflux);
        dbg<<"N+ = "<<posflux<<", N- = "<<negflux<<" -> eta = "<<eta<<std::endl;
        double eta_factor = 1.-2.*eta; // This is also the amount to scale each photon.
        double mod_flux = flux/(eta_factor*eta_factor);
        dbg<<"mod_flux = "<<mod_flux<<std::endl;

        // Use this for the factor by which to scale photon arrays.
        // Also need to scale flux by gain = photons/ADU so we add ADU to the image.
        double flux_scaling = eta_factor/gain;

        // If requested, let the target flux value vary as a Poisson deviate
        if (poisson_flux) {
            // If we have both positive and negative photons, then the mix of these
            // already gives us some variation in the flux value from the variance
            // of how many are positive and how many are negative.
            // The number of negative photons varies as a binomial distribution.
            // <F-> = eta * N * flux_scaling
            // <F+> = (1-eta) * N * flux_scaling
            // <F+ - F-> = (1-2eta) * N * flux_scaling = flux
            // Var(F-) = eta * (1-eta) * N * flux_scaling^2
            // F+ = N * flux_scaling - F- is not an independent variable, so 
            // Var(F+ - F-) = Var(N*flux_scaling - 2*F-)
            //              = 4 * Var(F-) 
            //              = 4 * eta * (1-eta) * N * flux_scaling^2
            //              = 4 * eta * (1-eta) * flux
            // We want the variance to be equal to flux, so we need an extra:
            // delta Var = (1 - 4*eta + 4*eta^2) * flux
            //           = (1-2eta)^2 * flux
            double mean = eta_factor*eta_factor * flux;
            PoissonDeviate pd(u, mean);
            double pd_val = pd() - mean + flux;
            dbg<<"Poisson flux = "<<pd_val<<", c.f. flux = "<<flux<<std::endl;
            double ratio = pd_val / flux;
            flux_scaling *= ratio;
            mod_flux *= ratio;
            dbg<<"flux_scaling => "<<flux_scaling<<std::endl;
            dbg<<"mod_flux => "<<mod_flux<<std::endl;
        }

        if (N == 0.) N = mod_flux;
        double origN = N;

        // If not adding to the current image, zero it out:
        if (!add_to_image) img.setZero();

        // Center the image at 0,0:
        img.setCenter(0,0);
        dbg<<"On input, image has central value = "<<img(0,0)<<std::endl;

        // Store the PhotonArrays to be added here rather than add them as we go,
        // since we might need to rescale them all before adding.
        // We only use this if max_extra_noise > 0 and add_to_image = true.
        std::vector<boost::shared_ptr<PhotonArray> > arrays;

        // total flux falling inside image bounds, this will be returned on exit.
        double added_flux = 0.; 
#ifdef DEBUGLOGGING
        double realized_flux = 0.;
        double positive_flux = 0.;
        double negative_flux = 0.;
#endif

        // If we're automatically figuring out N based on max_extra_noise, start with 100 photons
        // Otherwise we'll do a maximum of maxN at a time until we go through all N.
        int thisN = max_extra_noise > 0. ? 100 : maxN;
        Position<double> cen = centroid();
        Bounds<double> b(cen);
        b.addBorder(0.5);
        dbg<<"Bounds for fmax = "<<b<<std::endl;
        T raw_fmax = 0.;
        int fmax_count = 0;
        while (true) {
            // We break out of the loop when either N drops to 0 (if max_extra_noise = 0) or 
            // we find that the max pixel has a noise level < max_extra_noise

            if (thisN > maxN) thisN = maxN;
            // NB: don't need floor, since rhs is positive, so floor is superfluous.
            if (thisN > N) thisN = int(N+0.5);

            xdbg<<"shoot "<<thisN<<std::endl;
            assert(_pimpl.get());
            boost::shared_ptr<PhotonArray> pa = _pimpl->shoot(thisN, u);
            xdbg<<"pa.flux = "<<pa->getTotalFlux()<<std::endl;
            xdbg<<"scale flux by "<<(flux_scaling*thisN/origN)<<std::endl;
            pa->scaleFlux(flux_scaling * thisN / origN);
            xdbg<<"pa.flux => "<<pa->getTotalFlux()<<std::endl;

            if (add_to_image && max_extra_noise > 0.) {
                // Then we might need to rescale these, so store it and deal with it later.
                arrays.push_back(pa);
            } else {
                // Otherwise, we can go ahead and apply it here.
                added_flux += pa->addTo(img);
#ifdef DEBUGLOGGING
                realized_flux += pa->getTotalFlux();
                for(int i=0; i<pa->size(); ++i) {
                    double f = pa->getFlux(i);
                    if (f >= 0.) positive_flux += f;
                    else negative_flux += -f;
                }
#endif
            }
         
            N -= thisN;
            xdbg<<"N -> "<<N<<std::endl;

            // This is always a reason to break out.
            if (N < 1.) break;

            if (max_extra_noise > 0.) {
                xdbg<<"Check the noise level\n";
                // First need to find what the current fmax is.
                // (Only need to update based on the latest pa.)

                for(int i=0; i<pa->size(); ++i) {
                    if (b.includes(pa->getX(i),pa->getY(i))) {
                        ++fmax_count;
                        raw_fmax += pa->getFlux(i);
                    }
                }
                xdbg<<"fmax_count = "<<fmax_count<<std::endl;
                xdbg<<"raw_fmax = "<<raw_fmax<<std::endl;

                // Make sure we've got at least 25 photons for our fmax estimate and that
                // the fmax value is positive.
                // Otherwise keep the same initial value of thisN = 100 and try again.
                if (fmax_count < 25 || raw_fmax < 0.) continue;  

                double fmax = raw_fmax * origN / (origN-N);
                xdbg<<"fmax = "<<fmax<<std::endl;
                // Estimate a good value of Ntot based on what we know now
                // Ntot = fmax * flux / (1-2eta)^2 / max_extra_noise
                double Ntot = fmax * mod_flux / max_extra_noise;
                xdbg<<"Calculated Ntot = "<<Ntot<<std::endl;
                // So far we've done (origN-N)
                // Set thisN to do the rest on the next pass.
                Ntot -= (origN-N);
                if (Ntot > maxN) thisN = maxN; // Make sure we don't overflow thisN.
                else thisN = int(Ntot);
                xdbg<<"Next value of thisN = "<<thisN<<std::endl;
                // If we've already done enough, break out of the loop.
                if (thisN <= 0) break;
            }
        }

        if (N > 0.1) {
            // If we didn't shoot all the original number of photons, then our flux isn't right.
            // Need to rescale the arrays by factor of origN / (origN-N)
            dbg<<"Flux scalings were set according to origN = "<<origN<<std::endl;
            dbg<<"But only shot N = "<<origN-N<<std::endl;
            double factor = origN / (origN-N);
            dbg<<"Rescale by factor = "<<factor<<std::endl;

            if (arrays.size() > 0) {
                // If using arrays, rescale the flux in each
                for (size_t k=0; k<arrays.size(); ++k) arrays[k]->scaleFlux(factor);
            } else {
                // Otherwise, rescale the image itself
                assert(!add_to_image);
                img *= T(factor);
                // Also fix the added_flux value
                added_flux *= factor;
#ifdef DEBUGLOGGING
                realized_flux *= factor;
                positive_flux *= factor;
                negative_flux *= factor;
#endif
            }
        }

        if (arrays.size() > 0) {
            // Now we can go ahead and add all the arrays to the image:
            assert(added_flux == 0.);
            for (size_t k=0; k<arrays.size(); ++k) {
                PhotonArray* pa = arrays[k].get();
                added_flux += pa->addTo(img);
#ifdef DEBUGLOGGING
                realized_flux += pa->getTotalFlux();
                for(int i=0; i<pa->size(); ++i) {
                    double f = pa->getFlux(i);
                    if (f >= 0.) positive_flux += f;
                    else negative_flux += -f;
                }
#endif
            }
        }

#ifdef DEBUGLOGGING
        dbg<<"Done drawShoot.  Realized flux = "<<realized_flux*gain<<std::endl;
        dbg<<"c.f. target flux = "<<flux<<std::endl;
        dbg<<"Now image has central value = "<<img(0,0)*gain<<std::endl;
        dbg<<"Realized positive flux = "<<positive_flux*gain<<std::endl;
        dbg<<"Realized negative flux = "<<negative_flux*gain<<std::endl;
        dbg<<"Actual eta = "<<negative_flux / (positive_flux + negative_flux)<<std::endl;
        dbg<<"c.f. predicted eta = "<<eta<<std::endl;
#endif
        dbg<<"Added flux (falling within image bounds) = "<<added_flux*gain<<std::endl;

        // The "added_flux" above really counts ADU's.  So multiply by gain to get the 
        // actual flux in photons that was added.
        return added_flux * gain;
    }

    // instantiate template functions for expected image types
    template double SBProfile::SBProfileImpl::fillXImage(
        ImageView<float>& img, double gain) const;
    template double SBProfile::SBProfileImpl::fillXImage(
        ImageView<double>& img, double gain) const;

    template double SBProfile::drawShoot(
        ImageView<float> image, double N, UniformDeviate ud, double gain,
        double max_extra_noise, bool poisson_flux, bool add_to_image) const;
    template double SBProfile::drawShoot(
        ImageView<double> image, double N, UniformDeviate ud, double gain,
        double max_extra_noise, bool poisson_flux, bool add_to_image) const;

    template double SBProfile::draw(ImageView<float> img, double gain, double wmult) const;
    template double SBProfile::draw(ImageView<double> img, double gain, double wmult) const;

    template double SBProfile::plainDraw(ImageView<float> I, double gain) const;
    template double SBProfile::plainDraw(ImageView<double> I, double gain) const;

    template double SBProfile::fourierDraw(ImageView<float> I, double gain, double wmult) const;
    template double SBProfile::fourierDraw(ImageView<double> I, double gain, double wmult) const;

    template void SBProfile::drawK(
        ImageView<float> Re, ImageView<float> Im, double gain, double wmult) const;
    template void SBProfile::drawK(
        ImageView<double> Re, ImageView<double> Im, double gain, double wmult) const;

    template void SBProfile::plainDrawK(
        ImageView<float> Re, ImageView<float> Im, double gain) const;
    template void SBProfile::plainDrawK(
        ImageView<double> Re, ImageView<double> Im, double gain) const;

    template void SBProfile::fourierDrawK(
        ImageView<float> Re, ImageView<float> Im, double gain, double wmult) const;
    template void SBProfile::fourierDrawK(
        ImageView<double> Re, ImageView<double> Im, double gain, double wmult) const;

}
