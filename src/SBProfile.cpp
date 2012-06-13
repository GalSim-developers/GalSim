//
// Functions for the Surface Brightness Profile Class
//

//#define DEBUGLOGGING

#include "SBProfile.h"
#include "integ/Int.h"
#include "TMV.h"
#include "Solve.h"
#include "integ/Int.h"

#ifdef DEBUGLOGGING
#include <fstream>
std::ostream* dbgout = new std::ofstream("debug.out");
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

#include <numeric>

namespace galsim {

    // ????? Change treatement of aliased images to simply add in the aliased
    // FT components instead of doing a larger FT and then subsampling!
    // ??? Make a formula for asymptotic high-k SBSersic::kValue ??


    //
    // Virtual methods of Base Class "SBProfile"
    //

    void SBProfile::scaleFlux(double fluxRatio)
    { 
        SBDistort d(*this,1.,0.,0.,1.,Position<double>(0.,0.),fluxRatio); 
        _pimpl = d._pimpl;
    }

    void SBProfile::setFlux(double flux)
    { 
        SBDistort d(*this,1.,0.,0.,1.,Position<double>(0.,0.),flux/getFlux());
        _pimpl = d._pimpl;
    }

    void SBProfile::applyDistortion(const Ellipse& e)
    {
        SBDistort d(*this,e);
        _pimpl = d._pimpl;
    }

    void SBProfile::applyShear(double g1, double g2)
    {
        Shear s;
        s.setG1G2(g1,g2);
        SBDistort d(*this,Ellipse(s.getE1(),s.getE2()));
        _pimpl = d._pimpl;
    }

    void SBProfile::applyRotation(const Angle& theta)
    {
        SBDistort d(*this,
                    std::cos(theta.rad()),-std::sin(theta.rad()),
                    std::sin(theta.rad()),std::cos(theta.rad()));
        _pimpl = d._pimpl;
    }

    void SBProfile::applyShift(double dx, double dy)
    { 
        SBDistort d(*this,1.,0.,0.,1., Position<double>(dx,dy));
        _pimpl = d._pimpl;
    }

    //
    // Common methods of Base Class "SBProfile"
    //

    ImageView<float> SBProfile::draw(double dx, int wmult) const 
    {
        dbg<<"Start draw that returns ImageView"<<std::endl;
        Image<float> img;
        draw(img, dx, wmult);
        return img.view();
    }

    template <typename T>
    double SBProfile::draw(ImageView<T>& img, double dx, int wmult) const 
    {
        dbg<<"Start draw ImageView"<<std::endl;
        if (isAnalyticX())
            return plainDraw(img, dx, wmult);
        else
            return fourierDraw(img, dx, wmult);
    }

    template <typename T>
    double SBProfile::draw(Image<T>& img, double dx, int wmult) const 
    {
        dbg<<"Start draw Image"<<std::endl;
        if (isAnalyticX())
            return plainDraw(img, dx, wmult);
        else
            return fourierDraw(img, dx, wmult);
    }

    // First is a simple case wherein we have a formula for x values:
    template <typename T>
    double SBProfile::plainDraw(ImageView<T>& I, double dx, int wmult) const 
    {
        dbg<<"Start plainDraw ImageView"<<std::endl;
        // Determine desired dx:
        dbg<<"maxK = "<<maxK()<<std::endl;
        if (dx<=0.) dx = M_PI / maxK();
        dbg<<"dx = "<<dx<<std::endl;
        // recenter an existing image, to be consistent with fourierDraw:
        int xSize = I.getXMax()-I.getXMin()+1, ySize = I.getYMax()-I.getYMin()+1;
        dbg<<"xSize = "<<xSize<<std::endl;
        I.setOrigin(-xSize/2, -ySize/2);

        assert(_pimpl.get());
        return _pimpl->fillXImage(I, dx);
    }

    template <typename T>
    double SBProfile::plainDraw(Image<T>& I, double dx, int wmult) const 
    {
        dbg<<"Start plainDraw Image"<<std::endl;
        // Determine desired dx:
        dbg<<"maxK = "<<maxK()<<std::endl;
        if (dx<=0.) dx = M_PI / maxK();
        dbg<<"dx = "<<dx<<std::endl;
        if (!I.getBounds().isDefined()) {
            if (wmult<1) throw SBError("Requested wmult<1 in plainDraw()");
            // Need to choose an image size
            int N = int(std::ceil(2*M_PI/(dx*stepK())));
            dbg<<"N = "<<N<<std::endl;

            // Round up to an even value
            N = 2*( (N+1)/2);
            N *= wmult; // make even bigger if desired
            dbg<<"N => "<<N<<std::endl;
            Bounds<int> imgsize(-N/2, N/2-1, -N/2, N/2-1);
            dbg<<"imgsize => "<<imgsize<<std::endl;
            I.resize(imgsize);
        } else {
            // recenter an existing image, to be consistent with fourierDraw:
            int xSize = I.getXMax()-I.getXMin()+1, ySize = I.getYMax()-I.getYMin()+1;
            dbg<<"xSize = "<<xSize<<std::endl;
            I.setOrigin(-xSize/2, -ySize/2);
        }

        ImageView<T> Iv = I.view();
        assert(_pimpl.get());
        double ret = _pimpl->fillXImage(Iv, dx);
        I.setScale(Iv.getScale());
        dbg<<"scale => "<<I.getScale()<<std::endl;
        return ret;
    }
 
    template <typename T>
    double SBProfile::SBProfileImpl::doFillXImage2(ImageView<T>& I, double dx) const 
    {
        xdbg<<"Start doFillXImage2"<<std::endl;
        double totalflux=0;
        for (int y = I.getYMin(); y <= I.getYMax(); y++) {
            int x = I.getXMin(); 
            typedef typename Image<T>::iterator ImIter;
            ImIter ee=I.rowEnd(y);
            for (ImIter it=I.rowBegin(y); it!=ee; ++it, ++x) {
                Position<double> p(x*dx,y*dx); // since x,y are pixel indices
                *it = xValue(p);
                totalflux += *it;
            } 
        }
        I.setScale(dx);
        xdbg<<"scale => "<<I.getScale()<<std::endl;
        return totalflux * (dx*dx);
    }

    // Now the more complex case: real space via FT from k space.
    // Will enforce image size is power of 2 or 3x2^n.
    // Aliasing will be handled by folding the k values before transforming
    // And enforce no image folding
    template <typename T>
    double SBProfile::fourierDraw(ImageView<T>& I, double dx, int wmult) const 
    {
        Bounds<int> imgBounds; // Bounds for output image
        if (wmult<1) throw SBError("Requested wmult<1 in fourierDraw()");
        // First choose desired dx if we were not given one:
        if (dx<=0.) {
            // Choose for ourselves:
            dx = M_PI / maxK();
        }

        dbg << " maxK() " << maxK() << " dx " << dx << std::endl;

        // Now decide how big the FT must be to avoid folding:
        double xRange = 2*M_PI*wmult / stepK();
        // Some slop to keep from getting extra pixels due to roundoff errors in calculations.
        int Nnofold = int(std::ceil(xRange / dx -0.0001));
        dbg << " stepK() " << stepK() << " Nnofold " << Nnofold << std::endl;

        // W must make something big enough to cover the target image size:
        int xSize, ySize;
        xSize = I.getXMax()-I.getXMin()+1;
        ySize = I.getYMax()-I.getYMin()+1;
        if (xSize  > Nnofold) Nnofold = xSize;
        if (ySize  > Nnofold) Nnofold = ySize;
        xRange = Nnofold * dx;

        // Round up to a good size for making FFTs:
        int NFT = goodFFTSize(Nnofold);
        NFT = std::max(NFT,sbp::minimum_fft_size);
        dbg << " After adjustments: Nnofold " << Nnofold << " NFT " << NFT << std::endl;
        if (NFT > sbp::maximum_fft_size)
            FormatAndThrow<SBError>() << 
                "fourierDraw() requires an FFT that is too large, " << NFT;

        // Move the output image to be centered near zero
        I.setOrigin(-xSize/2, -ySize/2);
        double dk = 2.*M_PI/(NFT*dx);
        dbg << 
            " After adjustments: dx " << dx << " dk " << dk << 
            " maxK " << dk*NFT/2 << std::endl;
        assert(dk <= stepK());
        boost::shared_ptr<XTable> xtmp;
        if (NFT*dk/2 > maxK()) {
            dbg<<"NFT*dk/2 = "<<NFT*dk/2<<" > maxK() = "<<maxK()<<std::endl;
            dbg<<"Use NFT = "<<NFT<<std::endl;
            // No aliasing: build KTable and transform
            KTable kt(NFT,dk);
            assert(_pimpl.get());
            _pimpl->fillKGrid(kt); 
            xtmp = kt.transform();
        } else {
            dbg<<"NFT*dk/2 = "<<NFT*dk/2<<" <= maxK() = "<<maxK()<<std::endl;
            // There will be aliasing.  Construct a KTable out to maxK() and
            // then wrap it
            int Nk = int(std::ceil(maxK()/dk)) * 2;
            dbg<<"Use Nk = "<<Nk<<std::endl;
            KTable kt(Nk, dk);
            assert(_pimpl.get());
            _pimpl->fillKGrid(kt);
            xtmp = kt.wrap(NFT)->transform();
        }
        int Nxt = xtmp->getN();
        dbg<<"Nxt = "<<Nxt<<std::endl;
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
        for (int y = I.getYMin(); y <= I.getYMax(); y++)
            for (int x = I.getXMin(); x <= I.getXMax(); x++) {
                I(x,y) = xtmp->xval(x,y);
                sum += I(x,y);
            }

        I.setScale(dx);

        return sum*dx*dx;;
    }

    // TODO: I'd like to try to separate out the resize operation.  
    // Then this function can just take an ImageView<T>& argument, not also an Image<T>.
    // Similar to what plainDraw does by passing the bulk of the work to fillXImage.
    // In fact, if we could have a single resizer, than that could be called from draw()
    // and both plainDraw and fourierDraw could drop to only having the ImageView argument.
    template <typename T>
    double SBProfile::fourierDraw(Image<T>& I, double dx, int wmult) const 
    {
        Bounds<int> imgBounds; // Bounds for output image
        bool sizeIsFree = !I.getBounds().isDefined();
        if (wmult<1) throw SBError("Requested wmult<1 in fourierDraw()");
        // First choose desired dx if we were not given one:
        if (dx<=0.) {
            // Choose for ourselves:
            dx = M_PI / maxK();
        }

        dbg << " maxK() " << maxK() << " dx " << dx << std::endl;

        // Now decide how big the FT must be to avoid folding:
        double xRange = 2*M_PI*wmult / stepK();
        // Some slop to keep from getting extra pixels due to roundoff errors in calculations.
        int Nnofold = int(std::ceil(xRange / dx -0.0001));
        dbg << " stepK() " << stepK() << " Nnofold " << Nnofold << std::endl;

        // And if there is a target image size, we must make something big enough to cover
        // the target image size:
        if (!sizeIsFree) {
            int xSize, ySize;
            xSize = I.getXMax()-I.getXMin()+1;
            ySize = I.getYMax()-I.getYMin()+1;
            if (xSize  > Nnofold) Nnofold = xSize;
            if (ySize  > Nnofold) Nnofold = ySize;
            xRange = Nnofold * dx;
        }

        // Round up to a good size for making FFTs:
        int NFT = goodFFTSize(Nnofold);
        NFT = std::max(NFT,sbp::minimum_fft_size);
        dbg << " After adjustments: Nnofold " << Nnofold << " NFT " << NFT << std::endl;
        if (NFT > sbp::maximum_fft_size)
            FormatAndThrow<SBError>() << 
                "fourierDraw() requires an FFT that is too large, " << NFT;

        // If we are free to set up output image, make it size of FFT
        if (sizeIsFree) {
            int Nimg = NFT;
            // Reduce to make even
            Nimg = 2*(Nimg/2);
            imgBounds = Bounds<int>(-Nimg/2, Nimg/2-1, -Nimg/2, Nimg/2-1);
            I.resize(imgBounds);
        } else {
            // Going to move the output image to be centered near zero
            int xSize, ySize;
            xSize = I.getXMax()-I.getXMin()+1;
            ySize = I.getYMax()-I.getYMin()+1;
            I.setOrigin(-xSize/2, -ySize/2);
        }
        double dk = 2.*M_PI/(NFT*dx);
        dbg << 
            " After adjustments: dx " << dx << " dk " << dk << 
            " maxK " << dk*NFT/2 << std::endl;
        assert(dk <= stepK());
        boost::shared_ptr<XTable> xtmp;
        if (NFT*dk/2 > maxK()) {
            dbg<<"NFT*dk/2 = "<<NFT*dk/2<<" > maxK() = "<<maxK()<<std::endl;
            dbg<<"Use NFT = "<<NFT<<std::endl;
            // No aliasing: build KTable and transform
            KTable kt(NFT,dk);
            assert(_pimpl.get());
            _pimpl->fillKGrid(kt); 
            xtmp = kt.transform();
        } else {
            dbg<<"NFT*dk/2 = "<<NFT*dk/2<<" <= maxK() = "<<maxK()<<std::endl;
            // There will be aliasing.  Construct a KTable out to maxK() and
            // then wrap it
            int Nk = int(std::ceil(maxK()/dk)) * 2;
            dbg<<"Use Nk = "<<Nk<<std::endl;
            KTable kt(Nk, dk);
            assert(_pimpl.get());
            _pimpl->fillKGrid(kt);
            xtmp = kt.wrap(NFT)->transform();
        }
        int Nxt = xtmp->getN();
        dbg<<"Nxt = "<<Nxt<<std::endl;
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
        for (int y = I.getYMin(); y <= I.getYMax(); y++)
            for (int x = I.getXMin(); x <= I.getXMax(); x++) {
                I(x,y) = xtmp->xval(x,y);
                sum += I(x,y);
            }

        I.setScale(dx);

        return sum*dx*dx;;
    }

    template <typename T>
    void SBProfile::drawK(ImageView<T>& Re, ImageView<T>& Im, double dk, int wmult) const 
    {
        if (isAnalyticK()) 
            plainDrawK(Re, Im, dk, wmult);   // calculate in k space
        else               
            fourierDrawK(Re, Im, dk, wmult); // calculate via FT from real space
    }

    template <typename T>
    void SBProfile::drawK(Image<T>& Re, Image<T>& Im, double dk, int wmult) const 
    {
        if (isAnalyticK()) 
            plainDrawK(Re, Im, dk, wmult);   // calculate in k space
        else               
            fourierDrawK(Re, Im, dk, wmult); // calculate via FT from real space
    }

    template <typename T>
    void SBProfile::plainDrawK(ImageView<T>& Re, ImageView<T>& Im, double dk, int wmult) const 
    {
        // Make sure input images match or are both null
        assert(Re.getBounds() == Im.getBounds());
        if (dk<=0.) dk = stepK();

        // recenter an existing image, to be consistent with fourierDrawK:
        int xSize = Re.getXMax()-Re.getXMin()+1, ySize = Re.getYMax()-Re.getYMin()+1;
        Re.setOrigin(-xSize/2, -ySize/2);
        Im.setOrigin(-xSize/2, -ySize/2);

        // ??? Make this into a virtual function to allow pipelining?
        for (int y = Re.getYMin(); y <= Re.getYMax(); y++) {
            int x = Re.getXMin(); 
            typedef typename ImageView<T>::iterator ImIter;
            ImIter ee=Re.rowEnd(y);
            for (ImIter it=Re.rowBegin(y), it2=Im.rowBegin(y); it!=ee; ++it, ++it2, ++x) {
                Position<double> p(x*dk,y*dk); // since x,y are pixel indicies
                std::complex<double> c = this->kValue(p);  
                *it = c.real(); 
                *it2 = c.imag(); 
            } 
        }

        Re.setScale(dk);
        Im.setScale(dk);
    }

    template <typename T>
    void SBProfile::plainDrawK(Image<T>& Re, Image<T>& Im, double dk, int wmult) const 
    {
        // Make sure input images match or are both null
        assert(!(Re.getBounds().isDefined() || Im.getBounds().isDefined()) 
               || (Re.getBounds() == Im.getBounds()));
        if (dk<=0.) dk = stepK();

        if (!Re.getBounds().isDefined()) {
            if (wmult<1) throw SBError("Requested wmult<1 in plainDrawK()");
            // Need to choose an image size
            int N = int(std::ceil(2.*maxK()*wmult / dk));
            // Round up to an even value
            N = 2*( (N+1)/2);

            Bounds<int> imgsize(-N/2, N/2-1, -N/2, N/2-1);
            Re.resize(imgsize);
            Im.resize(imgsize);
        } else {
            // recenter an existing image, to be consistent with fourierDrawK:
            int xSize = Re.getXMax()-Re.getXMin()+1, ySize = Re.getYMax()-Re.getYMin()+1;
            Re.setOrigin(-xSize/2, -ySize/2);
            Im.setOrigin(-xSize/2, -ySize/2);
        }

        // ??? Make this into a virtual function to allow pipelining?
        for (int y = Re.getYMin(); y <= Re.getYMax(); y++) {
            int x = Re.getXMin(); 
            typedef typename ImageView<T>::iterator ImIter;
            ImIter ee=Re.rowEnd(y);
            for (ImIter it=Re.rowBegin(y), it2=Im.rowBegin(y); it!=ee; ++it, ++it2, ++x) {
                Position<double> p(x*dk,y*dk); // since x,y are pixel indicies
                std::complex<double> c = this->kValue(p);  
                *it = c.real(); 
                *it2 = c.imag(); 
            } 
        }

        Re.setScale(dk);
        Im.setScale(dk);
    }

    // Build K domain by transform from X domain.  This is likely
    // to be a rare event but what the heck.  Enforce no "aliasing"
    // by oversampling and extending x domain if needed.  Force
    // power of 2 for transform

    template <typename T>
    void SBProfile::fourierDrawK(ImageView<T>& Re, ImageView<T>& Im, double dk, int wmult) const 
    {
        assert(Re.getBounds() == Im.getBounds());

        int oversamp =1; // oversampling factor
        Bounds<int> imgBounds; // Bounds for output image
        if (wmult<1) throw SBError("Requested wmult<1 in fourierDrawK()");
        // First choose desired dx
        if (dk<=0.) {
            // Choose for ourselves:
            dk = stepK();
        } else {
            // We have a value we must produce.  Do we need to oversample in k
            // to avoid folding from real space?
            // Note a little room for numerical slop before triggering oversampling:
            oversamp = int( std::ceil(dk/stepK() - 0.0001));
        }

        // Now decide how big the FT must be to avoid folding
        double kRange = 2*maxK()*wmult;
        // Some slop to keep from getting extra pixels due to roundoff errors in calculations.
        int Nnofold = int(std::ceil(oversamp*kRange / dk -0.0001));

        // And if there is a target image size, we must make something big enough to cover
        // the target image size:
        int xSize, ySize;
        xSize = Re.getXMax()-Re.getXMin()+1;
        ySize = Re.getYMax()-Re.getYMin()+1;
        if (xSize * oversamp > Nnofold) Nnofold = xSize*oversamp;
        if (ySize * oversamp > Nnofold) Nnofold = ySize*oversamp;
        kRange = Nnofold * dk / oversamp;

        // Round up to a power of 2 to get required FFT size
        int NFT = sbp::minimum_fft_size;
        while (NFT < Nnofold && NFT<= sbp::maximum_fft_size) NFT *= 2;
        if (NFT > sbp::maximum_fft_size)
            throw SBError("fourierDrawK() requires an FFT that is too large");

        // Move the output image to be centered near zero
        Re.setOrigin(-xSize/2, -ySize/2);
        Im.setOrigin(-xSize/2, -ySize/2);

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

        for (int y = Re.getYMin(); y <= Re.getYMax(); y++)
            for (int x = Re.getXMin(); x <= Re.getXMax(); x++) {
                Re(x,y) = ktmp->kval(x*oversamp,y*oversamp).real();
                Im(x,y) = ktmp->kval(x*oversamp,y*oversamp).imag();
            }

        Re.setScale(dk);
        Im.setScale(dk);
    }

    template <typename T>
    void SBProfile::fourierDrawK(Image<T>& Re, Image<T>& Im, double dk, int wmult) const 
    {
        assert(!(Re.getBounds().isDefined() || Im.getBounds().isDefined()) 
               || (Re.getBounds() == Im.getBounds()));

        int oversamp =1; // oversampling factor
        Bounds<int> imgBounds; // Bounds for output image
        bool sizeIsFree = !Re.getBounds().isDefined();
        if (wmult<1) throw SBError("Requested wmult<1 in fourierDrawK()");
        bool canReduceDk=true;
        // First choose desired dx
        if (dk<=0.) {
            // Choose for ourselves:
            dk = stepK();
            canReduceDk = true;
        } else {
            // We have a value we must produce.  Do we need to oversample in k
            // to avoid folding from real space?
            // Note a little room for numerical slop before triggering oversampling:
            oversamp = int( std::ceil(dk/stepK() - 0.0001));
            canReduceDk = false; // Force output image to input dx.
        }

        // Now decide how big the FT must be to avoid folding
        double kRange = 2*maxK()*wmult;
        // Some slop to keep from getting extra pixels due to roundoff errors in calculations.
        int Nnofold = int(std::ceil(oversamp*kRange / dk -0.0001));

        // And if there is a target image size, we must make something big enough to cover
        // the target image size:
        if (!sizeIsFree) {
            int xSize, ySize;
            xSize = Re.getXMax()-Re.getXMin()+1;
            ySize = Re.getYMax()-Re.getYMin()+1;
            if (xSize * oversamp > Nnofold) Nnofold = xSize*oversamp;
            if (ySize * oversamp > Nnofold) Nnofold = ySize*oversamp;
            kRange = Nnofold * dk / oversamp;
            // If the input image *size* was specified but not the input *dk*, then
            // we will hold dk at the Nyquist scale:
            canReduceDk = false;
        }

        // Round up to a power of 2 to get required FFT size
        int NFT = sbp::minimum_fft_size;
        while (NFT < Nnofold && NFT<= sbp::maximum_fft_size) NFT *= 2;
        if (NFT > sbp::maximum_fft_size)
            throw SBError("fourierDrawK() requires an FFT that is too large");

        // If we are free to set up output image, make it size of FFT less oversampling
        if (sizeIsFree) {
            int Nimg = NFT / oversamp;
            // Reduce to make even
            Nimg = 2*(Nimg/2);
            imgBounds = Bounds<int>(-Nimg/2, Nimg/2-1, -Nimg/2, Nimg/2-1);
            Re.resize(imgBounds);
            Im.resize(imgBounds);
            // Reduce dk if 2^N made left room to do so.
            if (canReduceDk) {
                dk = kRange / Nimg; 
            }
        } else {
            // Going to move the output image to be centered near zero
            int xSize, ySize;
            xSize = Re.getXMax()-Re.getXMin()+1;
            ySize = Re.getYMax()-Re.getYMin()+1;
            Re.setOrigin(-xSize/2, -ySize/2);
            Im.setOrigin(-xSize/2, -ySize/2);
        }

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

        for (int y = Re.getYMin(); y <= Re.getYMax(); y++)
            for (int x = Re.getXMin(); x <= Re.getXMax(); x++) {
                Re(x,y) = ktmp->kval(x*oversamp,y*oversamp).real();
                Im(x,y) = ktmp->kval(x*oversamp,y*oversamp).imag();
            }

        Re.setScale(dk);
        Im.setScale(dk);
    }

    void SBProfile::SBProfileImpl::fillXGrid(XTable& xt) const 
    {
        int N = xt.getN();
        double dx = xt.getDx();
        for (int iy = -N/2; iy < N/2; iy++) {
            double y = iy*dx;
            for (int ix = -N/2; ix < N/2; ix++) {
                Position<double> x(ix*dx,y);
                xt.xSet(ix,iy,xValue(x));
            }
        }
    }

    void SBProfile::SBProfileImpl::fillKGrid(KTable& kt) const 
    {
        int N = kt.getN();
        double dk = kt.getDk();
#if 0
        // The simple version, saved for reference
        for (int iy = -N/2; iy < N/2; iy++) {
            // Only need ix>=0 because it's Hermitian:
            for (int ix = 0; ix <= N/2; ix++) {
                Position<double> k(ix*dk,iy*dk);
                kt.kSet(ix,iy,kValue(k));
            }
        }
#else
        // A faster version that pulls out all the if statements
        kt.clearCache();
        // First iy=0
        Position<double> k1(0.,0.);
        for (int ix = 0; ix <= N/2; ix++, k1.x += dk) kt.kSet2(ix,0,kValue(k1));

        // Then iy = 1..N/2-1
        k1.y = dk;
        Position<double> k2(0.,-dk);
        for (int iy = 1; iy < N/2; iy++, k1.y += dk, k2.y -= dk) {
            k1.x = k2.x = 0.;
            for (int ix = 0; ix <= N/2; ix++, k1.x += dk, k2.x += dk) {
                kt.kSet2(ix,iy,kValue(k1));
                kt.kSet2(ix,N-iy,kValue(k2));
            }
        }

        // Finally, iy = N/2
        k1.x = 0.;
        for (int ix = 0; ix <= N/2; ix++, k1.x += dk) kt.kSet2(ix,N/2,kValue(k1));
#endif
    }

    //
    // Methods for Derived Classes
    //

    void SBAdd::SBAddImpl::add(const SBProfile& rhs)
    {
        xdbg<<"Start SBAdd::add.  Adding item # "<<_plist.size()+1<<std::endl;
        // Add new summand(s) to the _plist:
        assert(rhs._pimpl.get());
        const SBAddImpl *sba = dynamic_cast<const SBAddImpl*>(rhs._pimpl.get());
        if (sba) {
            // If rhs is an SBAdd, copy its full list here
            _plist.insert(_plist.end(),sba->_plist.begin(),sba->_plist.end());
        } else {
            _plist.push_back(rhs);
        }
    }

    void SBAdd::SBAddImpl::initialize() 
    {
        _sumflux = _sumfx = _sumfy = 0.;
        _maxMaxK = _minStepK = 0.;
        _allAxisymmetric = _allAnalyticX = _allAnalyticK = true;
        _anyHardEdges = false;

        // Accumulate properties of all summands
        for(ConstIter it=_plist.begin(); it!=_plist.end(); ++it) {
            xdbg<<"SBAdd component has maxK, stepK = "<<
                it->maxK()<<" , "<<it->stepK()<<std::endl;
            _sumflux += it->getFlux();
            _sumfx += it->getFlux() * it->centroid().x;
            _sumfy += it->getFlux() * it->centroid().x;
            if ( it->maxK() > _maxMaxK) 
                _maxMaxK = it->maxK();
            if ( _minStepK<=0. || (it->stepK() < _minStepK) ) 
                _minStepK = it->stepK();
            _allAxisymmetric = _allAxisymmetric && it->isAxisymmetric();
            _anyHardEdges = _anyHardEdges || it->hasHardEdges();
            _allAnalyticX = _allAnalyticX && it->isAnalyticX();
            _allAnalyticK = _allAnalyticK && it->isAnalyticK();
        }
        xdbg<<"Net maxK, stepK = "<<_maxMaxK<<" , "<<_minStepK<<std::endl;
    }

    double SBAdd::SBAddImpl::xValue(const Position<double>& p) const 
    {
        ConstIter pptr = _plist.begin();
        assert(pptr != _plist.end());
        double xv = pptr->xValue(p);
        for (++pptr; pptr != _plist.end(); ++pptr)
            xv += pptr->xValue(p);
        return xv;
    } 

    std::complex<double> SBAdd::SBAddImpl::kValue(const Position<double>& k) const 
    {
        ConstIter pptr = _plist.begin();
        assert(pptr != _plist.end());
        std::complex<double> kv = pptr->kValue(k);
        for (++pptr; pptr != _plist.end(); ++pptr)
            kv += pptr->kValue(k);
        return kv;
    } 

    void SBAdd::SBAddImpl::fillKGrid(KTable& kt) const 
    {
        if (_plist.empty()) kt.clear();
        ConstIter pptr = _plist.begin();
        assert(pptr->_pimpl.get());
        pptr->_pimpl->fillKGrid(kt);
        if (++pptr != _plist.end()) {
            KTable k2(kt.getN(),kt.getDk());
            for ( ; pptr!= _plist.end(); ++pptr) {
                assert(pptr->_pimpl.get());
                pptr->_pimpl->fillKGrid(k2);
                kt.accumulate(k2);
            }
        }
    }

    void SBAdd::SBAddImpl::fillXGrid(XTable& xt) const 
    {
        if (_plist.empty()) xt.clear();
        ConstIter pptr = _plist.begin();
        assert(pptr->_pimpl.get());
        pptr->_pimpl->fillXGrid(xt);
        if (++pptr != _plist.end()) {
            XTable x2(xt.getN(),xt.getDx());
            for ( ; pptr!= _plist.end(); ++pptr) {
                assert(pptr->_pimpl.get());
                pptr->_pimpl->fillXGrid(x2);
                xt.accumulate(x2);
            }
        }
    }

    double SBAdd::SBAddImpl::getPositiveFlux() const 
    {
        double result = 0.;
        for (ConstIter pptr = _plist.begin(); pptr != _plist.end(); ++pptr) {
            result += pptr->getPositiveFlux();  
        }
        return result;
    }

    double SBAdd::SBAddImpl::getNegativeFlux() const 
    {
        double result = 0.;
        for (ConstIter pptr = _plist.begin(); pptr != _plist.end(); ++pptr) {
            result += pptr->getNegativeFlux();  
        }
        return result;
    }
        

    //
    // "SBDistort" Class 
    //
    SBDistort::SBDistortImpl::SBDistortImpl(
        const SBProfile& sbin, double mA, double mB, double mC, double mD,
        const Position<double>& cen, double fluxScaling) :
        _adaptee(sbin), _mA(mA), _mB(mB), _mC(mC), _mD(mD), _cen(cen), _fluxScaling(fluxScaling)
    {
        dbg<<"Start DistortImpl (1)\n";
        dbg<<"matrix = "<<_mA<<','<<_mB<<','<<_mC<<','<<_mD<<std::endl;
        dbg<<"cen = "<<_cen<<", fluxScaling = "<<_fluxScaling<<std::endl;

        // All the actual initialization is in a separate function so we can share code
        // with the other constructor.
        initialize();
    }

    SBDistort::SBDistortImpl::SBDistortImpl(
        const SBProfile& sbin, const Ellipse& e, double fluxScaling) :
        _adaptee(sbin), _cen(e.getX0()), _fluxScaling(fluxScaling)
    {
        dbg<<"Start DistortImpl (2)\n";
        dbg<<"e = "<<e<<", fluxScaling = "<<_fluxScaling<<std::endl;
        // First get what we need from the Ellipse:
        tmv::Matrix<double> m = e.getMatrix();
        _mA = m(0,0);
        _mB = m(0,1);
        _mC = m(1,0);
        _mD = m(1,1);

        // Then move on to the rest of the initialization process.
        initialize();
    }

    void SBDistort::SBDistortImpl::initialize() 
    {
        dbg<<"Start SBDistortImpl initialize\n";
        // First check if our adaptee is really another SBDistort:
        assert(_adaptee._pimpl.get());
        const SBDistortImpl* sbd = dynamic_cast<const SBDistortImpl*>(_adaptee._pimpl.get());
        dbg<<"sbd = "<<sbd<<std::endl;
        if (sbd) {
            dbg<<"wrapping another distortion.\n";
            // We are distorting something that's already a distortion.
            // So just compound the affine transformaions
            // New matrix is product (M_this) * (M_old)
            double mA = _mA; double mB=_mB; double mC=_mC; double mD=_mD;
            _cen += Position<double>(mA*sbd->_cen.x + mB*sbd->_cen.y,
                                     mC*sbd->_cen.x + mD*sbd->_cen.y);
            _mA = mA*sbd->_mA + mB*sbd->_mC;
            _mB = mA*sbd->_mB + mB*sbd->_mD;
            _mC = mC*sbd->_mA + mD*sbd->_mC;
            _mD = mC*sbd->_mB + mD*sbd->_mD;
            _fluxScaling *= sbd->_fluxScaling;
            dbg<<"before set adaptee"<<std::endl;
            _adaptee = sbd->_adaptee;
            dbg<<"after set adaptee"<<std::endl;
        } else {
            dbg<<"wrapping a non-distortion.\n";
        }
   
        // It will be reasonably common to have an identity matrix (for just
        // a flux scaling and/or shift) for (A,B,C,D).  If so, we can use simpler
        // versions of fwd and inv:
        if (_mA == 1. && _mB == 0. && _mC == 0. && _mD == 1.) {
            dbg<<"Using identity functions for fwd and inv\n";
            _fwd = &SBDistort::_ident;
            _inv = &SBDistort::_ident;
        } else {
            dbg<<"Using normal fwd and inv\n";
            _fwd = &SBDistort::_fwd_normal;
            _inv = &SBDistort::_inv_normal;
        }

        // Calculate some derived quantities:
        double det = _mA*_mD-_mB*_mC;
        if (det==0.) throw SBError("Attempt to SBDistort with degenerate matrix");
        _absdet = std::abs(det);
        _invdet = 1./det;

        double h1 = hypot( _mA+_mD, _mB-_mC);
        double h2 = hypot( _mA-_mD, _mB+_mC);
        _major = 0.5*std::abs(h1+h2);
        _minor = 0.5*std::abs(h1-h2);
        if (_major<_minor) std::swap(_major,_minor);
        _stillIsAxisymmetric = _adaptee.isAxisymmetric() 
            && (_mB==-_mC) 
            && (_mA==_mD)
            && (_cen.x==0.) && (_cen.y==0.); // Need pure rotation

        xdbg<<"Distortion init\n";
        xdbg<<"matrix = "<<_mA<<','<<_mB<<','<<_mC<<','<<_mD<<std::endl;
        xdbg<<"_cen = "<<_cen<<std::endl;
        xdbg<<"_invdet = "<<_invdet<<std::endl;
        xdbg<<"_absdet = "<<_absdet<<std::endl;
        xdbg<<"_fluxScaling = "<<_fluxScaling<<std::endl;
        xdbg<<"_major, _minor = "<<_major<<", "<<_minor<<std::endl;
        xdbg<<"maxK() = "<<_adaptee.maxK() / _minor<<std::endl;
        xdbg<<"stepK() = "<<_adaptee.stepK() / _major<<std::endl;

        // Calculate the values for getXRange and getYRange:
        if (_adaptee.isAxisymmetric()) {
            // The original is a circle, so first get its radius.
            _adaptee.getXRange(_xmin,_xmax,_xsplits);
            if (_xmax == integ::MOCK_INF) {
                // Then these are correct, and use +- inf for y range too.
                _ymin = -integ::MOCK_INF;
                _ymax = integ::MOCK_INF;
            } else {
                double R = _xmax;
                // The distortion takes each point on the circle to the following new coordinates:
                // (x,y) -> (A*x + B*y + x0 , C*x + D*y + y0)
                // Using x = R cos(t) and y = R sin(t), we can find the minimum wrt t as:
                // xmax = R sqrt(A^2 + B^2) + x0
                // xmin = -R sqrt(A^2 + B^2) + x0
                // ymax = R sqrt(C^2 + D^2) + y0
                // ymin = -R sqrt(C^2 + D^2) + y0
                double AApBB = _mA*_mA + _mB*_mB;
                double sqrtAApBB = sqrt(AApBB);
                double temp = sqrtAApBB * R;
                _xmin = -temp + _cen.x;
                _xmax = temp + _cen.x;
                double CCpDD = _mC*_mC + _mD*_mD;
                double sqrtCCpDD = sqrt(CCpDD);
                temp = sqrt(CCpDD) * R;
                _ymin = -temp + _cen.y;
                _ymax = temp + _cen.y;
                _ysplits.resize(_xsplits.size());
                for (size_t k=0;k<_xsplits.size();++k) {
                    // The split points work the same way.  Scale them by the same factor we
                    // scaled the R value above, then add _cen.x or _cen.y.
                    double split = _xsplits[k];
                    xxdbg<<"Adaptee split at "<<split<<std::endl;
                    _xsplits[k] = sqrtAApBB * split + _cen.x;
                    _ysplits[k] = sqrtCCpDD * split + _cen.y;
                    xxdbg<<"-> x,y splits at "<<_xsplits[k]<<"  "<<_ysplits[k]<<std::endl;
                }
                // Now a couple of calculations that get reused in getYRange(x,yminymax):
                _coeff_b = (_mA*_mC + _mB*_mD) / AApBB;
                _coeff_c = CCpDD / AApBB;
                _coeff_c2 = _absdet*_absdet / AApBB;
                xxdbg<<"adaptee is axisymmetric.\n";
                xxdbg<<"adaptees maxR = "<<R<<std::endl;
                xxdbg<<"xmin..xmax = "<<_xmin<<" ... "<<_xmax<<std::endl;
                xxdbg<<"ymin..ymax = "<<_ymin<<" ... "<<_ymax<<std::endl;
            }
        } else {
            // Apply the distortion to each of the four corners of the original
            // and find the minimum and maximum.
            double xmin_1, xmax_1;
            std::vector<double> xsplits0;
            _adaptee.getXRange(xmin_1,xmax_1,xsplits0);
            double ymin_1, ymax_1;
            std::vector<double> ysplits0;
            _adaptee.getYRange(ymin_1,ymax_1,ysplits0);
            // Note: This doesn't explicitly check for MOCK_INF values.
            // It shouldn't be a problem, since the integrator will still treat
            // large values near MOCK_INF as infinity, but it just means that 
            // the following calculations might be wasted flops.
            Position<double> bl = fwd(Position<double>(xmin_1,ymin_1));
            Position<double> br = fwd(Position<double>(xmax_1,ymin_1));
            Position<double> tl = fwd(Position<double>(xmin_1,ymax_1));
            Position<double> tr = fwd(Position<double>(xmax_1,ymax_1));
            _xmin = std::min(std::min(std::min(bl.x,br.x),tl.x),tr.x) + _cen.x;
            _xmax = std::max(std::max(std::max(bl.x,br.x),tl.x),tr.x) + _cen.x;
            _ymin = std::min(std::min(std::min(bl.y,br.y),tl.y),tr.y) + _cen.y;
            _ymax = std::max(std::max(std::max(bl.y,br.y),tl.y),tr.y) + _cen.y;
            xxdbg<<"adaptee is not axisymmetric.\n";
            xxdbg<<"adaptees x range = "<<xmin_1<<" ... "<<xmax_1<<std::endl;
            xxdbg<<"adaptees y range = "<<ymin_1<<" ... "<<ymax_1<<std::endl;
            xxdbg<<"Corners are: bl = "<<bl<<std::endl;
            xxdbg<<"             br = "<<br<<std::endl;
            xxdbg<<"             tl = "<<tl<<std::endl;
            xxdbg<<"             tr = "<<tr<<std::endl;
            xxdbg<<"xmin..xmax = "<<_xmin<<" ... "<<_xmax<<std::endl;
            xxdbg<<"ymin..ymax = "<<_ymin<<" ... "<<_ymax<<std::endl;
            if (bl.x + _cen.x > _xmin && bl.x + _cen.x < _xmax) {
                xxdbg<<"X Split from bl.x = "<<bl.x+_cen.x<<std::endl;
                _xsplits.push_back(bl.x+_cen.x);
            }
            if (br.x + _cen.x > _xmin && br.x + _cen.x < _xmax) {
                xxdbg<<"X Split from br.x = "<<br.x+_cen.x<<std::endl;
                _xsplits.push_back(br.x+_cen.x);
            }
            if (tl.x + _cen.x > _xmin && tl.x + _cen.x < _xmax) {
                xxdbg<<"X Split from tl.x = "<<tl.x+_cen.x<<std::endl;
                _xsplits.push_back(tl.x+_cen.x);
            }
            if (tr.x + _cen.x > _xmin && tr.x + _cen.x < _xmax) {
                xxdbg<<"X Split from tr.x = "<<tr.x+_cen.x<<std::endl;
                _xsplits.push_back(tr.x+_cen.x);
            }
            if (bl.y + _cen.y > _ymin && bl.y + _cen.y < _ymax) {
                xxdbg<<"Y Split from bl.y = "<<bl.y+_cen.y<<std::endl;
                _ysplits.push_back(bl.y+_cen.y);
            }
            if (br.y + _cen.y > _ymin && br.y + _cen.y < _ymax) {
                xxdbg<<"Y Split from br.y = "<<br.y+_cen.y<<std::endl;
                _ysplits.push_back(br.y+_cen.y);
            }
            if (tl.y + _cen.y > _ymin && tl.y + _cen.y < _ymax) {
                xxdbg<<"Y Split from tl.y = "<<tl.y+_cen.y<<std::endl;
                _ysplits.push_back(tl.y+_cen.y);
            }
            if (tr.y + _cen.y > _ymin && tr.y + _cen.y < _ymax) {
                xxdbg<<"Y Split from tr.y = "<<tr.y+_cen.y<<std::endl;
                _ysplits.push_back(tr.y+_cen.y);
            }
            // If the adaptee has any splits, try to propagate those up
            for(size_t k=0;k<xsplits0.size();++k) {
                xxdbg<<"Adaptee xsplit at "<<xsplits0[k]<<std::endl;
                Position<double> bx = fwd(Position<double>(xsplits0[k],ymin_1));
                Position<double> tx = fwd(Position<double>(xsplits0[k],ymax_1));
                if (bx.x + _cen.x > _xmin && bx.x + _cen.x < _xmax) {
                    xxdbg<<"X Split from bx.x = "<<bx.x+_cen.x<<std::endl;
                    _xsplits.push_back(bx.x+_cen.x);
                }
                if (tx.x + _cen.x > _xmin && tx.x + _cen.x < _xmax) {
                    xxdbg<<"X Split from tx.x = "<<tx.x+_cen.x<<std::endl;
                    _xsplits.push_back(tx.x+_cen.x);
                }
                if (bx.y + _cen.y > _ymin && bx.y + _cen.y < _ymax) {
                    xxdbg<<"Y Split from bx.y = "<<bx.y+_cen.y<<std::endl;
                    _ysplits.push_back(bx.y+_cen.y);
                }
                if (tx.y + _cen.y > _ymin && tx.y + _cen.y < _ymax) {
                    xxdbg<<"Y Split from tx.y = "<<tx.y+_cen.y<<std::endl;
                    _ysplits.push_back(tx.y+_cen.y);
                }
            }
            for(size_t k=0;k<ysplits0.size();++k) {
                xxdbg<<"Adaptee ysplit at "<<ysplits0[k]<<std::endl;
                Position<double> yl = fwd(Position<double>(xmin_1,ysplits0[k]));
                Position<double> yr = fwd(Position<double>(xmax_1,ysplits0[k]));
                if (yl.x + _cen.x > _xmin && yl.x + _cen.x < _xmax) {
                    xxdbg<<"X Split from tl.x = "<<tl.x+_cen.x<<std::endl;
                    _xsplits.push_back(yl.x+_cen.x);
                }
                if (yr.x + _cen.x > _xmin && yr.x + _cen.x < _xmax) {
                    xxdbg<<"X Split from yr.x = "<<yr.x+_cen.x<<std::endl;
                    _xsplits.push_back(yr.x+_cen.x);
                }
                if (yl.y + _cen.y > _ymin && yl.y + _cen.y < _ymax) {
                    xxdbg<<"Y Split from yl.y = "<<yl.y+_cen.y<<std::endl;
                    _ysplits.push_back(yl.y+_cen.y);
                }
                if (yr.y + _cen.y > _ymin && yr.y + _cen.y < _ymax) {
                    xxdbg<<"Y Split from yr.y = "<<yr.y+_cen.y<<std::endl;
                    _ysplits.push_back(yr.y+_cen.y);
                }
            }
        }
        // At this point we are done with _absdet per se.  Multiply it by _fluxScaling
        // so we can use it as the scale factor for kValue and getFlux.
        _absdet *= _fluxScaling;
        xdbg<<"_absdet -> "<<_absdet<<std::endl;

        // Figure out which function we need for kValue and kValueNoPhase
        if (std::abs(_absdet-1.) < sbp::kvalue_accuracy) {
            xdbg<<"absdet = "<<_absdet*_fluxScaling<<" = 1, so use NoDet version.\n";
            _kValueNoPhase = &SBDistort::_kValueNoPhaseNoDet;
        } else {
            xdbg<<"absdet = "<<_absdet*_fluxScaling<<" != 1, so use WithDet version.\n";
            _kValueNoPhase = &SBDistort::_kValueNoPhaseWithDet;
        }
        if (_cen.x == 0. && _cen.y == 0.) _kValue = _kValueNoPhase;
        else _kValue = &SBDistort::_kValueWithPhase;
    }

    void SBDistort::SBDistortImpl::getXRange(
        double& xmin, double& xmax, std::vector<double>& splits) const
    {
        xmin = _xmin; xmax = _xmax;
        splits.insert(splits.end(),_xsplits.begin(),_xsplits.end());
    }

    void SBDistort::SBDistortImpl::getYRange(
        double& ymin, double& ymax, std::vector<double>& splits) const
    {
        ymin = _ymin; ymax = _ymax;
        splits.insert(splits.end(),_ysplits.begin(),_ysplits.end());
    }

    void SBDistort::SBDistortImpl::getYRange(double x, double& ymin, double& ymax,
                              std::vector<double>& splits) const
    {
        xxdbg<<"Distortion getYRange for x = "<<x<<std::endl;
        if (_adaptee.isAxisymmetric()) {
            std::vector<double> splits0;
            _adaptee.getYRange(ymin,ymax,splits0);
            if (ymax == integ::MOCK_INF) return;
            double R = ymax;
            // The circlue with radius R is mapped onto an ellipse with (x,y) given by:
            // x = A R cos(t) + B R sin(t) + x0
            // y = C R cos(t) + D R sin(t) + y0
            //
            // Or equivalently:
            // (A^2+B^2) (y-y0)^2 - 2(AC+BD) (x-x0)(y-y0) + (C^2+D^2) (x-x0)^2 = R^2 (AD-BC)^2
            //
            // Given a particular value for x, we solve the latter equation for the 
            // corresponding range for y.
            // y^2 - 2 b y = c
            // -> y^2 - 2b y = c
            //    (y - b)^2 = c + b^2
            //    y = b +- sqrt(c + b^2)
            double b = _coeff_b * (x-_cen.x);
            double c = _coeff_c2 * R*R - _coeff_c * (x-_cen.x) * (x-_cen.x);
            double d = sqrt(c + b*b);
            ymax = b + d + _cen.y;
            ymin = b - d + _cen.y;
            for (size_t k=0;k<splits0.size();++k) if (splits0[k] >= 0.) {
                double r = splits0[k];
                double c = _coeff_c2 * r*r - _coeff_c * (x-_cen.x) * (x-_cen.x);
                double d = sqrt(c+b*b);
                splits.push_back(b + d + _cen.y);
                splits.push_back(b - d + _cen.y);
            }
            xxdbg<<"Axisymmetric adaptee with R = "<<R<<std::endl;
            xxdbg<<"ymin .. ymax = "<<ymin<<" ... "<<ymax<<std::endl;
        } else {
            // There are 4 lines to check for where they intersect the given x.
            // Start with the adaptee's given ymin.
            // This line is distorted onto the line:
            // (x',ymin) -> ( A x' + B ymin + x0 , C x' + D ymin + y0 )
            // x' = (x - x0 - B ymin) / A
            // y = C x' + D ymin + y0 
            //   = C (x - x0 - B ymin) / A + D ymin + y0
            // The top line is analagous for ymax instead of ymin.
            // 
            // The left line is distorted as:
            // (xmin,y) -> ( A xmin + B y' + x0 , C xmin + D y' + y0 )
            // y' = (x - x0 - A xmin) / B
            // y = C xmin + D (x - x0 - A xmin) / B + y0
            // And again, the right line is analgous.
            //
            // We also need to check for A or B = 0, since then only one pair of lines is
            // relevant.
            xxdbg<<"Non-axisymmetric adaptee\n";
            if (_mA == 0.) {
                xxdbg<<"_mA == 0:\n";
                double xmin_1, xmax_1;
                std::vector<double> xsplits0;
                _adaptee.getXRange(xmin_1,xmax_1,xsplits0);
                xxdbg<<"xmin_1, xmax_1 = "<<xmin_1<<','<<xmax_1<<std::endl;
                ymin = _mC * xmin_1 + _mD * (x - _cen.x - _mA*xmin_1) / _mB + _cen.y;
                ymax = _mC * xmax_1 + _mD * (x - _cen.x - _mA*xmax_1) / _mB + _cen.y;
                if (ymax < ymin) std::swap(ymin,ymax);
                for(size_t k=0;k<xsplits0.size();++k) {
                    double xx = xsplits0[k];
                    splits.push_back(_mC * xx + _mD * (x - _cen.x - _mA*xx) / _mB + _cen.y);
                }
            } else if (_mB == 0.) {
                xxdbg<<"_mB == 0:\n";
                double ymin_1, ymax_1;
                std::vector<double> ysplits0;
                _adaptee.getYRange(ymin_1,ymax_1,ysplits0);
                xxdbg<<"ymin_1, ymax_1 = "<<ymin_1<<','<<ymax_1<<std::endl;
                ymin = _mC * (x - _cen.x - _mB*ymin_1) / _mA + _mD*ymin_1 + _cen.y;
                ymax = _mC * (x - _cen.x - _mB*ymax_1) / _mA + _mD*ymax_1 + _cen.y;
                if (ymax < ymin) std::swap(ymin,ymax);
                for(size_t k=0;k<ysplits0.size();++k) {
                    double yy = ysplits0[k];
                    splits.push_back(_mC * (x - _cen.x - _mB*yy) / _mA + _mD*yy + _cen.y);
                }
            } else {
                xxdbg<<"_mA,B != 0:\n";
                double ymin_1, ymax_1;
                std::vector<double> xsplits0;
                _adaptee.getYRange(ymin_1,ymax_1,xsplits0);
                xxdbg<<"ymin_1, ymax_1 = "<<ymin_1<<','<<ymax_1<<std::endl;
                ymin = _mC * (x - _cen.x - _mB*ymin_1) / _mA + _mD*ymin_1 + _cen.y;
                ymax = _mC * (x - _cen.x - _mB*ymax_1) / _mA + _mD*ymax_1 + _cen.y;
                xxdbg<<"From top and bottom: ymin,ymax = "<<ymin<<','<<ymax<<std::endl;
                if (ymax < ymin) std::swap(ymin,ymax);
                double xmin_1, xmax_1;
                std::vector<double> ysplits0;
                _adaptee.getXRange(xmin_1,xmax_1,ysplits0);
                xxdbg<<"xmin_1, xmax_1 = "<<xmin_1<<','<<xmax_1<<std::endl;
                ymin_1 = _mC * xmin_1 + _mD * (x - _cen.x - _mA*xmin_1) / _mB + _cen.y;
                ymax_1 = _mC * xmax_1 + _mD * (x - _cen.x - _mA*xmax_1) / _mB + _cen.y;
                xxdbg<<"From left and right: ymin,ymax = "<<ymin_1<<','<<ymax_1<<std::endl;
                if (ymax_1 < ymin_1) std::swap(ymin_1,ymax_1);
                if (ymin_1 > ymin) ymin = ymin_1;
                if (ymax_1 < ymax) ymax = ymax_1;
                for(size_t k=0;k<ysplits0.size();++k) {
                    double yy = ysplits0[k];
                    splits.push_back(_mC * (x - _cen.x - _mB*yy) / _mA + _mD*yy + _cen.y);
                }
                for(size_t k=0;k<xsplits0.size();++k) {
                    double xx = xsplits0[k];
                    splits.push_back(_mC * xx + _mD * (x - _cen.x - _mA*xx) / _mB + _cen.y);
                }
            }
            xxdbg<<"ymin .. ymax = "<<ymin<<" ... "<<ymax<<std::endl;
        }
    }

    // Specialization of fillKGrid is desired since the phase terms from shift 
    // are factorizable:
    void SBDistort::SBDistortImpl::fillKGrid(KTable& kt) const 
    {
        double N = (double) kt.getN();
        double dk = kt.getDk();

#if 0
        // The simpler version, saved for reference
        if (_cen.x==0. && _cen.y==0.) {
            // Branch to faster calculation if there is no centroid shift:
            for (int iy = -N/2; iy < N/2; iy++) {
                // only need ix>=0 since it's Hermitian:
                for (int ix = 0; ix <= N/2; ix++) {
                    Position<double> k(ix*dk,iy*dk);
                    kt.kSet(ix,iy,kValueNoPhase(k));
                }
            }
        } else {
            std::complex<double> dxexp(0,-dk*_cen.x),   dyexp(0,-dk*_cen.y);
            std::complex<double> dxphase(std::exp(dxexp)), dyphase(std::exp(dyexp));
            // xphase, yphase: current phase value
            std::complex<double> yphase(std::exp(-dyexp*N/2.));
            for (int iy = -N/2; iy < N/2; iy++) {
                std::complex<double> phase = yphase; // since kx=0 to start
                // Only ix>=0 since it's Hermitian:
                for (int ix = 0; ix <= N/2; ix++) {
                    Position<double> k(ix*dk,iy*dk);
                    kt.kSet(ix,iy,kValueNoPhase(k) * phase);
                    phase *= dxphase;
                }
                yphase *= dyphase;
            }
        }
#else
        // A faster version that pulls out all the if statements
        // and keeps track of fwdT(k) as we go
        kt.clearCache();

        double dkA = dk*_mA;
        double dkB = dk*_mB;
        if (_cen.x==0. && _cen.y==0.) {
            // Branch to faster calculation if there is no centroid shift:
            Position<double> k1(0.,0.);
            Position<double> fwdTk1(0.,0.);
            for (int ix = 0; ix <= N/2; ix++, fwdTk1.x += dkA, fwdTk1.y += dkB) {
                // NB: the last two terms are not used by _kValueNoPhase,
                // so it's ok that k1.x is not kept up to date.
                kt.kSet2(ix,0,_kValueNoPhase(_adaptee,fwdTk1,_absdet,k1,_cen));
            }
            k1.y = dk; 
            Position<double> k2(0.,-dk);
            Position<double> fwdTk2;
            for (int iy = 1; iy < N/2; ++iy, k1.y += dk, k2.y -= dk) {
                fwdTk1 = fwdT(k1); fwdTk2 = fwdT(k2);
                for (int ix = 0; ix <= N/2; ++ix,
                     fwdTk1.x += dkA, fwdTk1.y += dkB, fwdTk2.x += dkA, fwdTk2.y += dkB) {
                    kt.kSet2(ix,iy, _kValueNoPhase(_adaptee,fwdTk1,_absdet,k1,_cen));
                    kt.kSet2(ix,N-iy, _kValueNoPhase(_adaptee,fwdTk2,_absdet,k2,_cen));
                }
            }
            fwdTk1 = fwdT(k1);
            for (int ix = 0; ix <= N/2; ix++, fwdTk1.x += dkA, fwdTk1.y += dkB) {
                kt.kSet2(ix,N/2,_kValueNoPhase(_adaptee,fwdTk1,_absdet,k1,_cen));
            }
        } else {
            std::complex<double> dxphase = std::polar(1.,-dk*_cen.x);
            std::complex<double> dyphase = std::polar(1.,-dk*_cen.y);
            // xphase, yphase: current phase value
            std::complex<double> yphase = 1.;
            Position<double> k1(0.,0.);
            Position<double> fwdTk1(0.,0.);
            std::complex<double> phase = yphase; // since kx=0 to start
            for (int ix = 0; ix <= N/2; ++ix,
                 fwdTk1.x += dkA, fwdTk1.y += dkB, phase *= dxphase) {
                kt.kSet2(ix,0, _kValueNoPhase(_adaptee,fwdTk1,_absdet,k1,_cen) * phase);
            }
            k1.y = dk; yphase *= dyphase;
            Position<double> k2(0.,-dk);  
            Position<double> fwdTk2;
            std::complex<double> phase2;
            for (int iy = 1; iy < N/2; iy++, k1.y += dk, k2.y -= dk, yphase *= dyphase) {
                fwdTk1 = fwdT(k1); fwdTk2 = fwdT(k2);
                phase = yphase; phase2 = conj(yphase);
                for (int ix = 0; ix <= N/2; ++ix,
                     fwdTk1.x += dkA, fwdTk1.y += dkB, fwdTk2.x += dkA, fwdTk2.y += dkB,
                     phase *= dxphase, phase2 *= dxphase) {
                    kt.kSet2(ix,iy, _kValueNoPhase(_adaptee,fwdTk1,_absdet,k1,_cen) * phase);
                    kt.kSet2(ix,N-iy, _kValueNoPhase(_adaptee,fwdTk2,_absdet,k1,_cen) * phase2);
                }
            }
            fwdTk1 = fwdT(k1);
            phase = yphase; 
            for (int ix = 0; ix <= N/2; ++ix, fwdTk1.x += dkA, fwdTk1.y += dkB, phase *= dxphase) {
                kt.kSet2(ix,N/2, _kValueNoPhase(_adaptee,fwdTk1,_absdet,k1,_cen) * phase);
            }
        }
#endif
    }

    std::complex<double> SBDistort::SBDistortImpl::kValue(const Position<double>& k) const
    { return _kValue(_adaptee,fwdT(k),_absdet,k,_cen); }

    std::complex<double> SBDistort::SBDistortImpl::kValueNoPhase(const Position<double>& k) const
    { return _kValueNoPhase(_adaptee,fwdT(k),_absdet,k,_cen); }

    std::complex<double> SBDistort::_kValueNoPhaseNoDet(
        const SBProfile& adaptee, const Position<double>& fwdTk, double absdet,
        const Position<double>& , const Position<double>& )
    { return adaptee.kValue(fwdTk); }

    std::complex<double> SBDistort::_kValueNoPhaseWithDet(
        const SBProfile& adaptee, const Position<double>& fwdTk, double absdet,
        const Position<double>& , const Position<double>& )
    { return absdet * adaptee.kValue(fwdTk); }

    std::complex<double> SBDistort::_kValueWithPhase(
        const SBProfile& adaptee, const Position<double>& fwdTk, double absdet,
        const Position<double>& k, const Position<double>& cen)
    { return adaptee.kValue(fwdTk) * std::polar(absdet , -k.x*cen.x-k.y*cen.y); }


    //
    // SBConvolve class - adding new members
    //
    void SBConvolve::SBConvolveImpl::add(const SBProfile& rhs) 
    {
        dbg<<"Start SBConvolveImpl::add.  Adding item # "<<_plist.size()+1<<std::endl;

        // Add new terms(s) to the _plist:
        assert(rhs._pimpl.get());
        const SBConvolveImpl *sbc = dynamic_cast<const SBConvolveImpl*>(rhs._pimpl.get());
        if (sbc) {  
            // If rhs is an SBConvolve, copy its list here
            for (ConstIter pptr = sbc->_plist.begin(); pptr!=sbc->_plist.end(); ++pptr) {
                if (!pptr->isAnalyticK() && !_real_space) 
                    throw SBError("SBConvolve requires members to be analytic in k");
                if (!pptr->isAnalyticX() && _real_space)
                    throw SBError("Real_space SBConvolve requires members to be analytic in x");
                _plist.push_back(*pptr);
            }
        } else {
            if (!rhs.isAnalyticK() && !_real_space) 
                throw SBError("SBConvolve requires members to be analytic in k");
            if (!rhs.isAnalyticX() && _real_space)
                throw SBError("Real-space SBConvolve requires members to be analytic in x");
            _plist.push_back(rhs);
        }
    }

    void SBConvolve::SBConvolveImpl::initialize()
    {
        _x0 = _y0 = 0.;
        _fluxProduct = 1.;
        _minMaxK = 0.;
        _isStillAxisymmetric = true;

        _netStepK = 0.;  // Accumulate Sum 1/stepk^2
        for(ConstIter it=_plist.begin(); it!=_plist.end(); ++it) {
            double maxk = it->maxK();
            double stepk = it->stepK();
            dbg<<"SBConvolve component has maxK, stepK = "<<maxk<<" , "<<stepk<<std::endl;
            _fluxProduct *= it->getFlux();
            _x0 += it->centroid().x;
            _y0 += it->centroid().y;
            if ( _minMaxK<=0. || maxk < _minMaxK) _minMaxK = maxk;
            _netStepK += 1./(stepk*stepk);
            _isStillAxisymmetric = _isStillAxisymmetric && it->isAxisymmetric();
        }
        _netStepK = 1./sqrt(_netStepK);  // Convert to (Sum 1/stepk^2)^(-1/2)
        dbg<<"Net maxK, stepK = "<<_minMaxK<<" , "<<_netStepK<<std::endl;
    }

    void SBConvolve::SBConvolveImpl::fillKGrid(KTable& kt) const 
    {
        if (_plist.empty()) kt.clear();
        ConstIter pptr = _plist.begin();
        assert(pptr->_pimpl.get());
        pptr->_pimpl->fillKGrid(kt);
        if (++pptr != _plist.end()) {
            KTable k2(kt.getN(),kt.getDk());
            for ( ; pptr!= _plist.end(); ++pptr) {
                assert(pptr->_pimpl.get());
                pptr->_pimpl->fillKGrid(k2);
                kt *= k2;
            }
        }
    }

    double SBConvolve::SBConvolveImpl::xValue(const Position<double>& pos) const
    {
        // Perform a direct calculation of the convolution at a particular point by
        // doing the real-space integral.
        // Note: This can only really be done one pair at a time, so it is 
        // probably rare that this will be more efficient if N > 2.
        // For now, we don't bother implementing this for N > 2.
        
        if (_plist.size() == 2) {
            const SBProfile& p1 = _plist.front();
            const SBProfile& p2 = _plist.back();
            if (p2.isAxisymmetric())
                return RealSpaceConvolve(p2,p1,pos,_fluxProduct);
            else 
                return RealSpaceConvolve(p1,p2,pos,_fluxProduct);
        } else if (_plist.empty()) 
            return 0.;
        else if (_plist.size() == 1) 
            return _plist.front().xValue(pos);
        else 
            throw SBError("Real-space integration of more than 2 profiles is not implemented.");
    }

    std::complex<double> SBConvolve::SBConvolveImpl::kValue(const Position<double>& k) const 
    {
        ConstIter pptr = _plist.begin();
        assert(pptr != _plist.end());
        std::complex<double> kv = pptr->kValue(k);
        for (++pptr; pptr != _plist.end(); ++pptr)
            kv *= pptr->kValue(k);
        return kv;
    } 


    double SBConvolve::SBConvolveImpl::getPositiveFlux() const 
    {
        if (_plist.empty()) return 0.;
        std::list<SBProfile>::const_iterator pptr = _plist.begin();
        double pResult = pptr->getPositiveFlux();
        double nResult = pptr->getNegativeFlux();
        for (++pptr; pptr!=_plist.end(); ++pptr) {
            double p = pptr->getPositiveFlux();
            double n = pptr->getNegativeFlux();
            double pNew = p*pResult + n*nResult;
            nResult = p*nResult + n*pResult;
            pResult = pNew;
        }
        return pResult;
    }

    // Note duplicated code here, could be caching results for tiny efficiency gain
    double SBConvolve::SBConvolveImpl::getNegativeFlux() const 
    {
        if (_plist.empty()) return 0.;
        std::list<SBProfile>::const_iterator pptr = _plist.begin();
        double pResult = pptr->getPositiveFlux();
        double nResult = pptr->getNegativeFlux();
        for (++pptr; pptr!=_plist.end(); ++pptr) {
            double p = pptr->getPositiveFlux();
            double n = pptr->getNegativeFlux();
            double pNew = p*pResult + n*nResult;
            nResult = p*nResult + n*pResult;
            pResult = pNew;
        }
        return nResult;
    }

    //
    // "SBGaussian" Class 
    //

  SBGaussian::SBGaussianImpl::SBGaussianImpl(double sigma, double flux) :
        _flux(flux), _sigma(sigma), _sigma_sq(sigma*sigma)
    {
        // For large k, we clip the result of kValue to 0.
        // We do this when the correct answer is less than kvalue_accuracy.
        // exp(-k^2*sigma^2/2) = kvalue_accuracy
        _ksq_max = -2. * std::log(sbp::kvalue_accuracy) / _sigma_sq;

        // For small k, we can use up to quartic in the taylor expansion to avoid the exp.
        // This is acceptable when the next term is less than kvalue_accuracy.
        // 1/48 (k^2 r0^2)^3 = kvalue_accuracy
        _ksq_min = std::pow(sbp::kvalue_accuracy * 48., 1./3.) / _sigma_sq;

        _norm = _flux / (_sigma_sq * 2. * M_PI);

        dbg<<"Gaussian:\n";
        dbg<<"_flux = "<<_flux<<std::endl;
        dbg<<"_sigma = "<<_sigma<<std::endl;
        dbg<<"_sigma_sq = "<<_sigma_sq<<std::endl;
        dbg<<"_ksq_max = "<<_ksq_max<<std::endl;
        dbg<<"_ksq_min = "<<_ksq_min<<std::endl;
        dbg<<"_norm = "<<_norm<<std::endl;
        dbg<<"maxK() = "<<maxK()<<std::endl;
        dbg<<"stepK() = "<<stepK()<<std::endl;
    }

    // Set maxK to the value where the FT is down to maxk_threshold
    double SBGaussian::SBGaussianImpl::maxK() const 
    { return sqrt(-2.*std::log(sbp::maxk_threshold))/_sigma; }

    // The amount of flux missed in a circle of radius pi/stepk should miss at 
    // most alias_threshold of the flux.
    double SBGaussian::SBGaussianImpl::stepK() const
    {
        // int( exp(-r^2/2) r, r=0..R) = 1 - exp(-R^2/2)
        // exp(-R^2/2) = alias_threshold
        double R = sqrt(-2.*std::log(sbp::alias_threshold));
        // Make sure it is at least 4 sigma;
        R = std::max(4., R);
        return M_PI / (R*_sigma);
    }

    double SBGaussian::SBGaussianImpl::xValue(const Position<double>& p) const
    {
        double rsq = p.x*p.x + p.y*p.y;
        return _norm * std::exp( -rsq/(2.*_sigma_sq) );
    }

    std::complex<double> SBGaussian::SBGaussianImpl::kValue(const Position<double>& k) const
    {
        double ksq = k.x*k.x+k.y*k.y;

        if (ksq > _ksq_max) {
            return 0.;
        } else if (ksq < _ksq_min) {
            ksq *= _sigma_sq;
            return _flux*(1. - 0.5*ksq*(1. - 0.25*ksq));
        } else {
            return _flux * std::exp(-ksq * _sigma_sq/2.);
        }
    }


    //
    // SBExponential Class
    //

  SBExponential::SBExponentialImpl::SBExponentialImpl(double r0, double flux) :
        _flux(flux), _r0(r0), _r0_sq(r0*r0)
    {
        // For large k, we clip the result of kValue to 0.
        // We do this when the correct answer is less than kvalue_accuracy.
        // (1+k^2 r0^2)^-1.5 = kvalue_accuracy
        _ksq_max = (std::pow(sbp::kvalue_accuracy,-1./1.5)-1.) / _r0_sq;

        // For small k, we can use up to quartic in the taylor expansion to avoid the sqrt.
        // This is acceptable when the next term is less than kvalue_accuracy.
        // 35/16 (k^2 r0^2)^3 = kvalue_accuracy
        _ksq_min = std::pow(sbp::kvalue_accuracy * 16./35., 1./3.) / _r0_sq;

        _norm = _flux / (_r0_sq * 2. * M_PI);

        dbg<<"Exponential:\n";
        dbg<<"_flux = "<<_flux<<std::endl;
        dbg<<"_r0 = "<<_r0<<std::endl;
        dbg<<"_r0_sq = "<<_r0_sq<<std::endl;
        dbg<<"_ksq_max = "<<_ksq_max<<std::endl;
        dbg<<"_ksq_min = "<<_ksq_min<<std::endl;
        dbg<<"_norm = "<<_norm<<std::endl;
        dbg<<"maxK() = "<<maxK()<<std::endl;
        dbg<<"stepK() = "<<stepK()<<std::endl;
    }

    // Set maxK to the value where the FT is down to maxk_threshold
    double SBExponential::SBExponentialImpl::maxK() const 
    { return std::pow(sbp::maxk_threshold, -1./3.)/_r0; }

    // The amount of flux missed in a circle of radius pi/stepk should miss at 
    // most alias_threshold of the flux.
    double SBExponential::SBExponentialImpl::stepK() const
    {
        // int( exp(-r) r, r=0..R) = (1 - exp(-R) - Rexp(-R))
        // Fraction excluded is thus (1+R) exp(-R)
        // A fast solution to (1+R)exp(-R) = x:
        // log(1+R) - R = log(x)
        // R = log(1+R) - log(x)
        double logx = std::log(sbp::alias_threshold);
        double R = -logx;
        for (int i=0; i<3; i++) R = std::log(1.+R) - logx;
        // Make sure it is at least 6 scale radii.
        R = std::max(6., R);
        return M_PI / (R*_r0);
    }

    double SBExponential::SBExponentialImpl::xValue(const Position<double>& p) const
    {
        double r = sqrt(p.x*p.x + p.y*p.y);
        return _norm * std::exp(-r/_r0);
    }

    std::complex<double> SBExponential::SBExponentialImpl::kValue(const Position<double>& k) const 
    {
        double ksq = k.x*k.x+k.y*k.y;

        if (ksq > _ksq_max) {
            return 0.;
        } else if (ksq < _ksq_min) {
            ksq *= _r0_sq;
            return _flux*(1. - 1.5*ksq*(1. - 1.25*ksq));
        } else {
            double temp = 1. + ksq*_r0_sq;
            return _flux/(temp*sqrt(temp));
            // NB: flux*std::pow(temp,-1.5) is slower.
        }
    }

    //
    // SBAiry Class
    //

    SBAiry::SBAiryImpl::SBAiryImpl(double D, double obscuration, double flux) :
        _D(D), _obscuration(obscuration), _flux(flux), _norm(flux*D*D),
        _radial(_obscuration) {}

    // This is a scale-free version of the Airy radial function.
    // Input radius is in units of lambda/D.  Output normalized
    // to integrate to unity over input units.
    double SBAiry::AiryRadialFunction::operator()(double radius) const 
    {
        double nu = radius*M_PI;
        double xval;
        // TODO: Check this limit.  Should use kvalue_accuracy?  
        // (Not in k space, though, so maybe need a new xvalue_accuracy...)
        if (nu<0.01) {
            // lim j1(u)/u = 1/2
            xval =  (1.-_obscuration*_obscuration);
        } else {
            // See Schroeder eq (10.1.10)
            xval = 2.*( j1(nu) - _obscuration*j1(_obscuration*nu)) / nu ; 
        }
        xval*=xval;
        // Normalize to give unit flux integrated over area.
        // TODO: Save this as _norm, so this can become xval *= _norm;
        // i.e. _norm = M_PI / (4.*(1.-_obscuration*_obscuration))
        xval /= (1-_obscuration*_obscuration)*4./M_PI;
        return xval;
    }

    double SBAiry::SBAiryImpl::xValue(const Position<double>& p) const 
    {
        double radius = sqrt(p.x*p.x+p.y*p.y) * _D;
        return _norm * _radial(radius);
    }

    std::complex<double> SBAiry::SBAiryImpl::kValue(const Position<double>& k) const
    {
        // TODO: I think the sqrt can be skipped, but need to follow through 
        //       some of the other functions here to use ksq, rather than K
        //       (which is called t in e.g. circle_intersection and annuli_intersect.
        //       However, I haven't gone through this yet, because we don't have
        //       any unit tests currently that would test this, and I didn't 
        //       feel like making them just now.
        double kk = sqrt(k.x*k.x+k.y*k.y);
        // calculate circular FT(PSF) on p'=(x',y')
        return _flux * annuli_autocorrelation(kk);
    }

    // Set maxK to hard limit for Airy disk.
    double SBAiry::SBAiryImpl::maxK() const 
    { return 2.*M_PI*_D; }

    // The amount of flux missed in a circle of radius pi/stepk should miss at 
    // most alias_threshold of the flux.
    double SBAiry::SBAiryImpl::stepK() const
    {
        // Schroeder (10.1.18) gives limit of EE at large radius.
        // This stepK could probably be relaxed, it makes overly accurate FFTs.
        double R = 1. / (sbp::alias_threshold * 0.5 * M_PI * M_PI * (1.-_obscuration));
        // Use at least 5 lam/D
        R = std::max(R,5.);
        return M_PI * _D / R;
    }

    double SBAiry::SBAiryImpl::chord(const double r, const double h) const 
    {
        if (r<h) throw SBError("Airy calculation r<h");
        else if (r==0.) return 0.;
        else if (r<0. || h<0.) throw SBError("Airy calculation (r||h)<0");
        return r*r*std::asin(h/r) -h*sqrt(r*r-h*h);
    }

    /* area inside intersection of 2 circles radii r & s, seperated by t*/
    double SBAiry::SBAiryImpl::circle_intersection(double r, double s, double t) const 
    {
        double h;
        if (r<0. || s<0.) throw SBError("Airy calculation negative radius");
        t = fabs(t);
        if (t>= r+s) return 0.;
        if (r<s) {
            double temp;
            temp = s;
            s = r;
            r = temp;
        }
        if (t<= r-s) return M_PI*s*s;

        /* in between we calculate half-height at intersection */
        h = 0.5*(r*r + s*s) - (std::pow(t,4.) + (r*r-s*s)*(r*r-s*s))/(4.*t*t);
        if (h<0.) {
            throw SBError("Airy calculation half-height invalid");
        }
        h = sqrt(h);

        if (t*t < r*r - s*s) 
            return M_PI*s*s - chord(s,h) + chord(r,h);
        else
            return chord(s,h) + chord(r,h);
    }

    /* area of two intersecting identical annuli */
    double SBAiry::SBAiryImpl::annuli_intersect(double r1, double r2, double t) const 
    {
        if (r1<r2) {
            double temp;
            temp = r2;
            r2 = r1;
            r1 = temp;
        }
        return circle_intersection(r1,r1,t)
            - 2. * circle_intersection(r1,r2,t)
            +  circle_intersection(r2,r2,t);
    }

    /* Beam pattern of annular aperture, in k space, which is just the
     * autocorrelation of two annuli.  Normalize to unity at k=0 for now */
    double SBAiry::SBAiryImpl::annuli_autocorrelation(const double k) const 
    {
        double k_scaled = k / (M_PI*_D);
        double norm = M_PI*(1. - _obscuration*_obscuration);
        return annuli_intersect(1.,_obscuration,k_scaled)/norm;
    }


    //
    // SBBox Class
    //

    double SBBox::SBBoxImpl::xValue(const Position<double>& p) const 
    {
        if (fabs(p.x) < 0.5*_xw && fabs(p.y) < 0.5*_yw) return _norm;
        else return 0.;  // do not use this function for fillXGrid()!
    }

    double SBBox::SBBoxImpl::sinc(const double u) const 
    {
        if (std::abs(u) < 1.e-3)
            return 1.-u*u/6.;
        else
            return std::sin(u)/u;
    }

    std::complex<double> SBBox::SBBoxImpl::kValue(const Position<double>& k) const
    {
        return _flux * sinc(0.5*k.x*_xw)*sinc(0.5*k.y*_yw);
    }

    // Set maxK to the value where the FT is down to maxk_threshold
    double SBBox::SBBoxImpl::maxK() const 
    { 
        return 2. / (sbp::maxk_threshold * std::min(_xw,_yw));
    }

    // The amount of flux missed in a circle of radius pi/stepk should miss at 
    // most alias_threshold of the flux.
    double SBBox::SBBoxImpl::stepK() const
    {
        // In this case max(xw,yw) encloses all the flux, so use that.
        return M_PI / std::max(_xw,_yw);
    }

    // Override fillXGrid so we can partially fill pixels at edge of box.
    void SBBox::SBBoxImpl::fillXGrid(XTable& xt) const 
    {
        int N = xt.getN();
        double dx = xt.getDx(); // pixel grid size

        // Pixel index where edge of box falls:
        int xedge = int( std::ceil(_xw / (2*dx) - 0.5) );
        int yedge = int( std::ceil(_yw / (2*dx) - 0.5) );
        // Fraction of edge pixel that is filled by box:
        double xfrac = _xw / (2*dx) - xedge + 0.5;
        assert(xfrac>0. && xfrac<=1.);
        double yfrac = _yw / (2*dx) - yedge + 0.5;
        assert(yfrac>0. && yfrac<=1.);
        if (xedge==0) xfrac = _xw/dx;
        if (yedge==0) yfrac = _yw/dx;

        double yfac;
        for (int iy = -N/2; iy < N/2; iy++) {
            if ( std::abs(iy) < yedge ) yfac = 0.;
            else if (std::abs(iy)==yedge) yfac = _norm*yfrac;
            else yfac = _norm;

            for (int ix = -N/2; ix < N/2; ix++) {
                if (yfac==0. || std::abs(ix)>xedge) xt.xSet(ix, iy ,0.);
                else if (std::abs(ix)==xedge) xt.xSet(ix, iy ,xfrac*yfac);
                else xt.xSet(ix,iy,yfac);
            }
        }
    }

    // Override x-domain writing so we can partially fill pixels at edge of box.
    template <typename T>
    double SBBox::SBBoxImpl::fillXImage(ImageView<T>& I, double dx) const 
    {
        // Pixel index where edge of box falls:
        int xedge = int( std::ceil(_xw / (2*dx) - 0.5) );
        int yedge = int( std::ceil(_yw / (2*dx) - 0.5) );
        // Fraction of edge pixel that is filled by box:
        double xfrac = _xw / (2*dx) - xedge + 0.5;
        assert(xfrac>0. && xfrac<=1.);
        double yfrac = _yw / (2*dx) - yedge + 0.5;
        assert(yfrac>0. && yfrac<=1.);
        if (xedge==0) xfrac = _xw/dx;
        if (yedge==0) yfrac = _yw/dx;

        double totalflux = 0.;
        double xfac;
        for (int i = I.getXMin(); i <= I.getXMax(); i++) {
            if ( std::abs(i) > xedge ) xfac = 0.;
            else if (std::abs(i)==xedge) xfac = _norm*xfrac;
            else xfac = _norm;

            for (int j = I.getYMin(); j <= I.getYMax(); j++) {
                if (xfac==0. || std::abs(j)>yedge) I(i,j)=T(0);
                else if (std::abs(j)==yedge) I(i,j)=xfac*yfrac;
                else I(i,j)=xfac;
                totalflux += I(i,j);
            }
        }
        I.setScale(dx);

        return totalflux * (dx*dx);
    }

    // Override fillKGrid for efficiency, since kValues are separable.
    void SBBox::SBBoxImpl::fillKGrid(KTable& kt) const 
    {
        int N = kt.getN();
        double dk = kt.getDk();

#if 0
        // The simple version, saved for reference
        for (int iy = -N/2; iy < N/2; iy++) {
            // Only need ix>=0 because it's Hermitian:
            for (int ix = 0; ix <= N/2; ix++) {
                Position<double> k(ix*dk,iy*dk);
                // The value returned by kValue(k)
                double kvalue = _flux * sinc(0.5*k.x*_xw) * sinc(0.5*k.y*_yw);
                kt.kSet(ix,iy,kvalue);
            }
        }
#else
        // A faster version that pulls out all the if statements and store the 
        // relevant sinc functions in two arrays, so we don't need to keep calling 
        // sinc on the same values over and over.
        
        kt.clearCache();
        std::vector<double> sinc_x(N/2+1);
        std::vector<double> sinc_y(N/2+1);
        if (_xw == _yw) { // Typical
            for (int i = 0; i <= N/2; i++) {
                sinc_x[i] = sinc(0.5 * i * dk * _xw);
                sinc_y[i] = sinc_x[i];
            }
        } else {
            for (int i = 0; i <= N/2; i++) {
                sinc_x[i] = sinc(0.5 * i * dk * _xw);
                sinc_y[i] = sinc(0.5 * i * dk * _yw);
            }
        }

        // Now do the unrolled version with kSet2
        for (int ix = 0; ix <= N/2; ix++) {
            kt.kSet2(ix,0, _flux * sinc_x[ix] * sinc_y[0]);
        }
        for (int iy = 1; iy < N/2; iy++) {
            for (int ix = 0; ix <= N/2; ix++) {
                double kval = _flux * sinc_x[ix] * sinc_y[iy];
                kt.kSet2(ix,iy,kval);
                kt.kSet2(ix,N-iy,kval);
            }
        }
        for (int ix = 0; ix <= N/2; ix++) {
            kt.kSet2(ix,N/2, _flux * sinc_x[ix] * sinc_y[N/2]);
        }
#endif
    }

    //
    // SBLaguerre Class
    //

    // ??? Have not really investigated these:
    double SBLaguerre::SBLaguerreImpl::maxK() const 
    {
        // Start with value for plain old Gaussian:
        double maxk = sqrt(-2.*std::log(sbp::maxk_threshold))/_sigma; 
        // Grow as sqrt of order
        if (_bvec.getOrder() > 1) maxk *= sqrt(double(_bvec.getOrder()));
        return maxk;
    }

    double SBLaguerre::SBLaguerreImpl::stepK() const 
    {
        // Start with value for plain old Gaussian:
        double R = std::max(4., sqrt(-2.*std::log(sbp::alias_threshold)));
        // Grow as sqrt of order
        if (_bvec.getOrder() > 1) R *= sqrt(double(_bvec.getOrder()));
        return M_PI / (R*_sigma);
    }

    double SBLaguerre::SBLaguerreImpl::xValue(const Position<double>& p) const 
    {
        LVector psi(_bvec.getOrder());
        psi.fillBasis(p.x/_sigma, p.y/_sigma, _sigma);
        double xval = _bvec.dot(psi);
        return xval;
    }

    std::complex<double> SBLaguerre::SBLaguerreImpl::kValue(const Position<double>& k) const 
    {
        int N=_bvec.getOrder();
        LVector psi(N);
        psi.fillBasis(k.x*_sigma, k.y*_sigma);  // Fourier[Psi_pq] is unitless
        // rotate kvalues of Psi with i^(p+q)
        // dotting b_pq with psi in k-space:
        double rr=0.;
        double ii=0.;
        {
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
        }
        // difference in Fourier convention with FFTW ???
        return std::complex<double>(2.*M_PI*rr, 2.*M_PI*ii);
    }

    double SBLaguerre::SBLaguerreImpl::getFlux() const 
    {
        double flux=0.;
        for (PQIndex pp(0,0); !pp.pastOrder(_bvec.getOrder()); pp.incN())
            flux += _bvec[pp].real();  // _bvec[pp] is real, but need type conv.
        return flux;
    }


    // SBSersic Class 
    // First need to define the static member that holds info on all the Sersic n's
    SBSersic::InfoBarn SBSersic::nmap;

    SBSersic::SBSersicImpl::SBSersicImpl(double n,  double re, double flux) :
        _n(n), _flux(flux), _re(re), _re_sq(_re*_re), _norm(_flux/_re_sq),
        _info(nmap.get(_n))
    {
        _ksq_max = _info->getKsqMax() * _re_sq;
    }

    double SBSersic::SBSersicImpl::xValue(const Position<double>& p) const
    {  return _norm * _info->xValue((p.x*p.x+p.y*p.y)/_re_sq); }

    std::complex<double> SBSersic::SBSersicImpl::kValue(const Position<double>& k) const
    { 
        double ksq = k.x*k.x + k.y*k.y;
        if (ksq > _ksq_max) 
            return 0.;
        else
            return _flux * _info->kValue(ksq * _re_sq);
    }

    double SBSersic::SBSersicImpl::maxK() const { return _info->maxK() / _re; }
    double SBSersic::SBSersicImpl::stepK() const { return _info->stepK() / _re; }

    double SBSersic::SersicInfo::xValue(double xsq) const 
    { return _norm * std::exp(-_b*std::pow(xsq,_inv2n)); }

    double SBSersic::SersicInfo::kValue(double ksq) const 
    {
        assert(ksq >= 0.);

        if (ksq>=_ksq_max)
            return 0.; // truncate the Fourier transform
        else if (ksq<_ksq_min)
            return 1. + ksq*(_kderiv2 + ksq*_kderiv4); // Use quartic approx at low k
        else {
            double lk=0.5*std::log(ksq); // Lookup table is logarithmic
            return _ft(lk);
        }
    }

    // Integrand class for the Hankel transform of Sersic
    class SersicIntegrand : public std::unary_function<double,double>
    {
    public:
        SersicIntegrand(double n, double b, double k):
            _invn(1./n), _b(b), _k(k) {}
        double operator()(double r) const 
        { return r*std::exp(-_b*std::pow(r, _invn))*j0(_k*r); }

    private:
        double _invn;
        double _b;
        double _k;
    };

    // Find what radius encloses (1-missing_flux_frac) of the total flux in a Sersic profile
    double SBSersic::SersicInfo::findMaxR(double missing_flux_frac, double gamma2n)
    { 
        // int(exp(-b r^1/n) r, r=R..inf) = x * int(exp(-b r^1/n) r, r=0..inf)
        //                                = x n b^-2n Gamma(2n)
        // Change variables: u = b r^1/n,
        // du = b/n r^(1-n)/n dr
        //    = b/n r^1/n dr/r
        //    = u/n dr/r
        // r dr = n du r^2 / u
        //      = n du (u/b)^2n / u
        // n b^-2n int(u^(2n-1) exp(-u), u=bR^1/n..inf) = x n b^-2n Gamma(2n)
        // Let z = b R^1/n
        //
        // int(u^(2n-1) exp(-u), u=z..inf) = x Gamma(2n)
        //
        // The lhs is an incomplete gamma function: Gamma(2n,z), which according to
        // Abramowitz & Stegun (6.5.32) has a high-z asymptotic form of:
        // Gamma(2n,z) ~= z^(2n-1) exp(-z) (1 + (2n-2)/z + (2n-2)(2n-3)/z^2 + ... )
        // ln(x Gamma(2n)) = (2n-1) ln(z) - z + 2(n-1)/z + 2(n-1)(n-2)/z^2
        // z = -ln(x Gamma(2n) + (2n-1) ln(z) + 2(n-1)/z + 2(n-1)(n-2)/z^2
        // Iterate this until converge.  Should be quick.
        dbg<<"Find maxR for missing_flux_frac = "<<missing_flux_frac<<std::endl;
        double z0 = -std::log(missing_flux_frac * gamma2n);
        // Successive approximation method:
        double z = 4.*(_n+1.);  // A decent starting guess for a range of n.
        double oldz = 0.;
        const int MAXIT = 15;
        dbg<<"Start with z = "<<z<<std::endl;
        for(int niter=0; niter < MAXIT; ++niter) {
            oldz = z;
            z = z0 + (2.*_n-1.) * std::log(z) + 2.*(_n-1.)/z + 2.*(_n-1.)*(_n-2.)/(z*z);
            dbg<<"z = "<<z<<", dz = "<<z-oldz<<std::endl;
            if (std::abs(z-oldz) < 0.01) break;
        }
        dbg<<"Converged at z = "<<z<<std::endl;
        double R=std::pow(z/_b, _n);
        dbg<<"R = (z/b)^n = "<<R<<std::endl;
        return R;
    }

    // Constructor to initialize Sersic constants and k lookup table
    SBSersic::SersicInfo::SersicInfo(double n) : _n(n), _inv2n(1./(2.*n)) 
    {
        // Going to constraint range of allowed n to those I have looked at
        if (_n<0.5 || _n>4.2) throw SBError("Requested Sersic index out of range");

        // Formula for b from Ciotti & Bertin (1999)
        _b = 2.*_n - (1./3.)
            + (4./405.)/_n
            + (46./25515.)/(_n*_n)
            + (131./1148175.)/(_n*_n*_n)
            - (2194697./30690717750.)/(_n*_n*_n*_n);

        double b2n = std::pow(_b,2.*_n);  // used frequently here
        double b4n = b2n*b2n;
        // The normalization factor to give unity flux integral:
        double gamma2n = tgamma(2.*_n);
        _norm = b2n / (2.*M_PI*_n*gamma2n);

        // The small-k expansion of the Hankel transform is (normalized to have flux=1):
        // 1 - Gamma(4n) / 4 b^2n Gamma(2n) + Gamma(6n) / 64 b^4n Gamma(2n)
        //   - Gamma(8n) / 2304 b^6n Gamma(2n)
        // The quadratic term of small-k expansion:
        _kderiv2 = -tgamma(4.*_n) / (4.*b2n*gamma2n) ; 
        // And a quartic term:
        _kderiv4 = tgamma(6.*_n) / (64.*b4n*gamma2n);

        dbg << "Building for n=" << _n << " b= " << _b << " norm= " << _norm << std::endl;
        dbg << "Deriv terms: " << _kderiv2 << " " << _kderiv4 << std::endl;

        // When is it safe to use low-k approximation?  
        // See when next term past quartic is at accuracy threshold
        double kderiv6 = tgamma(8*_n) / (2304.*b4n*b2n*gamma2n);
        dbg<<"kderiv6 = "<<kderiv6<<std::endl;
        double kmin = std::pow(sbp::kvalue_accuracy / kderiv6, 1./6.);
        dbg<<"kmin = "<<kmin<<std::endl;
        _ksq_min = kmin * kmin;

        // How far should nominal profile extend?
        // Estimate number of effective radii needed to enclose (1-alias_threshold) of flux
        double R = findMaxR(sbp::alias_threshold,gamma2n);
        // Go to at least 5 re
        if (R < 5) R = 5;
        dbg<<"R => "<<R<<std::endl;
        _stepK = M_PI / R;
        dbg<<"stepK = "<<_stepK<<std::endl;

        // Now start building the lookup table for FT of the profile.
        
        // Normalization for integral at k=0:
        double hankel_norm = _n*gamma2n/b2n;
        dbg<<"hankel_norm = "<<hankel_norm<<std::endl;

        // Along the way, find the last k that has a kValue > 1.e-3
        double maxlogk = 0.;
        // Keep going until at least 5 in a row have kvalues below kvalue_accuracy.
        int n_below_thresh = 0;

        double integ_maxR = findMaxR(sbp::kvalue_accuracy * hankel_norm,gamma2n);
        //double integ_maxR = integ::MOCK_INF;

        double dlogk = 0.1;
        // Don't go past k = 500
        for (double logk = std::log(kmin)-0.001; logk < std::log(500.); logk += dlogk) {
            SersicIntegrand I(_n, _b, std::exp(logk));
            double val = integ::int1d(
                I, 0., integ_maxR, sbp::integration_relerr, sbp::integration_abserr*hankel_norm);
            val /= hankel_norm;
            xdbg<<"logk = "<<logk<<", ft("<<exp(logk)<<") = "<<val<<std::endl;
            _ft.addEntry(logk,val);

            if (std::abs(val) > sbp::maxk_threshold) maxlogk = logk;

            if (std::abs(val) > sbp::kvalue_accuracy) n_below_thresh = 0;
            else ++n_below_thresh;
            if (n_below_thresh == 5) break;
        }
        _maxK = exp(maxlogk);
        xdbg<<"maxK with val >= "<<sbp::maxk_threshold<<" = "<<_maxK<<std::endl;
        _ksq_max = exp(_ft.argMax());

        // Next, set up the classes for photon shooting
        _radial.reset(new SersicRadialFunction(_n, _b));
        std::vector<double> range(2,0.);
        range[1] = R;
        _sampler.reset(new OneDimensionalDeviate( *_radial, range, true));
    }

    PhotonArray SBSersic::SersicInfo::shoot(int N, UniformDeviate& ud) const 
    {
        PhotonArray result = _sampler->shoot(N,ud);
        result.scaleFlux(_norm);
        return result;
    }

    SBMoffat::SBMoffatImpl::SBMoffatImpl(double beta, double size, RadiusType rType,
                                         double trunc, double flux) : 
        _beta(beta), _flux(flux), _trunc(trunc), _ft(Table<double,double>::spline)
    {
        xdbg<<"Start SBMoffat constructor: \n";
        xdbg<<"beta = "<<_beta<<"\n";
        xdbg<<"flux = "<<_flux<<"\n";
        xdbg<<"trunc = "<<_trunc<<"\n";

        // First, relation between FWHM and rD:
        double FWHMrD = 2.* sqrt(std::pow(2., 1./_beta)-1.);
        xdbg<<"FWHMrD = "<<FWHMrD<<"\n";

        // Set size of this instance according to type of size given in constructor:
        switch (rType) {
          case FWHM:
               _rD = size / FWHMrD;
               break;
          case HALF_LIGHT_RADIUS: 
               {
		   // Note that _fluxFactor is needed for calculating the half light radius,
                   // and thus _rD, and with trunc in physical units _rD is needed to calculate the
                   // _fluxFactor via maxRrD!! Need to think about how to break this circularity...
                   if (trunc > 0.){
                       throw SBError("Setting Moffat size via half light radius not supported for "
                                     "trunc > 0.");
                   } else {
                       // If trunc = 0., calculate half-light radius in units of rD:
                   // OLD: double rerD = sqrt( std::pow(1.-0.5*_fluxFactor , 1./(1.-_beta)) - 1.);
                       double rerD = sqrt( std::pow(1. - 0.5, 1./(1.-_beta)) - 1.);
                       _rD = size / rerD;
                   }
               }
               break;
          case SCALE_RADIUS:
               _rD = size;
               break;
          default:
               throw SBError("Unknown SBMoffat::RadiusType");
        }

        double maxRrD;
        if (trunc > 0.) {
            maxRrD = trunc / _rD;  // note new usage of trunc in physical units requires _rD here
            xdbg<<"maxRrD = "<<maxRrD<<"\n";

            // Analytic integration of total flux:
            _fluxFactor = 1. - std::pow( 1+maxRrD*maxRrD, (1.-_beta));
        } else {
            _fluxFactor = 1.;

            // Set maxRrD to the radius where surface brightness is kvalue_accuracy
            // of center value.  (I know this isn't  a kvalue, but the same level 
            // is probably appropriate here.)
            // (1+R^2)^-beta = kvalue_accuracy
            // And ignore the 1+ part of (1+R^2), so
            maxRrD = std::pow(sbp::kvalue_accuracy,-1./(2.*_beta));
            xdbg<<"Not truncate.  Calculated maxRrD = "<<maxRrD<<"\n";
        }

        _FWHM = FWHMrD * _rD;
        _maxR = maxRrD * _rD;
        _maxR_sq = _maxR * _maxR;
        _rD_sq = _rD * _rD;
        _norm = _flux * (_beta-1.) / (M_PI * _fluxFactor * _rD_sq);

        dbg << "Moffat rD " << _rD << " fluxFactor " << _fluxFactor
            << " norm " << _norm << " maxRrD " << _maxR << std::endl;

        if (_beta == 1) pow_beta = &SBMoffat::pow_1;
        else if (_beta == 2) pow_beta = &SBMoffat::pow_2;
        else if (_beta == 3) pow_beta = &SBMoffat::pow_3;
        else if (_beta == 4) pow_beta = &SBMoffat::pow_4;
        else if (_beta == int(_beta)) pow_beta = &SBMoffat::pow_int;
        else pow_beta = &SBMoffat::pow_gen;

        setupFT();
    }

    double SBMoffat::SBMoffatImpl::getHalfLightRadius() const 
    {
        // BARNEY NOTE: A little inefficient to calculate rerD every getHLR call?  Done since
        // rerD depends on _fluxFactor and thus requires _rD in advance, so this needs to happen
        // largely post setup. And it doesn't seem efficient to ALWAYS calculate it above...
        // Get half-light radius in units of rD:
        double rerD = sqrt(std::pow(1.-0.5*_fluxFactor , 1./(1.-_beta)) - 1.);
        return _rD * rerD;
    }

    double SBMoffat::SBMoffatImpl::xValue(const Position<double>& p) const 
    {
        double rsq = p.x*p.x + p.y*p.y;
        if (rsq > _maxR_sq) return 0.;
        else return _norm / pow_beta(1.+rsq/_rD_sq, _beta);
    }

    std::complex<double> SBMoffat::SBMoffatImpl::kValue(const Position<double>& k) const 
    {
        double ksq = k.x*k.x + k.y*k.y;
        if (ksq > _ft.argMax()) return 0.;
        else return _ft(ksq);
    }

    // Set maxK to the value where the FT is down to maxk_threshold
    double SBMoffat::SBMoffatImpl::maxK() const 
    {
        // _maxK is determined during setupFT() as the last k value to have a  kValue > 1.e-3.
#if 1
        return _maxK;
#else
        // Old version from Gary:
        // Require at least 16 points across FWHM when drawing:
        return 16.*M_PI / _FWHM;
#endif
    }

    // The amount of flux missed in a circle of radius pi/stepk should miss at 
    // most alias_threshold of the flux.
    double SBMoffat::SBMoffatImpl::stepK() const
    {
        dbg<<"Find Moffat stepK\n";
        dbg<<"beta = "<<_beta<<std::endl;
#if 1
        // The fractional flux out to radius R is (if not truncated)
        // 1 - (1+R^2)^(1-beta)
        // So solve (1+R^2)^(1-beta) = alias_threshold
        if (_beta <= 1.1) {
            // Then flux never converges (or nearly so), so just use truncation radius
            return M_PI / _maxR;
        } else {
            // Ignore the 1 in (1+R^2), so approximately:
            double R = std::pow(sbp::alias_threshold, 0.5/(1.-_beta)) * _rD;
            dbg<<"R = "<<R<<std::endl;
            // If it is truncated at less than this, drop to that value.
            if (R > _maxR) R = _maxR;
            dbg<<"_maxR = "<<_maxR<<std::endl;
            dbg<<"R => "<<R<<std::endl;
            dbg<<"stepk = "<<(M_PI/R)<<std::endl;
            return M_PI / R;
        }
#else
        // Old version from Gary:
        // Make FFT's periodic at 4x truncation radius or 1.5x diam at alias_threshold,
        // whichever is smaller
        return 2.*M_PI / std::min(4.*_maxR, 
                                  3.*sqrt(std::pow(sbp::alias_threshold, -1./_beta)-1.)*_rD);
#endif
    }

    // Integrand class for the Hankel transform of Moffat
    class MoffatIntegrand : public std::unary_function<double,double>
    {
    public:
        MoffatIntegrand(double beta, double k, double (*pb)(double, double)) : 
            _beta(beta), _k(k), pow_beta(pb) {}
        double operator()(double r) const 
        { return r/pow_beta(1.+r*r, _beta)*j0(_k*r); }

    private:
        double _beta;
        double _k;
        double (*pow_beta)(double x, double beta);
    };

    void SBMoffat::SBMoffatImpl::setupFT()
    {
        if (_ft.size() > 0) return;

        // Do a Hankel transform and store the results in a lookup table.
        
        double nn = _norm * 2.*M_PI * _rD_sq;
        //double maxR = _fluxFactor == 1. ? integ::MOCK_INF : _maxR / _rD;
        double maxR = _maxR / _rD;

        // Along the way, find the last k that has a kValue > 1.e-3
        double maxK_val = sbp::maxk_threshold * _flux;
        dbg<<"Looking for maxK_val = "<<maxK_val<<std::endl;
        // Keep going until at least 5 in a row have kvalues below kvalue_accuracy.
        // (It's oscillatory, so want to make sure not to stop at a zero crossing.)
        double thresh = sbp::kvalue_accuracy * _flux;

        // These are dimensionless k values for doing the integral.
        double dk = 0.1;
        dbg<<"dk = "<<dk<<std::endl;
        int n_below_thresh = 0;
        // Don't go past k = 50
        for(double k=0.; k < 50; k += dk) {
            MoffatIntegrand I(_beta, k, pow_beta);
            double val = integ::int1d(
                I, 0., maxR, sbp::integration_relerr, sbp::integration_abserr);
            val *= nn;

            double kreal = k / _rD;
            xdbg<<"ft("<<kreal<<") = "<<val<<std::endl;
            _ft.addEntry( kreal*kreal, val );

            if (std::abs(val) > maxK_val) _maxK = kreal;

            if (std::abs(val) > thresh) n_below_thresh = 0;
            else ++n_below_thresh;
            if (n_below_thresh == 5) break;
        }
        dbg<<"maxK = "<<_maxK<<std::endl;
    }

    /*************************************************************
     * Photon-shooting routines
     *************************************************************/

    template <class T>
    void SBProfile::drawShoot(ImageView<T> img, double N, UniformDeviate& u) const 
    {
        const int maxN = 100000;

        // Clear image before adding photons, for consistency with draw() methods.
        img.fill(0.);  
        double origN = N;
        xdbg<<"origN = "<<origN<<std::endl;
        while (N > maxN) {
            xdbg<<"shoot "<<maxN<<std::endl;
            assert(_pimpl.get());
            PhotonArray pa = _pimpl->shoot(maxN, u);
            pa.scaleFlux(maxN / origN);
            pa.addTo(img);
            N -= maxN;
        }
        xdbg<<"shoot "<<N<<std::endl;
        assert(_pimpl.get());
        PhotonArray pa = _pimpl->shoot(int(N), u);
        pa.scaleFlux(N / origN);
        pa.addTo(img);
    }
    
    PhotonArray SBAdd::SBAddImpl::shoot(int N, UniformDeviate& u) const 
    {
        double totalAbsoluteFlux = getPositiveFlux() + getNegativeFlux();
        double fluxPerPhoton = totalAbsoluteFlux / N;

        // Initialize the output array
        PhotonArray result(0);
        result.reserve(N);

        double remainingAbsoluteFlux = totalAbsoluteFlux;
        int remainingN = N;

        // Get photons from each summand, using BinomialDeviate to
        // randomize distribution of photons among summands
        for (ConstIter pptr = _plist.begin(); pptr!= _plist.end(); ++pptr) {
            double thisAbsoluteFlux = pptr->getPositiveFlux() + pptr->getNegativeFlux();

            // How many photons to shoot from this summand?
            int thisN = remainingN;  // All of what's left, if this is the last summand...
            std::list<SBProfile>::const_iterator nextPtr = pptr;
            ++nextPtr;
            if (nextPtr!=_plist.end()) {
                // otherwise allocate a randomized fraction of the remaining photons to this summand:
                BinomialDeviate bd(u, remainingN, thisAbsoluteFlux/remainingAbsoluteFlux);
                thisN = bd();
            }
            if (thisN > 0) {
                PhotonArray thisPA = pptr->shoot(thisN, u);
                // Now rescale the photon fluxes so that they are each nominally fluxPerPhoton
                // whereas the shoot() routine would have made them each nominally 
                // thisAbsoluteFlux/thisN
                thisPA.scaleFlux(fluxPerPhoton*thisN/thisAbsoluteFlux);
                result.append(thisPA);
            }
            remainingN -= thisN;
            remainingAbsoluteFlux -= thisAbsoluteFlux;
            if (remainingN <=0) break;
            if (remainingAbsoluteFlux <= 0.) break;
        }
        
        return result;
    }

    PhotonArray SBConvolve::SBConvolveImpl::shoot(int N, UniformDeviate& u) const 
    {
        std::list<SBProfile>::const_iterator pptr = _plist.begin();
        if (pptr==_plist.end())
            throw SBError("Cannot shoot() for empty SBConvolve");
        PhotonArray result = pptr->shoot(N, u);
        // It is necessary to shuffle when convolving because we do
        // do not have a gaurantee that the convolvee's photons are
        // uncorrelated, e.g. they might both have their negative ones
        // at the end.
        for (++pptr; pptr != _plist.end(); ++pptr)
            result.convolveShuffle( pptr->shoot(N, u), u);
        return result;
    }

    PhotonArray SBDistort::SBDistortImpl::shoot(int N, UniformDeviate& u) const 
    {
        // Simple job here: just remap coords of each photon, then change flux
        // If there is overall magnification in the transform
        PhotonArray result = _adaptee.shoot(N,u);
        for (int i=0; i<result.size(); i++) {
            Position<double> xy = fwd(Position<double>(result.getX(i), result.getY(i))+_cen);
            result.setPhoton(i,xy.x, xy.y, result.getFlux(i)*_absdet);
        }
        return result;
    }

    PhotonArray SBGaussian::SBGaussianImpl::shoot(int N, UniformDeviate& u) const 
    {
        PhotonArray result(N);
        double fluxPerPhoton = _flux/N;
        for (int i=0; i<N; i++) {
            // First get a point uniformly distributed on unit circle
            double xu, yu, rsq;
            do {
                xu = 2.*u()-1.;
                yu = 2.*u()-1.;
                rsq = xu*xu+yu*yu;
            } while (rsq>=1. || rsq==0.);
            
            // Then map it to desired Gaussian with analytic transformation
            double factor = _sigma*sqrt( -2.*std::log(rsq)/rsq);
            result.setPhoton(i,factor*xu, factor*yu, fluxPerPhoton);
        }
        return result;
    }

    PhotonArray SBSersic::SBSersicImpl::shoot(int N, UniformDeviate& ud) const
    {
        // Get photons from the SersicInfo structure, rescale flux and size for this instance
        PhotonArray result = _info->shoot(N,ud);
        result.scaleFlux(_flux);
        result.scaleXY(_re);
        return result;
    }

    PhotonArray SBExponential::SBExponentialImpl::shoot(int N, UniformDeviate& u) const
    {
        // Accuracy to which to solve for (log of) cumulative flux distribution:
        const double Y_TOLERANCE=1e-6;

        double fluxPerPhoton = getFlux() / N;
        PhotonArray result(N);
        // The cumulative distribution of flux is 1-(1+r)exp(-r).
        // Here is a way to solve for r by an initial guess followed
        // by Newton-Raphson iterations.  Probably not
        // the most efficient thing since there are logs in the iteration.
        for (int i=0; i<N; i++) {
            double y = u();
            if (y==0.) {
                // In case of infinite radius - just set to origin:
                result.setPhoton(i,0.,0.,fluxPerPhoton);
                continue;
            }
            // Initial guess
            y = -std::log(y);
            double r = y>2 ? y : sqrt(2*y);
            double dy = y - r + std::log(1+r);
            while ( std::abs(dy) > Y_TOLERANCE) {
                r = r + (1+r)*dy/r;
                dy = y - r + std::log(1+r);
            }
            // Draw another random for azimuthal angle (could use the unit-circle trick here...)
            double theta = 2*M_PI*u();
            result.setPhoton(i,_r0*r*std::cos(theta), _r0*r*std::sin(theta), fluxPerPhoton);
        }
        return result;
    }

    PhotonArray SBAiry::SBAiryImpl::shoot(int N, UniformDeviate& u) const
    {
        // Use the OneDimensionalDeviate to sample from scale-free distribution
        checkSampler();
        PhotonArray pa=_sampler->shoot(N, u);
        // Then rescale for this flux & size
        pa.scaleFlux(_flux);
        pa.scaleXY(1./_D);
        return pa;
    }

    void SBAiry::SBAiryImpl::flushSampler() const 
    { _sampler.reset(); }

    void SBAiry::SBAiryImpl::checkSampler() const 
    {
        if (_sampler.get()) return;
        std::vector<double> ranges(1,0.);
        // Break Airy function into ranges that will not have >1 extremum:
        double xmin = (1.1 - 0.5*_obscuration);
        // Use Schroeder (10.1.18) limit of EE at large radius.
        // to stop sampler at radius with EE>(1-alias_threshold).
        // TODO: Is alias_threshold appropriate for photon shooting?  
        //       Should we introduce another parameter that might work similarly
        //       and try to be consistent across all profiles?
        //       Or maybe the same new xvalue_accuracy I proposed above.
        double maximumRadius = 2./(sbp::alias_threshold * M_PI*M_PI * (1-_obscuration));
        while (xmin < maximumRadius) {
            ranges.push_back(xmin);
            xmin += 0.5;
        }
        ranges.push_back(xmin);
        _sampler.reset(new OneDimensionalDeviate(_radial, ranges, true));
    }

    PhotonArray SBBox::SBBoxImpl::shoot(int N, UniformDeviate& u) const
    {
        PhotonArray result(N);
        for (int i=0; i<result.size(); i++)
            result.setPhoton(i, _xw*(u()-0.5), _yw*(u()-0.5), _flux/N);
        return result;
    }

    PhotonArray SBMoffat::SBMoffatImpl::shoot(int N, UniformDeviate& u) const
    {
        // Moffat has analytic inverse-cumulative-flux function.
        PhotonArray result(N);
        double fluxPerPhoton = _flux/N;
        for (int i=0; i<N; i++) {
            // First get a point uniformly distributed on unit circle
            double xu, yu, rsq;
            do {
                xu = 2.*u()-1.;
                yu = 2.*u()-1.;
                rsq = xu*xu+yu*yu;
            } while (rsq>=1. || rsq==0.);
            
            // Then map it to the Moffat flux distribution
            double newRsq = std::pow( 1.-rsq*_fluxFactor , 1./(1.-_beta)) - 1.;
            double rFactor = _rD*sqrt(newRsq / rsq);
            result.setPhoton(i,rFactor*xu, rFactor*yu, fluxPerPhoton);
        }
        return result;
    }

    // instantiate template functions for expected image types
    template double SBProfile::SBProfileImpl::doFillXImage2(ImageView<float>& img, double dx) const;
    template double SBProfile::SBProfileImpl::doFillXImage2(ImageView<double>& img, double dx) const;

    template void SBProfile::drawShoot(ImageView<float> image, double N, UniformDeviate& ud) const;
    template void SBProfile::drawShoot(ImageView<double> image, double N, UniformDeviate& ud) const;
    template void SBProfile::drawShoot(Image<float>& image, double N, UniformDeviate& ud) const;
    template void SBProfile::drawShoot(Image<double>& image, double N, UniformDeviate& ud) const;

    template double SBProfile::draw(Image<float>& img, double dx, int wmult) const;
    template double SBProfile::draw(Image<double>& img, double dx, int wmult) const;
    template double SBProfile::draw(ImageView<float>& img, double dx, int wmult) const;
    template double SBProfile::draw(ImageView<double>& img, double dx, int wmult) const;

    template double SBProfile::plainDraw(Image<float>& I, double dx, int wmult) const;
    template double SBProfile::plainDraw(Image<double>& I, double dx, int wmult) const;
    template double SBProfile::plainDraw(ImageView<float>& I, double dx, int wmult) const;
    template double SBProfile::plainDraw(ImageView<double>& I, double dx, int wmult) const;

    template double SBProfile::fourierDraw(Image<float>& I, double dx, int wmult) const;
    template double SBProfile::fourierDraw(Image<double>& I, double dx, int wmult) const;
    template double SBProfile::fourierDraw(ImageView<float>& I, double dx, int wmult) const;
    template double SBProfile::fourierDraw(ImageView<double>& I, double dx, int wmult) const;

    template void SBProfile::drawK(
        Image<float>& Re, Image<float>& Im, double dk, int wmult) const;
    template void SBProfile::drawK(
        Image<double>& Re, Image<double>& Im, double dk, int wmult) const;
    template void SBProfile::drawK(
        ImageView<float>& Re, ImageView<float>& Im, double dk, int wmult) const;
    template void SBProfile::drawK(
        ImageView<double>& Re, ImageView<double>& Im, double dk, int wmult) const;

    template void SBProfile::plainDrawK(
        Image<float>& Re, Image<float>& Im, double dk, int wmult) const;
    template void SBProfile::plainDrawK(
        Image<double>& Re, Image<double>& Im, double dk, int wmult) const;
    template void SBProfile::plainDrawK(
        ImageView<float>& Re, ImageView<float>& Im, double dk, int wmult) const;
    template void SBProfile::plainDrawK(
        ImageView<double>& Re, ImageView<double>& Im, double dk, int wmult) const;

    template void SBProfile::fourierDrawK(
        Image<float>& Re, Image<float>& Im, double dk, int wmult) const;
    template void SBProfile::fourierDrawK(
        Image<double>& Re, Image<double>& Im, double dk, int wmult) const;
    template void SBProfile::fourierDrawK(
        ImageView<float>& Re, ImageView<float>& Im, double dk, int wmult) const;
    template void SBProfile::fourierDrawK(
        ImageView<double>& Re, ImageView<double>& Im, double dk, int wmult) const;

}
