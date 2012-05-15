//
// Functions for the Surface Brightness Profile Class
//
#include "SBProfile.h"
#include "integ/Int.h"
#include "TMV.h"
#include "Solve.h"
#include "integ/Int.h"

#include <fstream>

#ifdef DEBUGLOGGING
//std::ostream* dbgout = new std::ofstream("debug.out");
//std::ostream* dbgout = &std::cerr;
std::ostream* dbgout = 0;
int verbose_level = 0;
#endif

namespace galsim {

    // ????? Change treatement of aliased images to simply add in the aliased
    // FT components instead of doing a larger FT and then subsampling!
    // ??? Make a formula for asymptotic high-k SBSersic::kValue ??

    // Parameters controlling behavior of all classes:
    const int SBProfile::MINIMUM_FFT_SIZE = 128;
    const int SBProfile::MAXIMUM_FFT_SIZE = 4096;
    // Allow aliasing of Fourier modes below this amplitude, roughly.
    // Also set the FFT image size such that this fraction of flux (or less) is "folded."
    const double SBProfile::ALIAS_THRESHOLD = 0.005;


    //
    // Virtual methods of Base Class "SBProfile"
    //

    SBProfile* SBProfile::distort(const Ellipse e) const 
    { return new SBDistort(*this,e); }

    SBProfile* SBProfile::rotate(Angle theta) const 
    {
        return new SBDistort(*this,
                             std::cos(theta.rad()),-std::sin(theta.rad()),
                             std::sin(theta.rad()),std::cos(theta.rad())); 
    }

    SBProfile* SBProfile::shift(double dx, double dy) const 
    { return new SBDistort(*this,1.,0.,0.,1., Position<double>(dx,dy)); }

    //
    // Common methods of Base Class "SBProfile"
    //

#ifdef USE_IMAGES
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
        if (dx<=0.) dx = M_PI / maxK();
        // recenter an existing image, to be consistent with fourierDraw:
        int xSize = I.getXMax()-I.getXMin()+1, ySize = I.getYMax()-I.getYMin()+1;
        I.setOrigin(-xSize/2, -ySize/2);

        return fillXImage(I, dx);
    }

    template <typename T>
    double SBProfile::plainDraw(Image<T>& I, double dx, int wmult) const 
    {
        dbg<<"Start plainDraw Image"<<std::endl;
        // Determine desired dx:
        if (dx<=0.) dx = M_PI / maxK();
        if (!I.getBounds().isDefined()) {
            if (wmult<1) throw SBError("Requested wmult<1 in plainDraw()");
            // Need to choose an image size
            int N = static_cast<int> (std::ceil(2*M_PI/(dx*stepK())));

            // Round up to an even value
            N = 2*( (N+1)/2);
            N *= wmult; // make even bigger if desired
            Bounds<int> imgsize(-N/2, N/2-1, -N/2, N/2-1);
            I.resize(imgsize);
        } else {
            // recenter an existing image, to be consistent with fourierDraw:
            int xSize = I.getXMax()-I.getXMin()+1, ySize = I.getYMax()-I.getYMin()+1;
            I.setOrigin(-xSize/2, -ySize/2);
        }

        // TODO: If we decide not to keep the scale, then can switch to simply:
        // return fillXImage(I.view(), dx);
        // (And switch fillXImage to take a const ImageView<T>& argument.)
        ImageView<T> Iv = I.view();
        double ret = fillXImage(Iv, dx);
        I.setScale(Iv.getScale());
        return ret;
    }
 
    template <typename T>
    double SBProfile::doFillXImage2(ImageView<T>& I, double dx) const 
    {
        dbg<<"Start doFillXImage2"<<std::endl;
        double totalflux=0;
        for (int y = I.getYMin(); y <= I.getYMax(); y++) {
            int x = I.getXMin(); 
            typename Image<T>::iterator ee=I.rowEnd(y);
            for (typename Image<T>::iterator it=I.rowBegin(y);
                 it!=ee;
                 ++it, ++x) {
                Position<double> p(x*dx,y*dx); // since x,y are pixel indices
                *it = xValue(p);
#ifdef DANIELS_TRACING
                cout << "x=" << x << ", y=" << y << ": " << *it << std::endl;
                cout << "--------------------------" << std::endl;
#endif
                totalflux += *it;
            } 
        }
        I.setScale(dx);
        return totalflux * (dx*dx);
    }

    // Now the more complex case: real space via FT from k space.
    // Will enforce image size is power of 2 or 3x2^n.
    // Aliasing will be handled by folding the k values before transforming
    // And enforce no image folding
    //**/ #define DEBUG
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

#ifdef DEBUG
        std::cerr << " maxK() " << maxK() << " dx " << dx << std::endl;
#endif

        // Now decide how big the FT must be to avoid folding:
        double xRange = 2*M_PI*wmult / stepK();
        int Nnofold = static_cast<int> (std::ceil(xRange / dx -0.0001));
#ifdef DEBUG
        std::cerr << " stepK() " << stepK() << " Nnofold " << Nnofold << std::endl;
#endif

        // W must make something big enough to cover the target image size:
        int xSize, ySize;
        xSize = I.getXMax()-I.getXMin()+1;
        ySize = I.getYMax()-I.getYMin()+1;
        if (xSize  > Nnofold) Nnofold = xSize;
        if (ySize  > Nnofold) Nnofold = ySize;
        xRange = Nnofold * dx;

        // Round up to a good size for making FFTs:
        int NFT = goodFFTSize(Nnofold);
        NFT = std::max(NFT,MINIMUM_FFT_SIZE);
#ifdef DEBUG
        std::cerr << " After adjustments: Nnofold " << Nnofold << " NFT " << NFT << std::endl;
#endif
        if (NFT > MAXIMUM_FFT_SIZE)
            FormatAndThrow<SBError>() << "fourierDraw() requires an FFT that is too large, " << NFT;

        // Move the output image to be centered near zero
        I.setOrigin(-xSize/2, -ySize/2);
        double dk = 2.*M_PI/(NFT*dx);
#ifdef DEBUG
        std::cerr << 
            " After adjustments: dx " << dx << " dk " << dk << 
            " maxK " << dk*NFT/2 << std::endl;
#endif
        assert(dk <= stepK());
        XTable* xtmp=0;
        if (NFT*dk/2 > maxK()) {
            // No aliasing: build KTable and transform
            KTable kt(NFT,dk);
            fillKGrid(kt); 
            xtmp = kt.transform();
        } else {
            // There will be aliasing.  Construct a KTable out to maxK() and
            // then wrap it
            int Nk = static_cast<int> (std::ceil(maxK()/dk)) * 2;
            KTable kt(Nk, dk);
            fillKGrid(kt);
            KTable* kt2 = kt.wrap(NFT);
            xtmp = kt2->transform();
            delete kt2;
        }
        int Nxt = xtmp->getN();
        Bounds<int> xb(-Nxt/2, Nxt/2-1, -Nxt/2, Nxt/2-1);
        if (I.getYMin() < xb.getYMin()
            || I.getYMax() > xb.getYMax()
            || I.getXMin() < xb.getXMin()
            || I.getXMax() > xb.getXMax()) {
            std::cerr << "Bounds error!! target image bounds " << I.getBounds()
                << " and FFT range " << xb
                << std::endl;
            throw SBError("fourierDraw() FT bounds do not cover target image");
        }
        double sum=0.;
        for (int y = I.getYMin(); y <= I.getYMax(); y++)
            for (int x = I.getXMin(); x <= I.getXMax(); x++) {
                I(x,y) = xtmp->xval(x,y);
                sum += I(x,y);
            }

        I.setScale(dx);

        delete xtmp;  // no memory leak!
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

#ifdef DEBUG
        std::cerr << " maxK() " << maxK() << " dx " << dx << std::endl;
#endif

        // Now decide how big the FT must be to avoid folding:
        double xRange = 2*M_PI*wmult / stepK();
        int Nnofold = static_cast<int> (std::ceil(xRange / dx -0.0001));
#ifdef DEBUG
        std::cerr << " stepK() " << stepK() << " Nnofold " << Nnofold << std::endl;
#endif

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
        NFT = std::max(NFT,MINIMUM_FFT_SIZE);
#ifdef DEBUG
        std::cerr << " After adjustments: Nnofold " << Nnofold << " NFT " << NFT << std::endl;
#endif
        if (NFT > MAXIMUM_FFT_SIZE)
            FormatAndThrow<SBError>() << "fourierDraw() requires an FFT that is too large, " << NFT;

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
#ifdef DEBUG
        std::cerr << 
            " After adjustments: dx " << dx << " dk " << dk << 
            " maxK " << dk*NFT/2 << std::endl;
#endif
        assert(dk <= stepK());
        XTable* xtmp=0;
        if (NFT*dk/2 > maxK()) {
            // No aliasing: build KTable and transform
            KTable kt(NFT,dk);
            fillKGrid(kt); 
            xtmp = kt.transform();
        } else {
            // There will be aliasing.  Construct a KTable out to maxK() and
            // then wrap it
            int Nk = static_cast<int> (std::ceil(maxK()/dk)) * 2;
            KTable kt(Nk, dk);
            fillKGrid(kt);
            KTable* kt2 = kt.wrap(NFT);
            xtmp = kt2->transform();
            delete kt2;
        }
        int Nxt = xtmp->getN();
        Bounds<int> xb(-Nxt/2, Nxt/2-1, -Nxt/2, Nxt/2-1);
        if (I.getYMin() < xb.getYMin()
            || I.getYMax() > xb.getYMax()
            || I.getXMin() < xb.getXMin()
            || I.getXMax() > xb.getXMax()) {
            std::cerr << "Bounds error!! target image bounds " << I.getBounds()
                << " and FFT range " << xb
                << std::endl;
            throw SBError("fourierDraw() FT bounds do not cover target image");
        }
        double sum=0.;
        for (int y = I.getYMin(); y <= I.getYMax(); y++)
            for (int x = I.getXMin(); x <= I.getXMax(); x++) {
                I(x,y) = xtmp->xval(x,y);
                sum += I(x,y);
            }

        I.setScale(dx);

        delete xtmp;  // no memory leak!
        return sum*dx*dx;;
    }

    template <typename T>
    void SBProfile::drawK(ImageView<T>& Re, ImageView<T>& Im, double dk, int wmult) const 
    {
        if (isAnalyticK()) 
            plainDrawK(Re, Im, dk, wmult);   // calculate in k space
        else               
            fourierDrawK(Re, Im, dk, wmult); // calculate via FT from real space
        return;
    }

    template <typename T>
    void SBProfile::drawK(Image<T>& Re, Image<T>& Im, double dk, int wmult) const 
    {
        if (isAnalyticK()) 
            plainDrawK(Re, Im, dk, wmult);   // calculate in k space
        else               
            fourierDrawK(Re, Im, dk, wmult); // calculate via FT from real space
        return;
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
            typename ImageView<T>::iterator ee=Re.rowEnd(y);
            typename ImageView<T>::iterator it;
            typename ImageView<T>::iterator it2;
            for (it=Re.rowBegin(y), it2=Im.rowBegin(y); it!=ee; ++it, ++it2, ++x) {
                Position<double> p(x*dk,y*dk); // since x,y are pixel indicies
                std::complex<double> c = this->kValue(p);  
                *it = c.real(); 
                *it2 = c.imag(); 
            } 
        }

        Re.setScale(dk);
        Im.setScale(dk);

        return;
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
            int N = static_cast<int> (std::ceil(2.*maxK()*wmult / dk));
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
            typename ImageView<T>::iterator ee=Re.rowEnd(y);
            typename ImageView<T>::iterator it;
            typename ImageView<T>::iterator it2;
            for (it=Re.rowBegin(y), it2=Im.rowBegin(y); it!=ee; ++it, ++it2, ++x) {
                Position<double> p(x*dk,y*dk); // since x,y are pixel indicies
                std::complex<double> c = this->kValue(p);  
                *it = c.real(); 
                *it2 = c.imag(); 
            } 
        }

        Re.setScale(dk);
        Im.setScale(dk);

        return;
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
            oversamp = static_cast<int> ( std::ceil(dk/stepK() - 0.0001));
        }

        // Now decide how big the FT must be to avoid folding
        double kRange = 2*maxK()*wmult;
        int Nnofold = static_cast<int> (std::ceil(oversamp*kRange / dk -0.0001));

        // And if there is a target image size, we must make something big enough to cover
        // the target image size:
        int xSize, ySize;
        xSize = Re.getXMax()-Re.getXMin()+1;
        ySize = Re.getYMax()-Re.getYMin()+1;
        if (xSize * oversamp > Nnofold) Nnofold = xSize*oversamp;
        if (ySize * oversamp > Nnofold) Nnofold = ySize*oversamp;
        kRange = Nnofold * dk / oversamp;

        // Round up to a power of 2 to get required FFT size
        int NFT = MINIMUM_FFT_SIZE;
        while (NFT < Nnofold && NFT<= MAXIMUM_FFT_SIZE) NFT *= 2;
        if (NFT > MAXIMUM_FFT_SIZE)
            throw SBError("fourierDrawK() requires an FFT that is too large");

        // Move the output image to be centered near zero
        Re.setOrigin(-xSize/2, -ySize/2);
        Im.setOrigin(-xSize/2, -ySize/2);

        double dx = 2.*M_PI*oversamp/(NFT*dk);
        XTable xt(NFT,dx);
        this->fillXGrid(xt);
        KTable *ktmp = xt.transform();

        int Nkt = ktmp->getN();
        Bounds<int> kb(-Nkt/2, Nkt/2-1, -Nkt/2, Nkt/2-1);
        if (Re.getYMin() < kb.getYMin()
            || Re.getYMax()*oversamp > kb.getYMax()
            || Re.getXMin()*oversamp < kb.getXMin()
            || Re.getXMax()*oversamp > kb.getXMax()) {
            std::cerr << "Bounds error!! oversamp is " << oversamp
                << " target image bounds " << Re.getBounds()
                << " and FFT range " << kb
                << std::endl;
            throw SBError("fourierDrawK() FT bounds do not cover target image");
        }

        for (int y = Re.getYMin(); y <= Re.getYMax(); y++)
            for (int x = Re.getXMin(); x <= Re.getXMax(); x++) {
                Re(x,y) = ktmp->kval(x*oversamp,y*oversamp).real();
                Im(x,y) = ktmp->kval(x*oversamp,y*oversamp).imag();
            }

        Re.setScale(dk);
        Im.setScale(dk);

        delete ktmp;  // no memory leak!
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
            oversamp = static_cast<int> ( std::ceil(dk/stepK() - 0.0001));
            canReduceDk = false; // Force output image to input dx.
        }

        // Now decide how big the FT must be to avoid folding
        double kRange = 2*maxK()*wmult;
        int Nnofold = static_cast<int> (std::ceil(oversamp*kRange / dk -0.0001));

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
        int NFT = MINIMUM_FFT_SIZE;
        while (NFT < Nnofold && NFT<= MAXIMUM_FFT_SIZE) NFT *= 2;
        if (NFT > MAXIMUM_FFT_SIZE)
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
        this->fillXGrid(xt);
        KTable *ktmp = xt.transform();

        int Nkt = ktmp->getN();
        Bounds<int> kb(-Nkt/2, Nkt/2-1, -Nkt/2, Nkt/2-1);
        if (Re.getYMin() < kb.getYMin()
            || Re.getYMax()*oversamp > kb.getYMax()
            || Re.getXMin()*oversamp < kb.getXMin()
            || Re.getXMax()*oversamp > kb.getXMax()) {
            std::cerr << "Bounds error!! oversamp is " << oversamp
                << " target image bounds " << Re.getBounds()
                << " and FFT range " << kb
                << std::endl;
            throw SBError("fourierDrawK() FT bounds do not cover target image");
        }

        for (int y = Re.getYMin(); y <= Re.getYMax(); y++)
            for (int x = Re.getXMin(); x <= Re.getXMax(); x++) {
                Re(x,y) = ktmp->kval(x*oversamp,y*oversamp).real();
                Im(x,y) = ktmp->kval(x*oversamp,y*oversamp).imag();
            }

        Re.setScale(dk);
        Im.setScale(dk);

        delete ktmp;  // no memory leak!
    }

#endif

    void SBProfile::fillXGrid(XTable& xt) const 
    {
        int N = xt.getN();
        double dx = xt.getDx();
        for (int iy = -N/2; iy < N/2; iy++)
            for (int ix = -N/2; ix < N/2; ix++) {
                Position<double> x(ix*dx,iy*dx);
                xt.xSet(ix,iy,xValue(x));
            }
        return;
    }

    void SBProfile::fillKGrid(KTable& kt) const 
    {
        int N = kt.getN();
        double dk = kt.getDk();
        for (int iy = -N/2; iy < N/2; iy++) {
            // Only need ix>=0 because it's Hermitian:
            for (int ix = 0; ix <= N/2; ix++) {
                Position<double> k(ix*dk,iy*dk);
                kt.kSet(ix,iy,kValue(k));
            }
        }
        return;
    }

    //
    // Methods for Derived Classes
    //

    void SBAdd::initialize() 
    {
        sumflux = sumfx = sumfy = 0.;
        maxMaxK = minStepK = 0.;
        minMinX = maxMaxX = 0.;
        minMinY = maxMaxY = 0.;
        allAxisymmetric = allAnalyticX = allAnalyticK = true;
    }

    void SBAdd::add(const SBProfile& rhs, double scale) 
    {
        // Need a non-const copy of the rhs:
        SBProfile* p=rhs.duplicate();


        // Keep track of where first new summand is on list:
        std::list<SBProfile*>::iterator newptr = plist.end();

        // Add new summand(s) to the plist:
        SBAdd *sba = dynamic_cast<SBAdd*> (p);
        if (sba) {  
            // If rhs is an SBAdd, copy its full list here
            std::list<SBProfile*>::const_iterator pptr;
            for (pptr = sba->plist.begin(); pptr!=sba->plist.end(); ++pptr) {
                if (newptr==plist.end()) {
                    plist.push_back((*pptr)->duplicate()); 
                    // Rescale flux for duplicate copy if desired:
                    if (scale!=1.) 
                        plist.back()->setFlux( scale*plist.back()->getFlux());
                    newptr = --plist.end();  // That was first new summand
                } else {
                    plist.push_back((*pptr)->duplicate()); 
                }
            }
            delete sba; // no memory leak! 
        } else {
            plist.push_back(p);
            // Rescale flux for duplicate copy if desired:
            if (scale!=1.) 
                plist.back()->setFlux( scale*plist.back()->getFlux());
            newptr = --plist.end();  // That was first new summand
        }

        // Accumulate properties of all summands
        while (newptr != plist.end()) {
            sumflux += (*newptr)->getFlux();
            sumfx += (*newptr)->getFlux() * (*newptr)->centroid().x;
            sumfy += (*newptr)->getFlux() * (*newptr)->centroid().x;
            if ( (*newptr)->maxK() > maxMaxK) maxMaxK = (*newptr)->maxK();
            if ( minStepK<=0. || ((*newptr)->stepK() < minStepK)) minStepK = (*newptr)->stepK();
            if ( (*newptr)->minX() > minMinX) minMinX = (*newptr)->minX();
            if ( (*newptr)->maxX() > maxMaxX) maxMaxX = (*newptr)->maxX();
            if ( (*newptr)->minY() > minMinY) minMinY = (*newptr)->minY();
            if ( (*newptr)->maxY() > maxMaxY) maxMaxY = (*newptr)->maxY();
            allAxisymmetric = allAxisymmetric && (*newptr)->isAxisymmetric();
            allAnalyticX = allAnalyticX && (*newptr)->isAnalyticX();
            allAnalyticK = allAnalyticK && (*newptr)->isAnalyticK();
            newptr++;
        }
        return; 
    }

    double SBAdd::xValue(Position<double> _p) const 
    {
        double xv = 0.;  
        std::list<SBProfile*>::const_iterator pptr;
        for (pptr = plist.begin(); pptr != plist.end(); ++pptr)
        {
            xv += (*pptr)->xValue(_p);
        }
        return xv;
    } 

    std::complex<double> SBAdd::kValue(Position<double> _p) const 
    {
        std::complex<double> kv = 0.;  
        std::list<SBProfile*>::const_iterator pptr;
        for (pptr = plist.begin(); pptr != plist.end(); ++pptr)
            kv += (*pptr)->kValue(_p);
        return kv;
    } 

    void SBAdd::fillKGrid(KTable& kt) const 
    {
        if (plist.empty()) kt.clear();
        std::list<SBProfile*>::const_iterator pptr = plist.begin();
        (*pptr)->fillKGrid(kt);
        ++pptr;
        if (pptr==plist.end()) return;
        int N = kt.getN();
        double dk = kt.getDk();
        KTable k2(N,dk);
        for ( ; pptr!= plist.end(); ++pptr) {
            (*pptr)->fillKGrid(k2);
            kt.accumulate(k2);
        }
    }

    void SBAdd::fillXGrid(XTable& xt) const 
    {
        if (plist.empty()) xt.clear();
        std::list<SBProfile*>::const_iterator pptr = plist.begin();
        (*pptr)->fillXGrid(xt);
        ++pptr;
        if (pptr==plist.end()) return;
        int N = xt.getN();
        double dx = xt.getDx();
        XTable x2(N,dx);
        for ( ; pptr!= plist.end(); ++pptr) {
            (*pptr)->fillXGrid(x2);
            xt.accumulate(x2);
        }
    }

    void SBAdd::setFlux(double f) 
    {
        if (sumflux==0.) throw SBError("SBAdd::setFlux not possible when flux=0 to start");
        double m = f/getFlux();  // Factor by which to change flux
        std::list<SBProfile*>::iterator pptr; 
        for (pptr = plist.begin(); pptr != plist.end(); ++pptr) {
            double pf = (*pptr)->getFlux();  
            (*pptr)->setFlux(pf*m);
        }
        sumflux *=m;
        sumfx *= m;
        sumfy *= m;
        return;
    }

    //
    // "SBDistort" Class 
    //
    SBDistort::SBDistort(
        const SBProfile& sbin, double mA, double mB, double mC, double mD,
        Position<double> x0_) :
        matrixA(mA), matrixB(mB), matrixC(mC), matrixD(mD), x0(x0_)
    {
        SBProfile* p=sbin.duplicate();
        SBDistort* sbd = dynamic_cast<SBDistort*> (p);
        if (sbd) {
            // We are distorting something that's already a distortion.
            // So just compound the affine transformaions
            adaptee = sbd->adaptee->duplicate();
            x0 = x0_ + fwd(sbd->x0);
            // New matrix is product (M_this) * (M_old)
            matrixA = mA*sbd->matrixA + mB*sbd->matrixC;
            matrixB = mA*sbd->matrixB + mB*sbd->matrixD;
            matrixC = mC*sbd->matrixA + mD*sbd->matrixC;
            matrixD = mC*sbd->matrixB + mD*sbd->matrixD;
            delete sbd;
        } else {
            // Distorting something generic
            adaptee = p;
        }
        initialize();
    }

    SBDistort::SBDistort(const SBProfile& sbin, const Ellipse e_) 
    {
        // First get what we need from the Ellipse:
        tmv::Matrix<double> m = e_.getMatrix();
        matrixA = m(0,0);
        matrixB = m(0,1);
        matrixC = m(1,0);
        matrixD = m(1,1);
        x0 = e_.getX0();
        // Then repeat generic construction:
        SBProfile* p=sbin.duplicate();
        SBDistort* sbd = dynamic_cast<SBDistort*> (p);
        if (sbd) {
            // We are distorting something that's already a distortion.
            // So just compound the affine transformaions
            adaptee = sbd->adaptee->duplicate();
            x0 = e_.getX0() + fwd(sbd->x0);
            // New matrix is product (M_this) * (M_old)
            double mA = matrixA; double mB=matrixB; double mC=matrixC; double mD=matrixD;
            matrixA = mA*sbd->matrixA + mB*sbd->matrixC;
            matrixB = mA*sbd->matrixB + mB*sbd->matrixD;
            matrixC = mC*sbd->matrixA + mD*sbd->matrixC;
            matrixD = mC*sbd->matrixB + mD*sbd->matrixD;
            delete sbd;
        } else {
            // Distorting something generic
            adaptee = p;
        }
        initialize();
    }

    void SBDistort::initialize() 
    {
        double det = matrixA*matrixD-matrixB*matrixC;
        if (det==0.) throw SBError("Attempt to SBDistort with degenerate matrix");
        absdet = std::abs(det);
        invdet = 1./det;

        double h1 = hypot( matrixA+matrixD, matrixB-matrixC);
        double h2 = hypot( matrixA-matrixD, matrixB+matrixC);
        major = 0.5*std::abs(h1+h2);
        minor = 0.5*std::abs(h1-h2);
        if (major<minor) std::swap(major,minor);
        stillIsAxisymmetric = adaptee->isAxisymmetric() 
            && (matrixB==-matrixC) 
            && (matrixA==matrixD)
            && (x0.x==0.) && (x0.y==0.); // Need pure rotation
    }

    // Specialization of fillKGrid is desired since the phase terms from shift 
    // are factorizable:
    void SBDistort::fillKGrid(KTable& kt) const 
    {
        double N = (double) kt.getN();
        double dk = kt.getDk();

        if (x0.x==0. && x0.y==0.) {
            // Branch to faster calculation if there is no centroid shift:
            for (int iy = -N/2; iy < N/2; iy++) {
                // only need ix>=0 since it's Hermitian:
                for (int ix = 0; ix <= N/2; ix++) {
                    Position<double> k(ix*dk,iy*dk);
                    kt.kSet(ix,iy,kValNoPhase(k));
                }
            }
        } else {
            std::complex<double> dxexp(0,-dk*x0.x),   dyexp(0,-dk*x0.y);
            std::complex<double> dxphase(std::exp(dxexp)), dyphase(std::exp(dyexp));
            // xphase, yphase: current phase value
            std::complex<double> yphase(std::exp(-dyexp*N/2.));
            for (int iy = -N/2; iy < N/2; iy++) {
                std::complex<double> phase = yphase; // since kx=0 to start
                // Only ix>=0 since it's Hermitian:
                for (int ix = 0; ix <= N/2; ix++) {
                    Position<double> k(ix*dk,iy*dk);
                    kt.kSet(ix,iy,kValNoPhase(k) * phase);
                    phase *= dxphase;
                }
                yphase *= dyphase;
            }
        }
    }

    //
    // SBConvolve class - adding new members
    //
    void SBConvolve::add(const SBProfile& rhs) 
    {
        if (!rhs.isAnalyticK() && !_real_space) 
            throw SBError("SBConvolve requires members to be analytic in k");
        if (!rhs.isAnalyticX() && _real_space)
            throw SBError("SBConvolve with real_space=true requires members to be analytic in x");

        // If this is the first thing being added to the list, initialize some accumulators
        if (plist.empty()) {
            x0 = y0 = 0.;
            fluxProduct = 1.;
            minMaxK = 0.;
            minStepK = 0.;
            sumMinX = 0.;
            sumMaxX = 0.;
            sumMinY = 0.;
            sumMaxY = 0.;
            isStillAxisymmetric = true;
        }

        // Need a non-const copy of the rhs:
        SBProfile* p=rhs.duplicate();

        // Keep track of where first new term is on list:
        std::list<SBProfile*>::iterator newptr = plist.end();

        // Add new terms(s) to the plist:
        SBConvolve *sbc = dynamic_cast<SBConvolve*> (p);
        if (sbc) {  
            // If rhs is an SBConvolve, copy its list here
            fluxScale *= sbc->fluxScale;
            std::list<SBProfile*>::iterator pptr;
            for (pptr = sbc->plist.begin(); pptr!=sbc->plist.end(); ++pptr) {
                if (newptr==plist.end()) {
                    plist.push_back((*pptr)->duplicate()); 
                    newptr = --plist.end();  // That was first new term
                } else {
                    plist.push_back((*pptr)->duplicate()); 
                }
            }
            delete sbc; // no memory leak! 
        } else {
            plist.push_back(p);
            newptr = --plist.end();  // That was first new term
        }

        // Accumulate properties of all terms
        while (newptr != plist.end()) {
            fluxProduct *= (*newptr)->getFlux();
            x0 += (*newptr)->centroid().x;
            y0 += (*newptr)->centroid().y;
            if ( minMaxK<=0. || (*newptr)->maxK() < minMaxK) minMaxK = (*newptr)->maxK();
            if ( minStepK<=0. || ((*newptr)->stepK() < minStepK)) minStepK = (*newptr)->stepK();
            sumMinX += (*newptr)->minX();
            sumMaxX += (*newptr)->maxX();
            sumMinY += (*newptr)->minY();
            sumMaxY += (*newptr)->maxY();
            isStillAxisymmetric = isStillAxisymmetric && (*newptr)->isAxisymmetric();
            newptr++;
        }
    }

    void SBConvolve::fillKGrid(KTable& kt) const 
    {
        if (plist.empty()) kt.clear();
        std::list<SBProfile*>::const_iterator pptr = plist.begin();
        (*pptr)->fillKGrid(kt);
        kt *= fluxScale;
        ++pptr;
        if (pptr==plist.end()) return;
        int N = kt.getN();
        double dk = kt.getDk();
        KTable k2(N,dk);
        for ( ; pptr!= plist.end(); ++pptr) {
            (*pptr)->fillKGrid(k2);
            kt*=k2;
        }
    }

    class RealSpaceConvolve : 
        public std::binary_function<double,double,double>
    {
    public:
        RealSpaceConvolve(const SBProfile* p1, const SBProfile* p2, double x0, double y0) :
            _p1(p1), _p2(p2), _x0(x0), _y0(y0) {}

        double operator()(double x, double y) const 
        {
            return 
                _p1->xValue(Position<double>(x,y)) *
                _p2->xValue(Position<double>(_x0-x,_y0-y));
        }
    private:
        const SBProfile* _p1;
        const SBProfile* _p2;
        double _x0, _y0;
    };

    class YRegion :
        public std::unary_function<double, integ::IntRegion<double> >
    {
    public:
        YRegion(const SBProfile* p1, const SBProfile* p2, const Position<double>& pos) :
            _p1(p1), _p2(p2), _pos(pos) {}

        integ::IntRegion<double> operator()(double x) const
        {
            // First figure out each profiles y region separately.
            // Note: if profile is axisymmetric, then maxY = maxR.
            double ymin1,ymax1;
            if (_p1->isAxisymmetric()) {
                double r = _p1->maxY();
                ymax1 = sqrt(r*r-x*x);
                ymin1 = -ymax1;
            } else {
                ymin1 = _p1->minY();
                ymax1 = _p1->maxY();
            }
            double ymin2,ymax2;
            if (_p2->isAxisymmetric()) {
                double r = _p2->maxY();
                ymax2 = sqrt(r*r-x*x);
                ymin2 = -ymax2;
            } else {
                ymin2 = _p2->minY();
                ymax2 = _p2->maxY();
            }
            // Then take the overlap relevant for the calculation:
            //     _p1->xValue(x,y) * _p2->xValue(_x0-x,_y0-y)
            double ymin = std::max(ymin1, _pos.y-ymax2);
            double ymax = std::min(ymax1, _pos.y-ymin2);
            dbg<<"Y region for x = "<<x<<" = "<<ymin<<" ... "<<ymax<<std::endl;
            return integ::IntRegion<double>(ymin,ymax);
        }
    private:
        const SBProfile* _p1;
        const SBProfile* _p2;
        const Position<double>& _pos;
    };

    double SBConvolve::xValue(Position<double> pos) const
    {
        dbg<<"Start SBConvolve xValue for pos = "<<pos<<std::endl;
        // I think these values for the relative and absolute error are sufficient for
        // making images.  Not sure how they implicitly compare to Gary's choices in
        // the Fourier version.
        const double relerr = 1.e-3;
        const double abserr = 1.e-6;

        // Perform a direct calculation of the convolution at a particular point by
        // doing the real-space integral.
        // Note: This can only really be done one pair at a time, so it is 
        // probably rare that this will be more efficient if N > 2.
        // For now, we don't bother implementing this for N > 2.
        if (plist.empty()) return 0.;
        else if (plist.size() == 1) return plist.front()->xValue(pos);
        else if (plist.size() > 2) 
            throw SBError("Real-space integration of more than 2 profiles is not implemented.");
        else {
            double x = pos.x;
            double y = pos.y;
            if (x < sumMinX) { dbg<<"trivial 0\n"; return 0.; }
            if (x > sumMaxX) { dbg<<"trivial 0\n"; return 0.; }
            if (y < sumMinX) { dbg<<"trivial 0\n"; return 0.; }
            if (y > sumMaxY) { dbg<<"trivial 0\n"; return 0.; }

            const SBProfile* p1 = plist.front();
            const SBProfile* p2 = plist.back();
            RealSpaceConvolve conv(p1,p2,x,y);

            double xmin = std::max(p1->minX() , x-p2->maxX());
            double xmax = std::min(p1->maxX() , x-p2->minX());
            integ::IntRegion<double> xreg(xmin,xmax);
            dbg<<"xreg = "<<xmin<<" ... "<<xmax<<std::endl;

            YRegion yreg(p1,p2,pos);

            double result = integ::int2d(conv, xreg, yreg, relerr, abserr);
            dbg<<"Found result = "<<result<<std::endl;
            return result;
        }
    }

    //
    // "SBGaussian" Class 
    //
    double SBGaussian::xValue(Position<double> p) const
    {
        double r2 = p.x*p.x + p.y*p.y;
        double xval = flux * std::exp( -r2/2./sigma/sigma );
        xval /= 2*M_PI*sigma*sigma;  // normalize
        return xval;
    }

    std::complex<double> SBGaussian::kValue(Position<double> p) const
    {
        double r2 = p.x*p.x + p.y*p.y;
        std::complex<double> kval(flux * std::exp(-(r2)*sigma*sigma/2.),0);
        return kval;
    }


    //
    // SBExponential Class
    //

    // Set stepK so that folding occurs when excluded flux=ALIAS_THRESHOLD
    // Or at least 6 scale lengths
    double SBExponential::stepK() const
    {
        // A fast solution to (1+R)exp(-R)=ALIAS_THRESHOLD:
        double R = -std::log(ALIAS_THRESHOLD);
        for (int i=0; i<3; i++) R = -std::log(ALIAS_THRESHOLD) + std::log(1+R);
        R = std::max(6., R);
        return M_PI / (R*r0);
    }

    double SBExponential::xValue(Position<double> p) const
    {
        double r = std::sqrt(p.x*p.x + p.y*p.y);
        double xval = flux * std::exp(-r/r0);
        xval /= r0*r0*2*M_PI;   // normalize
        return xval;
    }

    std::complex<double> SBExponential::kValue(Position<double> p) const 
    {
        double kk = p.x*p.x+p.y*p.y;
        double temp = 1 + kk*r0*r0;         // [1+k^2*r0^2]
        std::complex<double> kval( flux/std::sqrt(temp*temp*temp), 0.);
        return kval;
    }

    //
    // SBAiry Class
    //

    // Note x & y are in units of lambda/D here.  Integral over area
    // will give unity in this normalization.

    double SBAiry::xValue(Position<double> p) const 
    {
        double radius = std::sqrt(p.x*p.x+p.y*p.y);
        double nu = radius*M_PI*D;
        double xval;
        if (nu<0.01)
            // lim j1(u)/u = 1/2
            xval =  D * (1-obscuration*obscuration);
        else {
            xval = 2*D*( j1(nu) - obscuration*j1(obscuration*nu)) /
                nu ; //See Schroeder eq (10.1.10)
        }
        xval*=xval;
        // Normalize to give unit flux integrated over area.
        xval /= (1-obscuration*obscuration)*4./M_PI;
        return xval*flux;
    }

    double SBAiry::chord(const double r, const double h) const 
    {
        if (r<h) throw SBError("Airy calculation r<h");
        else if (r==0.) return 0.;
        else if (r<0 || h<0) throw SBError("Airy calculation (r||h)<0");
        return r*r*std::asin(h/r) -h*std::sqrt(r*r-h*h);
    }

    /* area inside intersection of 2 circles radii r & s, seperated by t*/
    double SBAiry::circle_intersection(double r, double s, double t) const 
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
        if (h<0) {
            throw SBError("Airy calculation half-height invalid");
        }
        h = std::sqrt(h);

        if (t*t < r*r - s*s) 
            return M_PI*s*s - chord(s,h) + chord(r,h);
        else
            return chord(s,h) + chord(r,h);
    }

    /* area of two intersecting identical annuli */
    double SBAiry::annuli_intersect(double r1, double r2, double t) const 
    {
        if (r1<r2) {
            double temp;
            temp = r2;
            r2 = r1;
            r1 = temp;
        }
        return circle_intersection(r1,r1,t)
            - 2 * circle_intersection(r1,r2,t)
            +  circle_intersection(r2,r2,t);
    }

    /* Beam pattern of annular aperture, in k space, which is just the
     * autocorrelation of two annuli.  Normalize to unity at k=0 for now */
    double SBAiry::annuli_autocorrelation(const double k) const 
    {
        double k_scaled = k / (M_PI*D);
        double norm = M_PI*(1. - obscuration*obscuration);
        return annuli_intersect(1.,obscuration,k_scaled)/norm;
    }

    std::complex<double> SBAiry::kValue(Position<double> p) const
    {
        double radius = std::sqrt(p.x*p.x+p.y*p.y);
        // calculate circular FT(PSF) on p'=(x',y')
        double r = annuli_autocorrelation(radius);
        std::complex<double> kval(r, 0.);
        return kval*flux;
    }


    //
    // SBBox Class
    //

    double SBBox::xValue(Position<double> p) const 
    {
        if (fabs(p.x) < 0.5*xw && fabs(p.y) < 0.5*yw) return flux/(xw*yw);
        else return 0.;  // do not use this function for fillXGrid()!
    }

    double SBBox::sinc(const double u) const 
    {
        if (u<0.001 && u>-0.001)
            return 1.-u*u/6.;
        else
            return std::sin(u)/u;
    }

    std::complex<double> SBBox::kValue(Position<double> p) const
    {
        std::complex<double> kval( sinc(0.5*p.x*xw)*sinc(0.5*p.y*yw), 0.);
        return kval*flux;
    }

    // Override fillXGrid so we can partially fill pixels at edge of box.
    void SBBox::fillXGrid(XTable& xt) const 
    {
        int N = xt.getN();
        double dx = xt.getDx(); // pixel grid size
        double norm = flux/xw/yw;

        // Pixel index where edge of box falls:
        int xedge = static_cast<int> ( std::ceil(xw / (2*dx) - 0.5) );
        int yedge = static_cast<int> ( std::ceil(yw / (2*dx) - 0.5) );
        // Fraction of edge pixel that is filled by box:
        double xfrac = xw / (2*dx) - xedge + 0.5;
        assert(xfrac>0. && xfrac<=1.);
        double yfrac = yw / (2*dx) - yedge + 0.5;
        assert(yfrac>0. && yfrac<=1.);
        if (xedge==0) xfrac = xw/dx;
        if (yedge==0) yfrac = yw/dx;

        double yfac;
        for (int iy = -N/2; iy < N/2; iy++) {
            if ( std::abs(iy) < yedge ) yfac = 0.;
            else if (std::abs(iy)==yedge) yfac = norm*yfrac;
            else yfac = norm;

            for (int ix = -N/2; ix < N/2; ix++) {
                if (yfac==0. || std::abs(ix)>xedge) xt.xSet(ix, iy ,0.);
                else if (std::abs(ix)==xedge) xt.xSet(ix, iy ,xfrac*yfac);
                else xt.xSet(ix,iy,yfac);
            }
        }
    }

#ifdef USE_IMAGES
    // Override x-domain writing so we can partially fill pixels at edge of box.
    template <typename T>
    double SBBox::fillXImage(ImageView<T>& I, double dx) const 
    {
        double norm = flux/xw/yw;

        // Pixel index where edge of box falls:
        int xedge = static_cast<int> ( std::ceil(xw / (2*dx) - 0.5) );
        int yedge = static_cast<int> ( std::ceil(yw / (2*dx) - 0.5) );
        // Fraction of edge pixel that is filled by box:
        double xfrac = xw / (2*dx) - xedge + 0.5;
        assert(xfrac>0. && xfrac<=1.);
        double yfrac = yw / (2*dx) - yedge + 0.5;
        assert(yfrac>0. && yfrac<=1.);
        if (xedge==0) xfrac = xw/dx;
        if (yedge==0) yfrac = yw/dx;

        double totalflux = 0.;
        double xfac;
        for (int i = I.getXMin(); i <= I.getXMax(); i++) {
            if ( std::abs(i) > xedge ) xfac = 0.;
            else if (std::abs(i)==xedge) xfac = norm*xfrac;
            else xfac = norm;

            for (int j = I.getYMin(); j <= I.getYMax(); j++) {
                if (xfac==0. || std::abs(j)>yedge) I(i,j)=0.;
                else if (std::abs(j)==yedge) I(i,j)=xfac*yfrac;
                else I(i,j)=xfac;
                totalflux += I(i,j);
            }
        }
        I.setScale(dx);

        return totalflux * (dx*dx);
    }
#endif


#ifdef USE_LAGUERRE
    //
    // SBLaguerre Class
    //

    // ??? Have not really investigated these:
    double SBLaguerre::maxK() const 
    {
        // Start with value for plain old Gaussian:
        double m=std::max(4., std::sqrt(-2.*std::log(ALIAS_THRESHOLD))) / sigma;
        // Grow as sqrt of order
        if (bvec.getOrder()>1) m *= std::sqrt(bvec.getOrder()/1.);
        return m;
    }

    double SBLaguerre::stepK() const 
    {
        // Start with value for plain old Gaussian:
        double m= M_PI/std::max(4., std::sqrt(-2.*std::log(ALIAS_THRESHOLD))) / sigma;
        // Shrink as sqrt of order
        if (bvec.getOrder()>1) m /= std::sqrt(bvec.getOrder()/1.);
        return m;
    }

    double SBLaguerre::xValue(Position<double> p) const 
    {
        LVector psi(bvec.getOrder());
        psi.fillBasis(p.x/sigma, p.y/sigma, sigma);
        double xval = bvec.dot(psi);
        return xval;
    }

    std::complex<double> SBLaguerre::kValue(Position<double> k) const 
    {
        int N=bvec.getOrder();
        LVector psi(N);
        psi.fillBasis(k.x*sigma, k.y*sigma);  // Fourier[Psi_pq] is unitless
        // rotate kvalues of Psi with i^(p+q)
        // dotting b_pq with psi in k-space:
        double rr=0.;
        double ii=0.;
        {
            for (PQIndex pq(0,0); !pq.pastOrder(N); pq.nextDistinct()) {
                int j = pq.rIndex();
                double x = bvec[j]*psi[j] + (pq.isReal() ? 0 : bvec[j+1]*psi[j+1]);
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
        return std::complex<double>(2*M_PI*rr, 2*M_PI*ii);
    }

    double SBLaguerre::getFlux() const 
    {
        double flux=0.;
        for (PQIndex pp(0,0); !pp.pastOrder(bvec.getOrder()); pp.incN())
            flux += bvec[pp].real();  // bvec[pp] is real, but need type conv.
        return flux;
    }

    void SBLaguerre::setFlux(double flux_) 
    {
        double newflux=flux_;
        if (getFlux()!=0) newflux /= getFlux();
        bvec.rVector() *= newflux;
        return;
    }

#endif

    // SBSersic Class 
    // First need to define the static member that holds info on all the Sersic n's
    SBSersic::InfoBarn SBSersic::nmap;

    double SBSersic::SersicInfo::kValue(double ksq) const 
    {
        if (ksq<0.) 
            throw SBError("Negative k-squared passed to SersicInfo");
        if (ksq==0.) 
            return 1.;

        double lk=0.5*std::log(ksq); // Lookup table is logarithmic

        if (lk<logkMin)
            return 1 + ksq*(kderiv2 + ksq*kderiv4); // Use quartic approx at low k
        if (lk>=logkMax)
            return 0.; // truncate the Fourier transform

        // simple linear interpolation to this value
        double fstep = (lk-logkMin)/logkStep;
        int index = static_cast<int> (std::floor(fstep));
        assert(index < int(lookup.size())-1);
        fstep -= index;
        return lookup[index]*(1.-fstep) + fstep*lookup[index+1];
    }

    // Integrand class for the Hankel transform of Sersic
    class SersicIntegrand : public std::unary_function<double,double>
    {
    public:
        SersicIntegrand(double n, double b_, double k_):
            invn(1./n), b(b_), k(k_) {}
        double operator()(double r) const 
            { return r*std::exp(-b*std::pow(r, invn))*j0(k*r); }

    private:
        double invn;
        double b;
        double k;
    };


    // Constructor to initialize Sersic constants and k lookup table
    SBSersic::SersicInfo::SersicInfo(double n): inv2n(1./(2.*n)) 
    {
        // Going to constraint range of allowed n to those I have looked at
        if (n<0.5 || n>4.2) throw SBError("Requested Sersic index out of range");

        // Formula for b from Ciotti & Bertin (1999)
        b = 2*n - (1./3.)
            + (4./405.)/n
            + (46./25515.)/(n*n)
            + (131./1148175.)/(n*n*n)
            - (2194697./30690717750.)/(n*n*n*n);

        double b2n = std::pow(b,2*n);  // used frequently here
        // The normalization factor to give unity flux integral:
        norm = b2n / (2*M_PI*n*tgamma(2.*n));

        // The quadratic term of small-k expansion:
        kderiv2 = -tgamma(4*n) / (4*b2n*tgamma(2*n)) ; 
        // And a quartic term:
        kderiv4 = tgamma(6*n) / (64*b2n*b2n*tgamma(2*n));

#if 0
        std::cerr << "Building for n=" << n << " b= " << b << " norm= " << norm << std::endl;
        std::cerr << "Deriv terms: " << kderiv2 << " " << kderiv4 << std::endl;
#endif

        // When is it safe to use low-k approximation?  See when
        // quartic term is at threshold
        double lookupMin = 0.05; // Default lower limit for lookup table
        const double kAccuracy=0.001; // What errors in FT coefficients are acceptable?
        double smallK = std::pow(kAccuracy / kderiv4, 0.25);
        if (smallK < lookupMin) lookupMin = smallK;
        logkMin = std::log(lookupMin);

        // How far should nominal profile extend?
        // Estimate number of effective radii needed to enclose

        double xMax = 5.; // Go to at least 5r_e
        {
            // Successive approximation method:
            double a=2*n;
            double z=a;
            double oldz=0.;
            int niter=0;
            const int MAXIT = 15;
            while ( std::abs(oldz-z)>0.01 && niter<MAXIT) {
                niter++;
                oldz = z;
                z = a - std::log(ALIAS_THRESHOLD*std::sqrt(2*M_PI*a)*(1+1./(12*a)+1./(288*a*a)))
                    +(a-1)*std::log(z/a) + std::log(1 + (a-1)/z + (a-1)*(a-2)/(z*z));
            }
            double r=std::pow(z/b, n);
            if (r>xMax) xMax = r;
        }
        stepK = M_PI / xMax;

        // Going to calculate another outer radius for the integration of the 
        // Hankel transforms:
        double integrateMax=xMax;
        const double integrationLoss=0.001;
        {
            // Successive approximation method:
            double a=2*n;
            double z=a;
            double oldz=0.;
            int niter=0;
            const int MAXIT = 15;
            while ( std::abs(oldz-z)>0.01 && niter<MAXIT) {
                niter++;
                oldz = z;
                z = a - std::log(integrationLoss*std::sqrt(2*M_PI*a)*(1+1./(12*a)+1./(288*a*a)))
                    +(a-1)*std::log(z/a) + std::log(1 + (a-1)/z + (a-1)*(a-2)/(z*z));
            }
            double r=std::pow(z/b, n);
            //std::cerr << "99.9% radius " << r <<std::endl;
            if (r>integrateMax) integrateMax = r;    
        }

        // Normalization for integral at k=0:
        double norm;
        const double INTEGRATION_RELTOL=0.0001;
        const double INTEGRATION_ABSTOL=1e-5;
        {
            SersicIntegrand I(n, b, 0.);
            // Integrate with at least 2^10 steps and up to 2^16:
            norm = integ::int1d(
                I, 0., integrateMax, INTEGRATION_RELTOL, INTEGRATION_ABSTOL);
        }

        // Now start building the lookup table for FT of the profile.
        // Keep track of where the FT drops below ALIAS_THRESHOLD - this
        // will be our maxK.
        // Then extend the table another order of magnitude either in k
        //  or in FT, whichever comes first.
        logkStep = 0.05;
        // Here is preset range of acceptable maxK:
        const double MINMAXK = 10.;
        const double MAXMAXK = 50.; 
        maxK = MINMAXK;
        double lastVal=1.;
        double lk = logkMin;
        while (lk < std::log(maxK*10.) && lastVal>ALIAS_THRESHOLD/10.) {
            SersicIntegrand I(n, b, std::exp(lk));
            // Need to make sure we are resolving oscillations in the integral:
            double val = integ::int1d(
                I, 0., integrateMax, INTEGRATION_RELTOL, INTEGRATION_ABSTOL*norm);
            //std::cerr << "Integrate k " << exp(lk) << " result " << val/norm << std::endl;
            val /= norm;
            lookup.push_back(val);
            if (val >= ALIAS_THRESHOLD) maxK = std::max(maxK, std::exp(lk));
            logkMax = lk;
            lk += logkStep;
        }
        maxK = std::min(MAXMAXK, maxK); // largest acceptable
    }

    // Integrand class for the flux integrals of Moffat
    class MoffatFluxInt : public std::unary_function<double,double>
    {
    public:
        MoffatFluxInt(double beta_): beta(beta_) {}
        double operator()(double r) const 
        { return r*std::pow(1.+r*r,-beta); }
    private:
        double beta;
    };

    class MoffatFlux 
    {
    public:
        MoffatFlux(double beta): mfi(beta), target(0.) {}
        void setTarget(double target_) {target=target_;}
        double operator()(double r) const 
        { return 2.*M_PI*integ::int1d(mfi, 0., r) - target; }
    private:
        MoffatFluxInt mfi;
        double target;
    };


    SBMoffat::SBMoffat(double beta_, double truncationFWHM, double flux_, double size, RadiusType rType) : 
        beta(beta_), flux(flux_), norm(1.), rD(1.),
        ft(Table<double,double>::spline)
    {
        //First, relation between FWHM and rD:
        FWHMrD = 2.* std::sqrt(std::pow(2., 1./beta)-1.);
        maxRrD = FWHMrD * truncationFWHM;
#if 1
        // Make FFT's periodic at 4x truncation radius or 1.5x diam at ALIAS_THRESHOLD,
        // whichever is smaller
        stepKrD = 2*M_PI / std::min(4*maxRrD, 
                                    3.*std::sqrt(pow(ALIAS_THRESHOLD, -1./beta)-1));
#else
        // Make FFT's periodic at 4x truncation radius or 8x half-light radius:
        stepKrD = M_PI / (2*std::max(maxRrD, 16.));
#endif
        // And be sure to get at least 16 pts across FWHM when drawing:
        maxKrD = 16*M_PI / FWHMrD;

        // Get flux and half-light radius in units of rD:
        MoffatFlux mf(beta);
        double fluxFactor = mf(maxRrD);

        // Set size of this instance according to type of size given in constructor:
        switch (rType)
        {
        case FWHM:
            rD = size / FWHMrD;
            break;
        case HALF_LIGHT_RADIUS: {
            Solve<MoffatFlux> s(mf, 0.1, 2.);
            mf.setTarget(0.5*fluxFactor);
            double rerD = s.root();
            rD = size / rerD;
        }
            break;
        case SCALE_RADIUS:
            rD = size;
            break;
        default:
            throw SBError("Unknown SBMoffat::RadiusType");
        }
        norm = 1./fluxFactor;

#if 0
        std::cerr << "Moffat rD " << rD
            << " norm " << norm << " maxRrD " << maxRrD << std::endl;
#endif

        // Get FFT by doing 2k transform over 2x the truncation radius
        // ??? need to do better here
        // ??? also install quadratic behavior near k=0?
        const int N=2048;
        double dx = std::max(4*maxRrD, 64.) / N;
        XTable xt(N, dx);
        dx = xt.getDx();
        for (int iy=-N/2; iy<N/2; iy++)
            for (int ix=-N/2; ix<N/2; ix++) {
                double rsq = dx*dx*(ix*ix+iy*iy);
                xt.xSet(ix, iy, rsq<=maxRrD*maxRrD ? std::pow(1+rsq,-beta) : 0.);
            }
        KTable* kt = xt.transform();
        double dk = kt->getDk();
        double nn = kt->kval(0,0).real();
        for (int i=0; i<=N/2; i++) {
            ft.addEntry( i*dk, kt->kval(0,-i).real() / nn);
        }
        delete kt;
    }

    std::complex<double> SBMoffat::kValue(Position<double> k) const 
    {
        double kk = hypot(k.x, k.y)*rD;
        if (kk > ft.argMax()) return 0.;
        else return flux*ft(kk);
    }

    // instantiate template functions for expected image types
#ifdef USE_IMAGES
    template double SBProfile::doFillXImage2(ImageView<float>& img, double dx) const;
    template double SBProfile::doFillXImage2(ImageView<double>& img, double dx) const;

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
#endif

}
