//
// Functions for the Surface Brightness Profile Class
//

//#define DEBUGLOGGING

#include "SBProfile.h"
#include "integ/Int.h"
#include "TMV.h"
#include "Solve.h"
#include "integ/Int.h"

#include <fstream>

// To time the real-space convolution integrals...
//#define TIMING
#ifdef TIMING
#include <sys/time.h>
#endif

#ifdef DEBUGLOGGING
std::ostream* dbgout = new std::ofstream("debug.out");
//std::ostream* dbgout = &std::cerr;
//std::ostream* dbgout = 0;
int verbose_level = 1;
#else
std::ostream* dbgout = 0;
int verbose_level = 0;
#endif

#include <numeric>

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

    ImageView<float> SBProfile::draw(double dx, int wmult) const 
    {
        xdbg<<"Start draw that returns ImageView"<<std::endl;
        Image<float> img;
        draw(img, dx, wmult);
        return img.view();
    }

    template <typename T>
    double SBProfile::draw(ImageView<T>& img, double dx, int wmult) const 
    {
        xdbg<<"Start draw ImageView"<<std::endl;
        if (isAnalyticX())
            return plainDraw(img, dx, wmult);
        else
            return fourierDraw(img, dx, wmult);
    }

    template <typename T>
    double SBProfile::draw(Image<T>& img, double dx, int wmult) const 
    {
        xdbg<<"Start draw Image"<<std::endl;
        if (isAnalyticX())
            return plainDraw(img, dx, wmult);
        else
            return fourierDraw(img, dx, wmult);
    }

    // First is a simple case wherein we have a formula for x values:
    template <typename T>
    double SBProfile::plainDraw(ImageView<T>& I, double dx, int wmult) const 
    {
        xdbg<<"Start plainDraw ImageView"<<std::endl;
        // Determine desired dx:
        xdbg<<"maxK = "<<maxK()<<std::endl;
        if (dx<=0.) dx = M_PI / maxK();
        xdbg<<"dx = "<<dx<<std::endl;
        // recenter an existing image, to be consistent with fourierDraw:
        int xSize = I.getXMax()-I.getXMin()+1, ySize = I.getYMax()-I.getYMin()+1;
        xdbg<<"xSize = "<<xSize<<std::endl;
        I.setOrigin(-xSize/2, -ySize/2);

        return fillXImage(I, dx);
    }

    template <typename T>
    double SBProfile::plainDraw(Image<T>& I, double dx, int wmult) const 
    {
        xdbg<<"Start plainDraw Image"<<std::endl;
        // Determine desired dx:
        xdbg<<"maxK = "<<maxK()<<std::endl;
        if (dx<=0.) dx = M_PI / maxK();
        xdbg<<"dx = "<<dx<<std::endl;
        if (!I.getBounds().isDefined()) {
            if (wmult<1) throw SBError("Requested wmult<1 in plainDraw()");
            // Need to choose an image size
            int N = static_cast<int> (std::ceil(2*M_PI/(dx*stepK())));
            xdbg<<"N = "<<N<<std::endl;

            // Round up to an even value
            N = 2*( (N+1)/2);
            N *= wmult; // make even bigger if desired
            xdbg<<"N => "<<N<<std::endl;
            Bounds<int> imgsize(-N/2, N/2-1, -N/2, N/2-1);
            xdbg<<"imgsize => "<<imgsize<<std::endl;
            I.resize(imgsize);
        } else {
            // recenter an existing image, to be consistent with fourierDraw:
            int xSize = I.getXMax()-I.getXMin()+1, ySize = I.getYMax()-I.getYMin()+1;
            xdbg<<"xSize = "<<xSize<<std::endl;
            I.setOrigin(-xSize/2, -ySize/2);
        }

        // TODO: If we decide not to keep the scale, then can switch to simply:
        // return fillXImage(I.view(), dx);
        // (And switch fillXImage to take a const ImageView<T>& argument.)
        ImageView<T> Iv = I.view();
        double ret = fillXImage(Iv, dx);
        I.setScale(Iv.getScale());
        xdbg<<"scale => "<<I.getScale()<<std::endl;
        return ret;
    }
 
    template <typename T>
    double SBProfile::doFillXImage2(ImageView<T>& I, double dx) const 
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

        xdbg << " maxK() " << maxK() << " dx " << dx << std::endl;

        // Now decide how big the FT must be to avoid folding:
        double xRange = 2*M_PI*wmult / stepK();
        int Nnofold = static_cast<int> (std::ceil(xRange / dx -0.0001));
        xdbg << " stepK() " << stepK() << " Nnofold " << Nnofold << std::endl;

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
        xdbg << " After adjustments: Nnofold " << Nnofold << " NFT " << NFT << std::endl;
        if (NFT > MAXIMUM_FFT_SIZE)
            FormatAndThrow<SBError>() << "fourierDraw() requires an FFT that is too large, " << NFT;

        // Move the output image to be centered near zero
        I.setOrigin(-xSize/2, -ySize/2);
        double dk = 2.*M_PI/(NFT*dx);
        xdbg << 
            " After adjustments: dx " << dx << " dk " << dk << 
            " maxK " << dk*NFT/2 << std::endl;
        assert(dk <= stepK());
        XTable* xtmp=0;
        if (NFT*dk/2 > maxK()) {
            xdbg<<"NFT*dk/2 = "<<NFT*dk/2<<" > maxK() = "<<maxK()<<std::endl;
            xdbg<<"Use NFT = "<<NFT<<std::endl;
            // No aliasing: build KTable and transform
            KTable kt(NFT,dk);
            fillKGrid(kt); 
            xtmp = kt.transform();
        } else {
            xdbg<<"NFT*dk/2 = "<<NFT*dk/2<<" <= maxK() = "<<maxK()<<std::endl;
            // There will be aliasing.  Construct a KTable out to maxK() and
            // then wrap it
            int Nk = static_cast<int> (std::ceil(maxK()/dk)) * 2;
            xdbg<<"Use Nk = "<<Nk<<std::endl;
            KTable kt(Nk, dk);
            fillKGrid(kt);
            KTable* kt2 = kt.wrap(NFT);
            xtmp = kt2->transform();
            delete kt2;
        }
        int Nxt = xtmp->getN();
        xdbg<<"Nxt = "<<Nxt<<std::endl;
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

        xdbg << " maxK() " << maxK() << " dx " << dx << std::endl;

        // Now decide how big the FT must be to avoid folding:
        double xRange = 2*M_PI*wmult / stepK();
        int Nnofold = static_cast<int> (std::ceil(xRange / dx -0.0001));
        xdbg << " stepK() " << stepK() << " Nnofold " << Nnofold << std::endl;

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
        xdbg << " After adjustments: Nnofold " << Nnofold << " NFT " << NFT << std::endl;
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
        xdbg << 
            " After adjustments: dx " << dx << " dk " << dk << 
            " maxK " << dk*NFT/2 << std::endl;
        assert(dk <= stepK());
        XTable* xtmp=0;
        if (NFT*dk/2 > maxK()) {
            xdbg<<"NFT*dk/2 = "<<NFT*dk/2<<" > maxK() = "<<maxK()<<std::endl;
            xdbg<<"Use NFT = "<<NFT<<std::endl;
            // No aliasing: build KTable and transform
            KTable kt(NFT,dk);
            fillKGrid(kt); 
            xtmp = kt.transform();
        } else {
            xdbg<<"NFT*dk/2 = "<<NFT*dk/2<<" <= maxK() = "<<maxK()<<std::endl;
            // There will be aliasing.  Construct a KTable out to maxK() and
            // then wrap it
            int Nk = static_cast<int> (std::ceil(maxK()/dk)) * 2;
            xdbg<<"Use Nk = "<<Nk<<std::endl;
            KTable kt(Nk, dk);
            fillKGrid(kt);
            KTable* kt2 = kt.wrap(NFT);
            xtmp = kt2->transform();
            delete kt2;
        }
        int Nxt = xtmp->getN();
        xdbg<<"Nxt = "<<Nxt<<std::endl;
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

    void SBProfile::fillXGrid(XTable& xt) const 
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

    void SBProfile::fillKGrid(KTable& kt) const 
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

    void SBAdd::initialize() 
    {
        sumflux = sumfx = sumfy = 0.;
        maxMaxK = minStepK = 0.;
        allAxisymmetric = allAnalyticX = allAnalyticK = true;
    }

    void SBAdd::add(const SBProfile& rhs, double scale) 
    {
        xdbg<<"Start SBAdd::add.  Adding item # "<<plist.size()+1<<std::endl;
        // Need a non-const copy of the rhs:
        SBProfile* p=rhs.duplicate();


        // Keep track of where first new summand is on list:
        Iter newptr = plist.end();

        // Add new summand(s) to the plist:
        SBAdd *sba = dynamic_cast<SBAdd*> (p);
        if (sba) {
            // If rhs is an SBAdd, copy its full list here
            for (ConstIter pptr = sba->plist.begin(); pptr!=sba->plist.end(); ++pptr) {
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
            xdbg<<"SBAdd component has maxK, stepK = "<<
                (*newptr)->maxK()<<" , "<<(*newptr)->stepK()<<std::endl;
            sumflux += (*newptr)->getFlux();
            sumfx += (*newptr)->getFlux() * (*newptr)->centroid().x;
            sumfy += (*newptr)->getFlux() * (*newptr)->centroid().x;
            if ( (*newptr)->maxK() > maxMaxK) maxMaxK = (*newptr)->maxK();
            if ( minStepK<=0. || ((*newptr)->stepK() < minStepK)) minStepK = (*newptr)->stepK();
            allAxisymmetric = allAxisymmetric && (*newptr)->isAxisymmetric();
            allAnalyticX = allAnalyticX && (*newptr)->isAnalyticX();
            allAnalyticK = allAnalyticK && (*newptr)->isAnalyticK();
            newptr++;
        }
        xdbg<<"Net maxK, stepK = "<<maxMaxK<<" , "<<minStepK<<std::endl;
    }

    double SBAdd::xValue(const Position<double>& _p) const 
    {
        double xv = 0.;  
        for (ConstIter pptr = plist.begin(); pptr != plist.end(); ++pptr)
            xv += (*pptr)->xValue(_p);
        return xv;
    } 

    std::complex<double> SBAdd::kValue(const Position<double>& _p) const 
    {
        ConstIter pptr = plist.begin();
        assert(pptr != plist.end());
        std::complex<double> kv = (*pptr)->kValue(_p);
        for (++pptr; pptr != plist.end(); ++pptr)
            kv += (*pptr)->kValue(_p);
        return kv;
    } 

    void SBAdd::fillKGrid(KTable& kt) const 
    {
        if (plist.empty()) kt.clear();
        ConstIter pptr = plist.begin();
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
        ConstIter pptr = plist.begin();
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
        for (Iter pptr = plist.begin(); pptr != plist.end(); ++pptr) {
            double pf = (*pptr)->getFlux();  
            (*pptr)->setFlux(pf*m);
        }
        sumflux *=m;
        sumfx *= m;
        sumfy *= m;
    }

    double SBAdd::getPositiveFlux() const {
        double result = 0.;
        for (std::list<SBProfile*>::const_iterator pptr = plist.begin(); pptr != plist.end(); ++pptr) {
            result += (*pptr)->getPositiveFlux();  
        }
        return result;
    }
    double SBAdd::getNegativeFlux() const {
        double result = 0.;
        for (std::list<SBProfile*>::const_iterator pptr = plist.begin(); pptr != plist.end(); ++pptr) {
            result += (*pptr)->getNegativeFlux();  
        }
        return result;
    }
        

    //
    // "SBDistort" Class 
    //
    SBDistort::SBDistort(
        const SBProfile& sbin, double mA, double mB, double mC, double mD,
        const Position<double>& x0_) :
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

        xdbg<<"Distortion init\n";
        xdbg<<"matrix = "<<matrixA<<','<<matrixB<<','<<matrixC<<','<<matrixD<<std::endl;
        xdbg<<"x0 = "<<x0<<std::endl;
        xdbg<<"invdet = "<<invdet<<std::endl;
        xdbg<<"major, minor = "<<major<<", "<<minor<<std::endl;
        xdbg<<"maxK() = "<<adaptee->maxK() / minor<<std::endl;
        xdbg<<"stepK() = "<<adaptee->stepK() / major<<std::endl;

        // Calculate the values for getXRange and getYRange:
        if (adaptee->isAxisymmetric()) {
            // The original is a circle, so first get its radius.
            adaptee->getXRange(_xmin,_xmax,_xsplits);
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
                double AApBB = matrixA*matrixA + matrixB*matrixB;
                double sqrtAApBB = sqrt(AApBB);
                double temp = sqrtAApBB * R;
                _xmin = -temp + x0.x;
                _xmax = temp + x0.x;
                double CCpDD = matrixC*matrixC + matrixD*matrixD;
                double sqrtCCpDD = sqrt(CCpDD);
                temp = sqrt(CCpDD) * R;
                _ymin = -temp + x0.y;
                _ymax = temp + x0.y;
                _ysplits.resize(_xsplits.size());
                for (size_t k=0;k<_xsplits.size();++k) {
                    // The split points work the same way.  Scale them by the same factor we
                    // scaled the R value above, then add x0.x or x0.y.
                    double split = _xsplits[k];
                    xdbg<<"Adaptee split at "<<split<<std::endl;
                    _xsplits[k] = sqrtAApBB * split + x0.x;
                    _ysplits[k] = sqrtCCpDD * split + x0.y;
                    xdbg<<"-> x,y splits at "<<_xsplits[k]<<"  "<<_ysplits[k]<<std::endl;
                }
                // Now a couple of calculations that get reused in getYRange(x,yminymax):
                _coeff_b = (matrixA*matrixC + matrixB*matrixD) / AApBB;
                _coeff_c = CCpDD / AApBB;
                _coeff_c2 = absdet*absdet / AApBB;
                xdbg<<"adaptee is axisymmetric.\n";
                xdbg<<"adaptees maxR = "<<R<<std::endl;
                xdbg<<"xmin..xmax = "<<_xmin<<" ... "<<_xmax<<std::endl;
                xdbg<<"ymin..ymax = "<<_ymin<<" ... "<<_ymax<<std::endl;
            }
        } else {
            // Apply the distortion to each of the four corners of the original
            // and find the minimum and maximum.
            double xmin_1, xmax_1;
            std::vector<double> xsplits0;
            adaptee->getXRange(xmin_1,xmax_1,xsplits0);
            double ymin_1, ymax_1;
            std::vector<double> ysplits0;
            adaptee->getYRange(ymin_1,ymax_1,ysplits0);
            // Note: This doesn't explicitly check for MOCK_INF values.
            // It shouldn't be a problem, since the integrator will still treat
            // large values near MOCK_INF as infinity, but it just means that 
            // the following calculations might be wasted flops.
            Position<double> bl = fwd(Position<double>(xmin_1,ymin_1));
            Position<double> br = fwd(Position<double>(xmax_1,ymin_1));
            Position<double> tl = fwd(Position<double>(xmin_1,ymax_1));
            Position<double> tr = fwd(Position<double>(xmax_1,ymax_1));
            _xmin = std::min(std::min(std::min(bl.x,br.x),tl.x),tr.x) + x0.x;
            _xmax = std::max(std::max(std::max(bl.x,br.x),tl.x),tr.x) + x0.x;
            _ymin = std::min(std::min(std::min(bl.y,br.y),tl.y),tr.y) + x0.y;
            _ymax = std::max(std::max(std::max(bl.y,br.y),tl.y),tr.y) + x0.y;
            xdbg<<"adaptee is not axisymmetric.\n";
            xdbg<<"adaptees x range = "<<xmin_1<<" ... "<<xmax_1<<std::endl;
            xdbg<<"adaptees y range = "<<ymin_1<<" ... "<<ymax_1<<std::endl;
            xdbg<<"Corners are: bl = "<<bl<<std::endl;
            xdbg<<"             br = "<<br<<std::endl;
            xdbg<<"             tl = "<<tl<<std::endl;
            xdbg<<"             tr = "<<tr<<std::endl;
            xdbg<<"xmin..xmax = "<<_xmin<<" ... "<<_xmax<<std::endl;
            xdbg<<"ymin..ymax = "<<_ymin<<" ... "<<_ymax<<std::endl;
            if (bl.x + x0.x > _xmin && bl.x + x0.x < _xmax) {
                xdbg<<"X Split from bl.x = "<<bl.x+x0.x<<std::endl;
                _xsplits.push_back(bl.x+x0.x);
            }
            if (br.x + x0.x > _xmin && br.x + x0.x < _xmax) {
                xdbg<<"X Split from br.x = "<<br.x+x0.x<<std::endl;
                _xsplits.push_back(br.x+x0.x);
            }
            if (tl.x + x0.x > _xmin && tl.x + x0.x < _xmax) {
                xdbg<<"X Split from tl.x = "<<tl.x+x0.x<<std::endl;
                _xsplits.push_back(tl.x+x0.x);
            }
            if (tr.x + x0.x > _xmin && tr.x + x0.x < _xmax) {
                xdbg<<"X Split from tr.x = "<<tr.x+x0.x<<std::endl;
                _xsplits.push_back(tr.x+x0.x);
            }
            if (bl.y + x0.y > _ymin && bl.y + x0.y < _ymax) {
                xdbg<<"Y Split from bl.y = "<<bl.y+x0.y<<std::endl;
                _ysplits.push_back(bl.y+x0.y);
            }
            if (br.y + x0.y > _ymin && br.y + x0.y < _ymax) {
                xdbg<<"Y Split from br.y = "<<br.y+x0.y<<std::endl;
                _ysplits.push_back(br.y+x0.y);
            }
            if (tl.y + x0.y > _ymin && tl.y + x0.y < _ymax) {
                xdbg<<"Y Split from tl.y = "<<tl.y+x0.y<<std::endl;
                _ysplits.push_back(tl.y+x0.y);
            }
            if (tr.y + x0.y > _ymin && tr.y + x0.y < _ymax) {
                xdbg<<"Y Split from tr.y = "<<tr.y+x0.y<<std::endl;
                _ysplits.push_back(tr.y+x0.y);
            }
            // If the adaptee has any splits, try to propagate those up
            for(size_t k=0;k<xsplits0.size();++k) {
                xdbg<<"Adaptee xsplit at "<<xsplits0[k]<<std::endl;
                Position<double> bx = fwd(Position<double>(xsplits0[k],ymin_1));
                Position<double> tx = fwd(Position<double>(xsplits0[k],ymax_1));
                if (bx.x + x0.x > _xmin && bx.x + x0.x < _xmax) {
                    xdbg<<"X Split from bx.x = "<<bx.x+x0.x<<std::endl;
                    _xsplits.push_back(bx.x+x0.x);
                }
                if (tx.x + x0.x > _xmin && tx.x + x0.x < _xmax) {
                    xdbg<<"X Split from tx.x = "<<tx.x+x0.x<<std::endl;
                    _xsplits.push_back(tx.x+x0.x);
                }
                if (bx.y + x0.y > _ymin && bx.y + x0.y < _ymax) {
                    xdbg<<"Y Split from bx.y = "<<bx.y+x0.y<<std::endl;
                    _ysplits.push_back(bx.y+x0.y);
                }
                if (tx.y + x0.y > _ymin && tx.y + x0.y < _ymax) {
                    xdbg<<"Y Split from tx.y = "<<tx.y+x0.y<<std::endl;
                    _ysplits.push_back(tx.y+x0.y);
                }
            }
            for(size_t k=0;k<ysplits0.size();++k) {
                xdbg<<"Adaptee ysplit at "<<ysplits0[k]<<std::endl;
                Position<double> yl = fwd(Position<double>(xmin_1,ysplits0[k]));
                Position<double> yr = fwd(Position<double>(xmax_1,ysplits0[k]));
                if (yl.x + x0.x > _xmin && yl.x + x0.x < _xmax) {
                    xdbg<<"X Split from tl.x = "<<tl.x+x0.x<<std::endl;
                    _xsplits.push_back(yl.x+x0.x);
                }
                if (yr.x + x0.x > _xmin && yr.x + x0.x < _xmax) {
                    xdbg<<"X Split from yr.x = "<<yr.x+x0.x<<std::endl;
                    _xsplits.push_back(yr.x+x0.x);
                }
                if (yl.y + x0.y > _ymin && yl.y + x0.y < _ymax) {
                    xdbg<<"Y Split from yl.y = "<<yl.y+x0.y<<std::endl;
                    _ysplits.push_back(yl.y+x0.y);
                }
                if (yr.y + x0.y > _ymin && yr.y + x0.y < _ymax) {
                    xdbg<<"Y Split from yr.y = "<<yr.y+x0.y<<std::endl;
                    _ysplits.push_back(yr.y+x0.y);
                }
            }
        }
    }

    void SBDistort::getXRange(double& xmin, double& xmax, std::vector<double>& splits) const
    {
        xmin = _xmin; xmax = _xmax;
        splits.insert(splits.end(),_xsplits.begin(),_xsplits.end());
    }

    void SBDistort::getYRange(double& ymin, double& ymax, std::vector<double>& splits) const
    {
        ymin = _ymin; ymax = _ymax;
        splits.insert(splits.end(),_ysplits.begin(),_ysplits.end());
    }

    void SBDistort::getYRange(double x, double& ymin, double& ymax,
                              std::vector<double>& splits) const
    {
        xdbg<<"Distortion getYRange for x = "<<x<<std::endl;
        if (adaptee->isAxisymmetric()) {
            std::vector<double> splits0;
            adaptee->getYRange(ymin,ymax,splits0);
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
            double b = _coeff_b * (x-x0.x);
            double c = _coeff_c2 * R*R - _coeff_c * (x-x0.x) * (x-x0.x);
            double d = sqrt(c + b*b);
            ymax = b + d + x0.y;
            ymin = b - d + x0.y;
            for (size_t k=0;k<splits0.size();++k) if (splits0[k] >= 0.) {
                double r = splits0[k];
                double c = _coeff_c2 * r*r - _coeff_c * (x-x0.x) * (x-x0.x);
                double d = sqrt(c+b*b);
                splits.push_back(b + d + x0.y);
                splits.push_back(b - d + x0.y);
            }
            xdbg<<"Axisymmetric adaptee with R = "<<R<<std::endl;
            xdbg<<"ymin .. ymax = "<<ymin<<" ... "<<ymax<<std::endl;
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
            xdbg<<"Non-axisymmetric adaptee\n";
            if (matrixA == 0.) {
                xdbg<<"matrixA == 0:\n";
                double xmin_1, xmax_1;
                std::vector<double> xsplits0;
                adaptee->getXRange(xmin_1,xmax_1,xsplits0);
                xdbg<<"xmin_1, xmax_1 = "<<xmin_1<<','<<xmax_1<<std::endl;
                ymin = matrixC * xmin_1 + matrixD * (x - x0.x - matrixA*xmin_1) / matrixB + x0.y;
                ymax = matrixC * xmax_1 + matrixD * (x - x0.x - matrixA*xmax_1) / matrixB + x0.y;
                if (ymax < ymin) std::swap(ymin,ymax);
                for(size_t k=0;k<xsplits0.size();++k) {
                    double xx = xsplits0[k];
                    splits.push_back(
                        matrixC * xx + matrixD * (x - x0.x - matrixA*xx) / matrixB + x0.y);
                }
            } else if (matrixB == 0.) {
                xdbg<<"matrixB == 0:\n";
                double ymin_1, ymax_1;
                std::vector<double> ysplits0;
                adaptee->getYRange(ymin_1,ymax_1,ysplits0);
                xdbg<<"ymin_1, ymax_1 = "<<ymin_1<<','<<ymax_1<<std::endl;
                ymin = matrixC * (x - x0.x - matrixB*ymin_1) / matrixA + matrixD*ymin_1 + x0.y;
                ymax = matrixC * (x - x0.x - matrixB*ymax_1) / matrixA + matrixD*ymax_1 + x0.y;
                if (ymax < ymin) std::swap(ymin,ymax);
                for(size_t k=0;k<ysplits0.size();++k) {
                    double yy = ysplits0[k];
                    splits.push_back(
                        matrixC * (x - x0.x - matrixB*yy) / matrixA + matrixD*yy + x0.y);
                }
            } else {
                xdbg<<"matrixA,B != 0:\n";
                double ymin_1, ymax_1;
                std::vector<double> xsplits0;
                adaptee->getYRange(ymin_1,ymax_1,xsplits0);
                xdbg<<"ymin_1, ymax_1 = "<<ymin_1<<','<<ymax_1<<std::endl;
                ymin = matrixC * (x - x0.x - matrixB*ymin_1) / matrixA + matrixD*ymin_1 + x0.y;
                ymax = matrixC * (x - x0.x - matrixB*ymax_1) / matrixA + matrixD*ymax_1 + x0.y;
                xdbg<<"From top and bottom: ymin,ymax = "<<ymin<<','<<ymax<<std::endl;
                if (ymax < ymin) std::swap(ymin,ymax);
                double xmin_1, xmax_1;
                std::vector<double> ysplits0;
                adaptee->getXRange(xmin_1,xmax_1,ysplits0);
                xdbg<<"xmin_1, xmax_1 = "<<xmin_1<<','<<xmax_1<<std::endl;
                ymin_1 = matrixC * xmin_1 + matrixD * (x - x0.x - matrixA*xmin_1) / matrixB + x0.y;
                ymax_1 = matrixC * xmax_1 + matrixD * (x - x0.x - matrixA*xmax_1) / matrixB + x0.y;
                xdbg<<"From left and right: ymin,ymax = "<<ymin_1<<','<<ymax_1<<std::endl;
                if (ymax_1 < ymin_1) std::swap(ymin_1,ymax_1);
                if (ymin_1 > ymin) ymin = ymin_1;
                if (ymax_1 < ymax) ymax = ymax_1;
                for(size_t k=0;k<ysplits0.size();++k) {
                    double yy = ysplits0[k];
                    splits.push_back(
                        matrixC * (x - x0.x - matrixB*yy) / matrixA + matrixD*yy + x0.y);
                }
                for(size_t k=0;k<xsplits0.size();++k) {
                    double xx = xsplits0[k];
                    splits.push_back(
                        matrixC * xx + matrixD * (x - x0.x - matrixA*xx) / matrixB + x0.y);
                }
            }
            xdbg<<"ymin .. ymax = "<<ymin<<" ... "<<ymax<<std::endl;
        }
    }

    // Specialization of fillKGrid is desired since the phase terms from shift 
    // are factorizable:
    void SBDistort::fillKGrid(KTable& kt) const 
    {
        double N = (double) kt.getN();
        double dk = kt.getDk();

#if 0
        // The simpler version, saved for reference
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
#else
        // A faster version that pulls out all the if statements
        kt.clearCache();

        if (x0.x==0. && x0.y==0.) {
            // Branch to faster calculation if there is no centroid shift:
            Position<double> k1(0.,0.);
            for (int ix = 0; ix <= N/2; ix++, k1.x += dk) kt.kSet2(ix,0,kValNoPhase(k1));
            k1.y = dk;
            Position<double> k2(0.,-dk);
            for (int iy = 1; iy < N/2; ++iy, k1.y += dk, k2.y -= dk) {
                k1.x = k2.x = 0.;
                for (int ix = 0; ix <= N/2; ++ix, k1.x += dk, k2.x += dk) {
                    kt.kSet2(ix,iy, kValNoPhase(k1));
                    kt.kSet2(ix,N-iy, kValNoPhase(k2));
                }
            }
            k1.x = 0.;
            for (int ix = 0; ix <= N/2; ix++, k1.x += dk) kt.kSet2(ix,N/2,kValNoPhase(k1));
        } else {
            std::complex<double> dxphase = std::polar(1.,-dk*x0.x);
            std::complex<double> dyphase = std::polar(1.,-dk*x0.y);
            // xphase, yphase: current phase value
            std::complex<double> yphase = 1.;
            Position<double> k1(0.,0.);
            std::complex<double> phase = yphase; // since kx=0 to start
            for (int ix = 0; ix <= N/2; ++ix, k1.x += dk, phase *= dxphase) {
                kt.kSet2(ix,0, kValNoPhase(k1) * phase);
            }
            k1.y = dk; yphase *= dyphase;
            Position<double> k2(0.,-dk);  
            std::complex<double> phase2;
            for (int iy = 1; iy < N/2; iy++, k1.y += dk, k2.y -= dk, yphase *= dyphase) {
                k1.x = k2.x = 0.; phase = yphase; phase2 = conj(yphase);
                for (int ix = 0; ix <= N/2; ++ix, k1.x += dk, k2.x += dk,
                     phase *= dxphase, phase2 *= dxphase) {
                    kt.kSet2(ix,iy, kValNoPhase(k1) * phase);
                    kt.kSet2(ix,N-iy, kValNoPhase(k2) * phase2);
                }
            }
            k1.x = 0.; phase = yphase; 
            for (int ix = 0; ix <= N/2; ++ix, k1.x += dk, phase *= dxphase) {
                kt.kSet2(ix,N/2, kValNoPhase(k1) * phase);
            }
        }
#endif
    }

    //
    // SBConvolve class - adding new members
    //
    void SBConvolve::add(const SBProfile& rhs) 
    {
        xdbg<<"Start SBConvolve::add.  Adding item # "<<plist.size()+1<<std::endl;
        // If this is the first thing being added to the list, initialize some accumulators
        if (plist.empty()) {
            x0 = y0 = 0.;
            fluxProduct = 1.;
            minMaxK = 0.;
            minStepK = 0.;
            isStillAxisymmetric = true;
        }

        // Need a non-const copy of the rhs:
        SBProfile* p=rhs.duplicate();

        // Keep track of where first new term is on list:
        Iter newptr = plist.end();

        // Add new terms(s) to the plist:
        SBConvolve *sbc = dynamic_cast<SBConvolve*> (p);
        if (sbc) {  
            // If rhs is an SBConvolve, copy its list here
            fluxScale *= sbc->fluxScale;
            for (Iter pptr = sbc->plist.begin(); pptr!=sbc->plist.end(); ++pptr) {
                if (!(*pptr)->isAnalyticK() && !_real_space) 
                    throw SBError("SBConvolve requires members to be analytic in k");
                if (!(*pptr)->isAnalyticX() && _real_space)
                    throw SBError("Real_space SBConvolve requires members to be analytic in x");
                if (newptr==plist.end()) {
                    plist.push_back((*pptr)->duplicate()); 
                    newptr = --plist.end();  // That was first new term
                } else {
                    plist.push_back((*pptr)->duplicate()); 
                }
            }
            delete sbc; // no memory leak! 
        } else {
            if (!rhs.isAnalyticK() && !_real_space) 
                throw SBError("SBConvolve requires members to be analytic in k");
            if (!rhs.isAnalyticX() && _real_space)
                throw SBError("Real-space SBConvolve requires members to be analytic in x");
            plist.push_back(p);
            newptr = --plist.end();  // That was first new term
        }

        // Accumulate properties of all terms
        while (newptr != plist.end()) {
            xdbg<<"SBConvolve component has maxK, stepK = "<<
                (*newptr)->maxK()<<" , "<<(*newptr)->stepK()<<std::endl;
            fluxProduct *= (*newptr)->getFlux();
            x0 += (*newptr)->centroid().x;
            y0 += (*newptr)->centroid().y;
            if ( minMaxK<=0. || (*newptr)->maxK() < minMaxK) minMaxK = (*newptr)->maxK();
            if ( minStepK<=0. || ((*newptr)->stepK() < minStepK)) minStepK = (*newptr)->stepK();
            isStillAxisymmetric = isStillAxisymmetric && (*newptr)->isAxisymmetric();
            newptr++;
        }
        xdbg<<"Net maxK, stepK = "<<minMaxK<<" , "<<minStepK<<std::endl;
    }

    void SBConvolve::fillKGrid(KTable& kt) const 
    {
        if (plist.empty()) kt.clear();
        ConstIter pptr = plist.begin();
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

    class ConvolveFunc : 
        public std::binary_function<double,double,double>
    {
    public:
        ConvolveFunc(const SBProfile* p1, const SBProfile* p2, const Position<double>& pos) :
            _p1(p1), _p2(p2), _pos(pos) {}

        double operator()(double x, double y) const 
        {
            xdbg<<"Convolve function for pos = "<<_pos<<" at x,y = "<<x<<','<<y<<std::endl;
            double v1 = _p1->xValue(Position<double>(x,y));
            double v2 = _p2->xValue(Position<double>(_pos.x-x,_pos.y-y));
            xdbg<<"Value = "<<v1<<" * "<<v2<<" = "<<v1*v2<<std::endl;
            return 
                _p1->xValue(Position<double>(x,y)) *
                _p2->xValue(Position<double>(_pos.x-x,_pos.y-y));
        }
    private:
        const SBProfile* _p1;
        const SBProfile* _p2;
        const Position<double>& _pos;
    };

    class YRegion :
        public std::unary_function<double, integ::IntRegion<double> >
    {
    public:
        YRegion(const SBProfile* p1, const SBProfile* p2, const Position<double>& pos) :
            _p1(p1), _p2(p2), _pos(pos) {}

        integ::IntRegion<double> operator()(double x) const
        {
            xdbg<<"Get IntRegion for pos = "<<_pos<<" at x = "<<x<<std::endl;
            // First figure out each profiles y region separately.
            double ymin1,ymax1;
            splits1.clear();
            _p1->getYRange(x,ymin1,ymax1,splits1);
            double ymin2,ymax2;
            splits2.clear();
            _p2->getYRange(_pos.x-x,ymin2,ymax2,splits2);

            // Then take the overlap relevant for the calculation:
            //     _p1->xValue(x,y) * _p2->xValue(_x0-x,_y0-y)
            xdbg<<"p1's y range = "<<ymin1<<" ... "<<ymax1<<std::endl;
            xdbg<<"p2's y range = "<<ymin2<<" ... "<<ymax2<<std::endl;
            double ymin = std::max(ymin1, _pos.y-ymax2);
            double ymax = std::min(ymax1, _pos.y-ymin2);
            xdbg<<"Y region for x = "<<x<<" = "<<ymin<<" ... "<<ymax<<std::endl;
            if (ymax < ymin) ymax = ymin;
            std::ostream* integ_dbgout = verbose_level >= 2 ? dbgout : 0;
            integ::IntRegion<double> reg(ymin,ymax,integ_dbgout);
            for(size_t k=0;k<splits1.size();++k) {
                double s = splits1[k];
                if (s > ymin && s < ymax) reg.addSplit(s);
            }
            for(size_t k=0;k<splits2.size();++k) {
                double s = _pos.y-splits2[k];
                if (s > ymin && s < ymax) reg.addSplit(s);
            }
            return reg;
        }
    private:
        const SBProfile* _p1;
        const SBProfile* _p2;
        const Position<double>& _pos;
        mutable std::vector<double> splits1, splits2;
    };

    // This class finds the overlap between the ymin/ymax values of two profiles.
    // For overlaps of one profile's min with the other's max, this informs how to 
    // adjust the xmin/xmax values to avoid the region where the integral is trivially 0.
    // This is important, because the abrupt shift from a bunch of 0's to not is 
    // hard for the integrator.  So it helps to figure this out in advance.
    // The other use of this it to see where the two ymin's or the two ymax's cross 
    // each other.  This also leads to an abrupt bend in the function being integrated, so 
    // it's easier if we put a split point there at the start.
    // The four cases are distinguished by a "mode" variable.  
    // mode = 1 and 2 are for finding where the ranges are disjoint.
    // mode = 3 and 4 are for finding the bends.
    struct OverlapFinder
    {
        OverlapFinder(const SBProfile* p1, const SBProfile* p2, const Position<double>& pos,
                      int mode) :
            _p1(p1), _p2(p2), _pos(pos), _mode(mode) 
        { assert(_mode >= 1 && _mode <= 4); }
        double operator()(double x) const
        {
            double ymin1, ymax1, ymin2, ymax2;
            splits.clear();
            _p1->getYRange(x,ymin1,ymax1,splits);
            _p2->getYRange(_pos.x-x,ymin2,ymax2,splits);
            // Note: the real ymin,ymax for p2 are _pos.y-ymax2 and _pos.y-ymin2
            ymin2 = _pos.y - ymin2;
            ymax2 = _pos.y - ymax2;
            std::swap(ymin2,ymax2);
            return 
                _mode == 1 ? ymax2 - ymin1 :
                _mode == 2 ? ymax1 - ymin2 :
                _mode == 3 ? ymax2 - ymax1 :
                /*_mode == 4*/ ymin2 - ymin1;
        }

    private:
        const SBProfile* _p1;
        const SBProfile* _p2;
        const Position<double>& _pos;
        int _mode;
        mutable std::vector<double> splits;
    };

    // We pull out this segment, since we do it twice.  Once with which = true, and once
    // with which = false.
    static void UpdateXRange(const OverlapFinder& func, double& xmin, double& xmax, 
                             const std::vector<double>& splits)
    {
        xdbg<<"Start UpdateXRange given xmin,xmax = "<<xmin<<','<<xmax<<std::endl;
        // Find the overlap at x = xmin:
        double yrangea = func(xmin);
        xdbg<<"yrange at x = xmin = "<<yrangea<<std::endl;

        // Find the overlap at x = xmax:
        double yrangeb = func(xmax);
        xdbg<<"yrange at x = xmax = "<<yrangeb<<std::endl;

        if (yrangea < 0. && yrangeb < 0.) {
            xdbg<<"Both ends are disjoint.  Check the splits.\n";
            std::vector<double> use_splits = splits;
            if (use_splits.size() == 0) {
                xdbg<<"No splits provided.  Use the middle instead.\n";
                use_splits.push_back( (xmin+xmax)/2. );
            }
            for (size_t k=0;k<use_splits.size();++k) {
                double xmid = use_splits[k];
                double yrangec = func(xmid);
                xdbg<<"yrange at x = "<<xmid<<" = "<<yrangec<<std::endl;
                if (yrangec > 0.) {
                    xdbg<<"Found a non-disjoint split\n";
                    xdbg<<"Separately adjust both xmin and xmax by finding zero crossings.\n";
                    Solve<OverlapFinder> solver1(func,xmin,xmid);
                    solver1.setMethod(Brent);
                    double root = solver1.root();
                    xdbg<<"Found root at "<<root<<std::endl;
                    xmin = root;
                    Solve<OverlapFinder> solver2(func,xmid,xmax);
                    solver2.setMethod(Brent);
                    root = solver2.root();
                    xdbg<<"Found root at "<<root<<std::endl;
                    xmax = root;
                    return;
                }
            }
            xdbg<<"All split locations are also disjoint, so set xmin = xmax.\n";
            xmin = xmax;
        } else if (yrangea > 0. && yrangeb > 0.) {
            xdbg<<"Neither end is disjoint.  Integrate the full range\n";
        } else {
            xdbg<<"One end is disjoint.  Find the zero crossing.\n";
            Solve<OverlapFinder> solver(func,xmin,xmax);
            solver.setMethod(Brent);
            double root = solver.root();
            xdbg<<"Found root at "<<root<<std::endl;
            if (yrangea < 0.) xmin = root;
            else xmax = root;
        }
    }

    static void AddSplitsAtBends(const OverlapFinder& func, double xmin, double xmax, 
                                 std::vector<double>& splits)
    {
        xdbg<<"Start AddSplitsAtBends given xmin,xmax = "<<xmin<<','<<xmax<<std::endl;
        // Find the overlap at x = xmin:
        double yrangea = func(xmin);
        xdbg<<"yrange at x = xmin = "<<yrangea<<std::endl;

        // Find the overlap at x = xmax:
        double yrangeb = func(xmax);
        xdbg<<"yrange at x = xmax = "<<yrangeb<<std::endl;

        if (yrangea * yrangeb > 0.) {
            xdbg<<"Both ends are the same sign.  Check the splits.\n";
            std::vector<double> use_splits = splits;
            if (use_splits.size() == 0) {
                xdbg<<"No splits provided.  Use the middle instead.\n";
                use_splits.push_back( (xmin+xmax)/2. );
            }
            for (size_t k=0;k<use_splits.size();++k) {
                double xmid = use_splits[k];
                double yrangec = func(xmid);
                xdbg<<"yrange at x = "<<xmid<<" = "<<yrangec<<std::endl;
                if (yrangea * yrangec < 0.) {
                    xdbg<<"Found split with the opposite sign\n";
                    xdbg<<"Find crossings on both sides:\n";
                    Solve<OverlapFinder> solver1(func,xmin,xmid);
                    solver1.setMethod(Brent);
                    double root = solver1.root();
                    xdbg<<"Found root at "<<root<<std::endl;
                    splits.push_back(root);
                    Solve<OverlapFinder> solver2(func,xmid,xmax);
                    solver2.setMethod(Brent);
                    root = solver2.root();
                    xdbg<<"Found root at "<<root<<std::endl;
                    splits.push_back(root);
                    return;
                }
            }
            xdbg<<"All split locations have the same sign, so don't add any new splits\n";
        } else {
            xdbg<<"Ends have opposite signs.  Look for zero crossings.\n";
            Solve<OverlapFinder> solver(func,xmin,xmax);
            solver.setMethod(Brent);
            double root = solver.root();
            xdbg<<"Found root at "<<root<<std::endl;
            splits.push_back(root);
        }
    }

    static double RealSpaceConvolve(
        const SBProfile* p1, const SBProfile* p2, const Position<double>& pos, double flux)
    {
        // Coming in, if only one of them is axisymmetric, it should be p1.
        // This cuts down on some of the logic below.
        // Furthermore, the calculation of xmin, xmax isn't optimal if both are
        // axisymmetric.  But that involves a bit of geometry to get the right cuts,
        // so I didn't bother, since I don't think we'll be doing that too often.
        // So p2 is always taken to be a rectangle rather than possibly a circle.
        assert(p1->isAxisymmetric() || !p2->isAxisymmetric());
        
        xdbg<<"Start RealSpaceConvolve for pos = "<<pos<<std::endl;
        double xmin1, xmax1, xmin2, xmax2;
        std::vector<double> xsplits1, xsplits2;
        p1->getXRange(xmin1,xmax1,xsplits1);
        p2->getXRange(xmin2,xmax2,xsplits2);
        xdbg<<"p1 X range = "<<xmin1<<"  "<<xmax1<<std::endl;
        xdbg<<"p2 X range = "<<xmin2<<"  "<<xmax2<<std::endl;

        // Check for early exit
        if (pos.x < xmin1 + xmin2 || pos.x > xmax1 + xmax2) {
            xdbg<<"x is outside range, so trivially 0\n";
            return 0;
        }

        double ymin1, ymax1, ymin2, ymax2;
        std::vector<double> ysplits1, ysplits2;
        p1->getYRange(ymin1,ymax1,ysplits1);
        p2->getYRange(ymin2,ymax2,ysplits2);
        xdbg<<"p1 Y range = "<<ymin1<<"  "<<ymax1<<std::endl;
        xdbg<<"p2 Y range = "<<ymin2<<"  "<<ymax2<<std::endl;
        // Second check for early exit
        if (pos.y < ymin1 + ymin2 || pos.y > ymax1 + ymax2) {
            xdbg<<"y is outside range, so trivially 0\n";
            return 0;
        }

        double xmin = std::max(xmin1, pos.x - xmax2);
        double xmax = std::min(xmax1, pos.x - xmin2);
        xdbg<<"xmin..xmax = "<<xmin<<" ... "<<xmax<<std::endl;

        // Consolidate the splits from each profile in to a single list to use.
        std::vector<double> xsplits;
        for(size_t k=0;k<xsplits1.size();++k) {
            double s = xsplits1[k];
            xdbg<<"p1 has split at "<<s<<std::endl;
            if (s > xmin && s < xmax) xsplits.push_back(s);
        }
        for(size_t k=0;k<xsplits2.size();++k) {
            double s = pos.x-xsplits2[k];
            xdbg<<"p2 has split at "<<xsplits2[k]<<", which is really (pox.x-s) "<<s<<std::endl;
            if (s > xmin && s < xmax) xsplits.push_back(s);
        }

        // If either profile is infinite, then we don't need to worry about any boundary
        // overlaps, so can skip this section.
        if ( (xmin1 == -integ::MOCK_INF || xmax2 == integ::MOCK_INF) &&
             (xmax1 == integ::MOCK_INF || xmin2 == -integ::MOCK_INF) ) {

            // Update the xmin and xmax values if the top of one profile crosses through
            // the bootom of the other.  Then part of the nominal range will in fact
            // be disjoint.  This leads to a bunch of 0's for the inner integral which
            // makes it harder for the outer integral to converge.
            OverlapFinder func1(p1,p2,pos,1);
            UpdateXRange(func1,xmin,xmax,xsplits);
            OverlapFinder func2(p1,p2,pos,2);
            UpdateXRange(func2,xmin,xmax,xsplits);

            // Third check for early exit
            if (xmin >= xmax) { 
                xdbg<<"p1 and p2 are disjoint, so trivially 0\n";
                return 0.; 
            }

            // Also check for where the two tops or the two bottoms might cross.
            // Then we don't have zero's, but the curve being integrated over gets a bend,
            // which also makes it hard for the outer integral to converge, so we
            // want to add split points at those bends.
            OverlapFinder func3(p1,p2,pos,3);
            AddSplitsAtBends(func3,xmin,xmax,xsplits);
            OverlapFinder func4(p1,p2,pos,4);
            AddSplitsAtBends(func4,xmin,xmax,xsplits);
        }

        ConvolveFunc conv(p1,p2,pos);

        std::ostream* integ_dbgout = verbose_level >= 2 ? dbgout : 0;
        integ::IntRegion<double> xreg(xmin,xmax,integ_dbgout);
        if (dbgout && verbose_level >= 2) xreg.useFXMap();
        xdbg<<"xreg = "<<xmin<<" ... "<<xmax<<std::endl;

        // Need to re-check validity of splits, since xmin,xmax may have changed.
        for(size_t k=0;k<xsplits.size();++k) {
            double s = xsplits[k];
            if (s > xmin && s < xmax) xreg.addSplit(s);
        }

        YRegion yreg(p1,p2,pos);


#ifdef TIMING
        timeval tp;
        gettimeofday(&tp,0);
        double t1 = tp.tv_sec + tp.tv_usec/1.e6;
#endif

        double result = integ::int2d(conv, xreg, yreg, 
                                     sbp::realspace_conv_relerr,
                                     sbp::realspace_conv_abserr * flux);

#ifdef TIMING
        gettimeofday(&tp,0);
        double t2 = tp.tv_sec + tp.tv_usec/1.e6;
        xdbg<<"Time for ("<<pos.x<<','<<pos.y<<") = "<<t2-t1<<std::endl;
#endif

        xdbg<<"Found result = "<<result<<std::endl;
        return result;
    }

    double SBConvolve::xValue(const Position<double>& pos) const
    {
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
            const SBProfile* p1 = plist.front();
            const SBProfile* p2 = plist.back();
            if (p2->isAxisymmetric())
                return RealSpaceConvolve(p2,p1,pos,fluxProduct);
            else 
                return RealSpaceConvolve(p1,p2,pos,fluxProduct);
        }
    }

    double SBConvolve::getPositiveFlux() const {
        if (plist.empty()) return 0.;
        std::list<SBProfile*>::const_iterator pptr = plist.begin();
        double pResult = (*pptr)->getPositiveFlux() * fluxScale;
        double nResult = (*pptr)->getNegativeFlux() * fluxScale;
        for (++pptr; pptr!=plist.end(); ++pptr) {
            double p = (*pptr)->getPositiveFlux();
            double n = (*pptr)->getNegativeFlux();
            double pNew = p*pResult + n*nResult;
            nResult = p*nResult + n*pResult;
            pResult = pNew;
        }
        return pResult;
    }

    // Note duplicated code here, could be caching results for tiny efficiency gain
    double SBConvolve::getNegativeFlux() const {
        if (plist.empty()) return 0.;
        std::list<SBProfile*>::const_iterator pptr = plist.begin();
        double pResult = (*pptr)->getPositiveFlux() * fluxScale;
        double nResult = (*pptr)->getNegativeFlux() * fluxScale;
        for (++pptr; pptr!=plist.end(); ++pptr) {
            double p = (*pptr)->getPositiveFlux();
            double n = (*pptr)->getNegativeFlux();
            double pNew = p*pResult + n*nResult;
            nResult = p*nResult + n*pResult;
            pResult = pNew;
        }
        return nResult;
    }

    //
    // "SBGaussian" Class 
    //
    double SBGaussian::xValue(const Position<double>& p) const
    {
        double r2 = p.x*p.x + p.y*p.y;
        double xval = flux * std::exp( -r2/(2.*_sigma_sq) );
        xval /= 2.*M_PI*(_sigma_sq);  // normalize
        return xval;
    }

    std::complex<double> SBGaussian::kValue(const Position<double>& p) const
    {
        double r2 = p.x*p.x + p.y*p.y;
        return flux * std::exp(-r2 * _sigma_sq/2.);
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
        for (int i=0; i<3; i++) R = -std::log(ALIAS_THRESHOLD) + std::log(1.+R);
        R = std::max(6., R);
        return M_PI / (R*r0);
    }

    double SBExponential::xValue(const Position<double>& p) const
    {
        double r = std::sqrt(p.x*p.x + p.y*p.y);
        double xval = flux * std::exp(-r/r0);
        xval /= _r0_sq*2.*M_PI;   // normalize
        return xval;
    }

    std::complex<double> SBExponential::kValue(const Position<double>& p) const 
    {
        double kk = p.x*p.x+p.y*p.y;
        double temp = 1. + kk*_r0_sq;         // [1+k^2*r0^2]
        return flux/std::sqrt(temp*temp*temp);
        // NB: flux*std::pow(temp,-1.5) is slower.
    }

    //
    // SBAiry Class
    //

    // This is a scale-free version of the Airy radial function.
    // Input radius is in units of lambda/D.  Output normalized
    // to integrate to unity over input units.
    double SBAiry::AiryRadialFunction::operator()(double radius) const 
    {
        double nu = radius*M_PI;
        double xval;
        if (nu<0.01) {
            // lim j1(u)/u = 1/2
            xval =  (1-_obscuration*_obscuration);
        } else {
            xval = 2*( j1(nu) - _obscuration*j1(_obscuration*nu)) /
                nu ; //See Schroeder eq (10.1.10)
        }
        xval*=xval;
        // Normalize to give unit flux integrated over area.
        xval /= (1-_obscuration*_obscuration)*4./M_PI;
        return xval;
    }

    double SBAiry::xValue(const Position<double>& p) const 
    {
        double radius = std::sqrt(p.x*p.x+p.y*p.y) * D;
        return norm * _radial(radius);
    }

    double SBAiry::chord(const double r, const double h) const 
    {
        if (r<h) throw SBError("Airy calculation r<h");
        else if (r==0.) return 0.;
        else if (r<0. || h<0.) throw SBError("Airy calculation (r||h)<0");
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
        if (h<0.) {
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
            - 2. * circle_intersection(r1,r2,t)
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

    std::complex<double> SBAiry::kValue(const Position<double>& p) const
    {
        double radius = std::sqrt(p.x*p.x+p.y*p.y);
        // calculate circular FT(PSF) on p'=(x',y')
        return flux * annuli_autocorrelation(radius);
    }


    //
    // SBBox Class
    //

    double SBBox::xValue(const Position<double>& p) const 
    {
        if (fabs(p.x) < 0.5*xw && fabs(p.y) < 0.5*yw) return _norm;
        else return 0.;  // do not use this function for fillXGrid()!
    }

    double SBBox::sinc(const double u) const 
    {
        if (u<0.001 && u>-0.001)
            return 1.-u*u/6.;
        else
            return std::sin(u)/u;
    }

    std::complex<double> SBBox::kValue(const Position<double>& p) const
    {
        return flux * sinc(0.5*p.x*xw)*sinc(0.5*p.y*yw);
    }

    // Override fillXGrid so we can partially fill pixels at edge of box.
    void SBBox::fillXGrid(XTable& xt) const 
    {
        int N = xt.getN();
        double dx = xt.getDx(); // pixel grid size

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
    double SBBox::fillXImage(ImageView<T>& I, double dx) const 
    {
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
    void SBBox::fillKGrid(KTable& kt) const 
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
                double kvalue = flux * sinc(0.5*k.x*xw) * sinc(0.5*k.y*yw);
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
        if (xw == yw) { // Typical
            for (int i = 0; i <= N/2; i++) {
                sinc_x[i] = sinc(0.5 * i * dk * xw);
                sinc_y[i] = sinc_x[i];
            }
        } else {
            for (int i = 0; i <= N/2; i++) {
                sinc_x[i] = sinc(0.5 * i * dk * xw);
                sinc_y[i] = sinc(0.5 * i * dk * yw);
            }
        }

        // Now do the unrolled version with kSet2
        for (int ix = 0; ix <= N/2; ix++) {
            kt.kSet2(ix,0, flux * sinc_x[ix] * sinc_y[0]);
        }
        for (int iy = 1; iy < N/2; iy++) {
            for (int ix = 0; ix <= N/2; ix++) {
                double kval = flux * sinc_x[ix] * sinc_y[iy];
                kt.kSet2(ix,iy,kval);
                kt.kSet2(ix,N-iy,kval);
            }
        }
        for (int ix = 0; ix <= N/2; ix++) {
            kt.kSet2(ix,N/2, flux * sinc_x[ix] * sinc_y[N/2]);
        }
#endif
    }

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

    double SBLaguerre::xValue(const Position<double>& p) const 
    {
        LVector psi(bvec.getOrder());
        psi.fillBasis(p.x/sigma, p.y/sigma, sigma);
        double xval = bvec.dot(psi);
        return xval;
    }

    std::complex<double> SBLaguerre::kValue(const Position<double>& k) const 
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
        return std::complex<double>(2.*M_PI*rr, 2.*M_PI*ii);
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
        if (getFlux()!=0.) newflux /= getFlux();
        bvec.rVector() *= newflux;
    }

#endif

    // SBSersic Class 
    // First need to define the static member that holds info on all the Sersic n's
    SBSersic::InfoBarn SBSersic::nmap;

    double SBSersic::SersicInfo::kValue(double ksq) const 
    {
        assert(ksq >= 0.);

        if (ksq>=ksqMax)
            return 0.; // truncate the Fourier transform
        if (ksq<ksqMin)
            return 1. + ksq*(kderiv2 + ksq*kderiv4); // Use quartic approx at low k

        double lk=0.5*std::log(ksq); // Lookup table is logarithmic

        // simple linear interpolation to this value
        double fstep = (lk-logkMin)/logkStep;
        double findex = std::floor(fstep);
        int index = int(findex);
        assert(index < int(lookup.size())-1);
        fstep -= findex;
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
        b = 2.*n - (1./3.)
            + (4./405.)/n
            + (46./25515.)/(n*n)
            + (131./1148175.)/(n*n*n)
            - (2194697./30690717750.)/(n*n*n*n);

        double b2n = std::pow(b,2.*n);  // used frequently here
        // The normalization factor to give unity flux integral:
        norm = b2n / (2.*M_PI*n*tgamma(2.*n));

        // The quadratic term of small-k expansion:
        kderiv2 = -tgamma(4.*n) / (4.*b2n*tgamma(2.*n)) ; 
        // And a quartic term:
        kderiv4 = tgamma(6.*n) / (64.*b2n*b2n*tgamma(2.*n));

        xdbg << "Building for n=" << n << " b= " << b << " norm= " << norm << std::endl;
        xdbg << "Deriv terms: " << kderiv2 << " " << kderiv4 << std::endl;

        // When is it safe to use low-k approximation?  See when
        // quartic term is at threshold
        double lookupMin = 0.05; // Default lower limit for lookup table
        const double kAccuracy=0.001; // What errors in FT coefficients are acceptable?
        double smallK = std::pow(kAccuracy / kderiv4, 0.25);
        if (smallK < lookupMin) lookupMin = smallK;
        logkMin = std::log(lookupMin);
        ksqMin = lookupMin * lookupMin;

        // How far should nominal profile extend?
        // Estimate number of effective radii needed to enclose

        xdbg<<"Determine xMax\n";
        double xMax = 5.; // Go to at least 5r_e
        {
            // Successive approximation method:
            double a=2.*n;
            double z=a;
            double oldz=0.;
            int niter=0;
            const int MAXIT = 15;
            xdbg<<"Start with z = "<<z<<std::endl;
            while ( std::abs(oldz-z)>0.01 && niter<MAXIT) {
                niter++;
                oldz = z;
                z = a - std::log(ALIAS_THRESHOLD*std::sqrt(2*M_PI*a)*(1.+1./(12.*a)+1./(288.*a*a)))
                    +(a-1.)*std::log(z/a) + std::log(1. + (a-1.)/z + (a-1.)*(a-2.)/(z*z));
            }
            xdbg<<"Converged at z = "<<z<<std::endl;
            double r=std::pow(z/b, n);
            xdbg<<"r = (z/b)^n = "<<r<<std::endl;
            if (r>xMax) xMax = r;
            xdbg<<"xMax = "<<xMax<<std::endl;
        }
        stepK = M_PI / xMax;
        xdbg<<"stepK = "<<stepK<<std::endl;

        // Going to calculate another outer radius for the integration of the 
        // Hankel transforms:
        xdbg<<"Determine integrateMax\n";
        double integrateMax=xMax;
        const double integrationLoss=0.001;
        {
            // Successive approximation method:
            double a=2.*n;
            double z=a;
            double oldz=0.;
            int niter=0;
            const int MAXIT = 15;
            xdbg<<"Start with z = "<<z<<std::endl;
            while ( std::abs(oldz-z)>0.01 && niter<MAXIT) {
                niter++;
                oldz = z;
                z = a - std::log(integrationLoss*std::sqrt(2.*M_PI*a)*(1.+1./(12.*a)+1./(288.*a*a)))
                    +(a-1.)*std::log(z/a) + std::log(1. + (a-1.)/z + (a-1.)*(a-2.)/(z*z));
            }
            xdbg<<"Converged at z = "<<z<<std::endl;
            double r=std::pow(z/b, n);
            xdbg << "99.9% radius " << r <<std::endl;
            if (r>integrateMax) integrateMax = r;    
        }

        // Normalization for integral at k=0:
        double norm;
        const double INTEGRATION_RELTOL=0.0001;
        const double INTEGRATION_ABSTOL=1e-5;
        {
            SersicIntegrand I(n, b, 0.);
            norm = integ::int1d(
                I, 0., integrateMax, INTEGRATION_RELTOL, INTEGRATION_ABSTOL);
        }

        // Now start building the lookup table for FT of the profile.
        // Keep track of where the FT drops below ALIAS_THRESHOLD - this
        // will be our maxK.
        // Then extend the table another order of magnitude either in k
        //  or in FT, whichever comes first.
        xdbg<<"Determine maxK\n";
        logkStep = 0.05;
        // Here is preset range of acceptable maxK:
        const double MINMAXK = 10.;
        const double MAXMAXK = 50.; 
        maxK = MINMAXK;
        double lastVal=1.;
        double lk = logkMin;
        xdbg<<"logkMin = "<<logkMin<<std::endl;
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
        xdbg<<"maxK with val >= ALIAS_THRESHOLD ("<<ALIAS_THRESHOLD<<") = "<<maxK<<std::endl;
        maxK = std::min(MAXMAXK, maxK); // largest acceptable
        xdbg<<"Final maxK = "<<maxK<<std::endl;
        ksqMax = exp(2.*logkMax);

        // Next, set up the classes for photon shooting
        _radialPtr = new SersicRadialFunction(n, b);
        std::vector<double> range(2,0.);
        range[1] = integrateMax;
        _sampler = new OneDimensionalDeviate( *_radialPtr, range, true);
    }

    PhotonArray SBSersic::SersicInfo::shoot(int N, UniformDeviate& ud) const {
        PhotonArray result = _sampler->shoot(N,ud);
        result.scaleFlux(norm);
        return result;
    }

    SBMoffat::SBMoffat(double beta_, double truncationFWHM, double flux_,
                       double size, RadiusType rType) : 
        beta(beta_), flux(flux_), ft(Table<double,double>::spline)
    {
        xdbg<<"Start SBMoffat constructor: \n";
        xdbg<<"beta = "<<beta<<"\n";
        xdbg<<"flux = "<<flux<<"\n";

        //First, relation between FWHM and rD:
        FWHMrD = 2.* std::sqrt(std::pow(2., 1./beta)-1.);
        xdbg<<"FWHMrD = "<<FWHMrD<<"\n";
        maxRrD = FWHMrD * truncationFWHM;
        xdbg<<"maxRrD = "<<maxRrD<<"\n";
#if 1
        // Make FFT's periodic at 4x truncation radius or 1.5x diam at ALIAS_THRESHOLD,
        // whichever is smaller
        stepKrD = 2.*M_PI / std::min(4.*maxRrD, 
                                     3.*std::sqrt(pow(ALIAS_THRESHOLD, -1./beta)-1.));
#else
        // Make FFT's periodic at 4x truncation radius or 8x half-light radius:
        stepKrD = M_PI / (2.*std::max(maxRrD, 16.));
#endif
        xdbg<<"stepKrD = "<<stepKrD<<"\n";
        // And be sure to get at least 16 pts across FWHM when drawing:
        maxKrD = 16.*M_PI / FWHMrD;
        xdbg<<"maxKrD = "<<maxKrD<<"\n";

        // Analytic integration of total flux:
        fluxFactor = 1. - pow( 1+maxRrD*maxRrD, (1.-beta));

        // Get half-light radius in units of rD:

        // Set size of this instance according to type of size given in constructor:
        switch (rType) {
          case FWHM:
               rD = size / FWHMrD;
               break;
          case HALF_LIGHT_RADIUS: 
               {
                   double rerD = sqrt( pow(1.-0.5*fluxFactor , 1./(1.-beta)) - 1.);
                   rD = size / rerD;
               }
               break;
          case SCALE_RADIUS:
               rD = size;
               break;
          default:
               throw SBError("Unknown SBMoffat::RadiusType");
        }
        _maxR = maxRrD * rD;
        _maxR_sq = _maxR * _maxR;
        _maxRrD_sq = maxRrD * maxRrD;
        _rD_sq = rD * rD;
        norm = flux * (beta - 1.) / (M_PI * fluxFactor * _rD_sq);

        xdbg << "Moffat rD " << rD << " fluxFactor " << fluxFactor
            << " norm " << norm << " maxRrD " << maxRrD << std::endl;
        xdbg << "maxR = "<<_maxR<<", maxK = "<<maxKrD/rD<<", stepK = "<<stepKrD/rD<<std::endl;

        if (beta == 1) pow_beta = &SBMoffat::pow_1;
        else if (beta == 2) pow_beta = &SBMoffat::pow_2;
        else if (beta == 3) pow_beta = &SBMoffat::pow_3;
        else if (beta == 4) pow_beta = &SBMoffat::pow_4;
        else if (beta == int(beta)) pow_beta = &SBMoffat::pow_int;
        else pow_beta = &SBMoffat::pow_gen;
    }

    void SBMoffat::setupFT() const
    {
        if (ft.size() > 0) return;

        // Get FFT by doing 2k transform over 2x the truncation radius
        // ??? need to do better here
        // ??? also install quadratic behavior near k=0?
        const int N=2048;
        double dx = std::max(4.*maxRrD, 64.) / N;
        XTable xt(N, dx);
        dx = xt.getDx();
        for (int iy=-N/2; iy<N/2; iy++) {
            for (int ix=-N/2; ix<N/2; ix++) {
                double rsq = dx*dx*(ix*ix+iy*iy);
                if (rsq <= _maxRrD_sq) xt.xSet(ix,iy,1./pow_beta(1.+rsq,beta));
                // XTable values are initialized to 0, so don't need to set ones with rsq > max
            }
        }
        KTable* kt = xt.transform();
        double dk = kt->getDk();
        double nn = flux / kt->kval(0,0).real();
        for (int i=0; i<=N/2; i++) {
            ft.addEntry( i*dk, kt->kval(0,-i).real() * nn);
        }
        delete kt;
    }

    std::complex<double> SBMoffat::kValue(const Position<double>& k) const 
    {
        setupFT();
        double kk = sqrt(k.x*k.x + k.y*k.y)*rD;
        if (kk > ft.argMax()) return 0.;
        else return ft(kk);
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
            PhotonArray pa = shoot(maxN, u);
            pa.scaleFlux(maxN / origN);
            pa.addTo(img);
            N -= maxN;
        }
        xdbg<<"shoot "<<N<<std::endl;
        PhotonArray pa = shoot(int(N), u);
        pa.scaleFlux(N / origN);
        pa.addTo(img);
    }
    
    PhotonArray SBAdd::shoot(int N, UniformDeviate& u) const 
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
        for (std::list<SBProfile*>::const_iterator pptr = plist.begin(); 
             pptr!= plist.end();
             ++pptr) {
            double thisAbsoluteFlux = (*pptr)->getPositiveFlux() + (*pptr)->getNegativeFlux();

            // How many photons to shoot from this summand?
            int thisN = remainingN;  // All of what's left, if this is the last summand...
            std::list<SBProfile*>::const_iterator nextPtr = pptr;
            ++nextPtr;
            if (nextPtr!=plist.end()) {
                // otherwise allocate a randomized fraction of the remaining photons to this summand:
                BinomialDeviate bd(u, remainingN, thisAbsoluteFlux/remainingAbsoluteFlux);
                thisN = bd();
            }
            if (thisN > 0) {
                PhotonArray thisPA = (*pptr)->shoot(thisN, u);
                // Now rescale the photon fluxes so that they are each nominally fluxPerPhoton
                // whereas the shoot() routine would have made them each nominally thisAbsoluteFlux/thisN
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

    PhotonArray SBConvolve::shoot(int N, UniformDeviate& u) const 
    {
        std::list<SBProfile*>::const_iterator pptr = plist.begin();
        if (pptr==plist.end())
            throw SBError("Cannot shoot() for empty SBConvolve");
        PhotonArray result = (*pptr)->shoot(N, u);
        if (fluxScale!=1.) result.scaleFlux(fluxScale);
        // It is necessary to shuffle when convolving because we do
        // do not have a gaurantee that the convolvee's photons are
        // uncorrelated, e.g. they might both have their negative ones
        // at the end.
        for (++pptr; pptr != plist.end(); ++pptr)
            result.convolveShuffle( (*pptr)->shoot(N, u), u);
        return result;
    }

    PhotonArray SBDistort::shoot(int N, UniformDeviate& u) const 
    {
        // Simple job here: just remap coords of each photon, then change flux
        // If there is overall magnification in the transform
        PhotonArray result = adaptee->shoot(N,u);
        for (int i=0; i<result.size(); i++) {
            Position<double> xy = fwd(Position<double>(result.getX(i),
                                                       result.getY(i))+x0);
            result.setPhoton(i,xy.x, xy.y, result.getFlux(i)*absdet);
        }
        return result;
    }

    PhotonArray SBGaussian::shoot(int N, UniformDeviate& u) const 
    {
        PhotonArray result(N);
        double fluxPerPhoton = flux/N;
        for (int i=0; i<N; i++) {
            // First get a point uniformly distributed on unit circle
            double xu, yu, rsq;
            do {
                xu = 2.*u()-1.;
                yu = 2.*u()-1.;
                rsq = xu*xu+yu*yu;
            } while (rsq>=1. || rsq==0.);
            
            // Then map it to desired Gaussian with analytic transformation
            double factor = sigma*sqrt( -2.*log(rsq)/rsq);
            result.setPhoton(i,factor*xu, factor*yu, fluxPerPhoton);
        }
        return result;
    }

    PhotonArray SBSersic::shoot(int N, UniformDeviate& ud) const
    {
        // Get photons from the SersicInfo structure, rescale flux and size for this instance
        PhotonArray result = info->shoot(N,ud);
        result.scaleFlux(flux);
        result.scaleXY(re);
        return result;
    }

    PhotonArray SBExponential::shoot(int N, UniformDeviate& u) const
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
                // Runt case of infinite radius - just set to origin:
                result.setPhoton(i,0.,0.,fluxPerPhoton);
                continue;
            }
            // Initial guess
            y = -std::log(y);
            double r = y>2 ? y : std::sqrt(2*y);
            double dy = y - r + std::log(1+r);
            while ( std::abs(dy) > Y_TOLERANCE) {
                r = r + (1+r)*dy/r;
                dy = y - r + std::log(1+r);
            }
            // Draw another random for azimuthal angle (could use the unit-circle trick here...)
            double theta = 2*M_PI*u();
            result.setPhoton(i,r0*r*std::cos(theta), r0*r*std::sin(theta), fluxPerPhoton);
        }
        return result;
    }

    PhotonArray SBAiry::shoot(int N, UniformDeviate& u) const
    {
        // Use the OneDimensionalDeviate to sample from scale-free distribution
        checkSampler();
        PhotonArray pa=_sampler->shoot(N, u);
        // Then rescale for this flux & size
        pa.scaleFlux(flux);
        pa.scaleXY(1./D);
        return pa;
    }

    void SBAiry::flushSampler() const {
        if (_sampler) {
            delete _sampler;
            _sampler = 0;
        }
    }

    void SBAiry::checkSampler() const {
        if (_sampler) return;
        std::vector<double> ranges(1,0.);
        // Break Airy function into ranges that will not have >1 extremum:
        double xmin = (1.1 - 0.5*obscuration);
        // Use Schroeder (10.1.18) limit of EE at large radius.
        // to stop sampler at radius with EE>(1-ALIAS_THRESHOLD).
        double maximumRadius = 2./(ALIAS_THRESHOLD * M_PI*M_PI * (1-obscuration));
        while (xmin < maximumRadius) {
            ranges.push_back(xmin);
            xmin += 0.5;
        }
        ranges.push_back(xmin);
        _sampler = new OneDimensionalDeviate(_radial, ranges, true);
    }

    PhotonArray SBBox::shoot(int N, UniformDeviate& u) const
    {
        PhotonArray result(N);
        for (int i=0; i<result.size(); i++)
            result.setPhoton(i, xw*(u()-0.5), yw*(u()-0.5), flux/N);
        return result;
    }

    PhotonArray SBMoffat::shoot(int N, UniformDeviate& u) const
    {
        // Moffat has analytic inverse-cumulative-flux function.
        PhotonArray result(N);
        double fluxPerPhoton = flux/N;
        for (int i=0; i<N; i++) {
            // First get a point uniformly distributed on unit circle
            double xu, yu, rsq;
            do {
                xu = 2.*u()-1.;
                yu = 2.*u()-1.;
                rsq = xu*xu+yu*yu;
            } while (rsq>=1. || rsq==0.);
            
            // Then map it to the Moffat flux distribution
            double newRsq = pow( 1.-rsq*fluxFactor , 1./(1.-beta)) - 1.;
            double rFactor = rD*sqrt(newRsq / rsq);
            result.setPhoton(i,rFactor*xu, rFactor*yu, fluxPerPhoton);
        }
        return result;
    }

    // instantiate template functions for expected image types
    template double SBProfile::doFillXImage2(ImageView<float>& img, double dx) const;
    template double SBProfile::doFillXImage2(ImageView<double>& img, double dx) const;

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
