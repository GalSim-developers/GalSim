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

#ifdef DEBUGLOGGING
std::ostream* dbgout = new std::ofstream("debug.out");
//std::ostream* dbgout = &std::cerr;
//std::ostream* dbgout = 0;
int verbose_level = 2;
#endif

#include <numeric>

namespace galsim {

    // ????? Change treatement of aliased images to simply add in the aliased
    // FT components instead of doing a larger FT and then subsampling!
    // ??? Make a formula for asymptotic high-k SBSersic::kValue ??


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
        xdbg<<"Start draw that returns ImageView"<<'\n';
        Image<float> img;
        draw(img, dx, wmult);
        return img.view();
    }

    template <typename T>
    double SBProfile::draw(ImageView<T>& img, double dx, int wmult) const 
    {
        xdbg<<"Start draw ImageView"<<'\n';
        if (isAnalyticX())
            return plainDraw(img, dx, wmult);
        else
            return fourierDraw(img, dx, wmult);
    }

    template <typename T>
    double SBProfile::draw(Image<T>& img, double dx, int wmult) const 
    {
        xdbg<<"Start draw Image"<<'\n';
        if (isAnalyticX())
            return plainDraw(img, dx, wmult);
        else
            return fourierDraw(img, dx, wmult);
    }

    // First is a simple case wherein we have a formula for x values:
    template <typename T>
    double SBProfile::plainDraw(ImageView<T>& I, double dx, int wmult) const 
    {
        xdbg<<"Start plainDraw ImageView"<<'\n';
        // Determine desired dx:
        xdbg<<"maxK = "<<maxK()<<'\n';
        if (dx<=0.) dx = M_PI / maxK();
        xdbg<<"dx = "<<dx<<'\n';
        // recenter an existing image, to be consistent with fourierDraw:
        int xSize = I.getXMax()-I.getXMin()+1, ySize = I.getYMax()-I.getYMin()+1;
        xdbg<<"xSize = "<<xSize<<'\n';
        I.setOrigin(-xSize/2, -ySize/2);

        return fillXImage(I, dx);
    }

    template <typename T>
    double SBProfile::plainDraw(Image<T>& I, double dx, int wmult) const 
    {
        xdbg<<"Start plainDraw Image"<<'\n';
        // Determine desired dx:
        xdbg<<"maxK = "<<maxK()<<'\n';
        if (dx<=0.) dx = M_PI / maxK();
        xdbg<<"dx = "<<dx<<'\n';
        if (!I.getBounds().isDefined()) {
            if (wmult<1) throw SBError("Requested wmult<1 in plainDraw()");
            // Need to choose an image size
            int N = static_cast<int> (std::ceil(2*M_PI/(dx*stepK())));
            xdbg<<"N = "<<N<<'\n';

            // Round up to an even value
            N = 2*( (N+1)/2);
            N *= wmult; // make even bigger if desired
            xdbg<<"N => "<<N<<'\n';
            Bounds<int> imgsize(-N/2, N/2-1, -N/2, N/2-1);
            xdbg<<"imgsize => "<<imgsize<<'\n';
            I.resize(imgsize);
        } else {
            // recenter an existing image, to be consistent with fourierDraw:
            int xSize = I.getXMax()-I.getXMin()+1, ySize = I.getYMax()-I.getYMin()+1;
            xdbg<<"xSize = "<<xSize<<'\n';
            I.setOrigin(-xSize/2, -ySize/2);
        }

        // TODO: If we decide not to keep the scale, then can switch to simply:
        // return fillXImage(I.view(), dx);
        // (And switch fillXImage to take a const ImageView<T>& argument.)
        ImageView<T> Iv = I.view();
        double ret = fillXImage(Iv, dx);
        I.setScale(Iv.getScale());
        xdbg<<"scale => "<<I.getScale()<<'\n';
        return ret;
    }
 
    template <typename T>
    double SBProfile::doFillXImage2(ImageView<T>& I, double dx) const 
    {
        xdbg<<"Start doFillXImage2"<<'\n';
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
        xdbg<<"scale => "<<I.getScale()<<'\n';
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

        xdbg << " maxK() " << maxK() << " dx " << dx << '\n';

        // Now decide how big the FT must be to avoid folding:
        double xRange = 2*M_PI*wmult / stepK();
        int Nnofold = static_cast<int> (std::ceil(xRange / dx -0.0001));
        xdbg << " stepK() " << stepK() << " Nnofold " << Nnofold << '\n';

        // W must make something big enough to cover the target image size:
        int xSize, ySize;
        xSize = I.getXMax()-I.getXMin()+1;
        ySize = I.getYMax()-I.getYMin()+1;
        if (xSize  > Nnofold) Nnofold = xSize;
        if (ySize  > Nnofold) Nnofold = ySize;
        xRange = Nnofold * dx;

        // Round up to a good size for making FFTs:
        int NFT = goodFFTSize(Nnofold);
        NFT = std::max(NFT,sbp::MINIMUM_FFT_SIZE);
        xdbg << " After adjustments: Nnofold " << Nnofold << " NFT " << NFT << '\n';
        if (NFT > sbp::MAXIMUM_FFT_SIZE)
            FormatAndThrow<SBError>() << 
                "fourierDraw() requires an FFT that is too large, " << NFT;

        // Move the output image to be centered near zero
        I.setOrigin(-xSize/2, -ySize/2);
        double dk = 2.*M_PI/(NFT*dx);
        xdbg << 
            " After adjustments: dx " << dx << " dk " << dk << 
            " maxK " << dk*NFT/2 << '\n';
        assert(dk <= stepK());
        XTable* xtmp=0;
        if (NFT*dk/2 > maxK()) {
            xdbg<<"NFT*dk/2 = "<<NFT*dk/2<<" > maxK() = "<<maxK()<<'\n';
            xdbg<<"Use NFT = "<<NFT<<'\n';
            // No aliasing: build KTable and transform
            KTable kt(NFT,dk);
            fillKGrid(kt); 
            xtmp = kt.transform();
        } else {
            xdbg<<"NFT*dk/2 = "<<NFT*dk/2<<" <= maxK() = "<<maxK()<<'\n';
            // There will be aliasing.  Construct a KTable out to maxK() and
            // then wrap it
            int Nk = static_cast<int> (std::ceil(maxK()/dk)) * 2;
            xdbg<<"Use Nk = "<<Nk<<'\n';
            KTable kt(Nk, dk);
            fillKGrid(kt);
            KTable* kt2 = kt.wrap(NFT);
            xtmp = kt2->transform();
            delete kt2;
        }
        int Nxt = xtmp->getN();
        xdbg<<"Nxt = "<<Nxt<<'\n';
        Bounds<int> xb(-Nxt/2, Nxt/2-1, -Nxt/2, Nxt/2-1);
        if (I.getYMin() < xb.getYMin()
            || I.getYMax() > xb.getYMax()
            || I.getXMin() < xb.getXMin()
            || I.getXMax() > xb.getXMax()) {
            std::cerr << "Bounds error!! target image bounds " << I.getBounds()
                << " and FFT range " << xb << '\n';
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

        xdbg << " maxK() " << maxK() << " dx " << dx << '\n';

        // Now decide how big the FT must be to avoid folding:
        double xRange = 2*M_PI*wmult / stepK();
        int Nnofold = static_cast<int> (std::ceil(xRange / dx -0.0001));
        xdbg << " stepK() " << stepK() << " Nnofold " << Nnofold << '\n';

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
        NFT = std::max(NFT,sbp::MINIMUM_FFT_SIZE);
        xdbg << " After adjustments: Nnofold " << Nnofold << " NFT " << NFT << '\n';
        if (NFT > sbp::MAXIMUM_FFT_SIZE)
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
        xdbg << 
            " After adjustments: dx " << dx << " dk " << dk << 
            " maxK " << dk*NFT/2 << '\n';
        assert(dk <= stepK());
        XTable* xtmp=0;
        if (NFT*dk/2 > maxK()) {
            xdbg<<"NFT*dk/2 = "<<NFT*dk/2<<" > maxK() = "<<maxK()<<'\n';
            xdbg<<"Use NFT = "<<NFT<<'\n';
            // No aliasing: build KTable and transform
            KTable kt(NFT,dk);
            fillKGrid(kt); 
            xtmp = kt.transform();
        } else {
            xdbg<<"NFT*dk/2 = "<<NFT*dk/2<<" <= maxK() = "<<maxK()<<'\n';
            // There will be aliasing.  Construct a KTable out to maxK() and
            // then wrap it
            int Nk = static_cast<int> (std::ceil(maxK()/dk)) * 2;
            xdbg<<"Use Nk = "<<Nk<<'\n';
            KTable kt(Nk, dk);
            fillKGrid(kt);
            KTable* kt2 = kt.wrap(NFT);
            xtmp = kt2->transform();
            delete kt2;
        }
        int Nxt = xtmp->getN();
        xdbg<<"Nxt = "<<Nxt<<'\n';
        Bounds<int> xb(-Nxt/2, Nxt/2-1, -Nxt/2, Nxt/2-1);
        if (I.getYMin() < xb.getYMin()
            || I.getYMax() > xb.getYMax()
            || I.getXMin() < xb.getXMin()
            || I.getXMax() > xb.getXMax()) {
            std::cerr << "Bounds error!! target image bounds " << I.getBounds()
                << " and FFT range " << xb << '\n';
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
        int NFT = sbp::MINIMUM_FFT_SIZE;
        while (NFT < Nnofold && NFT<= sbp::MAXIMUM_FFT_SIZE) NFT *= 2;
        if (NFT > sbp::MAXIMUM_FFT_SIZE)
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
                << " and FFT range " << kb << '\n';
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
        int NFT = sbp::MINIMUM_FFT_SIZE;
        while (NFT < Nnofold && NFT<= sbp::MAXIMUM_FFT_SIZE) NFT *= 2;
        if (NFT > sbp::MAXIMUM_FFT_SIZE)
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
                << " and FFT range " << kb << '\n';
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
        _sumflux = _sumfx = _sumfy = 0.;
        _maxMaxK = _minStepK = 0.;
        _allAxisymmetric = _allAnalyticX = _allAnalyticK = true;
    }

    void SBAdd::add(const SBProfile& rhs, double scale) 
    {
        xdbg<<"Start SBAdd::add.  Adding item # "<<_plist.size()+1<<'\n';
        // Need a non-const copy of the rhs:
        SBProfile* p=rhs.duplicate();

        // Keep track of where first new summand is on list:
        Iter newptr = _plist.end();

        // Add new summand(s) to the _plist:
        SBAdd *sba = dynamic_cast<SBAdd*>(p);
        if (sba) {
            // If rhs is an SBAdd, copy its full list here
            for (ConstIter pptr = sba->_plist.begin(); pptr!=sba->_plist.end(); ++pptr) {
                if (newptr==_plist.end()) {
                    _plist.push_back((*pptr)->duplicate()); 
                    // Rescale flux for duplicate copy if desired:
                    if (scale!=1.) 
                        _plist.back()->setFlux( scale*_plist.back()->getFlux());
                    newptr = --_plist.end();  // That was first new summand
                } else {
                    _plist.push_back((*pptr)->duplicate()); 
                }
            }
            delete sba; // no memory leak! 
        } else {
            _plist.push_back(p);
            // Rescale flux for duplicate copy if desired:
            if (scale!=1.) 
                _plist.back()->setFlux( scale*_plist.back()->getFlux());
            newptr = --_plist.end();  // That was first new summand
        }

        // Accumulate properties of all summands
        while (newptr != _plist.end()) {
            xdbg<<"SBAdd component has maxK, stepK = "<<
                (*newptr)->maxK()<<" , "<<(*newptr)->stepK()<<'\n';
            _sumflux += (*newptr)->getFlux();
            _sumfx += (*newptr)->getFlux() * (*newptr)->centroid().x;
            _sumfy += (*newptr)->getFlux() * (*newptr)->centroid().x;
            if ( (*newptr)->maxK() > _maxMaxK) 
                _maxMaxK = (*newptr)->maxK();
            if ( _minStepK<=0. || ((*newptr)->stepK() < _minStepK) ) 
                _minStepK = (*newptr)->stepK();
            _allAxisymmetric = _allAxisymmetric && (*newptr)->isAxisymmetric();
            _allAnalyticX = _allAnalyticX && (*newptr)->isAnalyticX();
            _allAnalyticK = _allAnalyticK && (*newptr)->isAnalyticK();
            newptr++;
        }
        xdbg<<"Net maxK, stepK = "<<_maxMaxK<<" , "<<_minStepK<<'\n';
    }

    double SBAdd::xValue(const Position<double>& p) const 
    {
        ConstIter pptr = _plist.begin();
        assert(pptr != _plist.end());
        double xv = (*pptr)->xValue(p);
        for (++pptr; pptr != _plist.end(); ++pptr)
            xv += (*pptr)->xValue(p);
        return xv;
    } 

    std::complex<double> SBAdd::kValue(const Position<double>& k) const 
    {
        ConstIter pptr = _plist.begin();
        assert(pptr != _plist.end());
        std::complex<double> kv = (*pptr)->kValue(k);
        for (++pptr; pptr != _plist.end(); ++pptr)
            kv += (*pptr)->kValue(k);
        return kv;
    } 

    void SBAdd::fillKGrid(KTable& kt) const 
    {
        if (_plist.empty()) kt.clear();
        ConstIter pptr = _plist.begin();
        (*pptr)->fillKGrid(kt);
        if (++pptr != _plist.end()) {
            KTable k2(kt.getN(),kt.getDk());
            for ( ; pptr!= _plist.end(); ++pptr) {
                (*pptr)->fillKGrid(k2);
                kt.accumulate(k2);
            }
        }
    }

    void SBAdd::fillXGrid(XTable& xt) const 
    {
        if (_plist.empty()) xt.clear();
        ConstIter pptr = _plist.begin();
        (*pptr)->fillXGrid(xt);
        if (++pptr != _plist.end()) {
            XTable x2(xt.getN(),xt.getDx());
            for ( ; pptr!= _plist.end(); ++pptr) {
                (*pptr)->fillXGrid(x2);
                xt.accumulate(x2);
            }
        }
    }

    void SBAdd::setFlux(double flux) 
    {
        if (_sumflux==0.) throw SBError("SBAdd::setFlux not possible when flux=0 to start");
        double m = flux/_sumflux;  // Factor by which to change flux
        for (Iter pptr = _plist.begin(); pptr != _plist.end(); ++pptr) {
            double pf = (*pptr)->getFlux();  
            (*pptr)->setFlux(pf*m);
        }
        _sumflux = flux;
        _sumfx *= m;
        _sumfy *= m;
    }

    double SBAdd::getPositiveFlux() const 
    {
        double result = 0.;
        for (ConstIter pptr = _plist.begin(); pptr != _plist.end(); ++pptr) {
            result += (*pptr)->getPositiveFlux();  
        }
        return result;
    }

    double SBAdd::getNegativeFlux() const 
    {
        double result = 0.;
        for (ConstIter pptr = _plist.begin(); pptr != _plist.end(); ++pptr) {
            result += (*pptr)->getNegativeFlux();  
        }
        return result;
    }
        

    //
    // "SBDistort" Class 
    //
    SBDistort::SBDistort(
        const SBProfile& sbin, double mA, double mB, double mC, double mD,
        const Position<double>& cen) :
        _mA(mA), _mB(mB), _mC(mC), _mD(mD), _cen(cen)
    {
        SBProfile* p=sbin.duplicate();
        SBDistort* sbd = dynamic_cast<SBDistort*> (p);
        if (sbd) {
            // We are distorting something that's already a distortion.
            // So just compound the affine transformaions
            _adaptee = sbd->_adaptee->duplicate();
            _cen = cen + fwd(sbd->_cen);
            // New matrix is product (M_this) * (M_old)
            _mA = mA*sbd->_mA + mB*sbd->_mC;
            _mB = mA*sbd->_mB + mB*sbd->_mD;
            _mC = mC*sbd->_mA + mD*sbd->_mC;
            _mD = mC*sbd->_mB + mD*sbd->_mD;
            delete sbd;
        } else {
            // Distorting something generic
            _adaptee = p;
        }
        initialize();
    }

    SBDistort::SBDistort(const SBProfile& sbin, const Ellipse e_) 
    {
        // First get what we need from the Ellipse:
        tmv::Matrix<double> m = e_.getMatrix();
        _mA = m(0,0);
        _mB = m(0,1);
        _mC = m(1,0);
        _mD = m(1,1);
        _cen = e_.getX0();
        // Then repeat generic construction:
        SBProfile* p=sbin.duplicate();
        SBDistort* sbd = dynamic_cast<SBDistort*> (p);
        if (sbd) {
            // We are distorting something that's already a distortion.
            // So just compound the affine transformaions
            _adaptee = sbd->_adaptee->duplicate();
            _cen = e_.getX0() + fwd(sbd->_cen);
            // New matrix is product (M_this) * (M_old)
            double mA = _mA; double mB=_mB; double mC=_mC; double mD=_mD;
            _mA = mA*sbd->_mA + mB*sbd->_mC;
            _mB = mA*sbd->_mB + mB*sbd->_mD;
            _mC = mC*sbd->_mA + mD*sbd->_mC;
            _mD = mC*sbd->_mB + mD*sbd->_mD;
            delete sbd;
        } else {
            // Distorting something generic
            _adaptee = p;
        }
        initialize();
    }

    void SBDistort::initialize() 
    {
        double det = _mA*_mD-_mB*_mC;
        if (det==0.) throw SBError("Attempt to SBDistort with degenerate matrix");
        _absdet = std::abs(det);
        _invdet = 1./det;

        double h1 = hypot( _mA+_mD, _mB-_mC);
        double h2 = hypot( _mA-_mD, _mB+_mC);
        _major = 0.5*std::abs(h1+h2);
        _minor = 0.5*std::abs(h1-h2);
        if (_major<_minor) std::swap(_major,_minor);
        _stillIsAxisymmetric = _adaptee->isAxisymmetric() 
            && (_mB==-_mC) 
            && (_mA==_mD)
            && (_cen.x==0.) && (_cen.y==0.); // Need pure rotation

        if (std::abs(det-1.) < sbp::kvalue_accuracy) 
            _kValueNoPhase = &SBDistort::_kValueNoPhaseNoDet;
        else
            _kValueNoPhase = &SBDistort::_kValueNoPhaseWithDet;
        if (_cen.x == 0. && _cen.y == 0.) _kValue = _kValueNoPhase;
        else _kValue = &SBDistort::_kValueWithPhase;

        xdbg<<"Distortion init\n";
        xdbg<<"matrix = "<<_mA<<','<<_mB<<','<<_mC<<','<<_mD<<'\n';
        xdbg<<"_cen = "<<_cen<<'\n';
        xdbg<<"_invdet = "<<_invdet<<'\n';
        xdbg<<"_major, _minor = "<<_major<<", "<<_minor<<'\n';
        xdbg<<"maxK() = "<<_adaptee->maxK() / _minor<<'\n';
        xdbg<<"stepK() = "<<_adaptee->stepK() / _major<<'\n';

        // Calculate the values for getXRange and getYRange:
        if (_adaptee->isAxisymmetric()) {
            // The original is a circle, so first get its radius.
            _adaptee->getXRange(_xmin,_xmax,_xsplits);
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
                    xxdbg<<"Adaptee split at "<<split<<'\n';
                    _xsplits[k] = sqrtAApBB * split + _cen.x;
                    _ysplits[k] = sqrtCCpDD * split + _cen.y;
                    xxdbg<<"-> x,y splits at "<<_xsplits[k]<<"  "<<_ysplits[k]<<'\n';
                }
                // Now a couple of calculations that get reused in getYRange(x,yminymax):
                _coeff_b = (_mA*_mC + _mB*_mD) / AApBB;
                _coeff_c = CCpDD / AApBB;
                _coeff_c2 = _absdet*_absdet / AApBB;
                xxdbg<<"adaptee is axisymmetric.\n";
                xxdbg<<"adaptees maxR = "<<R<<'\n';
                xxdbg<<"xmin..xmax = "<<_xmin<<" ... "<<_xmax<<'\n';
                xxdbg<<"ymin..ymax = "<<_ymin<<" ... "<<_ymax<<'\n';
            }
        } else {
            // Apply the distortion to each of the four corners of the original
            // and find the minimum and maximum.
            double xmin_1, xmax_1;
            std::vector<double> xsplits0;
            _adaptee->getXRange(xmin_1,xmax_1,xsplits0);
            double ymin_1, ymax_1;
            std::vector<double> ysplits0;
            _adaptee->getYRange(ymin_1,ymax_1,ysplits0);
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
            xxdbg<<"adaptees x range = "<<xmin_1<<" ... "<<xmax_1<<'\n';
            xxdbg<<"adaptees y range = "<<ymin_1<<" ... "<<ymax_1<<'\n';
            xxdbg<<"Corners are: bl = "<<bl<<'\n';
            xxdbg<<"             br = "<<br<<'\n';
            xxdbg<<"             tl = "<<tl<<'\n';
            xxdbg<<"             tr = "<<tr<<'\n';
            xxdbg<<"xmin..xmax = "<<_xmin<<" ... "<<_xmax<<'\n';
            xxdbg<<"ymin..ymax = "<<_ymin<<" ... "<<_ymax<<'\n';
            if (bl.x + _cen.x > _xmin && bl.x + _cen.x < _xmax) {
                xxdbg<<"X Split from bl.x = "<<bl.x+_cen.x<<'\n';
                _xsplits.push_back(bl.x+_cen.x);
            }
            if (br.x + _cen.x > _xmin && br.x + _cen.x < _xmax) {
                xxdbg<<"X Split from br.x = "<<br.x+_cen.x<<'\n';
                _xsplits.push_back(br.x+_cen.x);
            }
            if (tl.x + _cen.x > _xmin && tl.x + _cen.x < _xmax) {
                xxdbg<<"X Split from tl.x = "<<tl.x+_cen.x<<'\n';
                _xsplits.push_back(tl.x+_cen.x);
            }
            if (tr.x + _cen.x > _xmin && tr.x + _cen.x < _xmax) {
                xxdbg<<"X Split from tr.x = "<<tr.x+_cen.x<<'\n';
                _xsplits.push_back(tr.x+_cen.x);
            }
            if (bl.y + _cen.y > _ymin && bl.y + _cen.y < _ymax) {
                xxdbg<<"Y Split from bl.y = "<<bl.y+_cen.y<<'\n';
                _ysplits.push_back(bl.y+_cen.y);
            }
            if (br.y + _cen.y > _ymin && br.y + _cen.y < _ymax) {
                xxdbg<<"Y Split from br.y = "<<br.y+_cen.y<<'\n';
                _ysplits.push_back(br.y+_cen.y);
            }
            if (tl.y + _cen.y > _ymin && tl.y + _cen.y < _ymax) {
                xxdbg<<"Y Split from tl.y = "<<tl.y+_cen.y<<'\n';
                _ysplits.push_back(tl.y+_cen.y);
            }
            if (tr.y + _cen.y > _ymin && tr.y + _cen.y < _ymax) {
                xxdbg<<"Y Split from tr.y = "<<tr.y+_cen.y<<'\n';
                _ysplits.push_back(tr.y+_cen.y);
            }
            // If the adaptee has any splits, try to propagate those up
            for(size_t k=0;k<xsplits0.size();++k) {
                xxdbg<<"Adaptee xsplit at "<<xsplits0[k]<<'\n';
                Position<double> bx = fwd(Position<double>(xsplits0[k],ymin_1));
                Position<double> tx = fwd(Position<double>(xsplits0[k],ymax_1));
                if (bx.x + _cen.x > _xmin && bx.x + _cen.x < _xmax) {
                    xxdbg<<"X Split from bx.x = "<<bx.x+_cen.x<<'\n';
                    _xsplits.push_back(bx.x+_cen.x);
                }
                if (tx.x + _cen.x > _xmin && tx.x + _cen.x < _xmax) {
                    xxdbg<<"X Split from tx.x = "<<tx.x+_cen.x<<'\n';
                    _xsplits.push_back(tx.x+_cen.x);
                }
                if (bx.y + _cen.y > _ymin && bx.y + _cen.y < _ymax) {
                    xxdbg<<"Y Split from bx.y = "<<bx.y+_cen.y<<'\n';
                    _ysplits.push_back(bx.y+_cen.y);
                }
                if (tx.y + _cen.y > _ymin && tx.y + _cen.y < _ymax) {
                    xxdbg<<"Y Split from tx.y = "<<tx.y+_cen.y<<'\n';
                    _ysplits.push_back(tx.y+_cen.y);
                }
            }
            for(size_t k=0;k<ysplits0.size();++k) {
                xxdbg<<"Adaptee ysplit at "<<ysplits0[k]<<'\n';
                Position<double> yl = fwd(Position<double>(xmin_1,ysplits0[k]));
                Position<double> yr = fwd(Position<double>(xmax_1,ysplits0[k]));
                if (yl.x + _cen.x > _xmin && yl.x + _cen.x < _xmax) {
                    xxdbg<<"X Split from tl.x = "<<tl.x+_cen.x<<'\n';
                    _xsplits.push_back(yl.x+_cen.x);
                }
                if (yr.x + _cen.x > _xmin && yr.x + _cen.x < _xmax) {
                    xxdbg<<"X Split from yr.x = "<<yr.x+_cen.x<<'\n';
                    _xsplits.push_back(yr.x+_cen.x);
                }
                if (yl.y + _cen.y > _ymin && yl.y + _cen.y < _ymax) {
                    xxdbg<<"Y Split from yl.y = "<<yl.y+_cen.y<<'\n';
                    _ysplits.push_back(yl.y+_cen.y);
                }
                if (yr.y + _cen.y > _ymin && yr.y + _cen.y < _ymax) {
                    xxdbg<<"Y Split from yr.y = "<<yr.y+_cen.y<<'\n';
                    _ysplits.push_back(yr.y+_cen.y);
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
        xxdbg<<"Distortion getYRange for x = "<<x<<'\n';
        if (_adaptee->isAxisymmetric()) {
            std::vector<double> splits0;
            _adaptee->getYRange(ymin,ymax,splits0);
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
            xxdbg<<"Axisymmetric adaptee with R = "<<R<<'\n';
            xxdbg<<"ymin .. ymax = "<<ymin<<" ... "<<ymax<<'\n';
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
                _adaptee->getXRange(xmin_1,xmax_1,xsplits0);
                xxdbg<<"xmin_1, xmax_1 = "<<xmin_1<<','<<xmax_1<<'\n';
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
                _adaptee->getYRange(ymin_1,ymax_1,ysplits0);
                xxdbg<<"ymin_1, ymax_1 = "<<ymin_1<<','<<ymax_1<<'\n';
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
                _adaptee->getYRange(ymin_1,ymax_1,xsplits0);
                xxdbg<<"ymin_1, ymax_1 = "<<ymin_1<<','<<ymax_1<<'\n';
                ymin = _mC * (x - _cen.x - _mB*ymin_1) / _mA + _mD*ymin_1 + _cen.y;
                ymax = _mC * (x - _cen.x - _mB*ymax_1) / _mA + _mD*ymax_1 + _cen.y;
                xxdbg<<"From top and bottom: ymin,ymax = "<<ymin<<','<<ymax<<'\n';
                if (ymax < ymin) std::swap(ymin,ymax);
                double xmin_1, xmax_1;
                std::vector<double> ysplits0;
                _adaptee->getXRange(xmin_1,xmax_1,ysplits0);
                xxdbg<<"xmin_1, xmax_1 = "<<xmin_1<<','<<xmax_1<<'\n';
                ymin_1 = _mC * xmin_1 + _mD * (x - _cen.x - _mA*xmin_1) / _mB + _cen.y;
                ymax_1 = _mC * xmax_1 + _mD * (x - _cen.x - _mA*xmax_1) / _mB + _cen.y;
                xxdbg<<"From left and right: ymin,ymax = "<<ymin_1<<','<<ymax_1<<'\n';
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
            xxdbg<<"ymin .. ymax = "<<ymin<<" ... "<<ymax<<'\n';
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

    std::complex<double> SBDistort::kValue(const Position<double>& k) const
    { return _kValue(_adaptee,fwdT(k),_absdet,k,_cen); }

    std::complex<double> SBDistort::kValueNoPhase(const Position<double>& k) const
    { return _kValueNoPhase(_adaptee,fwdT(k),_absdet,k,_cen); }

    std::complex<double> SBDistort::_kValueNoPhaseNoDet(
        const SBProfile* adaptee, const Position<double>& fwdTk, double absdet,
        const Position<double>& , const Position<double>& )
    { return adaptee->kValue(fwdTk); }

    std::complex<double> SBDistort::_kValueNoPhaseWithDet(
        const SBProfile* adaptee, const Position<double>& fwdTk, double absdet,
        const Position<double>& , const Position<double>& )
    { return absdet * adaptee->kValue(fwdTk); }

    std::complex<double> SBDistort::_kValueWithPhase(
        const SBProfile* adaptee, const Position<double>& fwdTk, double absdet,
        const Position<double>& k, const Position<double>& cen)
    { return adaptee->kValue(fwdTk) * std::polar(absdet , -k.x*cen.x-k.y*cen.y); }


    //
    // SBConvolve class - adding new members
    //
    void SBConvolve::add(const SBProfile& rhs) 
    {
        xdbg<<"Start SBConvolve::add.  Adding item # "<<_plist.size()+1<<'\n';
        // If this is the first thing being added to the list, initialize some accumulators
        if (_plist.empty()) {
            _x0 = _y0 = 0.;
            _fluxProduct = 1.;
            _minMaxK = 0.;
            _minStepK = 0.;
            _isStillAxisymmetric = true;
        }

        // Need a non-const copy of the rhs:
        SBProfile* p=rhs.duplicate();

        // Keep track of where first new term is on list:
        Iter newptr = _plist.end();

        // Add new terms(s) to the _plist:
        SBConvolve *sbc = dynamic_cast<SBConvolve*> (p);
        if (sbc) {  
            // If rhs is an SBConvolve, copy its list here
            _fluxScale *= sbc->_fluxScale;
            for (Iter pptr = sbc->_plist.begin(); pptr!=sbc->_plist.end(); ++pptr) {
                if (!(*pptr)->isAnalyticK() && !_real_space) 
                    throw SBError("SBConvolve requires members to be analytic in k");
                if (!(*pptr)->isAnalyticX() && _real_space)
                    throw SBError("Real_space SBConvolve requires members to be analytic in x");
                if (newptr==_plist.end()) {
                    _plist.push_back((*pptr)->duplicate()); 
                    newptr = --_plist.end();  // That was first new term
                } else {
                    _plist.push_back((*pptr)->duplicate()); 
                }
            }
            delete sbc; // no memory leak! 
        } else {
            if (!rhs.isAnalyticK() && !_real_space) 
                throw SBError("SBConvolve requires members to be analytic in k");
            if (!rhs.isAnalyticX() && _real_space)
                throw SBError("Real-space SBConvolve requires members to be analytic in x");
            _plist.push_back(p);
            newptr = --_plist.end();  // That was first new term
        }

        // Accumulate properties of all terms
        while (newptr != _plist.end()) {
            xdbg<<"SBConvolve component has maxK, stepK = "<<
                (*newptr)->maxK()<<" , "<<(*newptr)->stepK()<<'\n';
            _fluxProduct *= (*newptr)->getFlux();
            _x0 += (*newptr)->centroid().x;
            _y0 += (*newptr)->centroid().y;
            if ( _minMaxK<=0. || (*newptr)->maxK() < _minMaxK)
                _minMaxK = (*newptr)->maxK();
            if ( _minStepK<=0. || ((*newptr)->stepK() < _minStepK))
                _minStepK = (*newptr)->stepK();
            _isStillAxisymmetric = _isStillAxisymmetric && (*newptr)->isAxisymmetric();
            newptr++;
        }
        xdbg<<"Net maxK, stepK = "<<_minMaxK<<" , "<<_minStepK<<'\n';
    }

    void SBConvolve::fillKGrid(KTable& kt) const 
    {
        if (_plist.empty()) kt.clear();
        ConstIter pptr = _plist.begin();
        (*pptr)->fillKGrid(kt);
        kt *= _fluxScale;
        if (++pptr != _plist.end()) {
            KTable k2(kt.getN(),kt.getDk());
            for ( ; pptr!= _plist.end(); ++pptr) {
                (*pptr)->fillKGrid(k2);
                kt *= k2;
            }
        }
    }

    double SBConvolve::xValue(const Position<double>& pos) const
    {
        // Perform a direct calculation of the convolution at a particular point by
        // doing the real-space integral.
        // Note: This can only really be done one pair at a time, so it is 
        // probably rare that this will be more efficient if N > 2.
        // For now, we don't bother implementing this for N > 2.
        
        if (_plist.empty()) return 0.;
        else if (_plist.size() == 1) return _plist.front()->xValue(pos);
        else if (_plist.size() > 2) 
            throw SBError("Real-space integration of more than 2 profiles is not implemented.");
        else {
            const SBProfile* p1 = _plist.front();
            const SBProfile* p2 = _plist.back();
            if (p2->isAxisymmetric())
                return RealSpaceConvolve(p2,p1,pos,_fluxProduct);
            else 
                return RealSpaceConvolve(p1,p2,pos,_fluxProduct);
        }
    }

    std::complex<double> SBConvolve::kValue(const Position<double>& k) const 
    {
        ConstIter pptr = _plist.begin();
        assert(pptr != _plist.end());
        std::complex<double> kv = (*pptr)->kValue(k);
        for (++pptr; pptr != _plist.end(); ++pptr)
            kv *= (*pptr)->kValue(k);
        return kv;
    } 


    double SBConvolve::getPositiveFlux() const 
    {
        if (_plist.empty()) return 0.;
        std::list<SBProfile*>::const_iterator pptr = _plist.begin();
        double pResult = (*pptr)->getPositiveFlux() * _fluxScale;
        double nResult = (*pptr)->getNegativeFlux() * _fluxScale;
        for (++pptr; pptr!=_plist.end(); ++pptr) {
            double p = (*pptr)->getPositiveFlux();
            double n = (*pptr)->getNegativeFlux();
            double pNew = p*pResult + n*nResult;
            nResult = p*nResult + n*pResult;
            pResult = pNew;
        }
        return pResult;
    }

    // Note duplicated code here, could be caching results for tiny efficiency gain
    double SBConvolve::getNegativeFlux() const 
    {
        if (_plist.empty()) return 0.;
        std::list<SBProfile*>::const_iterator pptr = _plist.begin();
        double pResult = (*pptr)->getPositiveFlux() * _fluxScale;
        double nResult = (*pptr)->getNegativeFlux() * _fluxScale;
        for (++pptr; pptr!=_plist.end(); ++pptr) {
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

    SBGaussian::SBGaussian(double flux, double sigma) :
        _flux(flux), _sigma(sigma), _sigma_sq(sigma*sigma)
    {
        // For large k, we clip the result of kValue to 0.
        // We do this when the correct anser is less than kvalue_accuracy.
        // exp(-k^2*sigma^2/2) = kvalue_accuracy
        _ksq_max = -2. * log(sbp::kvalue_accuracy) / _sigma_sq;

        // For small k, we can use up to quartic in the taylor expansion to avoid the exp.
        // This is acceptable when the next term is less than kvalue_accuracy.
        // 1/48 (k^2 r0^2)^3 = kvalue_accuracy
        _ksq_min = std::pow(sbp::kvalue_accuracy * 48., 1./3.) / _sigma_sq;

        _norm = _flux / (_sigma_sq * 2. * M_PI);

        xdbg<<"Gaussian:\n";
        xdbg<<"_flux = "<<_flux<<'\n';
        xdbg<<"_sigma = "<<_sigma<<'\n';
        xdbg<<"_sigma_sq = "<<_sigma_sq<<'\n';
        xdbg<<"_ksq_max = "<<_ksq_max<<'\n';
        xdbg<<"_ksq_min = "<<_ksq_min<<'\n';
        xdbg<<"_norm = "<<_norm<<'\n';
        xdbg<<"maxK() = "<<maxK()<<'\n';
        xdbg<<"stepK() = "<<stepK()<<'\n';
    }

    // Set maxK where the FT is down to 0.001 or threshold, whichever is harder.
    double SBGaussian::maxK() const 
    { return std::max(4., sqrt(-2.*log(sbp::ALIAS_THRESHOLD)))/_sigma; }

    // The amount of flux missed in a circle of radius pi/stepk should miss at 
    // most ALIAS_THRESHOLD of the flux.
    double SBGaussian::stepK() const
    {
        // int( exp(-r^2/2) r, r=0..R) = 1 - exp(-R^2/2)
        // exp(-R^2/2) = ALIAS_THRESHOLD
        double R = sqrt(-2.*std::log(sbp::ALIAS_THRESHOLD));
        // Make sure it is at least 4 sigma;
        R = std::max(4., R);
        return M_PI / (R*_sigma);
    }

    double SBGaussian::xValue(const Position<double>& p) const
    {
        double rsq = p.x*p.x + p.y*p.y;
        return _norm * std::exp( -rsq/(2.*_sigma_sq) );
    }

    std::complex<double> SBGaussian::kValue(const Position<double>& k) const
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

    SBExponential::SBExponential(double flux, double r0) :
        _flux(flux), _r0(r0), _r0_sq(r0*r0)
    {
        // For large k, we clip the result of kValue to 0.
        // We do this when the correct anser is less than kvalue_accuracy.
        // (1+k^2 r0^2)^-1.5 = kvalue_accuracy
        _ksq_max = (std::pow(sbp::kvalue_accuracy,-1./1.5)-1.) / _r0_sq;

        // For small k, we can use up to quartic in the taylor expansion to avoid the sqrt.
        // This is acceptable when the next term is less than kvalue_accuracy.
        // 35/16 (k^2 r0^2)^3 = kvalue_accuracy
        _ksq_min = std::pow(sbp::kvalue_accuracy * 16./35., 1./3.) / _r0_sq;

        _norm = _flux / (_r0_sq * 2. * M_PI);

        xdbg<<"Exponential:\n";
        xdbg<<"_flux = "<<_flux<<'\n';
        xdbg<<"_r0 = "<<_r0<<'\n';
        xdbg<<"_r0_sq = "<<_r0_sq<<'\n';
        xdbg<<"_ksq_max = "<<_ksq_max<<'\n';
        xdbg<<"_ksq_min = "<<_ksq_min<<'\n';
        xdbg<<"_norm = "<<_norm<<'\n';
        xdbg<<"maxK() = "<<maxK()<<'\n';
        xdbg<<"stepK() = "<<stepK()<<'\n';
    }

    // Set maxK where the FT is down to 0.001 or threshold, whichever is harder.
    double SBExponential::maxK() const 
    { return std::max(10., pow(sbp::ALIAS_THRESHOLD, -1./3.))/_r0; }

    // The amount of flux missed in a circle of radius pi/stepk should miss at 
    // most ALIAS_THRESHOLD of the flux.
    double SBExponential::stepK() const
    {
        // int( exp(-r) r, r=0..R) = (1 - exp(-R) - Rexp(-R))
        // Fraction excluded is thus (1+R) exp(-R)
        // A fast solution to (1+R)exp(-R)=ALIAS_THRESHOLD:
        double R = -std::log(sbp::ALIAS_THRESHOLD);
        for (int i=0; i<3; i++) R = -std::log(sbp::ALIAS_THRESHOLD) + std::log(1.+R);
        // Make sure it is at least 6 scale radii.
        R = std::max(6., R);
        return M_PI / (R*_r0);
    }

    double SBExponential::xValue(const Position<double>& p) const
    {
        double r = std::sqrt(p.x*p.x + p.y*p.y);
        return _norm * std::exp(-r/_r0);
    }

    std::complex<double> SBExponential::kValue(const Position<double>& k) const 
    {
        double ksq = k.x*k.x+k.y*k.y;

        if (ksq > _ksq_max) {
            return 0.;
        } else if (ksq < _ksq_min) {
            ksq *= _r0_sq;
            return _flux*(1. - 1.5*ksq*(1. - 1.25*ksq));
        } else {
            double temp = 1. + ksq*_r0_sq;
            return _flux/(temp*std::sqrt(temp));
            // NB: flux*std::pow(temp,-1.5) is slower.
        }
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

    std::complex<double> SBAiry::kValue(const Position<double>& k) const
    {
        double kk = std::sqrt(k.x*k.x+k.y*k.y);
        // calculate circular FT(PSF) on p'=(x',y')
        return flux * annuli_autocorrelation(kk);
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

    std::complex<double> SBBox::kValue(const Position<double>& k) const
    {
        return flux * sinc(0.5*k.x*xw)*sinc(0.5*k.y*yw);
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
        double m=std::max(4., std::sqrt(-2.*std::log(sbp::ALIAS_THRESHOLD))) / sigma;
        // Grow as sqrt of order
        if (bvec.getOrder()>1) m *= std::sqrt(bvec.getOrder()/1.);
        return m;
    }

    double SBLaguerre::stepK() const 
    {
        // Start with value for plain old Gaussian:
        double m= M_PI/std::max(4., std::sqrt(-2.*std::log(sbp::ALIAS_THRESHOLD))) / sigma;
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

        xdbg << "Building for n=" << n << " b= " << b << " norm= " << norm << '\n';
        xdbg << "Deriv terms: " << kderiv2 << " " << kderiv4 << '\n';

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
            xdbg<<"Start with z = "<<z<<'\n';
            while ( std::abs(oldz-z)>0.01 && niter<MAXIT) {
                niter++;
                oldz = z;
                z = a - std::log(sbp::ALIAS_THRESHOLD*std::sqrt(2*M_PI*a)*
                                 (1.+1./(12.*a)+1./(288.*a*a))) +
                    (a-1.)*std::log(z/a) + std::log(1. + (a-1.)/z + (a-1.)*(a-2.)/(z*z));
            }
            xdbg<<"Converged at z = "<<z<<'\n';
            double r=std::pow(z/b, n);
            xdbg<<"r = (z/b)^n = "<<r<<'\n';
            if (r>xMax) xMax = r;
            xdbg<<"xMax = "<<xMax<<'\n';
        }
        stepK = M_PI / xMax;
        xdbg<<"stepK = "<<stepK<<'\n';

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
            xdbg<<"Start with z = "<<z<<'\n';
            while ( std::abs(oldz-z)>0.01 && niter<MAXIT) {
                niter++;
                oldz = z;
                z = a - std::log(integrationLoss*std::sqrt(2.*M_PI*a)*
                                 (1.+1./(12.*a)+1./(288.*a*a))) +
                    (a-1.)*std::log(z/a) + std::log(1. + (a-1.)/z + (a-1.)*(a-2.)/(z*z));
            }
            xdbg<<"Converged at z = "<<z<<'\n';
            double r=std::pow(z/b, n);
            xdbg << "99.9% radius " << r <<'\n';
            if (r>integrateMax) integrateMax = r;    
        }

        // Normalization for integral at k=0:
        SersicIntegrand I(n, b, 0.);
        double norm = integ::int1d(
            I, 0., integrateMax, sbp::integration_relerr, sbp::integration_abserr);

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
        xdbg<<"logkMin = "<<logkMin<<'\n';
        while (lk < std::log(maxK*10.) && lastVal>sbp::ALIAS_THRESHOLD/10.) {
            SersicIntegrand I(n, b, std::exp(lk));
            // Need to make sure we are resolving oscillations in the integral:
            double val = integ::int1d(
                I, 0., integrateMax, sbp::integration_relerr, sbp::integration_abserr*norm);
            //std::cerr << "Integrate k " << exp(lk) << " result " << val/norm << '\n';
            val /= norm;
            lookup.push_back(val);
            if (val >= sbp::ALIAS_THRESHOLD) maxK = std::max(maxK, std::exp(lk));
            logkMax = lk;
            lk += logkStep;
        }
        xdbg<<"maxK with val >= ALIAS_THRESHOLD ("<<sbp::ALIAS_THRESHOLD<<") = "<<maxK<<'\n';
        maxK = std::min(MAXMAXK, maxK); // largest acceptable
        xdbg<<"Final maxK = "<<maxK<<'\n';
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
                                     3.*std::sqrt(pow(sbp::ALIAS_THRESHOLD, -1./beta)-1.));
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

        dbg << "Moffat rD " << rD << " fluxFactor " << fluxFactor
            << " norm " << norm << " maxRrD " << maxRrD << '\n';
        dbg << "maxR = "<<_maxR<<", maxK = "<<maxKrD/rD<<", stepK = "<<stepKrD/rD<<'\n';

        if (beta == 1) pow_beta = &SBMoffat::pow_1;
        else if (beta == 2) pow_beta = &SBMoffat::pow_2;
        else if (beta == 3) pow_beta = &SBMoffat::pow_3;
        else if (beta == 4) pow_beta = &SBMoffat::pow_4;
        else if (beta == int(beta)) pow_beta = &SBMoffat::pow_int;
        else pow_beta = &SBMoffat::pow_gen;

        // TODO: Once SBProfile is immutable, this should move back to the start
        // of kValue.  But for now, it's better to do it here, so copies don't have
        // to recalculate ft.
        setupFT();
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

    void SBMoffat::setupFT() const
    {
        if (ft.size() > 0) return;

        // Do a Hankel transform and store the results in a lookup table.
        // Use 2048 values across 2 x the truncation radius.
        const int N=2048;
        double dk = 2.*M_PI / std::max(4.*maxRrD, 64.);
        double nn = norm * 2.*M_PI * _rD_sq;
        for (int i=0; i<=N/2; i++) {
            double k = i*dk;
            MoffatIntegrand I(beta, k, pow_beta);
            double val = integ::int1d(
                I, 0., maxRrD, sbp::integration_relerr, sbp::integration_abserr);
            xdbg<<"ft("<<k<<") = "<<val*nn<<'\n';
            ft.addEntry( k, val * nn );
        }
    }

    std::complex<double> SBMoffat::kValue(const Position<double>& k) const 
    {
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
        xdbg<<"origN = "<<origN<<'\n';
        while (N > maxN) {
            xdbg<<"shoot "<<maxN<<'\n';
            PhotonArray pa = shoot(maxN, u);
            pa.scaleFlux(maxN / origN);
            pa.addTo(img);
            N -= maxN;
        }
        xdbg<<"shoot "<<N<<'\n';
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
        for (ConstIter pptr = _plist.begin(); pptr!= _plist.end(); ++pptr) {
            double thisAbsoluteFlux = (*pptr)->getPositiveFlux() + (*pptr)->getNegativeFlux();

            // How many photons to shoot from this summand?
            int thisN = remainingN;  // All of what's left, if this is the last summand...
            std::list<SBProfile*>::const_iterator nextPtr = pptr;
            ++nextPtr;
            if (nextPtr!=_plist.end()) {
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
        std::list<SBProfile*>::const_iterator pptr = _plist.begin();
        if (pptr==_plist.end())
            throw SBError("Cannot shoot() for empty SBConvolve");
        PhotonArray result = (*pptr)->shoot(N, u);
        if (_fluxScale!=1.) result.scaleFlux(_fluxScale);
        // It is necessary to shuffle when convolving because we do
        // do not have a gaurantee that the convolvee's photons are
        // uncorrelated, e.g. they might both have their negative ones
        // at the end.
        for (++pptr; pptr != _plist.end(); ++pptr)
            result.convolveShuffle( (*pptr)->shoot(N, u), u);
        return result;
    }

    PhotonArray SBDistort::shoot(int N, UniformDeviate& u) const 
    {
        // Simple job here: just remap coords of each photon, then change flux
        // If there is overall magnification in the transform
        PhotonArray result = _adaptee->shoot(N,u);
        for (int i=0; i<result.size(); i++) {
            Position<double> xy = fwd(Position<double>(result.getX(i), result.getY(i))+_cen);
            result.setPhoton(i,xy.x, xy.y, result.getFlux(i)*_absdet);
        }
        return result;
    }

    PhotonArray SBGaussian::shoot(int N, UniformDeviate& u) const 
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
            double factor = _sigma*sqrt( -2.*log(rsq)/rsq);
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
            result.setPhoton(i,_r0*r*std::cos(theta), _r0*r*std::sin(theta), fluxPerPhoton);
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
        double maximumRadius = 2./(sbp::ALIAS_THRESHOLD * M_PI*M_PI * (1-obscuration));
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
