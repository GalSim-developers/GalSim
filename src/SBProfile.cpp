
//#define DEBUGLOGGING

#include "SBProfile.h"
#include "SBTransform.h"
#include "SBProfileImpl.h"
#include "FFT.h"

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

    SBProfile::SBProfile(SBProfileImpl* pimpl) : _pimpl(pimpl) 
    {}

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

    void SBProfile::applyTransformation(const CppEllipse& e)
    {
        SBTransform d(*this,e);
        _pimpl = d._pimpl;
    }

    void SBProfile::applyShear(double g1, double g2)
    {
        CppShear s(g1, g2);
        CppEllipse e(s);
        SBTransform d(*this,e);
        _pimpl = d._pimpl;
    }

    void SBProfile::applyShear(CppShear s)
    {
        CppEllipse e(s);
        SBTransform d(*this,e);
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

    template <typename T>
    double SBProfile::SBProfileImpl::doFillXImage2(ImageView<T>& I, double gain) const 
    {
        xdbg<<"Start doFillXImage2"<<std::endl;
        double dx = I.getScale();
        xdbg<<"dx = "<<dx<<", gain = "<<gain<<std::endl;
        double totalflux=0;
        for (int y = I.getYMin(); y <= I.getYMax(); y++) {
            int x = I.getXMin(); 
            typedef typename Image<T>::iterator ImIter;
            ImIter ee=I.rowEnd(y);
            for (ImIter it=I.rowBegin(y); it!=ee; ++it, ++x) {
                Position<double> p(x*dx,y*dx); // since x,y are pixel indices
                double temp = xValue(p) / gain;
                *it += T(temp);
                totalflux += temp;
            } 
        }
        return totalflux * (dx*dx);
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
        dbg << " maxK() " << maxK() << " dx " << dx << std::endl;

        int Nnofold = getGoodImageSize(dx,wmult);
        dbg<<"Nnofold = "<<Nnofold<<std::endl;

        // W must make something big enough to cover the target image size:
        int xSize, ySize;
        xSize = I.getXMax()-I.getXMin()+1;
        ySize = I.getYMax()-I.getYMin()+1;
        if (xSize  > Nnofold) Nnofold = xSize;
        if (ySize  > Nnofold) Nnofold = ySize;

        // Round up to a good size for making FFTs:
        int NFT = goodFFTSize(Nnofold);
        NFT = std::max(NFT,sbp::minimum_fft_size);
        dbg << " After adjustments: Nnofold " << Nnofold << " NFT " << NFT << std::endl;
        if (NFT > sbp::maximum_fft_size)
            FormatAndThrow<SBError>() << 
                "fourierDraw() requires an FFT that is too large, " << NFT;

        // Move the output image to be centered near zero
        I.setCenter(0,0);
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
        for (int y = I.getYMin(); y <= I.getYMax(); y++) {
            for (int x = I.getXMin(); x <= I.getXMax(); x++) {
                double temp = xtmp->xval(x,y) / gain;
                I(x,y) += T(temp);
                sum += temp;
            }
        }

        I.setScale(dx);

        return sum*dx*dx;;
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

        // recenter an existing image, to be consistent with fourierDrawK:
        Re.setCenter(0,0);
        Im.setCenter(0,0);

        // ??? Make this into a virtual function to allow pipelining?
        for (int y = Re.getYMin(); y <= Re.getYMax(); y++) {
            int x = Re.getXMin(); 
            typedef typename ImageView<T>::iterator ImIter;
            ImIter ee=Re.rowEnd(y);
            for (ImIter it=Re.rowBegin(y), it2=Im.rowBegin(y); it!=ee; ++it, ++it2, ++x) {
                Position<double> p(x*dk,y*dk); // since x,y are pixel indicies
                std::complex<double> c = this->kValue(p) / gain;
                *it = c.real(); 
                *it2 = c.imag(); 
            } 
        }
    }

    // Build K domain by transform from X domain.  This is likely
    // to be a rare event but what the heck.  Enforce no "aliasing"
    // by oversampling and extending x domain if needed.  Force
    // power of 2 for transform
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
        NFT = std::max(NFT,sbp::minimum_fft_size);
        dbg << " After adjustments: Nnofold " << Nnofold << " NFT " << NFT << std::endl;
        if (NFT > sbp::maximum_fft_size)
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

    template <class T>
    double SBProfile::drawShoot(ImageView<T> img, double N, UniformDeviate u,
                                double gain, double max_extra_noise, bool poisson_flux) const 
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
        double flux_scaling = eta_factor;

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

        // Center the image at 0,0:
        img.setCenter(0,0);
        dbg<<"On input, image has central value = "<<img(0,0)<<std::endl;

        // Store the PhotonArrays to be added here rather than add them as we go,
        // since we might need to rescale them all before adding.
        std::vector<boost::shared_ptr<PhotonArray> > arrays;

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
            arrays.push_back(pa);
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
            double factor = origN / (origN-N) / gain;
            dbg<<"Rescale arrays by factor = "<<factor<<std::endl;
            for (size_t k=0; k<arrays.size(); ++k) arrays[k]->scaleFlux(factor);
        } else if (gain != 1.0) {
            // Also need to rescale if the gain != 1
            dbg<<"Rescale arrays by 1./gain = "<<1./gain<<std::endl;
            for (size_t k=0; k<arrays.size(); ++k) arrays[k]->scaleFlux(1./gain);
        }

        // Now we can go ahead and add all the arrays to the image:
        double added_flux = 0.; // total flux falling inside image bounds, returned
#ifdef DEBUGLOGGING
        double realized_flux = 0.;
        double positive_flux = 0.;
        double negative_flux = 0.;
#endif
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
    template double SBProfile::SBProfileImpl::doFillXImage2(
        ImageView<float>& img, double gain) const;
    template double SBProfile::SBProfileImpl::doFillXImage2(
        ImageView<double>& img, double gain) const;

    template double SBProfile::drawShoot(
        ImageView<float> image, double N, UniformDeviate ud, double gain,
        double max_extra_noise, bool poisson_flux) const;
    template double SBProfile::drawShoot(
        ImageView<double> image, double N, UniformDeviate ud, double gain,
        double max_extra_noise, bool poisson_flux) const;

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
