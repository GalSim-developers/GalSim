
#include <algorithm>

#include "SBInterpolatedImage.h"

namespace galsim {

    const double TWOPI = 2.*M_PI;
    const double OVERSAMPLE_X = 4.;  // FT must be at least this much larger than input

    // Default k-space interpolant is quintic:
    Quintic defaultKInterpolant1d(1e-4);

    InterpolantXY SBInterpolatedImage::defaultKInterpolant2d(defaultKInterpolant1d);

    SBInterpolatedImage::SBInterpolatedImage(
        int Npix, double dx_, const Interpolant2d& i, int Nimages_) :  
        
        Ninitial(Npix+Npix%2), dx(dx_), Nimages(Nimages_),
        xInterp(&i), kInterp(&defaultKInterpolant2d),
        wts(Nimages, 1.), fluxes(Nimages, 1.), 
        xFluxes(Nimages, 0.), yFluxes(Nimages,0.),
        xsum(0), ksum(0), xsumValid(false), ksumValid(false),
        ready(false) 
    {
        assert(Ninitial%2==0);
        assert(Ninitial>=2);
        // Choose the padded size for input array - size 2^N or 3*2^N
        // Make FFT either 2^n or 3x2^n
        Nk = goodFFTSize(OVERSAMPLE_X*Ninitial);
        dk = TWOPI / (Nk*dx);

        // allocate xTables
        for (int i=0; i<Nimages; i++) 
            vx.push_back(new XTable(Nk, dx));
    }

#ifdef USE_IMAGES
    template <typename T>
    SBInterpolatedImage::SBInterpolatedImage(
        const BaseImage<T>& img, const Interpolant2d& i, double dx_, double padFactor) : 

        dx(dx_), Nimages(1),
        xInterp(&i), kInterp(&defaultKInterpolant2d),
        wts(Nimages, 1.), fluxes(Nimages, 1.), 
        xFluxes(Nimages, 0.), yFluxes(Nimages,0.),
        xsum(0), ksum(0), xsumValid(false), ksumValid(false),
        ready(false) 
    {
        Ninitial = std::max( img.getYMax()-img.getYMin()+1, img.getXMax()-img.getXMin()+1);
        Ninitial = Ninitial + Ninitial%2;
        assert(Ninitial%2==0);
        assert(Ninitial>=2);
        if (dx<=0.) {
            dx = img.getScale();
        }
        if (padFactor <= 0.) padFactor = OVERSAMPLE_X;
        // Choose the padded size for input array - size 2^N or 3*2^N
        // Make FFT either 2^n or 3x2^n
        Nk = goodFFTSize(static_cast<int> (std::floor(padFactor*Ninitial)));
        dk = TWOPI / (Nk*dx);

        // allocate xTables
        for (int i=0; i<Nimages; i++) 
            vx.push_back(new XTable(Nk, dx));
        // fill data from image, shifting to center the image in the table
        int xStart = -((img.getXMax()-img.getXMin()+1)/2);
        int yTab = -((img.getYMax()-img.getYMin()+1)/2);
        for (int iy = img.getYMin(); iy<= img.getYMax(); iy++, yTab++) {
            int xTab = xStart;
            for (int ix = img.getXMin(); ix<= img.getXMax(); ix++, xTab++) 
                vx.front()->xSet(xTab, yTab, img(ix,iy));
        }
    }
#endif

    SBInterpolatedImage::SBInterpolatedImage(const SBInterpolatedImage& rhs):
        Ninitial(rhs.Ninitial), dx(rhs.dx), Nk(rhs.Nk), Nimages(rhs.Nimages),
        xInterp(rhs.xInterp), kInterp(rhs.kInterp),
        wts(rhs.wts), fluxes(rhs.fluxes), xFluxes(rhs.xFluxes), yFluxes(rhs.yFluxes),
        xsum(0), ksum(0), xsumValid(false), ksumValid(false), ready(rhs.ready) 
    {
        // copy tables
        for (int i=0; i<Nimages; i++) {
            vx.push_back(new XTable(*rhs.vx[i]));
            if (ready)     vk.push_back(new KTable(*rhs.vk[i]));
        }
    }

    SBInterpolatedImage::~SBInterpolatedImage() 
    {
        for (size_t i=0; i<vx.size(); i++) if (vx[i]) { delete vx[i]; vx[i]=0; }
        for (size_t i=0; i<vk.size(); i++) if (vk[i]) { delete vk[i]; vk[i]=0; }
        if (xsum) { delete xsum; xsum=0; }
        if (ksum) { delete ksum; ksum=0; }
    }

    double SBInterpolatedImage::getFlux() const 
    {
        checkReady();
        return wts * fluxes;
    }

    void SBInterpolatedImage::setFlux(double flux) 
    {
        checkReady();
        double factor = flux/getFlux();
        wts *= factor;
        if (xsumValid) *xsum *= factor;
        if (ksumValid) *ksum *= factor;
    }

    Position<double> SBInterpolatedImage::centroid() const 
    {
        checkReady();
        double wtsfluxes = wts * fluxes;
        Position<double> p((wts * xFluxes) / wtsfluxes, (wts * yFluxes) / wtsfluxes);
        return p;
    }

    void SBInterpolatedImage::setPixel(double value, int ix, int iy, int iz) 
    {
        if (iz < 0 || iz>=Nimages)
            FormatAndThrow<SBError>() << 
                "SBInterpolatedImage::setPixel image number " << iz << " out of bounds";
        if (ix < -Ninitial/2 || ix >= Ninitial / 2)
            FormatAndThrow<SBError>() << 
                "SBInterpolatedImage::setPixel x coordinate " << ix << " out of bounds";
        if (iy < -Ninitial/2 || iy >= Ninitial / 2)
            FormatAndThrow<SBError>() << 
                "SBInterpolatedImage::setPixel x coordinate " << iy << " out of bounds";

        ready = false;
        vx[iz]->xSet(ix, iy, value);
    }

    double SBInterpolatedImage::getPixel(int ix, int iy, int iz) const 
    {
        if (iz < 0 || iz>=Nimages)
            FormatAndThrow<SBError>() << 
                "SBInterpolatedImage::getPixel image number " << iz << " out of bounds";

        return vx[iz]->xval(ix, iy);
    }

    void SBInterpolatedImage::setWeights(const tmv::Vector<double>& wts_) 
    {
        assert(wts_.size()==Nimages);
        wts = wts_;
        xsumValid = false;
        ksumValid = false;
    }

    void SBInterpolatedImage::checkReady() const 
    {
        if (ready) return;
        // Flush old kTables if any;
        for (size_t i=0; i<vk.size(); i++) { delete vk[i]; vk[i]=0; }
        vk.clear();

        for (int i=0; i<Nimages; i++) {
            // Get sums:
            double sum = 0.;
            double sumx = 0.;
            double sumy = 0.;
            for (int iy=-Ninitial/2; iy<Ninitial/2; iy++) {
                for (int ix=-Ninitial/2; ix<Ninitial/2; ix++) {
                    double value = vx[i]->xval(ix, iy);
                    sum += value;
                    sumx += value*ix;
                    sumy += value*iy;
                }
            }
            fluxes[i] = sum*dx*dx;
            xFluxes[i] = sumx * std::pow(dx, 3.);
            yFluxes[i] = sumy * std::pow(dx, 3.);

            // Conduct FFT
            vk.push_back( vx[i]->transform());
        }
        ready = true;
        xsumValid = false;
        ksumValid = false;
        assert(int(vk.size())==Nimages);
    }

    void SBInterpolatedImage::checkXsum() const 
    {
        checkReady();
        if (xsumValid) return;
        if (!xsum) {
            xsum = new XTable(*vx[0]);
            *xsum *= wts[0];
        } else {
            xsum->clear();
            xsum->accumulate(*vx[0], wts[0]);
        }
        for (int i=1; i<Nimages; i++)
            xsum->accumulate(*vx[i], wts[i]);
        xsumValid = true;
    }

    void SBInterpolatedImage::checkKsum() const 
    {
        checkReady();
        if (ksumValid) return;
        if (!ksum) {
            ksum = new KTable(*vk[0]);
            *ksum *= wts[0];
        } else {
            ksum->clear();
            ksum->accumulate(*vk[0], wts[0]);
        }
        for (int i=1; i<Nimages; i++)
            ksum->accumulate(*vk[i], wts[i]);
        ksumValid = true;
    }

    void SBInterpolatedImage::fillKGrid(KTable& kt) const 
    {
        // This override of base class is to permit potential efficiency gain from
        // separable interpolant kernel.  If so, the KTable interpolation routine
        // will go faster if we make y iteration the inner loop.
        if (dynamic_cast<const InterpolantXY*> (kInterp)) {
            int N = kt.getN();
            double dk = kt.getDk();
            // Only need ix>=0 because it's Hermitian:
            for (int ix = 0; ix <= N/2; ix++) {
                for (int iy = -N/2; iy < N/2; iy++) {
                    Position<double> k(ix*dk,iy*dk);
                    kt.kSet(ix,iy,kValue(k));
                }
            }
        } else {
            // Otherwise just use the normal routine to fill the grid:
            SBProfile::fillKGrid(kt);
        }
    }

    // Same deal: reverse axis order if we have separable interpolant in X domain
    void SBInterpolatedImage::fillXGrid(XTable& xt) const 
    {
#ifdef DANIELS_TRACING
        cout << "SBInterpolatedImage::fillXGrid called" << endl;
#endif
        if ( dynamic_cast<const InterpolantXY*> (xInterp)) {
            int N = xt.getN();
            double dx = xt.getDx();
            for (int ix = -N/2; ix < N/2; ix++) {
                for (int iy = -N/2; iy < N/2; iy++) {
                    Position<double> x(ix*dx,iy*dx);
                    xt.xSet(ix,iy,xValue(x));
                }
            }
        } else {
            // Otherwise just use the normal routine to fill the grid:
            SBProfile::fillXGrid(xt);
        }
    }

#ifdef USE_IMAGES
    // One more time: for images now
    // Returns total flux
    template <typename T>
    double SBInterpolatedImage::fillXImage(ImageView<T>& I, double dx) const 
    {
#ifdef DANIELS_TRACING
        cout << "SBInterpolatedImage::fillXImage called" << endl;
#endif
        if ( dynamic_cast<const InterpolantXY*> (xInterp)) {
            double sum=0.;
            for (int ix = I.getXMin(); ix <= I.getXMax(); ix++) {
                for (int iy = I.getYMin(); iy <= I.getYMax(); iy++) {
                    Position<double> x(ix*dx,iy*dx);
                    T val = xValue(x);
                    sum += val;
                    I(ix,iy) = val;
                }
            }
            return sum;
        } else {
            // Otherwise just use the normal routine to fill the grid:
            // Note that we need to call doFillXImage, not fillXImage here,
            // to avoid the virtual function resolution.
            return SBProfile::doFillXImage(I,dx);
        }
    }
#endif

#ifndef OLD_WAY
    double SBInterpolatedImage::xValue(Position<double> p) const 
    {
#ifdef DANIELS_TRACING
        cout << "getting xValue at " << p << endl;
#endif
        checkXsum();
        return xsum->interpolate(p.x, p.y, *xInterp);
    }

    std::complex<double> SBInterpolatedImage::kValue(Position<double> p) const 
    {
        // Don't bother if the desired k value is cut off by the x interpolant:
        double ux = p.x*dx/TWOPI;
        if (std::abs(ux) > xInterp->urange()) return std::complex<double>(0.,0.);
        double uy = p.y*dx/TWOPI;
        if (std::abs(uy) > xInterp->urange()) return std::complex<double>(0.,0.);
        double xKernelTransform = xInterp->uval(ux, uy);

        checkKsum();
        return xKernelTransform * ksum->interpolate(p.x, p.y, *kInterp);
    }

#else
    double SBInterpolatedImage::xValue(Position<double> p) const 
    {
        // Interpolate WITHOUT wrapping the image.
        int ixMin = static_cast<int> ( std::ceil(p.x/dx - xInterp->xrange()));
        ixMin = std::max(ixMin, -Ninitial/2);
        int ixMax = static_cast<int> ( std::floor(p.x/dx + xInterp->xrange()));
        ixMax = std::min(ixMax, Ninitial/2-1);
        int iyMin = static_cast<int> ( std::ceil(p.y/dx - xInterp->xrange()));
        iyMin = std::max(iyMin, -Ninitial/2);
        int iyMax = static_cast<int> ( std::floor(p.y/dx + xInterp->xrange()));
        iyMax = std::min(iyMax, Ninitial/2-1);

        if (ixMax < ixMin || iyMax < iyMin) return 0.;  // kernel does not overlap data
        int npts = (ixMax - ixMin+1)*(iyMax-iyMin+1);
        tmv::Vector<double> kernel(npts,0.);
        tmv::Matrix<double> data(Nimages, npts, 0.);
        int ipt = 0;
        for (int iy = iyMin; iy <= iyMax; iy++) {
            double deltaY = p.y/dx - iy;
            for (int ix = ixMin; ix <= ixMax; ix++, ipt++) {
                double deltaX = p.x/dx - ix;
                kernel[ipt] = xInterp->xval(deltaX, deltaY);
                for (int iz=0; iz<Nimages; iz++) {
                    data(iz, ipt) = vx[iz]->xval(ix, iy);
                }
            }
        }
        return wts * data * kernel;
    }

    std::complex<double> SBInterpolatedImage::kValue(Position<double> p) const 
    {
        checkReady();
        // Interpolate in k space, first apply kInterp kernel to wrapped
        // k-space data, then multiply by FT of xInterp kernel.

        // Don't bother if the desired k value is cut off by the x interpolant:
        double ux = p.x*dx/TWOPI;
        if (std::abs(ux) > xInterp->urange()) return std::complex<double>(0.,0.);
        double uy = p.y*dx/TWOPI;
        if (std::abs(uy) > xInterp->urange()) return std::complex<double>(0.,0.);
        double xKernelTransform = xInterp->uval(ux, uy);

        // Range of k points within kernel
        int ixMin = static_cast<int> (std::ceil(p.x/dk - kInterp->xrange()));
        int ixMax = static_cast<int> (std::floor(p.x/dk + kInterp->xrange()));
        int iyMin = static_cast<int> (std::ceil(p.y/dk - kInterp->xrange()));
        int iyMax = static_cast<int> (std::floor(p.y/dk + kInterp->xrange()));

        int ixLast = std::min(ixMax, ixMin+Nk-1);
        int iyLast = std::min(iyMax, iyMin+Nk-1);
        int npts = (ixLast-ixMin+1) * (iyLast-iyMin+1);
        tmv::Vector<double> kernel(npts, 0.);
        tmv::Matrix<std::complex<double> > data(Nimages, npts, std::complex<double>(0.,0.));

        int ipt = 0;
        for (int iy = iyMin; iy <= iyLast; iy++) {
            for (int ix = ixMin; ix <= ixLast; ix++) {
                // sum kernel values for all aliases of this frequency
                double sumk = 0.;
                int iyy=iy;
                while (iyy <= iyMax) {
                    double deltaY = p.y/dk - iyy;
                    int ixx = ix;
                    while (ixx <= ixMax) {
                        double deltaX = p.x/dk - ixx;
                        sumk += kInterp->xval(deltaX, deltaY);
                        ixx += Nk;
                    }
                    iyy += Nk;
                }
                // Shift ix,iy into un-aliased zone to get k value
                iyy = iy % Nk;  
                if(iyy>=Nk/2) iyy-=Nk; 
                if(iyy<-Nk/2) iyy+=Nk;
                int ixx = ix % Nk;  
                if(ixx>=Nk/2) ixx-=Nk; 
                if(ixx<-Nk/2) ixx+=Nk;
                for (int iz=0; iz<Nimages; iz++) 
                    data(iz, ipt) = vk[iz]->kval(ixx, iyy);
                kernel[ipt] = sumk;
                ipt++;
            }
        }
        return xKernelTransform*(wts * data * kernel);
    }

#endif

    // instantiate template functions for expected image types
#ifdef USE_IMAGES
    template SBInterpolatedImage::SBInterpolatedImage(
        const BaseImage<float>& img, const Interpolant2d& i, double dx_, double padFactor);
    template SBInterpolatedImage::SBInterpolatedImage(
        const BaseImage<double>& img, const Interpolant2d& i, double dx_, double padFactor);
    template SBInterpolatedImage::SBInterpolatedImage(
        const BaseImage<short>& img, const Interpolant2d& i, double dx_, double padFactor);
    template SBInterpolatedImage::SBInterpolatedImage(
        const BaseImage<int>& img, const Interpolant2d& i, double dx_, double padFactor);
#endif
}

