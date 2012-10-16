
//#define DEBUGLOGGING

#include "SBBox.h"
#include "SBBoxImpl.h"
#include "FFT.h"

#ifdef DEBUGLOGGING
#include <fstream>
std::ostream* dbgout = new std::ofstream("debug.out");
int verbose_level = 2;
#endif

namespace galsim {


    SBBox::SBBox(double xw, double yw, double flux) :
        SBProfile(new SBBoxImpl(xw,yw,flux)) {}

    SBBox::SBBox(const SBBox& rhs) : SBProfile(rhs) {}

    SBBox::~SBBox() {}

    double SBBox::getXWidth() const 
    {
        assert(dynamic_cast<const SBBoxImpl*>(_pimpl.get()));
        return dynamic_cast<const SBBoxImpl&>(*_pimpl).getXWidth(); 
    }

    double SBBox::getYWidth() const 
    {
        assert(dynamic_cast<const SBBoxImpl*>(_pimpl.get()));
        return dynamic_cast<const SBBoxImpl&>(*_pimpl).getYWidth(); 
    }

    double SBBox::SBBoxImpl::xValue(const Position<double>& p) const 
    {
        if (fabs(p.x) < 0.5*_xw && fabs(p.y) < 0.5*_yw) return _norm;
        else return 0.;  // do not use this function for fillXGrid()!
    }

    double SBBox::SBBoxImpl::sinc(double u) const 
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
    double SBBox::SBBoxImpl::fillXImage(ImageView<T>& I, double gain) const 
    {
        double dx = I.getScale();
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
        double norm = _norm / gain; // norm is now total normalization.
        for (int i=I.getXMin(); i<=I.getXMax(); ++i) if (std::abs(i) <= xedge) {
            double xfac = std::abs(i)==xedge ? norm*xfrac : norm;

            for (int j=I.getYMin(); j<=I.getYMax(); ++j) if (std::abs(j) <= yedge) {
                double temp = std::abs(j)==yedge ? xfac*yfrac : xfac;
                I(i,j) += T(temp);
                totalflux += temp;
            }
        }

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

    boost::shared_ptr<PhotonArray> SBBox::SBBoxImpl::shoot(int N, UniformDeviate u) const
    {
        dbg<<"Box shoot: N = "<<N<<std::endl;
        dbg<<"Target flux = "<<getFlux()<<std::endl;
        boost::shared_ptr<PhotonArray> result(new PhotonArray(N));
        for (int i=0; i<result->size(); i++)
            result->setPhoton(i, _xw*(u()-0.5), _yw*(u()-0.5), _flux/N);
        dbg<<"Box Realized flux = "<<result->getTotalFlux()<<std::endl;
        return result;
    }
}
