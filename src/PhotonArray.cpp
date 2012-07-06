//
// PhotonArray Class members
//

//#define DEBUGLOGGING

#include <algorithm>
#include <numeric>
#include "PhotonArray.h"

#ifdef DEBUGLOGGING
#include <fstream>
//std::ostream* dbgout = new std::ofstream("debug.out");
//int verbose_level = 2;
#endif

namespace galsim {

    PhotonArray::PhotonArray(
        std::vector<double>& vx, std::vector<double>& vy, std::vector<double>& vflux) :
        _is_correlated(false)
    {
        if (vx.size() != vy.size() || vx.size() != vflux.size())
            throw std::runtime_error("Size mismatch of input vectors to PhotonArray");
        _x = vx;
        _y = vy;
        _flux = vflux;
    }

    double PhotonArray::getTotalFlux() const 
    {
        double total = 0.;
        return std::accumulate(_flux.begin(), _flux.end(), total);
    }

    void PhotonArray::setTotalFlux(double flux) 
    {
        double oldFlux = getTotalFlux();
        if (oldFlux==0.) return; // Do nothing if the flux is zero to start with
        scaleFlux(flux / oldFlux);
    }

    void PhotonArray::scaleFlux(double scale)
    {
        for (std::vector<double>::size_type i=0; i<_flux.size(); i++) {
            _flux[i] *= scale;
        }
    }

    void PhotonArray::scaleXY(double scale)
    {
        for (std::vector<double>::size_type i=0; i<_x.size(); i++) {
            _x[i] *= scale;
        }
        for (std::vector<double>::size_type i=0; i<_y.size(); i++) {
            _y[i] *= scale;
        }
    }

    void PhotonArray::append(const PhotonArray& rhs) 
    {
        if (rhs.size()==0) return;      // Nothing needed for empty RHS.
        int oldSize = size();
        int finalSize = oldSize + rhs.size();
        _x.resize(finalSize);
        _y.resize(finalSize);
        _flux.resize(finalSize);
        std::vector<double>::iterator destination=_x.begin()+oldSize;
        std::copy(rhs._x.begin(), rhs._x.end(), destination);
        destination=_y.begin()+oldSize;
        std::copy(rhs._y.begin(), rhs._y.end(), destination);
        destination=_flux.begin()+oldSize;
        std::copy(rhs._flux.begin(), rhs._flux.end(), destination);
    }

    void PhotonArray::convolve(const PhotonArray& rhs, UniformDeviate ud) 
    {
        // If both arrays have corrlated photons, then we need to shuffle the photons
        // as we convolve them.
        if (_is_correlated && rhs._is_correlated) return convolveShuffle(rhs,ud);

        // If neither or only one is correlated, we are ok to just use them in order.
        int N = size();
        if (rhs.size() != N) 
            throw std::runtime_error("PhotonArray::convolve with unequal size arrays");
        // Add x coordinates:
        std::vector<double>::iterator lIter = _x.begin();
        std::vector<double>::const_iterator rIter = rhs._x.begin();
        for ( ; lIter!=_x.end(); ++lIter, ++rIter) *lIter += *rIter;
        // Add y coordinates:
        lIter = _y.begin();
        rIter = rhs._y.begin();
        for ( ; lIter!=_y.end(); ++lIter, ++rIter) *lIter += *rIter;
        // Multiply fluxes, with a factor of N needed:
        lIter = _flux.begin();
        rIter = rhs._flux.begin();
        for ( ; lIter!=_flux.end(); ++lIter, ++rIter) *lIter *= *rIter*N;

        // If rhs was correlated, then the output will be correlated.
        // This is ok, but we need to mark it as such.
        if (rhs._is_correlated) _is_correlated = true;
    }

    void PhotonArray::convolveShuffle(const PhotonArray& rhs, UniformDeviate ud) 
    {
        int N = size();
        if (rhs.size() != N) 
            throw std::runtime_error("PhotonArray::convolve with unequal size arrays");
        double xSave=0.;
        double ySave=0.;
        double fluxSave=0.;

        for (int iOut = N-1; iOut>=0; iOut--) {
            // Randomly select an input photon to use at this output
            int iIn = int(floor( (iOut+1)*ud()));
            if (iIn > iOut) iIn=iOut;  // should not happen, but be safe
            if (iIn < iOut) {
                // Save input information
                xSave = _x[iOut];
                ySave = _y[iOut];
                fluxSave = _flux[iOut];
            }
            _x[iOut] = _x[iIn] + rhs._x[iOut];
            _y[iOut] = _y[iIn] + rhs._y[iOut];
            _flux[iOut] = _flux[iIn] * rhs._flux[iOut] * N;
            if (iIn < iOut) {
                // Move saved info to new location in array
                _x[iIn] = xSave;
                _y[iIn] = ySave ;
                _flux[iIn] = fluxSave;
            }
        }
    }

    void PhotonArray::takeYFrom(const PhotonArray& rhs) 
    {
        int N = size();
        assert(rhs.size()==N);
        for (int i=0; i<N; i++) {
            _y[i] = rhs._x[i];
            _flux[i] *= rhs._flux[i]*N;
        }
    }

    template <class T>
    double PhotonArray::addTo(ImageView<T>& target) const 
    {
        double dx = target.getScale();
        Bounds<int> b = target.getBounds();

        if (dx==0. || !b.isDefined()) 
            throw std::runtime_error("Attempting to PhotonArray::addTo an Image with"
                                     " zero pixel scale or undefined Bounds");

        // Factor to turn flux into surface brightness in an Image pixel
        double fluxScale = 1./(dx*dx);  
        dbg<<"In PhotonArray::addTo\n";
        dbg<<"dx = "<<dx<<std::endl;
        dbg<<"fluxScale = "<<fluxScale<<std::endl;
        dbg<<"bounds = "<<b<<std::endl;

        double addedFlux = 0.;
#ifdef DEBUGLOGGING
        double totalFlux = 0.;
        double lostFlux = 0.;
        int nx = target.getXMax()-target.getXMin()+1;
        int ny = target.getYMax()-target.getYMin()+1;
        std::vector<std::vector<double> > posFlux(nx,std::vector<double>(ny,0.));
        std::vector<std::vector<double> > negFlux(nx,std::vector<double>(ny,0.));
#endif
        for (int i=0; i<int(size()); i++) {
            int ix = int(floor(_x[i]/dx + 0.5));
            int iy = int(floor(_y[i]/dx + 0.5));
#ifdef DEBUGLOGGING
            totalFlux += _flux[i];
            xdbg<<"  photon: ("<<_x[i]<<','<<_y[i]<<")  f = "<<_flux[i]<<std::endl;
#endif
            if (b.includes(ix,iy)) {
#ifdef DEBUGLOGGING
                if (_flux[i] > 0.) posFlux[ix-target.getXMin()][iy-target.getXMin()] += _flux[i];
                else negFlux[ix-target.getXMin()][iy-target.getXMin()] -= _flux[i];
#endif
                target(ix,iy) += _flux[i]*fluxScale;
                addedFlux += _flux[i];
            } else {
#ifdef DEBUGLOGGING
                xdbg<<"lost flux at ix = "<<ix<<", iy = "<<iy<<" with flux = "<<_flux[i]<<std::endl;
                lostFlux += _flux[i];
#endif
            }
        }
#ifdef DEBUGLOGGING
        dbg<<"totalFlux = "<<totalFlux<<std::endl;
        dbg<<"addedlFlux = "<<addedFlux<<std::endl;
        dbg<<"lostFlux = "<<lostFlux<<std::endl;
        for(int ix=0;ix<nx;++ix) {
            for(int iy=0;iy<ny;++iy) {
                double pos = posFlux[ix][iy];
                double neg = negFlux[ix][iy];
                double tot = pos + neg;
                if (tot > 0.) {
                    xdbg<<"eta("<<ix+target.getXMin()<<','<<iy+target.getXMin()<<") = "<<
                        neg<<" / "<<tot<<" = "<<neg/tot<<std::endl;
                }
            }
        }
#endif

        return addedFlux;
    }

    // instantiate template functions for expected image types
    template double PhotonArray::addTo(ImageView<float>& image) const;
    template double PhotonArray::addTo(ImageView<double>& image) const;

}
