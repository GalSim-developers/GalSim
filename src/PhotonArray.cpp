//
// PhotonArray Class members
//
#include <algorithm>
#include <numeric>
#include "PhotonArray.h"

namespace galsim {

    PhotonArray::PhotonArray(std::vector<double>& vx, std::vector<double>& vy, std::vector<double>& vflux)
    {
        if (vx.size() != vy.size() || vx.size() != vflux.size())
            throw std::runtime_error("Size mismatch of input vectors to PhotonArray");
        _x = vx;
        _y = vy;
        _flux = vflux;
    }

    double PhotonArray::getTotalFlux() const {
        double total = 0.;
        return std::accumulate(_flux.begin(), _flux.end(), total);
    }

    void PhotonArray::setTotalFlux(double flux) {
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

    void PhotonArray::convolve(const PhotonArray& rhs) 
    {
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
    }

    void PhotonArray::convolveShuffle(const PhotonArray& rhs, UniformDeviate& ud) 
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

    void PhotonArray::takeYFrom(const PhotonArray& rhs) {
        int N = size();
        assert(rhs.size()==N);
        for (int i=0; i<N; i++) {
            _y[i] = rhs._x[i];
            _flux[i] *= rhs._flux[i]*N;
        }
    }

    template <class T>
    void PhotonArray::addTo(ImageView<T>& target) const {
        double dx = target.getScale();
        Bounds<int> b = target.getBounds();

        if (dx==0. || !b.isDefined()) 
            throw std::runtime_error("Attempting to PhotonArray::addTo an Image with"
                                     " zero pixel scale or undefined Bounds");

        double fluxScale = 1./(dx*dx);  // Factor to turn flux into surface brightness in an Image pixel

        for (int i=0; i<size(); i++) {
            int ix = int(floor(_x[i]/dx + 0.5));
            int iy = int(floor(_y[i]/dx + 0.5));
            if (b.includes(ix,iy)) target(ix,iy) += _flux[i]*fluxScale;
        }
    }

    // instantiate template functions for expected image types
    template void PhotonArray::addTo(ImageView<float>& image) const;
    template void PhotonArray::addTo(ImageView<double>& image) const;

}
