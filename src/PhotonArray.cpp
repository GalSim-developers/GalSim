/* -*- c++ -*-
 * Copyright (c) 2012-2016 by the GalSim developers team on GitHub
 * https://github.com/GalSim-developers
 *
 * This file is part of GalSim: The modular galaxy image simulation toolkit.
 * https://github.com/GalSim-developers/GalSim
 *
 * GalSim is free software: redistribution and use in source and binary forms,
 * with or without modification, are permitted provided that the following
 * conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions, and the disclaimer given in the accompanying LICENSE
 *    file.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions, and the disclaimer given in the documentation
 *    and/or other materials provided with the distribution.
 */
//
// PhotonArray Class members
//

//#define DEBUGLOGGING

#include <algorithm>
#include <numeric>
#include "PhotonArray.h"

#ifdef DEBUGLOGGING
#include <vector>
#endif

namespace galsim {

    template <typename T>
    struct ArrayDeleter {
        void operator()(T* p) const { delete [] p; }
    };

    PhotonArray::PhotonArray(int N) : _x(N), _y(N), _flux(N), _is_correlated(false)
    {}

    void PhotonArray::allocateAngleVectors()
    {
        if (!hasAllocatedAngles()) {
            _dxdz.resize(size());
            _dydz.resize(size());
        }
    }

    void PhotonArray::allocateWavelengthVector()
    {
        if (!hasAllocatedWavelengths()) {
            _wavelength.resize(size());
        }
    }

    bool PhotonArray::hasAllocatedAngles()
    {
        // dydz should always be in sync, so not need to check it.
        return _dxdz.size() == size();
    }

    bool PhotonArray::hasAllocatedWavelengths()
    {
        return _wavelength.size() == size();
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
        std::transform(_flux.begin(), _flux.end(), _flux.begin(),
                       std::bind2nd(std::multiplies<double>(),scale));
    }

    void PhotonArray::scaleXY(double scale)
    {
        std::transform(_x.begin(), _x.end(), _x.begin(),
                       std::bind2nd(std::multiplies<double>(),scale));
        std::transform(_y.begin(), _y.end(), _y.begin(),
                       std::bind2nd(std::multiplies<double>(),scale));
    }

    void PhotonArray::assignAt(int istart, const PhotonArray& rhs)
    {
        if (istart + rhs.size() > size())
            throw std::runtime_error("Trying to assign past the end of PhotonArray");

        std::copy(rhs._x.begin(), rhs._x.end(), _x.begin()+istart);
        std::copy(rhs._y.begin(), rhs._y.end(), _y.begin()+istart);
        std::copy(rhs._flux.begin(), rhs._flux.end(), _flux.begin()+istart);
        if (rhs._dxdz.size() > 0) {
            allocateAngleVectors();
            std::copy(rhs._dxdz.begin(), rhs._dxdz.end(), _dxdz.begin()+istart);
            std::copy(rhs._dydz.begin(), rhs._dydz.end(), _dydz.begin()+istart);
        }
        if (rhs._wavelength.size() > 0) {
            allocateWavelengthVector();
            std::copy(rhs._wavelength.begin(), rhs._wavelength.end(), _wavelength.begin()+istart);
        }
    }

    // Helper for multiplying x * y * N
    struct MultXYScale
    {
        MultXYScale(double scale) : _scale(scale) {}
        double operator()(double x, double y) { return x * y * _scale; }
        double _scale;
    };

    void PhotonArray::convolve(const PhotonArray& rhs, UniformDeviate ud)
    {
        // If both arrays have correlated photons, then we need to shuffle the photons
        // as we convolve them.
        if (_is_correlated && rhs._is_correlated) return convolveShuffle(rhs,ud);

        // If neither or only one is correlated, we are ok to just use them in order.
        if (rhs.size() != size())
            throw std::runtime_error("PhotonArray::convolve with unequal size arrays");
        // Add x coordinates:
        std::transform(_x.begin(), _x.end(), rhs._x.begin(), _x.begin(), std::plus<double>());
        // Add y coordinates:
        std::transform(_y.begin(), _y.end(), rhs._y.begin(), _y.begin(), std::plus<double>());
        // Multiply fluxes, with a factor of N needed:
        std::transform(_flux.begin(), _flux.end(), rhs._flux.begin(), _flux.begin(),
                       MultXYScale(size()));

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
            // NB: don't need floor, since rhs is positive, so floor is superfluous.
            int iIn = int((iOut+1)*ud());
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
        assert(rhs.size()==size());
        int N = size();
        for (int i=0; i<N; i++) {
            _y[i] = rhs._x[i];
            _flux[i] *= rhs._flux[i]*N;
        }
    }

    template <class T>
    double PhotonArray::addTo(ImageView<T> target) const
    {
        Bounds<int> b = target.getBounds();

        if (!b.isDefined())
            throw std::runtime_error("Attempting to PhotonArray::addTo an Image with"
                                     " undefined Bounds");

        // Factor to turn flux into surface brightness in an Image pixel
        dbg<<"In PhotonArray::addTo\n";
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
            int ix = int(floor(_x[i] + 0.5));
            int iy = int(floor(_y[i] + 0.5));
#ifdef DEBUGLOGGING
            totalFlux += _flux[i];
            xdbg<<"  photon: ("<<_x[i]<<','<<_y[i]<<")  f = "<<_flux[i]<<std::endl;
#endif
            if (b.includes(ix,iy)) {
#ifdef DEBUGLOGGING
                if (_flux[i] > 0.) posFlux[ix-target.getXMin()][iy-target.getXMin()] += _flux[i];
                else negFlux[ix-target.getXMin()][iy-target.getXMin()] -= _flux[i];
#endif
                target(ix,iy) += _flux[i];
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
    template double PhotonArray::addTo(ImageView<float> image) const;
    template double PhotonArray::addTo(ImageView<double> image) const;

}
