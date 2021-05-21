/* -*- c++ -*-
 * Copyright (c) 2012-2021 by the GalSim developers team on GitHub
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

namespace galsim {

    template <typename T>
    struct ArrayDeleter {
        void operator()(T* p) const { delete [] p; }
    };

    PhotonArray::PhotonArray(int N) : 
        _N(N), _dxdz(0), _dydz(0), _wave(0), _is_correlated(false), _vx(N), _vy(N), _vflux(N)
    {
        _x = &_vx[0];
        _y = &_vy[0];
        _flux = &_vflux[0];
    }

    template <typename T>
    struct AddImagePhotons
    {
        AddImagePhotons(double* x, double* y, double* f,
                        double maxFlux, BaseDeviate rng) :
            _x(x), _y(y), _f(f), _maxFlux(maxFlux), _ud(rng), _count(0) {}

        void operator()(T flux, int i, int j)
        {
            int N = (flux <= _maxFlux) ? 1 : int(std::ceil(flux / _maxFlux));
            double fluxPer = double(flux) / N;
            for (int k=0; k<N; ++k) {
                double x = i + _ud() - 0.5;
                double y = j + _ud() - 0.5;
                _x[_count] = x;
                _y[_count] = y;
                _f[_count] = fluxPer;
                ++_count;
            }
        }

        int getCount() const { return _count; }

        double* _x;
        double* _y;
        double* _f;
        const double _maxFlux;
        UniformDeviate _ud;
        int _count;
    };

    template <class T>
    int PhotonArray::setFrom(const BaseImage<T>& image, double maxFlux, BaseDeviate rng)
    {
        dbg<<"bounds = "<<image.getBounds()<<std::endl;
        dbg<<"flux, maxflux = "<<_flux<<','<<maxFlux<<std::endl;
        AddImagePhotons<T> adder(_x, _y, _flux, maxFlux, rng);
        for_each_pixel_ij_ref(image, adder);
        dbg<<"Done: size = "<<adder.getCount()<<std::endl;
        _N = adder.getCount();
        return _N;
    }

    double PhotonArray::getTotalFlux() const
    {
        double total = 0.;
        return std::accumulate(_flux, _flux+_N, total);
    }

    void PhotonArray::setTotalFlux(double flux)
    {
        double oldFlux = getTotalFlux();
        if (oldFlux==0.) return; // Do nothing if the flux is zero to start with
        scaleFlux(flux / oldFlux);
    }

    struct Scaler
    {
        Scaler(double _scale): scale(_scale) {}
        double operator()(double x) { return x * scale; }
        double scale;
    };

    void PhotonArray::scaleFlux(double scale)
    {
        std::transform(_flux, _flux+_N, _flux, Scaler(scale));
    }

    void PhotonArray::scaleXY(double scale)
    {
        std::transform(_x, _x+_N, _x, Scaler(scale));
        std::transform(_y, _y+_N, _y, Scaler(scale));
    }

    void PhotonArray::assignAt(int istart, const PhotonArray& rhs)
    {
        if (istart + rhs.size() > size())
            throw std::runtime_error("Trying to assign past the end of PhotonArray");

        const int N2 = rhs.size();
        std::copy(rhs._x, rhs._x+N2, _x+istart);
        std::copy(rhs._y, rhs._y+N2, _y+istart);
        std::copy(rhs._flux, rhs._flux+N2, _flux+istart);
        if (hasAllocatedAngles() && rhs.hasAllocatedAngles()) {
            std::copy(rhs._dxdz, rhs._dxdz+N2, _dxdz+istart);
            std::copy(rhs._dydz, rhs._dydz+N2, _dydz+istart);
        }
        if (hasAllocatedWavelengths() && rhs.hasAllocatedWavelengths()) {
            std::copy(rhs._wave, rhs._wave+N2, _wave+istart);
        }
    }

    // Helper for multiplying x * y * N
    struct MultXYScale
    {
        MultXYScale(double scale) : _scale(scale) {}
        double operator()(double x, double y) { return x * y * _scale; }
        double _scale;
    };

    void PhotonArray::convolve(const PhotonArray& rhs, BaseDeviate rng)
    {
        // If both arrays have correlated photons, then we need to shuffle the photons
        // as we convolve them.
        if (_is_correlated && rhs._is_correlated) return convolveShuffle(rhs,rng);

        // If neither or only one is correlated, we are ok to just use them in order.
        if (rhs.size() != size())
            throw std::runtime_error("PhotonArray::convolve with unequal size arrays");
        // Add x coordinates:
        std::transform(_x, _x+_N, rhs._x, _x, std::plus<double>());
        // Add y coordinates:
        std::transform(_y, _y+_N, rhs._y, _y, std::plus<double>());
        // Multiply fluxes, with a factor of N needed:
        std::transform(_flux, _flux+_N, rhs._flux, _flux, MultXYScale(_N));

        // If rhs was correlated, then the output will be correlated.
        // This is ok, but we need to mark it as such.
        if (rhs._is_correlated) _is_correlated = true;
    }

    void PhotonArray::convolveShuffle(const PhotonArray& rhs, BaseDeviate rng)
    {
        UniformDeviate ud(rng);
        if (rhs.size() != size())
            throw std::runtime_error("PhotonArray::convolve with unequal size arrays");
        double xSave=0.;
        double ySave=0.;
        double fluxSave=0.;

        for (int iOut = _N-1; iOut>=0; iOut--) {
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
            _flux[iOut] = _flux[iIn] * rhs._flux[iOut] * _N;
            if (iIn < iOut) {
                // Move saved info to new location in array
                _x[iIn] = xSave;
                _y[iIn] = ySave ;
                _flux[iIn] = fluxSave;
            }
        }
    }

    template <class T>
    double PhotonArray::addTo(ImageView<T> target) const
    {
        dbg<<"Start addTo\n";
        Bounds<int> b = target.getBounds();
        dbg<<"bounds = "<<b<<std::endl;
        if (!b.isDefined())
            throw std::runtime_error("Attempting to PhotonArray::addTo an Image with"
                                     " undefined Bounds");

        double addedFlux = 0.;
        for (int i=0; i<int(size()); i++) {
            int ix = int(floor(_x[i] + 0.5));
            int iy = int(floor(_y[i] + 0.5));
            if (b.includes(ix,iy)) {
                target(ix,iy) += _flux[i];
                addedFlux += _flux[i];
            }
        }
        return addedFlux;
    }

    // instantiate template functions for expected image types
    template double PhotonArray::addTo(ImageView<float> image) const;
    template double PhotonArray::addTo(ImageView<double> image) const;
    template int PhotonArray::setFrom(const BaseImage<float>& image, double maxFlux,
                                      BaseDeviate rng);
    template int PhotonArray::setFrom(const BaseImage<double>& image, double maxFlux,
                                      BaseDeviate rng);
}
