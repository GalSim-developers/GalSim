/* -*- c++ -*-
 * Copyright (c) 2012-2017 by the GalSim developers team on GitHub
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

    bool PhotonArray::hasAllocatedAngles() const
    {
        // dydz should always be in sync, so not need to check it.
        return _dxdz.size() == size();
    }

    bool PhotonArray::hasAllocatedWavelengths() const
    {
        return _wavelength.size() == size();
    }

    template <typename T>
    struct AddImagePhotons
    {
        AddImagePhotons(std::vector<double>& vx, std::vector<double>& vy, std::vector<double>& vf,
                        double maxFlux, UniformDeviate ud) :
            _vx(vx), _vy(vy), _vf(vf), _maxFlux(maxFlux), _ud(ud) {}

        void operator()(T flux, int i, int j)
        {
            int N = (flux <= _maxFlux) ? 1 : int(std::ceil(flux / _maxFlux));
            double fluxPer = double(flux) / N;
            for (int k=0; k<N; ++k) {
                double x = i + _ud() - 0.5;
                double y = j + _ud() - 0.5;
                _vx.push_back(x);
                _vy.push_back(y);
                _vf.push_back(fluxPer);
            }
        }

        std::vector<double>& _vx;
        std::vector<double>& _vy;
        std::vector<double>& _vf;
        const double _maxFlux;
        UniformDeviate _ud;
    };

    template <class T>
    PhotonArray::PhotonArray(const BaseImage<T>& image, double maxFlux, UniformDeviate ud) :
        _is_correlated(true)
    {
        double totalFlux = image.sumElements();
        dbg<<"totalFlux = "<<totalFlux<<std::endl;
        dbg<<"maxFlux = "<<maxFlux<<std::endl;
        int N = image.getNRow() * image.getNCol() + totalFlux / maxFlux;
        dbg<<"image size = "<<image.getNRow() * image.getNCol()<<std::endl;
        dbg<<"count from photons = "<<totalFlux / maxFlux<<std::endl;
        dbg<<"N = "<<N<<std::endl;
        // This goes a bit over what we actually need, but not by much.  Worth it to save
        // on the vector reallocations.
        _x.reserve(N);
        _y.reserve(N);
        _flux.reserve(N);
        dbg<<"bounds = "<<image.getBounds()<<std::endl;
        AddImagePhotons<T> adder(_x, _y, _flux, maxFlux, ud);
        for_each_pixel_ij(image, adder);
        dbg<<"Done: size = "<<_x.size()<<std::endl;
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
        if (rhs.size() != size())
            throw std::runtime_error("PhotonArray::convolve with unequal size arrays");
        double xSave=0.;
        double ySave=0.;
        double fluxSave=0.;
        int N = size();

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
    template PhotonArray::PhotonArray(const BaseImage<float>& image, double maxFlux,
                                      UniformDeviate ud);
    template PhotonArray::PhotonArray(const BaseImage<double>& image, double maxFlux,
                                      UniformDeviate ud);
}
