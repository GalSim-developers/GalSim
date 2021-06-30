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

#ifndef GalSim_PhotonArray_H
#define GalSim_PhotonArray_H

/**
 * @file PhotonArray.h @brief Contains a class definition for lists of photons from "shooting."
 */

#include <cmath>
#include <algorithm>

#include "Std.h"
#include "Random.h"
#include "Image.h"

namespace galsim {

    /** @brief Class to hold a list of "photon" arrival positions
     *
     * Class holds arrays of information about photon arrivals: x and y positions, dxdz and dydz
     * inclination "angles" (really slopes), a flux, and a wavelength carried by each photon.
     * It is the intention that fluxes of photons be nearly equal in absolute value so that noise
     * statistics can be estimated by counting number of positive and negative photons.
     */
    class PUBLIC_API PhotonArray
    {
    public:
        /**
         * @brief Construct a PhotonArray of the given size, allocating the arrays locally.
         *
         * Note: PhotonArrays made this way can only be used locally in the C++ layer, not
         * returned back to Python.  Also, only x,y,flux will be allocated.
         *
         * @param[in] N         Size of array
         */
        PhotonArray(int N);

        /**
         * @brief Construct a PhotonArray of the given size with the given arrays, which should
         * be allocated separately (in Python typically).
         *
         * If angles or wavelengths are not set, these may be 0.
         *
         * @param[in] N         Size of array
         * @param[in] x         An array of the initial x values
         * @param[in] y         An array of the initial y values
         * @param[in] flux      An array of the initial flux values
         * @param[in] dxdz      An array of the initial dxdz values (may be 0)
         * @param[in] dydz      An array of the initial dydz values (may be 0)
         * @param[in] wave      An array of the initial wavelength values (may be 0)
         * @param[in] is_corr   A boolean indicating whether the current values are correlated.
         */
        PhotonArray(size_t N, double* x, double* y, double* flux,
                    double* dxdz, double* dydz, double* wave, bool is_corr) :
            _N(N), _x(x), _y(y), _flux(flux), _dxdz(dxdz), _dydz(dydz), _wave(wave),
            _is_correlated(is_corr) {}

        /**
         * @brief Accessor for array size
         *
         * @returns Array size
         */
        size_t size() const { return _N; }

        /**
         * @{
         * @brief Accessors that provide access as numpy arrays in Python layer
         */
        double* getXArray() { return _x; }
        double* getYArray() { return _y; }
        double* getFluxArray() { return _flux; }
        double* getDXDZArray() { return _dxdz; }
        double* getDYDZArray() { return _dydz; }
        double* getWavelengthArray() { return _wave; }
        bool hasAllocatedAngles() const { return _dxdz != 0 && _dydz != 0; }
        bool hasAllocatedWavelengths() const { return _wave != 0; }
        /**
         * @}
         */

        /**
         * @brief Set characteristics of a photon that are decided during photon shooting
         * (i.e. only x,y,flux)
         *
         * @param[in] i     Index of desired photon (no bounds checking)
         * @param[in] x     x coordinate of photon
         * @param[in] y     y coordinate of photon
         * @param[in] flux  flux of photon
         */
        void setPhoton(int i, double x, double y, double flux)
        {
            _x[i]=x;
            _y[i]=y;
            _flux[i]=flux;
        }

        /**
         * @brief Access x coordinate of a photon
         *
         * @param[in] i Index of desired photon (no bounds checking)
         * @returns x coordinate of photon
         */
        double getX(int i) const { return _x[i]; }

        /**
         * @brief Access y coordinate of a photon
         *
         * @param[in] i Index of desired photon (no bounds checking)
         * @returns y coordinate of photon
         */
        double getY(int i) const { return _y[i]; }

        /**
         * @brief Access flux of a photon
         *
         * @param[in] i Index of desired photon (no bounds checking)
         * @returns flux of photon
         */
        double getFlux(int i) const { return _flux[i]; }

        /**
         * @brief Access dxdz of a photon
         *
         * @param[in] i Index of desired photon (no bounds checking)
         * @returns dxdz of photon
         */
        double getDXDZ(int i) const { return _dxdz[i]; }

        /**
         * @brief Access dydz coordinate of a photon
         *
         * @param[in] i Index of desired photon (no bounds checking)
         * @returns dydz coordinate of photon
         */
        double getDYDZ(int i) const { return _dydz[i]; }

        /**
         * @brief Access wavelength of a photon
         *
         * @param[in] i Index of desired photon (no bounds checking)
         * @returns wavelength of photon
         */
        double getWavelength(int i) const { return _wave[i]; }

        /**
         * @brief Return sum of all photons' fluxes
         *
         * @returns flux of photon
         */
        double getTotalFlux() const;

        /**
         * @brief Rescale all photon fluxes so that total flux matches argument
         *
         * If current total flux is zero, no rescaling is done.
         *
         * @param[in] flux desired total flux of all photons.
         */
        void setTotalFlux(double flux);

        /**
         * @brief Rescale all photon fluxes by the given factor
         *
         * @param[in] scale Scaling factor for all fluxes
         */
        void scaleFlux(double scale);

        /**
         * @brief Rescale all photon positions by the given factor
         *
         * @param[in] scale Scaling factor for all positions
         */
        void scaleXY(double scale);

        /**
         * @brief Assign the contents of another array to a portion of this one.
         *
         * @param[in] istart    The starting index at which to assign the contents of rhs
         * @param[in] rhs       PhotonArray whose contents to assign into this one
         */
        void assignAt(int istart, const PhotonArray& rhs);

        /**
         * @brief Convolve this array with another.
         *
         * Convolution of two arrays is defined as adding the coordinates on a photon-by-photon
         * basis and multiplying the fluxes on a photon-by-photon basis. Output photons' flux is
         * renormalized so that the expectation value of output total flux is product of two input
         * totals, if the two photon streams are uncorrelated.
         *
         * @param[in] rhs PhotonArray to convolve with this one.  Must be same size.
         * @param[in] rng  A BaseDeviate in case we need to shuffle.
         */
        void convolve(const PhotonArray& rhs, BaseDeviate ud);

        /**
         * @brief Convolve this array with another, shuffling the order in which photons are
         * combined.
         *
         * Same convolution behavior as convolve(), but the order in which the photons are
         * multiplied into the array is randomized to destroy any flux or position correlations.
         *
         * @param[in] rhs PhotonArray to convolve with this one.  Must be same size.
         * @param[in] rng  A BaseDeviate used to shuffle the input photons.
         */
        void convolveShuffle(const PhotonArray& rhs, BaseDeviate rng);

        /**
         * @brief Add flux of photons to an image by binning into pixels.
         *
         * Photon in this PhotonArray are binned into the pixels of the input
         * Image and their flux summed into the pixels.  Image is assumed to represent
         * surface brightness, so photons' fluxes are divided by image pixel area.
         * Photons past the edges of the image are discarded.
         *
         * @param[in] target the Image to which the photons' flux will be added.
         * @returns The total flux of photons the landed inside the image bounds.
         */
        template <class T>
        double addTo(ImageView<T> target) const;

        /**
         * @brief Set photon positions based on flux in an image.
         *
         * The flux in each non-zero pixel will be turned into 1 or more photons according
         * to the maxFlux parameter which sets an upper limit for the absolute value of the
         * flux of any photon.  Pixels with abs values > maxFlux will spawn multiple photons.
         *
         * The positions of the photons will be random within the area of each pixel.
         * TODO: This corresponds to the Nearest interpolant.  Consider implementing other
         * interpolation options here.
         *
         * @param image     The image to use for the photon fluxes and positions.
         * @param maxFlux   The maximum flux that any photon should have.
         * @param rng       A BaseDeviate in case we need to shuffle.
         *
         * @returns the total number of photons set.
         */
        template <class T>
        int setFrom(const BaseImage<T>& image, double maxFlux, BaseDeviate ud);

        /**
         * @brief Check if the current array has correlated photons.
         */
        bool isCorrelated() const { return _is_correlated; }

        /**
         * @brief Set whether the current array has correlated photons.
         */
        void setCorrelated(bool is_corr=true) { _is_correlated = is_corr; }

    private:
        size_t _N;              // The length of the arrays
        double* _x;             // Array holding x coords of photons
        double* _y;             // Array holding y coords of photons
        double* _flux;          // Array holding flux of photons
        double* _dxdz;          // Array holding dxdz of photons
        double* _dydz;          // Array holding dydz of photons
        double* _wave;          // Array holding wavelength of photons
        bool _is_correlated;    // Are the photons correlated?

        // Most of the time the arrays are constructed in Python and passed in, so we don't
        // do any memory management of them.  However, for some use cases, we need to make a
        // temporary PhotonArray with arrays allocated in the C++ layer.  The easiest way
        // to do this safely is to make these vectors and let the standard library handle
        // the memory allocation and deletion.
        std::vector<double> _vx;
        std::vector<double> _vy;
        std::vector<double> _vflux;
    };

} // end namespace galsim

#endif
