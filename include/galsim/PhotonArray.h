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

#ifndef GalSim_PhotonArray_H
#define GalSim_PhotonArray_H

/**
 * @file PhotonArray.h @brief Contains a class definition for lists of photons from "shooting."
 */

#include <cmath>
#include <vector>
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
    class PhotonArray
    {
    public:
        /**
         * @brief Construct an array of given size with zero-flux photons
         *
         * This will only allocate memory for x,y,flux, which are often the only things needed.
         * Memory for angles and wavelength will be allocated as needed.
         *
         * @param[in] N Size of desired array.
         */
        explicit PhotonArray(int N);

        /**
         * @brief Turn an image into an array of photons
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
         * @param ud        A UniformDeviate in case we need to shuffle.
         */
        template <class T>
        PhotonArray(const BaseImage<T>& image, double maxFlux, UniformDeviate ud);

        /**
         * @brief Accessor for array size
         *
         * @returns Array size
         */
        size_t size() const { return _x.size(); }

        /**
         * @{
         * @brief Allocate memory for optional arrays
         */
        void allocateAngleVectors();
        void allocateWavelengthVector();
        /**
         * @}
         */

        /**
         * @{
         * @brief Return whether the optional arrays are allocated
         */
        bool hasAllocatedAngles() const;
        bool hasAllocatedWavelengths() const;
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
        double getWavelength(int i) const { return _wavelength[i]; }

        /**
         * @{
         * @brief Accessors that provide access as numpy arrays in Python layer
         */
        std::vector<double>& getXVector() { return _x; }
        std::vector<double>& getYVector() { return _y; }
        std::vector<double>& getFluxVector() { return _flux; }
        std::vector<double>& getDXDZVector() { allocateAngleVectors(); return _dxdz; }
        std::vector<double>& getDYDZVector() { allocateAngleVectors(); return _dydz; }
        std::vector<double>& getWavelengthVector()
        { allocateWavelengthVector(); return _wavelength; }
        /**
         * @}
         */

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
         * @param[in] ud  A UniformDeviate in case we need to shuffle.
         */
        void convolve(const PhotonArray& rhs, UniformDeviate ud);

        /**
         * @brief Convolve this array with another, shuffling the order in which photons are
         * combined.
         *
         * Same convolution behavior as convolve(), but the order in which the photons are
         * multiplied into the array is randomized to destroy any flux or position correlations.
         *
         * @param[in] rhs PhotonArray to convolve with this one.  Must be same size.
         * @param[in] ud  A UniformDeviate used to shuffle the input photons.
         */
        void convolveShuffle(const PhotonArray& rhs, UniformDeviate ud);

        /**
         * @brief Take x displacement from this, and y displacement from x of another array,
         * multiplying fluxes.
         *
         * @param[in] rhs Source of y displacements
         */
        void takeYFrom(const PhotonArray& rhs);

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
         * @brief Declare that the photons in this array are correlated.
         */
        void setCorrelated(bool new_val=true) { _is_correlated = new_val; }

        /**
         * @brief Check if the current array has correlated photons.
         */
        bool isCorrelated() const { return _is_correlated; }

    private:
        std::vector<double> _x;         // Vector holding x coords of photons
        std::vector<double> _y;         // Vector holding y coords of photons
        std::vector<double> _flux;      // Vector holding flux of photons
        std::vector<double> _dxdz;      // Vector holding dxdz of photons
        std::vector<double> _dydz;      // Vector holding dydz of photons
        std::vector<double> _wavelength; // Vector holding wavelength of photons
        bool _is_correlated;            // Are the photons correlated?
    };

} // end namespace galsim

#endif
