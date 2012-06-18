// -*- c++ -*-
#ifndef PHOTON_ARRAY_H
#define PHOTON_ARRAY_H

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
     * Class holds a vector of information about photon arrivals: x and y positions, and a flux
     * carried by each photon.  It is the intention that fluxes of photons be nearly equal in absolute 
     * value so that noise statistics can be estimated by counting number of positive and negative photons.
     * This class holds the code that allows its flux to be added to a surface-brightness Image.
     */
    class PhotonArray 
    {
    public:
        /** 
         * @brief Construct an array of given size with zero-flux photons
         *
         * @param[in] N Size of desired array.
         */
        explicit PhotonArray(int N): _x(N,0.), _y(N,0.), _flux(N,0.) {}

        /** 
         * @brief Construct from three vectors.  Exception if vector sizes do not match.
         *
         * @param[in] vx vector of photon x coordinates
         * @param[in] vy vector of photon y coordinates
         * @param[in] vflux vector of photon fluxes
         */
        PhotonArray(std::vector<double>& vx, std::vector<double>& vy, std::vector<double>& vflux);

        /**
         * @brief Accessor for array size
         *
         * @returns Array size
         */
        int size() const {return _x.size();}

        /** @brief reserve space in arrays for future elements
         *
         * @param[in] N number of elements to reserve space for.
         */
        void reserve(int N) 
        {
            _x.reserve(N);
            _y.reserve(N);
            _flux.reserve(N);
        }

        /**
         * @brief Set characteristics of a photon
         *
         * @param[in] i Index of desired photon (no bounds checking)
         * @param[in] x x coordinate of photon
         * @param[in] y y coordinate of photon
         * @param[in] flux flux of photon
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
        double getX(int i) const {return _x[i];}

        /**
         * @brief Access y coordinate of a photon
         *
         * @param[in] i Index of desired photon (no bounds checking)
         * @returns y coordinate of photon
         */
        double getY(int i) const {return _y[i];}

        /**
         * @brief Access flux of a photon
         *
         * @param[in] i Index of desired photon (no bounds checking)
         * @returns flux of photon
         */
        double getFlux(int i) const {return _flux[i];}

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
         * @brief Extend this array with the contents of another.
         *
         * @param[in] rhs PhotonArray whose contents to append to this one.
         */
        void append(const PhotonArray& rhs);

        /**
         * @brief Convolve this array with another.
         *
         * Convolution of two arrays is defined as adding the coordinates on a photon-by-photon basis
         * and multiplying the fluxes on a photon-by-photon basis. Output photons' flux is renormalized
         * so that the expectation value of output total flux is product of two input totals, if
         * the two photon streams are uncorrelated.
         *
         * @param[in] rhs PhotonArray to convolve with this one.  Must be same size.
         */
        void convolve(const PhotonArray& rhs);

        /**
         * @brief Convolve this array with another, shuffling the order in which photons are combined.
         *
         * Same convolution behavior as convolve(), but the order in which the photons are
         * multiplied into the array is randomized to destroy any flux or position correlations.
         *
         * @param[in] rhs PhotonArray to convolve with this one.  Must be same size.
         * @param[in] ud  A UniformDeviate used to shuffle the input photons.
         */
        void convolveShuffle(const PhotonArray& rhs, UniformDeviate& ud);

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
         */
        template <class T>
        void addTo(ImageView<T>& target) const;
    private:
        std::vector<double> _x;      // Vector holding x coords of photons
        std::vector<double> _y;      // Vector holding y coords of photons
        std::vector<double> _flux;   // Vector holding flux of photons
    };

} // end namespace galsim

#endif
