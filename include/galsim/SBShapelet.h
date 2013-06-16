// -*- c++ -*-
/*
 * Copyright 2012, 2013 The GalSim developers:
 * https://github.com/GalSim-developers
 *
 * This file is part of GalSim: The modular galaxy image simulation toolkit.
 *
 * GalSim is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * GalSim is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with GalSim.  If not, see <http://www.gnu.org/licenses/>
 */

#ifndef SBSHAPELET_H
#define SBSHAPELET_H
/** 
 * @file SBShapelet.h @brief SBProfile that implements a polar shapelet profile 
 */

#include "SBProfile.h"
#include "Laguerre.h"

namespace galsim {

    /// @brief Class for describing polar shapelet surface brightness profiles.
    class SBShapelet : public SBProfile 
    {
    public:
        /** 
         * @brief Constructor.
         *
         * @param[in] sigma    scale size of Gauss-Laguerre basis set.
         * @param[in] bvec     `bvec[n,m]` contains flux information for the `(n, m)` basis
         *                     function.
         * @param[in] gsparams GSParams object storing constants that control the accuracy of image
         *                     operations and rendering, if different from the default.
         */
        SBShapelet(double sigma, LVector bvec, const GSParamsPtr& gsparams);

        /// @brief Copy Constructor. 
        SBShapelet(const SBShapelet& rhs);

        /// @brief Destructor. 
        ~SBShapelet();

        double getSigma() const;
        const LVector& getBVec() const;

    protected:
        class SBShapeletImpl;

    private:
        // op= is undefined
        void operator=(const SBShapelet& rhs);
    };

    template <typename T>
    void ShapeletFitImage(double sigma, LVector& bvec, const BaseImage<T>& image, 
                          const Position<double>& center);
}

#endif // SBSHAPELET_H

