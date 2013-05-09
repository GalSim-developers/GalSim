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

#ifndef SBTRANSFORM_H
#define SBTRANSFORM_H
/** 
 * @file SBTransform.h @brief SBProfile adapter that transforms another SBProfile.
 * Includes shear, dilation, rotation, translation, and flux scaling.
 *
 */

#include "SBProfile.h"

namespace galsim {

    /**
     * @brief An affine transformation of another SBProfile.
     *
     * Origin of original shape will now appear at `_cen`.
     * Flux is NOT conserved in transformation - surface brightness is preserved.
     * We keep track of all distortions in a 2x2 matrix `M = [(A B), (C D)]` = [row1, row2] 
     * plus a 2-element Positon object `cen` for the shift, and a flux scaling,
     * in addition to the scaling implicit in the matrix M = abs(det(M)).
     */
    class SBTransform : public SBProfile
    {
    public:
        /** 
         * @brief General constructor.
         *
         * @param[in] sbin SBProfile being transform
         * @param[in] mA A element of 2x2 distortion matrix `M = [(A B), (C D)]` = [row1, row2]
         * @param[in] mB B element of 2x2 distortion matrix `M = [(A B), (C D)]` = [row1, row2]
         * @param[in] mC C element of 2x2 distortion matrix `M = [(A B), (C D)]` = [row1, row2]
         * @param[in] mD D element of 2x2 distortion matrix `M = [(A B), (C D)]` = [row1, row2]
         * @param[in] cen 2-element (x, y) Position for the translational shift.
         * @param[in] fluxScaling Amount by which the flux should be multiplied.
         */
        SBTransform(const SBProfile& sbin, double mA, double mB, double mC, double mD,
                    const Position<double>& cen=Position<double>(0.,0.), double fluxScaling=1.,
                    boost::shared_ptr<GSParams> gsparams = boost::shared_ptr<GSParams>());

        /// @brief Copy constructor
        SBTransform(const SBTransform& rhs);

        /// @brief Destructor
        ~SBTransform();

    protected:

        class SBTransformImpl;

    private:
        // op= is undefined
        void operator=(const SBTransform& rhs);
    };

}

#endif // SBTRANSFORM_H

