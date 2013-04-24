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

#ifndef SBBOX_H
#define SBBOX_H
/** 
 * @file SBBox.h @brief SBProfile of a 2-d tophat profile.
 */

#include "SBProfile.h"

namespace galsim {

    /** 
     * @brief Surface Brightness Profile for the Boxcar function.
     *
     * The boxcar function is a rectangular box.  Convolution with a Boxcar function of dimensions
     * `xw` x `yw` and sampling at pixel centres is equivalent to pixelation (i.e. Surface
     * Brightness integration) across rectangular pixels of the same dimensions.  This class is
     * therefore useful for pixelating SBProfiles.
     */ 
    class SBBox : public SBProfile 
    {
    public:
        /** 
         * @brief Constructor.
         *
         * @param[in] xw    width of Boxcar function along x.
         * @param[in] yw    width of Boxcar function along y.
         * @param[in] flux  flux (default `flux = 1.`).
         */
        SBBox(double xw, double yw=0., double flux=1.,
              boost::shared_ptr<GSParams> gsparams = boost::shared_ptr<GSParams>());

        /// @brief Copy constructor.
        SBBox(const SBBox& rhs);

        /// @brief Destructor.
        ~SBBox();

        /// @brief Returns the x dimension width of the Boxcar.
        double getXWidth() const;

        /// @brief Returns the y dimension width of the Boxcar.
        double getYWidth() const;

    protected:

        class SBBoxImpl;

    private:
        // op= is undefined
        void operator=(const SBBox& rhs);
    };
}

#endif // SBBOX_H

