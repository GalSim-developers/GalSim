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

#ifndef SBLAGUERRE_H
#define SBLAGUERRE_H
/** 
 * @file SBLaguerre.h @brief SBProfile that implements a 2-d Gauss-Laguerre profile (aka shapelets)
 */

#include "SBProfile.h"
#include "Laguerre.h"

namespace galsim {

    /// @brief Class for describing Gauss-Laguerre polynomial Surface Brightness Profiles.
    class SBLaguerre : public SBProfile 
    {
    public:
        /** 
         * @brief Constructor.
         *
         * @param[in] bvec   `bvec[n,n]` contains flux information for the `(n, n)` basis function.
         * @param[in] sigma  scale size of Gauss-Laguerre basis set (default `sigma = 1.`).
         */
        SBLaguerre(LVector bvec=LVector(), double sigma=1., 
                   boost::shared_ptr<GSParams> gsparams = boost::shared_ptr<GSParams>());

        /// @brief Copy Constructor. 
        SBLaguerre(const SBLaguerre& rhs);

        /// @brief Destructor. 
        ~SBLaguerre();

    protected:
        class SBLaguerreImpl;

    private:
        // op= is undefined
        void operator=(const SBLaguerre& rhs);
    };
}

#endif // SBLAGUERRE_H

