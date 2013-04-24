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

#ifndef SBDECONVOLVE_H
#define SBDECONVOLVE_H
/** 
 * @file SBDeconvolve.h @brief SBProfile adapter which inverts its subject in k space to effect a
 * deconvolution.
 */


#include "SBProfile.h"

namespace galsim {

    /**
     * @brief SBProfile adapter which inverts its subject in k space to effect a deconvolvution.
     *
     * (TODO: Add more docs here!)
     */
    class SBDeconvolve : public SBProfile 
    {
    public:
        /// @brief Constructor.
        SBDeconvolve(const SBProfile& adaptee,
                     boost::shared_ptr<GSParams> gsparams = boost::shared_ptr<GSParams>());

        /// @brief Copy constructor.
        SBDeconvolve(const SBDeconvolve& rhs);

        /// @brief Destructor.
        ~SBDeconvolve();

    protected:

        class SBDeconvolveImpl;

    private:
        // op= is undefined
        void operator=(const SBDeconvolve& rhs);
    };

}

#endif // SBDECONVOLVE_H
