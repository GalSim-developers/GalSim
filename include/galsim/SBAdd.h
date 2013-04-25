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

#ifndef SBADD_H
#define SBADD_H
/** 
 * @file SBAdd.h @brief SBProfile adapter that is the sum of 2 or more other SBProfiles.
 */

#include "SBProfile.h"

namespace galsim {

    /** 
     * @brief Sums SBProfiles. 
     *
     * The SBAdd class can be used to add arbitrary numbers of SBProfiles together.
     */
    class SBAdd : public SBProfile 
    {
    public:

        /** 
         * @brief Constructor, list of inputs.
         *
         * @param[in] slist list of SBProfiles.
         */
        SBAdd(const std::list<SBProfile>& slist,
              boost::shared_ptr<GSParams> gsparams = boost::shared_ptr<GSParams>());

        /// @brief Copy constructor.
        SBAdd(const SBAdd& rhs);

        /// @brief Destructor.
        ~SBAdd();

    protected:

        class SBAddImpl;

    private:
        // op= is undefined
        void operator=(const SBAdd& rhs);
    };
}

#endif // SBPROFILE_H

