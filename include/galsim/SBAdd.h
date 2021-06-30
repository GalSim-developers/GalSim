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

#ifndef GalSim_SBAdd_H
#define GalSim_SBAdd_H
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
    class PUBLIC_API SBAdd : public SBProfile
    {
    public:

        /**
         * @brief Constructor, list of inputs.
         *
         * @param[in] slist    List of SBProfiles.
         * @param[in] gsparams GSParams object storing constants that control the accuracy of image
         *                     operations and rendering, if different from the default.
         */
        SBAdd(const std::list<SBProfile>& slist, const GSParams& gsparams);

        /// @brief Copy constructor.
        SBAdd(const SBAdd& rhs);

        /// @brief Destructor.
        ~SBAdd();

        /// @brief Get the list of SBProfiles that are being added together
        std::list<SBProfile> getObjs() const;

    protected:

        class SBAddImpl;

    private:
        // op= is undefined
        void operator=(const SBAdd& rhs);
    };
}

#endif

