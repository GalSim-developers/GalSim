/* -*- c++ -*-
 * Copyright (c) 2012-2014 by the GalSim developers team on GitHub
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

#ifndef GalSim_SBLinearOpticalet_H
#define GalSim_SBLinearOpticalet_H
/**
 * @file SBLinearOpticalet.h @brief SBProfile that implements a 2-d LinearOpticalet.
 */

#include "SBProfile.h"

namespace galsim {

    namespace sbp {
        // How many LinearOpticalet profiles to save in the cache
        // Start with 100, though it may make sense to increase this depending on what value of
        // jmax we end up settling on.
        const int max_linearopticalet_cache = 100;
    }

    /**
     * @brief LinearOpticalet Surface Brightness Profile.
     *
     * Basis functions in the wavefront series expansion of the optical PSF.
     *
     * Reference:
     *   S. van Haver and A.J.E.M. Janssen, "Advanced analytic treatment and efficient computation
     *      of the diffraction integrals in the extended Nijboer-Zernike theory",
     *   J. Europ. Opt. Soc. Rap. Public. 8, 13044 (2013)
     */
    class SBLinearOpticalet : public SBProfile
    {
    public:
        /**
         * @brief Constructor - note that `r0` is scale length, NOT half-light radius `re` as in
         * SBSersic.
         *
         * @param[in] r0       scale radius.
         * @param[in] n1       First radial index.
         * @param[in] m1       First azimuthal index.
         * @param[in] n2       Second radial index.
         * @param[in] m2       Second azimuthal index.
         * @param[in] gsparams GSParams object storing constants that control the accuracy of image
         *                     operations and rendering, if different from the default.
         */
        SBLinearOpticalet(double r0, int n1, int m1, int n2, int m2,
                    const GSParamsPtr& gsparams);

        /// @brief Copy constructor.
        SBLinearOpticalet(const SBLinearOpticalet& rhs);

        /// @brief Destructor.
        ~SBLinearOpticalet();

        /// @brief Returns the scale radius of the Moffat profile.
        double getScaleRadius() const;

        /// @brief Return indices of the LinearOpticalet.
        int getN1() const;
        int getM1() const;
        int getN2() const;
        int getM2() const;

    protected:


        class SBLinearOpticaletImpl;
    private:
        // op= is undefined
        void operator=(const SBLinearOpticalet& rhs);
    };

}

#endif
