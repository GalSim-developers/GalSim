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

#ifndef GalSim_SBSpergelet_H
#define GalSim_SBSpergelet_H
/**
 * @file SBSpergelet.h @brief SBProfile that implements a 2-d Spergelet.
 */

#include "SBProfile.h"

namespace galsim {

    namespace sbp {

        // Constrain range of allowed Spergel index nu.  Spergel (2010) Table 1 lists values of nu
        // from -0.9 to +0.85. I've found that nu = -0.9 is too tricky for the GKP integrator to
        // handle, however, so I'm setting the lower range to -0.85 instead.  I haven't run into any
        // problems with the upper limit though, which could probably be extended.

        const double minimum_spergel_nu = -0.85;
        const double maximum_spergel_nu = 0.85;

        // How many Spergel profiles to save in the cache
        // Start with 100, though it may make sense to increase this depending on what value of
        // jmax we end up settling on.
        const int max_spergelet_cache = 100;

    }

    /**
     * @brief Spergelet Surface Brightness Profile.
     *
     * Essentially equation 47 from Spergel (2010), but with a small bug correction.
     *
     * Reference:
     *   D. N. Spergel, "ANALYTICAL GALAXY PROFILES FOR PHOTOMETRIC AND LENSING ANALYSIS,""
     *   ASTROPHYS J SUPPL S 191(1), 58-65 (2010) [doi:10.1088/0067-0049/191/1/58].
     */
    class SBSpergelet : public SBProfile
    {
    public:
        // enum  RadiusType
        // {
        //     HALF_LIGHT_RADIUS,
        //     SCALE_RADIUS
        // };
        /**
         * @brief Constructor - note that `r0` is scale length, NOT half-light radius `re` as in
         * SBSersic.
         *
         * @param[in] nu       index parameter setting the logarithmic slope of the profile.
         * @param[in] r0       scale radius.
         * @param[in] flux     flux.
         * @param[in] j        Radial index.
         * @param[in] m        First azimuthal index.
         * @param[in] n        Second azimuthal index.
         * @param[in] gsparams GSParams object storing constants that control the accuracy of image
         *                     operations and rendering, if different from the default.
         */
        SBSpergelet(double nu, double r0, double flux, int j, int m, int n,
                    const GSParamsPtr& gsparams);

        /// @brief Copy constructor.
        SBSpergelet(const SBSpergelet& rhs);

        /// @brief Destructor.
        ~SBSpergelet();

        /// @brief Returns the Spergel index `nu` of the profile.
        double getNu() const;

        /// @brief Returns the scale radius of the Spergel profile.
        double getScaleRadius() const;

        /// @brief Return jmn indices of the Spergelet.
        int getJ() const;
        int getM() const;
        int getN() const;

    protected:

        class SBSpergeletImpl;

    private:
        // op= is undefined
        void operator=(const SBSpergelet& rhs);
    };

}

#endif
