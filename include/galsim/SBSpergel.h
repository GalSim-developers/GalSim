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

#ifndef GalSim_SBSpergel_H
#define GalSim_SBSpergel_H
/**
 * @file SBSpergel.h @brief SBProfile that implements a 2-d Spergel profile.
 */

#include "SBProfile.h"

namespace galsim {

    namespace sbp {

        // Constrain range of allowed Spergel index nu.  Spergel (2010) Table 1 lists values of nu
        // from -0.9 to +0.85. I've found that nu = -0.9 is too tricky for the GKP integrator to
        // handle, however, so I'm setting the lower range to -0.85 instead.  The upper limit is
        // set by the boost::math::cyl_bessel_k function, which I found runs into overflow errors
        // for nu larger than about 4.0.

        const double minimum_spergel_nu = -0.85;
        const double maximum_spergel_nu = 4.0;

        // How many Spergel profiles to save in the cache
        const int max_spergel_cache = 100;

    }

    PUBLIC_API double SpergelCalculateHLR(double nu);

    /**
     * @brief Spergel Surface Brightness Profile.
     *
     * Surface brightness profile with `I(r) propto (r/r_c)^{nu} K_{nu}(r/r_c)` for some
     * scale-length `r_c = r_0 / c_{nu}`, where `K_{nu}(u)` is the modified Bessel function of the
     * second kind (also confusingly referred to as the 'spherical modified Bessel function of the
     * third kind') and `nu > -1`. For different parameters `nu` this profile can approximate Sersic
     * profiles with different indices.
     *
     * Reference:
     *   D. N. Spergel, "ANALYTICAL GALAXY PROFILES FOR PHOTOMETRIC AND LENSING ANALYSIS,""
     *   ASTROPHYS J SUPPL S 191(1), 58-65 (2010) [doi:10.1088/0067-0049/191/1/58].
     */
    class PUBLIC_API SBSpergel : public SBProfile
    {
    public:
        /**
         * @brief Constructor - note that `r0` is scale length, NOT half-light radius `re` as in
         * SBSersic.
         *
         * @param[in] nu       index parameter setting the logarithmic slope of the profile.
         * @param[in] scale_radius  scale radius
         * @param[in] flux     flux.
         * @param[in] gsparams GSParams object storing constants that control the accuracy of image
         *                     operations and rendering, if different from the default.
         */
        SBSpergel(double nu, double scale_radius, double flux, const GSParams& gsparams);

        /// @brief Copy constructor.
        SBSpergel(const SBSpergel& rhs);

        /// @brief Destructor.
        ~SBSpergel();

        /// @brief Returns the Spergel index `nu` of the profile.
        double getNu() const;

        /// @brief Returns the scale radius of the Spergel profile.
        double getScaleRadius() const;

        /// @brief Return integrated flux of circular profile.
        double calculateIntegratedFlux(double r) const;

        /// @brief Return radius enclosing flux f
        double calculateFluxRadius(double f) const;

    protected:

        class SBSpergelImpl;

    private:
        // op= is undefined
        void operator=(const SBSpergel& rhs);
    };

}

#endif
