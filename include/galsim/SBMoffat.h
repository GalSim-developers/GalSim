// -*- c++ -*-
#ifndef SBMOFFAT_H
#define SBMOFFAT_H
/** 
 * @file SBMoffat.h @brief SBProfile that implements a Moffat profile.
 */

#include "SBProfile.h"

namespace galsim {

    /**
     * @brief Surface Brightness for the Moffat Profile (an approximate description of ground-based
     * PSFs).
     */
    class SBMoffat : public SBProfile 
    {
    public:
        enum  RadiusType
        {
            FWHM,
            HALF_LIGHT_RADIUS,
            SCALE_RADIUS
        };

        /** @brief Constructor.
         *
         * @param[in] beta           Moffat beta parameter for profile `[1 + (r / rD)^2]^beta`.
         * @param[in] size           Size specification.
         * @param[in] rType          Kind of size being specified (one of FWHM, HALF_LIGHT_RADIUS,
         *                           SCALE_RADIUS).
         * @param[in] trunc          Outer truncation radius in same physical units as size,
         *                           trunc = 0. for no truncation (default `trunc = 0.`). 
         * @param[in] flux           Flux (default `flux = 1.`).
         */
        SBMoffat(double beta, double size, RadiusType rType, double trunc=0., double flux=1.);


        /// @brief Copy constructor.
        SBMoffat(const SBMoffat& rhs);

        /// @brief Destructor.
        ~SBMoffat();

        /// @brief Returns beta of the Moffat profile `[1 + (r / rD)^2]^beta`.
        double getBeta() const;

        /// @brief Returns the FWHM of the Moffat profile.
        double getFWHM() const;

        /// @brief Returns the scale radius rD of the Moffat profile `[1 + (r / rD)^2]^beta`.
        double getScaleRadius() const;

        /// @brief Returns the half light radius of the Moffat profile.
        double getHalfLightRadius() const;

    protected:

        class SBMoffatImpl;

        static double pow_1(double x, double ) { return x; }
        static double pow_2(double x, double ) { return x*x; }
        static double pow_3(double x, double ) { return x*x*x; }
        static double pow_4(double x, double ) { return x*x*x*x; }
        static double pow_int(double x, double beta) { return std::pow(x,int(beta)); }
        static double pow_gen(double x, double beta) { return std::pow(x,beta); }

    private:
        // op= is undefined
        void operator=(const SBMoffat& rhs);
    };

}

#endif // SBMOFFAT_H

