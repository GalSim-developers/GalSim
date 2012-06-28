// -*- c++ -*-
#ifndef SBSERSIC_H
#define SBSERSIC_H
/** 
 * @file SBSersic.h @brief SBProfile that implements a Sersic profile.
 */

#include "SBProfile.h"

namespace galsim {

    /**
     * @brief Sersic Surface Brightness Profile.
     *
     * The Sersic Surface Brightness Profile is characterized by three properties: its Sersic 
     * index `n`, its `flux` and the half-light radius `re`.
     */
    class SBSersic : public SBProfile 
    {
    public:
        /**
         * @brief Constructor.
         *
         * @param[in] n     Sersic index.
         * @param[in] re    half-light radius.
         * @param[in] flux  flux (default `flux = 1.`).
         */
        SBSersic(double n, double re, double flux=1.);

        /// @brief Copy constructor.
        SBSersic(const SBSersic& rhs);

        /// @brief Destructor.
        ~SBSersic();

        /// @brief Returns the Sersic index `n` of the profile.
        double getN() const;

        /// @brief Returns the half light radius of the Sersic profile.
        double getHalfLightRadius() const;

    protected:
        class SersicInfo;
        class SersicRadialFunction;
        class SBSersicImpl;
        class InfoBarn;

        /// One static map of all `SersicInfo` structures for whole program.
        static InfoBarn nmap; 

    private:
        // op= is undefined
        void operator=(const SBSersic& rhs);
    };

    /**
     * @brief Surface Brightness for the de Vaucouleurs Profile, a special case of the Sersic with 
     * `n = 4`.
     */
    class SBDeVaucouleurs : public SBSersic 
    {
    public:
        /** 
         * @brief Constructor.
         *
         * @param[in] r0    Half-light radius.
         * @param[in] flux  flux (default `flux = 1.`).
         */
      SBDeVaucouleurs(double r0, double flux=1.) : SBSersic(4., r0, flux) {}
    };

}

#endif // SBSERSIC_H

