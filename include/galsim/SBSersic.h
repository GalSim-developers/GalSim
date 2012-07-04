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
     * The Sersic Surface Brightness Profile is characterized by three properties: its Sersic index
     * `n`, its `flux` and the half-light radius `re`.  Given these properties, the surface
     * brightness profile scales as I(r) propto exp[-(r/r_0)^{1/n}].  Currently the code is limited
     * to 0.5<=n<=4.2, with an exception thrown for values outside that range.
     *
     * There are several special cases of the Sersic profile that have their own SBProfiles: n=4
     * (SBDeVaucouleurs), n=1 (SBExponential), n=0.5 (SBGaussian).  These special cases use several
     * simplifications in all calculations, whereas for general n, the Fourier transform must be
     * treated numerically.
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
         * @param[in] re    Half-light radius.
         * @param[in] flux  flux (default `flux = 1.`).
         */
        SBDeVaucouleurs(double re, double flux=1.) : SBSersic(4., re, flux) {}
    };

}

#endif // SBSERSIC_H

