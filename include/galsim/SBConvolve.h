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

#ifndef SBCONVOLVE_H
#define SBCONVOLVE_H
/** 
 * @file SBConvolve.h @brief SBProfile adapter which convolves 2 or more other SBProfiles.
 */

#include "SBProfile.h"

namespace galsim {

    // Defined in RealSpaceConvolve.cpp
    double RealSpaceConvolve(
        const SBProfile& p1, const SBProfile& p2, const Position<double>& pos, double flux,
        const GSParams* gsparams);

    /**
     * @brief Convolve SBProfiles.
     *
     * Convolve two, three or more SBProfiles together.
     *
     * The profiles to be convolved may be provided either as the first 2 or 3 parameters in the
     * constructor, or as a std::list<SBProfile>.
     *
     * The convolution will normally be done using discrete Fourier transforms of each of the
     * component profiles, multiplying them together, and then transforming back to real space.  The
     * nominal flux of the resulting SBConvolve is the product of the fluxes of each of the
     * component profiles.  Thus, when using the SBConvolve to convolve a galaxy of some desired
     * flux with a PSF, it is important to normalize the flux in the PSF to 1 beforehand.
     *
     * The stepK used for the k-space image will be (Sum 1/stepK()^2)^(-1/2) where the sum is over
     * all the components being convolved.  Since the size of the convolved image scales roughly as
     * the quadrature sum of the components, this should be close to Pi/Rmax where Rmax is the
     * radius that encloses all but (1-alias_threshold) of the flux in the final convolved image.
     *
     * The maxK used for the k-space image will be the minimum of the maxK() calculated for each
     * component.  Since the k-space images are multiplied, if one of them is essentially zero
     * beyond some k value, then that will be true of the final image as well.
     *
     * There is also an option to do the convolution as integrals in real space.  Each constructor
     * has an optional boolean parameter, real_space, that comes immediately after the list of
     * profiles to convolve.  Currently, the real-space integration is only enabled for 2 profiles.
     * If you try to use it for more than 2 profiles, an exception will be thrown.
     *
     * The real-space convolution is normally slower than the DFT convolution.  The exception is if
     * both component profiles have hard edges (e.g. a truncated Moffat with a Box).  In that case,
     * the maxK for each component is quite large since the ringing dies off fairly slowly.  So it
     * can be quicker to use real-space convolution instead.
     *
     */
    class SBConvolve : public SBProfile 
    {
    public:
        /**
         * @brief Constructor, list of inputs.
         *
         * @param[in] slist       Input: list of SBProfiles.
         * @param[in] real_space  Do convolution in real space? (default `real_space = false`).
         * @param[in] gsparams    GSParams to use, if different from the default.
         */
        SBConvolve(const std::list<SBProfile>& slist, bool real_space=false,
                   boost::shared_ptr<GSParams> gsparams = boost::shared_ptr<GSParams>());

        /// @brief Copy constructor.
        SBConvolve(const SBConvolve& rhs);

        /// @brief Destructor.
        ~SBConvolve();

    protected:

        class SBConvolveImpl;

    private:
        // op= is undefined
        void operator=(const SBConvolve& rhs);
    };

    // A special case of a convolution of a profile with itself, which allows for some 
    // efficiency gains over SBConvolve(s,s)
    class SBAutoConvolve : public SBProfile
    {
    public:
        /**
         * @brief Constructor
         *
         * @param[in] s         SBProfile to be convolved with itself.
         * @param[in] gsparams  GSParams to use, if different from the default.
         */
        SBAutoConvolve(const SBProfile& s, boost::shared_ptr<GSParams> gsparams = boost::shared_ptr<GSParams>());

        /// @brief Copy constructor.
        SBAutoConvolve(const SBAutoConvolve& rhs);

        /// @brief Destructor.
        ~SBAutoConvolve();

    protected:

        class SBAutoConvolveImpl;

    private:
        // op= is undefined
        void operator=(const SBAutoConvolve& rhs);
    };

    // A special case of the autocorrelation of profile (i.e. with itself), primarily used by the
    // correlated noise models
    class SBAutoCorrelate : public SBProfile
    {
    public:
        /**
         * @brief Constructor
         *
         * @param[in] s         SBProfile to be correlated with itself.
         * @param[in] gsparams  GSParams to use, if different from the default.
         */
        SBAutoCorrelate(const SBProfile& s,
                        boost::shared_ptr<GSParams> gsparams = boost::shared_ptr<GSParams>());

        /// @brief Copy constructor.
        SBAutoCorrelate(const SBAutoCorrelate& rhs);

        /// @brief Destructor.
        ~SBAutoCorrelate();

    protected:

        class SBAutoCorrelateImpl;

    private:
        // op= is undefined
        void operator=(const SBAutoCorrelate& rhs);
    };


}

#endif // SBCONVOLVE_H

