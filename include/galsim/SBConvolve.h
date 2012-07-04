// -*- c++ -*-
#ifndef SBCONVOLVE_H
#define SBCONVOLVE_H
/** 
 * @file SBConvolve.h @brief SBProfile adapter which convolves 2 or more other SBProfiles.
 */

#include "SBProfile.h"

namespace galsim {

    // Defined in RealSpaceConvolve.cpp
    double RealSpaceConvolve(
        const SBProfile& p1, const SBProfile& p2, const Position<double>& pos, double flux);

    /**
     * @brief Convolve SBProfiles.
     *
     * Convolve one, two, three or more SBProfiles together.
     *
     * The profiles to be convolved may be provided either as the first 1, 2, or 3 parameters in the
     * constructor, or as a std::list<SBProfile*>.
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
     * (Aside from the trivial implementaion for 1 profile.)  If you try to use it for more than 2
     * profiles, an exception will be thrown.
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
         * @brief Constructor, 2 inputs.
         *
         * @param[in] s1 first SBProfile.
         * @param[in] s2 second SBProfile.
         * @param[in] real_space  Do convolution in real space? (default `real_space = false`).
         */
        SBConvolve(const SBProfile& s1, const SBProfile& s2, bool real_space=false);

        /**
         * @brief Constructor, 3 inputs.
         *
         * @param[in] s1 first SBProfile.
         * @param[in] s2 second SBProfile.
         * @param[in] s3 third SBProfile.
         * @param[in] real_space  Do convolution in real space? (default `real_space = false`).
         */
        SBConvolve(const SBProfile& s1, const SBProfile& s2, const SBProfile& s3,
                   bool real_space=false);

        /**
         * @brief Constructor, list of inputs.
         *
         * @param[in] slist Input: list of SBProfiles.
         * @param[in] real_space  Do convolution in real space? (default `real_space = false`).
         */
        SBConvolve(const std::list<SBProfile>& slist, bool real_space=false);

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

}

#endif // SBCONVOLVE_H

