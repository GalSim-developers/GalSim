// -*- c++ -*-
#ifndef SBBOX_H
#define SBBOX_H
/** 
 * @file SBBox.h @brief SBProfile of a 2-d tophat profile.
 */

#include "SBProfile.h"

namespace galsim {

    /** 
     * @brief Surface Brightness Profile for the Boxcar function.
     *
     * Convolution with a Boxcar function of dimensions `xw` x `yw` and sampling at pixel centres
     * is equivalent to pixelation (i.e. Surface Brightness integration) across rectangular pixels
     * of the same dimensions.  This class is therefore useful for pixelating SBProfiles.
     */ 
    class SBBox : public SBProfile 
    {
    public:
        /** 
         * @brief Constructor.
         *
         * @param[in] xw    width of Boxcar function along x.
         * @param[in] yw    width of Boxcar function along y.
         * @param[in] flux  flux (default `flux = 1.`).
         */
        SBBox(double xw, double yw=0., double flux=1.);

        /// @brief Copy constructor.
        SBBox(const SBBox& rhs);

        /// @brief Destructor.
        ~SBBox();

        /// @brief Returns the x dimension width of the Boxcar.
        double getXWidth() const;

        /// @brief Returns the y dimension width of the Boxcar.
        double getYWidth() const;

    protected:

        class SBBoxImpl;

    private:
        // op= is undefined
        void operator=(const SBBox& rhs);
    };
}

#endif // SBBOX_H

