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

#ifndef GalSim_SBBox_H
#define GalSim_SBBox_H
/**
 * @file SBBox.h @brief SBProfile of a 2-d tophat profile.
 */

#include "SBProfile.h"

namespace galsim {

    /**
     * @brief Surface Brightness Profile for the Boxcar function.
     *
     * The boxcar function is a rectangular box.  Convolution with a Boxcar function of dimensions
     * `width` x `height` and sampling at pixel centres is equivalent to pixelation (i.e. Surface
     * Brightness integration) across rectangular pixels of the same dimensions.  This class is
     * therefore useful for pixelating SBProfiles.
     */
    class PUBLIC_API SBBox : public SBProfile
    {
    public:
        /**
         * @brief Constructor.
         *
         * @param[in] width    width of Boxcar function along x.
         * @param[in] height   height of Boxcar function along y.
         * @param[in] flux     flux.
         * @param[in] gsparams GSParams object storing constants that control the accuracy of image
         *                     operations and rendering, if different from the default.
         */
        SBBox(double width, double height, double flux, const GSParams& gsparams);

        /// @brief Copy constructor.
        SBBox(const SBBox& rhs);

        /// @brief Destructor.
        ~SBBox();

        /// @brief Returns the x dimension width of the Boxcar.
        double getWidth() const;

        /// @brief Returns the y dimension width of the Boxcar.
        double getHeight() const;

    protected:

        class SBBoxImpl;

    private:
        // op= is undefined
        void operator=(const SBBox& rhs);
    };

    /**
     * @brief Surface Brightness Profile for the TopHat function.
     *
     * The tophat function is much like the boxcar, but a circular plateau, rather than
     * a rectangle.  It is defined by a radius and a flux.
     */
    class PUBLIC_API SBTopHat : public SBProfile
    {
    public:
        /**
         * @brief Constructor.
         *
         * @param[in] radius    radius of TopHat function
         * @param[in] flux      flux.
         * @param[in] gsparams  GSParams object storing constants that control the accuracy of
         *                      image operations and rendering, if different from the default.
         */
        SBTopHat(double radius, double flux, const GSParams& gsparams);

        /// @brief Copy constructor.
        SBTopHat(const SBTopHat& rhs);

        /// @brief Destructor.
        ~SBTopHat();

        /// @brief Returns the radius of the TopHat.
        double getRadius() const;

    protected:

        class SBTopHatImpl;

    private:
        // op= is undefined
        void operator=(const SBTopHat& rhs);
    };
}

#endif

