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

#ifndef GalSim_SBInterpolatedImage_H
#define GalSim_SBInterpolatedImage_H
/**
 * @file SBInterpolatedImage.h @brief SBProfile that interpolates a given image.
 */

#include "SBProfile.h"
#include "Interpolant.h"

namespace galsim {

    PUBLIC_API double CalculateSizeContainingFlux(
        const BaseImage<double>& im, double target_flux);

    /**
     * @brief Surface Brightness Profile represented by interpolation over one or more data
     * tables/images.
     *
     * The SBInterpolatedImage class represents an arbitrary surface brightness profile (supplied as
     * an image), including rules for how to interpolate the profile between the supplied pixel
     * values.
     *
     * It is assumed that input images oversample the profiles they represent.  maxK() is set at
     * the Nyquist frequency of the input image, although it should be noted that interpolants
     * other than the ideal sinc function may make the max frequency higher than this.  The output
     * is required to be periodic on a scale > original image extent + kernel footprint, and
     * stepK() is set accordingly.
     *
     * The normal way to make an SBInterpolatedImage is to provide the image to interpolate
     * and the interpolation scheme.  See Interpolant.h for more about the different
     * kind of interpolation.
     *
     * You can provide different interpolation schemes for real and Fourier space
     * (passed as xInterp and kInterp respectively).  These are required, but there are
     * sensible defaults in the python layer wrapper class, InterpolatedImage.
     *
     * The ideal k-space interpolant is a sinc function; however, the quintic interpolant is the
     * default, based on detailed investigations on the tradeoffs between accuracy and speed.  Note
     * that, as in Bernstein & Gruen (2012), the accuracy achieved by this interpolant is dependent
     * on our choice of 4x pad factor.  Users who do not wish to pad the arrays to this degree may
     * need to use a higher-order Lanczos interpolant instead, but this is not the recommended
     * usage.  (Note: this padding is done by the python layer now, not here.)
     *
     * The surface brightness profile will be in terms of the image pixels.  The python layer
     * InterpolatedImage class takes care of converting between these units and the arcsec units
     * that are usually desired.
     */
    class PUBLIC_API SBInterpolatedImage : public SBProfile
    {
    public:
        /**
         * @brief Initialize internal quantities and allocate data tables based on a supplied 2D
         * image.
         *
         * @param[in] image       Input Image (ImageF or ImageD).
         * @param[in] init_bounds The bounds of the original unpadded image.
         * @param[in] nonzero_bounds  The bounds in which the padded image is non-zero.
         * @param[in] xInterp     Interpolation scheme to adopt between pixels
         * @param[in] kInterp     Interpolation scheme to adopt in k-space
         * @param[in] stepk       If > 0, force stepk to this value.
         * @param[in] maxk        If > 0, force maxk to this value.
         * @param[in] gsparams    GSParams object storing constants that control the accuracy of
         *                        image operations and rendering.
         */
        SBInterpolatedImage(
            const BaseImage<double>& image,
            const Bounds<int>& init_bounds, const Bounds<int>& nonzero_bounds,
            const Interpolant& xInterp, const Interpolant& kInterp,
            double stepk, double maxk, const GSParams& gsparams);

        /// @brief Copy Constructor.
        SBInterpolatedImage(const SBInterpolatedImage& rhs);

        /// @brief Destructor
        ~SBInterpolatedImage();

        const Interpolant& getXInterp() const;
        const Interpolant& getKInterp() const;
        double getPadFactor() const;

        /**
         * @brief Refine the value of stepK if the input image was larger than necessary.
         *
         * @param[in] max_stepk  Optional maximum value of stepk if you have some a priori
         *                       knowledge about an appropriate maximum.
         */
        void calculateStepK(double max_stepk=0.) const;

        /**
         * @brief Refine the value of maxK if the input image had a smaller scale than necessary.
         *
         * @param[in] max_maxk  Optional maximum value of maxk if you have some a priori
         *                      knowledge about an appropriate maximum.
         */
        void calculateMaxK(double max_maxk=0.) const;

        ConstImageView<double> getPaddedImage() const;
        ConstImageView<double> getNonZeroImage() const;
        ConstImageView<double> getImage() const;

    protected:

        class SBInterpolatedImageImpl;

    private:
        // op= is undefined
        void operator=(const SBInterpolatedImage& rhs);
    };

    class PUBLIC_API SBInterpolatedKImage : public SBProfile
    {
    public:
        /**
         * @brief Initialize internal quantities and allocate data tables based on a supplied 2D
         * image.
         *
         * @param[in] kimage      Input Fourier-space Image (ImageC).
         * @param[in] stepk       If > 0, force stepk to this value.
         * @param[in] kInterp     Interpolation scheme to adopt in k-space
         * @param[in] gsparams    GSParams object storing constants that control the accuracy of
         *                        image operations and rendering.
         */
        SBInterpolatedKImage(
            const BaseImage<std::complex<double> >& kimage, double stepk,
            const Interpolant& kInterp, const GSParams& gsparams);

        // @brief Serialization constructor.
        SBInterpolatedKImage(
            const BaseImage<double>& data,
            double stepk, double maxk,
            const Interpolant& kInterp,
            const GSParams& gsparams);

        /// @brief Copy Constructor.
        SBInterpolatedKImage(const SBInterpolatedKImage& rhs);

        /// @brief Destructor
        ~SBInterpolatedKImage();

        const Interpolant& getKInterp() const;

        ConstImageView<double> getKData() const;

    protected:

        class SBInterpolatedKImageImpl;

    private:
        // op= is undefined
        void operator=(const SBInterpolatedKImage& rhs);
    };
}

#endif
