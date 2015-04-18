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

#ifndef GalSim_SBInterpolatedImage_H
#define GalSim_SBInterpolatedImage_H
/** 
 * @file SBInterpolatedImage.h @brief SBProfile that interpolates a given image.
 */

#include "SBProfile.h"
#include "Interpolant.h"
#include "FFT.h"

namespace galsim {

    class SBInterpolated : public SBProfile
    {
    public:
        /// @brief Copy Constructor.
        SBInterpolated(const SBInterpolated& rhs);

        /// @brief Destructor
        ~SBInterpolated();

    protected:
        class SBInterpolatedImpl;

        // Regular SBProfile pimpl constructor so as to be available to derived classes
        SBInterpolated(SBProfileImpl* pimpl) : SBProfile(pimpl) {}

    private:
        // op= is undefined
        void operator=(const SBInterpolated& rhs);
    };

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
     * You can provide different interpolation schemes for real and fourier space
     * (passed as xInterp and kInterp respectively).  These are required, but there are 
     * sensible defaults in the python layer wrapper class, InterpolatedImage.
     *
     * The ideal k-space interpolant is a sinc function; however, the quintic interpolant is the
     * default, based on detailed investigations on the tradeoffs between accuracy and speed.  Note
     * that, as in Bernstein & Gruen (2012), the accuracy achieved by this interpolant is dependent
     * on our choice of 4x pad factor.  Users who do not wish to pad the arrays to this degree may
     * need to use a higher-order Lanczos interpolant instead, but this is not the recommended
     * usage.
     *
     * The surface brightness profile will be in terms of the image pixels.  The python layer
     * InterpolatedImage class takes care of converting between these units and the arcsec units
     * that are usually desired.
     */
    class SBInterpolatedImage : public SBInterpolated
    {
    public:
        /** 
         * @brief Initialize internal quantities and allocate data tables based on a supplied 2D 
         * image.
         *
         * @param[in] image       Input Image (any of ImageF, ImageD, ImageS, ImageI).
         * @param[in] xInterp     Interpolation scheme to adopt between pixels 
         * @param[in] kInterp     Interpolation scheme to adopt in k-space
         * @param[in] pad_factor  Multiple by which to increase the image size when zero-padding
         *                        for the Fourier transform.
         * @param[in] stepk       If > 0, force stepk to this value.
         * @param[in] maxk        If > 0, force maxk to this value.
         * @param[in] gsparams    GSParams object storing constants that control the accuracy of
         *                        image operations and rendering.
         */
        template <typename T> 
        SBInterpolatedImage(
            const BaseImage<T>& image,
            boost::shared_ptr<Interpolant> xInterp,
            boost::shared_ptr<Interpolant> kInterp,
            double pad_factor, double stepk, double maxk,
            const GSParamsPtr& gsparams);

        /// @brief Same as above, but take 2-d interpolants.
        template <typename T> 
        SBInterpolatedImage(
            const BaseImage<T>& image,
            boost::shared_ptr<Interpolant2d> xInterp,
            boost::shared_ptr<Interpolant2d> kInterp,
            double pad_factor, double stepk, double maxk,
            const GSParamsPtr& gsparams);

        /// @brief Copy Constructor.
        SBInterpolatedImage(const SBInterpolatedImage& rhs);

        /// @brief Destructor
        ~SBInterpolatedImage();

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

        ConstImageView<double> getImage() const;
        boost::shared_ptr<Interpolant> getXInterp() const;
        boost::shared_ptr<Interpolant> getKInterp() const;

    protected:

        class SBInterpolatedImageImpl;

        // I'm not even sure how this works, but the tests seem to pass...
        SBInterpolatedImage(SBProfileImpl* pimpl) : SBInterpolated(pimpl) {}

    private:
        // op= is undefined
        void operator=(const SBInterpolatedImage& rhs);
    };
}

#endif
