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

#ifndef GalSim_SBProfile_H
#define GalSim_SBProfile_H
/**
 * @file SBProfile.h @brief Contains a class definition for two-dimensional Surface Brightness
 * Profiles.
 *
 * The SBProfiles include common star, galaxy, and PSF shapes.
 */


#include <cmath>
#include <list>
#include <map>
#include <vector>
#include <algorithm>
#include <complex>

#include "Std.h"
#include "Random.h"
#include "GSParams.h"
#include "Image.h"
#include "PhotonArray.h"

namespace galsim {

    // All code between the @cond and @endcond is excluded from Doxygen documentation
    //! @cond

    /// @brief Exception class thrown by SBProfiles.
    class PUBLIC_API SBError : public std::runtime_error
    {
    public:
        SBError(const std::string& m) : std::runtime_error("SB Error: " + m) {}
    };

    //! @endcond

    class PUBLIC_API SBTransform;

    /**
     * @brief A base class representing all of the 2D surface brightness profiles that we know how
     * to draw.
     *
     * The SBProfile class is a representation of a surface brightness distribution across a
     * 2-dimensional image plane, with real and/or Fourier-domain models of a wide variety of galaxy
     * shapes, point-spread functions (PSFs), and their convolutions.  There are several
     * realizations of the SBProfile classes: There are the "atomic" classes that represent specific
     * analytic profiles: (SBDeltaFunction, SBGaussian, SBSersic, SBAiry, SBExponential, SBBox,
     * SBDeVaucouleurs and SBMoffat). SBInterpolatedImage represents a pattern defined by a grid of
     * pixel values and a chosen interpolation scheme between pixel centers.  SBTransform represents
     * any affine transformation (shear, magnification, rotation, translation, and/or flux
     * rescaling) of any other SBProfile. SBAdd represents the sum of any number of SBProfiles.
     * SBConvolve represents the convolution of any number of SBProfiles, and SBDeconvolve is the
     * deconvolution of one SBProfile with another.
     *
     * Every SBProfile knows how to draw an Image<float> of itself in real and k space.  Each also
     * knows what is needed to prevent aliasing or truncation of itself when drawn.  The classes
     * model the surface brightness of the object, so the xValue routine returns the value of the
     * surface brightness at a given point.  Usually we define this as flux/arcsec^2.
     *
     * However, when drawing an SBProfile, any distance measures used to define the SBProfile will
     * be taken to be in units of pixels.  Thus the image, which is nominally a **surface
     * brightness** image is also a **flux** image.  The flux in each pixel is the flux/pixel^2.
     * The python layer takes care of the typical case of defining an SBProfile in terms of arcsec,
     * then converting the image to pixels (scaling the size and flux appropriately), so that the
     * draw command here has the correct flux.
     *
     * This isn't an abstract base class.  An SBProfile is a concrete object which internally has
     * a pointer to the implementation details (which _is_ an abstract base class).  Furthermore,
     * all SBProfiles are immutable objects.  Any changes are made through modifiers that return a
     * new object.  (e.g. setFlux, shear, shift, etc.)  This means that we can safely make
     * SBProfiles use shallow copies, since that will never be confusing, which in turn means that
     * SBProfiles can be safely returned by value, used in containers (e.g. list<SBProfile>), etc.
     *
     * The only constructor for SBProfile is the copy constructor.  All SBProfiles need to be
     * created as one of the derived types that have real constructors.
     *
     * Well, technically, there is also a default constructor to make it easier to use containers
     * of SBProfiles.  However, it is an error to use an SBProfile that has been default
     * constructed for any purpose.
     *
     * The assignment operator does a shallow copy, replacing the current contents of the SBProfile
     * with that of the rhs profile.
     *
     */
    class PUBLIC_API SBProfile
    {
    public:

        /**
         * @brief Default constructor for convenience only.  Do not use!
         *
         * This constructor is only provided so you can do things like:
         * @code
         * std::list<SBProfile> prof_list;
         * prof_list.push_back(psf);
         * prof_list.push_back(gal);
         * prof_list.push_back(pix);
         * @endcode
         * The default constructor for std::list strangely requires a default
         * constructor for the argument type, even though it isn't ever really used.
         */
        SBProfile();

        /// Only legitimate public constructor is a copy constructor.
        SBProfile(const SBProfile& rhs);

        /// operator= replaces the current contents with those of the rhs.
        SBProfile& operator=(const SBProfile& rhs);

        /// Destructor isn't virtual, since derived classes don't have anything to cleanup.
        ~SBProfile();

        /// Get the GSParams object for this SBProfile
        /// Makes a copy, since this is used for python pickling, so the current SBProfile
        /// may go out of scope before we need to use the returned GSParams.
        GSParams getGSParams() const;

        /**
         * @brief Return value of SBProfile at a chosen 2D position in real space.
         *
         * Assume all are real-valued.  xValue() may not be implemented for derived classes
         * (SBConvolve) that require an FFT to determine real-space values.  In this case, an
         * SBError will be thrown.
         *
         * @param[in] p 2D position in real space.
         */
        double xValue(const Position<double>& p) const;

        /**
         * @brief Return value of SBProfile at a chosen 2D position in k space.
         *
         * @param[in] k 2D position in k space.
         */
        std::complex<double> kValue(const Position<double>& k) const;

        //@{
        /**
         *  @brief Define the range over which the profile is not trivially zero.
         *
         *  These values are used when a real-space convolution is requested to define
         *  the appropriate range of integration.
         *  The implementation here is +- infinity for both x and y.
         *  Derived classes may override this if they a have different range.
         */
        void getXRange(double& xmin, double& xmax, std::vector<double>& splits) const;

        void getYRange(double& ymin, double& ymax, std::vector<double>& splits) const;

        void getYRangeX(double x, double& ymin, double& ymax, std::vector<double>& splits) const;
        //@}

        /// @brief Value of k beyond which aliasing can be neglected.
        double maxK() const;

        /// @brief Real-space image pixel spacing that does not alias maxK.
        double nyquistDx() const { return M_PI / maxK(); }

        /// @brief Sampling in k-space necessary to avoid folding too much of image in x space.
        double stepK() const;

        /**
         * @brief Determine a good size for a drawn image based on dx and stepK()
         *
         * @param[in] dx      The pixel scale of the image
         *
         * @returns the recommended image size.
         *
         * The basic formula is 2pi / (dx * stepK()).
         * But then we round up to the next even integer value.
         */
        int getGoodImageSize(double dx) const;

        /**
         * @brief Check whether the SBProfile is known to have rotational symmetry about x=y=0
         *
         * If the SBProfile has rotational symmetry, certain calculations can be simplified.
         */
        bool isAxisymmetric() const;

        /**
         *  @brief The presence of hard edges help determine whether real space convolution might be
         *  a better choice.
         */
        bool hasHardEdges() const;

        /**
         * @brief Check whether the SBProfile is analytic in the real domain.
         *
         *
         * An SBProfile is "analytic" in the real domain if values can be determined immediately at
         * any position through formula or a stored table (no DFT); this makes certain calculations
         * more efficient.
         */
        bool isAnalyticX() const;

        /**
         * @brief Check whether the SBProfile is analytic in the Fourier domain.
         *
         * An SBProfile is "analytic" in the k domain if values can be determined immediately at any
         * position through formula or a stored table (no DFT); this makes certain calculations
         * more efficient.
         */
        bool isAnalyticK() const;

        /// @brief Returns (X, Y) centroid of SBProfile.
        Position<double> centroid() const;

        /// @brief Get the total flux of the SBProfile.
        double getFlux() const;

        /// @brief Get an estimate of the maximum surface brightness.
        double maxSB() const;

        // ****Methods implemented in base class****

        // Transformations (all are special cases of affine transformations via SBTransform):

        /**
         * @brief Multiply the flux by fluxRatio
         */
        SBTransform scaleFlux(double fluxRatio) const;

        /**
         * @brief Apply an overall scale change to the profile, preserving surface brightness.
         */
        SBTransform expand(double scale) const;

        /**
         * @brief Apply a given rotation in radians
         */
        SBTransform rotate(double theta) const;

        /**
         * @brief Apply a transformation given by an arbitrary Jacobian matrix
         */
        SBTransform transform(double dudx, double dudy, double dvdx, double dvdy) const;

        /**
         * @brief Apply a translation.
         */
        SBTransform shift(const Position<double>& delta) const;

        /**
         * @brief Shoot photons through this SBProfile.
         *
         * Returns an array of photon coordinates and fluxes that are drawn from the light
         * distribution of this SBProfile.  Absolute value of each photons' flux should be
         * approximately equal, but some photons can be negative as needed to represent negative
         * regions.  Note that the ray-shooting method is not intended to produce a randomized value
         * of the total object flux, so do not assume that there will be sqrt(N) error on the flux.
         * In fact most implementations will return a PhotonArray with exactly correct flux, with
         * only the *distribution* of flux on the sky that will definitely have sampling noise.
         *
         * The one definitive gaurantee is that, in the limit of large number of photons, the
         * surface brightness distribution of the photons will converge on the SB pattern defined by
         * the object.
         *
         * Objects with regions of negative flux will result in creation of photons with negative
         * flux.  Absolute value of negative photons' flux should be nearly equal to the standard
         * flux of positive photons.  Shot-noise fluctuations between the number of positive and
         * negative photons will produce noise in the total net flux carried by the output
         * [PhotonArray](@ref PhotonArray).
         *
         * The typical implementation will be to take the integral of the absolute value of flux,
         * and divide it nearly equally into N photons.  The photons are then drawn from the
         * distribution of the *absolute value* of flux.  If a photon is drawn from a region of
         * negative flux, then that photon's flux is negated.  Because of cancellation, this means
         * that each photon will carry more than `getFlux()/N` flux if there are negative-flux
         * regions in the object.  It also means that during convolution, addition, or
         * interpolation, positive- and negative-flux photons can be contributing to the same region
         * of the image.  Their cancellation means that the shot noise may be substantially higher
         * than you would expect if you had only positive-flux photons.
         *
         * The photon flux may also vary slightly as a means of speeding up photon-shooting, as an
         * alternative to rejection sampling.  See `OneDimensionalDeviate` documentation.
         *
         * @param[in] photons PhotonArray in which to write the photon information
         * @param[in] rng BaseDeviate that will be used to draw photons from distribution.
         */
        void shoot(PhotonArray& photons, BaseDeviate rng) const;

        /**
         * @brief Return expectation value of flux in positive photons when shoot() is called
         *
         * Returns expectation value of flux returned in positive-valued photons when
         * [shoot()](@ref shoot)
         * is called for this object.  Default implementation is to return getFlux(), if it is
         * positive, or 0 otherwise, which will be
         * the case when the SBProfile is constructed entirely from elements of the same sign.
         *
         * It should be generally true that `getPositiveFlux() - getNegativeFlux()` returns the same
         * thing as `getFlux()`.  Small difference may accrue from finite numerical accuracy in
         * cases involving lookup tables, etc.
         *
         * @returns Expected positive-photon flux.
         */
        double getPositiveFlux() const;

        /**
         * @brief Return expectation value of absolute value of flux in negative photons from
         * shoot()
         *
         * Returns expectation value of (absolute value of) flux returned in negative-valued photons
         * when shoot() is called for this object.
         * Default implementation is to return getFlux() if it is negative, 0 otherwise,
         * which will be the case when the SBProfile is constructed entirely from elements that
         * have the same sign.
         *
         * It should be generally true that `getPositiveFlux() - getNegativeFlux()` returns the
         * same thing as `getFlux()`.  Small difference may accrue from finite numerical accuracy
         * in cases involving lookup tables, etc.
         *
         * @returns Expected absolute value of negative-photon flux.
         */
        double getNegativeFlux() const;

        /**
         * @brief Draw an image of the SBProfile in real space forcing the use of real methods
         * where we have a formula for x values.
         * For SBProfiles without an analytic real-space representation, an exception will be
         * thrown.
         *
         * The image is not cleared out before drawing.  So this profile will be added to anything
         * already on the input image.
         *
         * @param[in,out]    image (any of ImageViewF, ImageViewD, ImageViewS, ImageViewI)
         * @param[in]        dx, the pixel scale
         */
        template <typename T>
        void draw(ImageView<T> image, double dx, double* jac,
                  double xoff, double yoff, double flux_ratio) const;

        /**
         * @brief Draw an image of the SBProfile in k space.
         *
         * For drawing in k space: routines are analagous to real space, except the image is
         * complex. The image is normalized such that I(0,0) is the total flux.
         *
         * @param[in,out]    image in k space (must be an ImageViewC)
         * @param[in]        dk, the step size in k space
         */
        template <typename T>
        void drawK(ImageView<std::complex<T> > image, double dk, double* jac) const;

    protected:

        class SBProfileImpl;

        // Regular constructor only available to derived classes
        SBProfile(SBProfileImpl* pimpl);

        // Protected static class to access pimpl of one SBProfile object from another one.
        static SBProfileImpl* GetImpl(const SBProfile& rhs);

        shared_ptr<SBProfileImpl> _pimpl;
    };

}

#endif
