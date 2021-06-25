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

#ifndef GalSim_SBInterpolatedImageImpl_H
#define GalSim_SBInterpolatedImageImpl_H

#include "SBProfileImpl.h"
#include "SBInterpolatedImage.h"
#include "ProbabilityTree.h"

namespace galsim {

    class SBInterpolatedImage::SBInterpolatedImageImpl : public SBProfile::SBProfileImpl
    {
    public:

        SBInterpolatedImageImpl(
            const BaseImage<double>& image,
            const Bounds<int>& init_bounds, const Bounds<int>& nonzero_bounds,
            const Interpolant& xInterp, const Interpolant& kInterp,
            double stepk, double maxk, const GSParams& gsparams);

        ~SBInterpolatedImageImpl();

        ////////////////////////////////////////////////////////////////////////
        // Methods of SBProfileImpl that are overriden/implemented in this subclass:

        double xValue(const Position<double>& p) const;
        std::complex<double> kValue(const Position<double>& p) const;

        template <typename T>
        void fillXImage(ImageView<T> im,
                        double x0, double dx, int izero,
                        double y0, double dy, int jzero) const;
        template <typename T>
        void fillXImage(ImageView<T> im,
                        double x0, double dx, double dxy,
                        double y0, double dy, double dyx) const;
        template <typename T>
        void fillKImage(ImageView<std::complex<T> > im,
                        double kx0, double dkx, int izero,
                        double ky0, double dky, int jzero) const;
        template <typename T>
        void fillKImage(ImageView<std::complex<T> > im,
                        double kx0, double dkx, double dkxy,
                        double ky0, double dky, double dkyx) const;

        double maxK() const { return _maxk; }
        double stepK() const { return _stepk; }
        bool isAxisymmetric() const { return false; }
        // We'll use false here, but really, there's not an easy way to tell.
        // Certainly an Image _could_ have hard edges.
        bool hasHardEdges() const { return false; }
        // This class will be set up so that both x and k domain values
        // are found by interpolation of a table:
        bool isAnalyticX() const { return true; }
        bool isAnalyticK() const { return true; }
        Position<double> centroid() const;
        double getFlux() const;
        double maxSB() const;

        /**
         *
         * @brief Shoot photons through this object
         *
         * SBInterpolatedImage will assign photons to its input pixels with probability
         * proportional to their flux.  Each photon will then be displaced from its pixel center
         * by an (x,y) amount drawn from the interpolation kernel.  Note that if either the input
         * image or the interpolation kernel have negative regions, then negative-flux photons can
         * be generated.  Noisy images or ring-y kernels will generate a lot of shot noise in
         * the shoot() output.  Not all kernels have photon-shooting implemented.  It may be best to
         * stick to nearest-neighbor and linear interpolation kernels if you wish to avoid these
         * issues.
         *
         * Use the `Delta` Interpolant if you do not want to waste time moving the photons from
         * their pixel centers.  But you will regret any attempt to draw images analytically with
         * that one.
         *
         * Photon shooting with the Sinc kernel is a bad idea and is currently forbidden.
         *
         * @param[in] photons PhotonArray in which to write the photon information
         * @param[in] ud UniformDeviate that will be used to draw photons from distribution.
         */
        void shoot(PhotonArray& photons, UniformDeviate ud) const;

        void getXRange(double& xmin, double& xmax, std::vector<double>& ) const;
        void getYRange(double& ymin, double& ymax, std::vector<double>& ) const;

        double getPositiveFlux() const { checkReadyToShoot(); return _positiveFlux; }
        double getNegativeFlux() const { checkReadyToShoot(); return _negativeFlux; }


        //////////////////////////////
        // Additional subclass methods

        const Interpolant& getXInterp() const;
        const Interpolant& getKInterp() const;
        ConstImageView<double> getPaddedImage() const;
        ConstImageView<double> getNonZeroImage() const;
        ConstImageView<double> getImage() const;

        void calculateMaxK(double max_stepk) const;
        void calculateStepK(double max_maxk) const;

        double calculateFlux() const;

    private:

        int _Nk;
        const ConstImageView<double> _image;
        Bounds<int> _image_bounds;
        Bounds<int> _init_bounds;
        Bounds<int> _nonzero_bounds;

        const Interpolant& _xInterp; ///< Interpolant used in real space.
        const Interpolant& _kInterp; ///< Interpolant used in k space.
        mutable shared_ptr<ImageAlloc<std::complex<double> > > _kimage;
        mutable double _stepk;
        mutable double _maxk;
        mutable double _flux;
        mutable double _xcentroid;
        mutable double _ycentroid;

        double _maxk1; ///< maxk based just on the xInterp urange
        double _uscale; ///< conversion from k to u for xInterpolant

        /// @brief Make kimage if necessary.
        void checkK() const;

        /// @brief Set true if the data structures for photon-shooting are valid
        mutable bool _readyToShoot;

        /// @brief Set up photon-shooting quantities, if not ready
        void checkReadyToShoot() const;

        // Structures used for photon shooting
        /**
         * @brief Simple structure used to index all pixels for photon shooting
         */
        struct Pixel {
            double x;
            double y;
            bool isPositive;
            double flux;

            Pixel(double x_, double y_, double flux_):
                x(x_), y(y_), flux(flux_) { isPositive = flux>=0.; }
            double getFlux() const { return flux; }
        };
        mutable double _positiveFlux;    ///< Sum of all positive pixels' flux
        mutable double _negativeFlux;    ///< Sum of all negative pixels' flux
        mutable ProbabilityTree<Pixel> _pt; ///< Binary tree of pixels, for photon-shooting

    private:

        void doFillXImage(ImageView<double> im,
                          double x0, double dx, int izero,
                          double y0, double dy, int jzero) const
        { fillXImage(im,x0,dx,izero,y0,dy,jzero); }
        void doFillXImage(ImageView<float> im,
                          double x0, double dx, int izero,
                          double y0, double dy, int jzero) const
        { fillXImage(im,x0,dx,izero,y0,dy,jzero); }
        void doFillXImage(ImageView<double> im,
                          double x0, double dx, double dxy,
                          double y0, double dy, double dyx) const
        { fillXImage(im,x0,dx,dxy,y0,dy,dyx); }
        void doFillXImage(ImageView<float> im,
                          double x0, double dx, double dxy,
                          double y0, double dy, double dyx) const
        { fillXImage(im,x0,dx,dxy,y0,dy,dyx); }
        void doFillKImage(ImageView<std::complex<double> > im,
                          double kx0, double dkx, int izero,
                          double ky0, double dky, int jzero) const
        { fillKImage(im,kx0,dkx,izero,ky0,dky,jzero); }
        void doFillKImage(ImageView<std::complex<double> > im,
                          double kx0, double dkx, double dkxy,
                          double ky0, double dky, double dkyx) const
        { fillKImage(im,kx0,dkx,dkxy,ky0,dky,dkyx); }
        void doFillKImage(ImageView<std::complex<float> > im,
                          double kx0, double dkx, int izero,
                          double ky0, double dky, int jzero) const
        { fillKImage(im,kx0,dkx,izero,ky0,dky,jzero); }
        void doFillKImage(ImageView<std::complex<float> > im,
                          double kx0, double dkx, double dkxy,
                          double ky0, double dky, double dkyx) const
        { fillKImage(im,kx0,dkx,dkxy,ky0,dky,dkyx); }

        // Copy constructor and op= are undefined.
        SBInterpolatedImageImpl(const SBInterpolatedImageImpl& rhs);
        void operator=(const SBInterpolatedImageImpl& rhs);
    };


    class SBInterpolatedKImage::SBInterpolatedKImageImpl : public SBProfile::SBProfileImpl
    {
    public:

        SBInterpolatedKImageImpl(
            const BaseImage<std::complex<double> >& kimage, double stepk,
            const Interpolant& kInterp, const GSParams& gsparams);

        ~SBInterpolatedKImageImpl();

        ////////////////////////////////////////////////////////////////////////////
        // Methods of SBProfileImpl that are overriden/implemented in this subclass:

        double xValue(const Position<double>& p) const
        { throw SBError("SBInterpolatedKImage::xValue() is not implemented"); }
        std::complex<double> kValue(const Position<double>& p) const;

        const Interpolant& getKInterp() const;

        double maxK() const { return _maxk; }
        double stepK() const { return _stepk; }
        bool isAxisymmetric() const { return false; }
        // We'll use false here, but really, there's not an easy way to tell.
        // Certainly an Image _could_ have hard edges.
        bool hasHardEdges() const { return false; }
        // This class will be set up so that k domain values are found by interpolation of
        // a table.  We do not currently implement xValue for real-space interpolation.
        bool isAnalyticX() const { return false; }
        bool isAnalyticK() const { return true; }
        void setCentroid() const;
        Position<double> centroid() const;
        double getFlux() const { return _flux; }
        double maxSB() const;
        void shoot(PhotonArray& photons, UniformDeviate ud) const
        { throw SBError("SBInterpolatedKImage::shoot() is not implemented"); }


        //////////////////////////////
        // Additional subclass methods

        ConstImageView<double> getKData() const;

    protected:

        const ConstImageView<std::complex<double> > _kimage;
        const Interpolant& _kInterp; ///< Interpolant used in k space.
        double _stepk; ///< Stored value of stepK
        double _maxk; ///< Stored value of maxK
        double _flux;

        mutable double _xcentroid;
        mutable double _ycentroid;

    private:

        // Copy constructor and op= are undefined.
        SBInterpolatedKImageImpl(const SBInterpolatedKImageImpl& rhs);
        void operator=(const SBInterpolatedKImageImpl& rhs);
    };
}

#endif
