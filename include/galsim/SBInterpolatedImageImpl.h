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

#ifndef GalSim_SBInterpolatedImageImpl_H
#define GalSim_SBInterpolatedImageImpl_H

#include "SBProfileImpl.h"
#include "SBInterpolatedImage.h"
#include "ProbabilityTree.h"

namespace galsim {

    class SBInterpolatedImage::SBInterpolatedImageImpl : public SBProfile::SBProfileImpl 
    {
    public:
        template <typename T> 
        SBInterpolatedImageImpl(
            const BaseImage<T>& image, 
            boost::shared_ptr<Interpolant2d> xInterp,
            boost::shared_ptr<Interpolant2d> kInterp,
            double pad_factor, const GSParamsPtr& gsparams);

        SBInterpolatedImageImpl(
            const MultipleImageHelper& multi,
            const std::vector<double>& weights,
            boost::shared_ptr<Interpolant2d> xInterp,
            boost::shared_ptr<Interpolant2d> kInterp,
            const GSParamsPtr& gsparams);

        ~SBInterpolatedImageImpl();

        double xValue(const Position<double>& p) const;
        std::complex<double> kValue(const Position<double>& p) const;

        double maxK() const { return _maxk; }
        double stepK() const { return _stepk; }

        void calculateMaxK(double max_stepk) const;
        void calculateStepK(double max_maxk) const;
        void forceStepK(double stepk) const;
        void forceMaxK(double maxk) const;

        void getXRange(double& xmin, double& xmax, std::vector<double>& ) const;
        void getYRange(double& ymin, double& ymax, std::vector<double>& ) const;

        bool isAxisymmetric() const { return false; }

        // We'll use false here, but really, there's not an easy way to tell.
        // Certainly an Image _could_ have hard edges.
        bool hasHardEdges() const { return false; }

        // This class will be set up so that both x and k domain values
        // are found by interpolation of a table:
        bool isAnalyticX() const { return true; }
        bool isAnalyticK() const { return true; }

        Position<double> centroid() const;

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
         * @param[in] N Total umber of photons to produce.
         * @param[in] u UniformDeviate that will be used to draw photons from distribution.
         * @returns PhotonArray containing all the photons' info.
         */
        boost::shared_ptr<PhotonArray> shoot(int N, UniformDeviate u) const;

        double getFlux() const { return _flux; }
        double calculateFlux() const;

        double getPositiveFlux() const { checkReadyToShoot(); return _positiveFlux; }
        double getNegativeFlux() const { checkReadyToShoot(); return _negativeFlux; }

        // Overrides for better efficiency
        void fillXValue(tmv::MatrixView<double> val,
                        double x0, double dx, int ix_zero,
                        double y0, double dy, int iy_zero) const;
        void fillXValue(tmv::MatrixView<double> val,
                        double x0, double dx, double dxy,
                        double y0, double dy, double dyx) const;
        void fillKValue(tmv::MatrixView<std::complex<double> > val,
                        double x0, double dx, int ix_zero,
                        double y0, double dy, int iy_zero) const;
        void fillKValue(tmv::MatrixView<std::complex<double> > val,
                        double x0, double dx, double dxy,
                        double y0, double dy, double dyx) const;

    protected:  // Made protected so that these can be used in the derived CorrelationFunction class

        MultipleImageHelper _multi;
        std::vector<double> _wts;

        boost::shared_ptr<Interpolant2d> _xInterp; ///< Interpolant used in real space.
        boost::shared_ptr<Interpolant2d> _kInterp; ///< Interpolant used in k space.

        boost::shared_ptr<XTable> _xtab; ///< Final padded real-space image.
        mutable boost::shared_ptr<KTable> _ktab; ///< Final k-space image.

        /// @brief Make ktab if necessary.
        void checkK() const;

        mutable double _stepk; ///< Stored value of stepK
        mutable double _maxk; ///< Stored value of maxK
        double _maxk1; ///< maxk based just on the xInterp urange
        double _uscale; ///< conversion from k to u for xInterpolant
        double _flux;
        int _maxNin;

        void initialize(); ///< Put code common to both constructors here.

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

        // Copy constructor and op= are undefined.
        SBInterpolatedImageImpl(const SBInterpolatedImageImpl& rhs);
        void operator=(const SBInterpolatedImageImpl& rhs);
    };

}

#endif
