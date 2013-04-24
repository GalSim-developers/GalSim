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

#ifndef SBINTERPOLATED_IMAGE_IMPL_H
#define SBINTERPOLATED_IMAGE_IMPL_H

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
            double dx, double pad_factor,
            boost::shared_ptr<Image<T> > pad_image, boost::shared_ptr<GSParams> gsparams);

        SBInterpolatedImageImpl(
            const MultipleImageHelper& multi, const std::vector<double>& weights,
            boost::shared_ptr<Interpolant2d> xInterp, boost::shared_ptr<Interpolant2d> kInterp,
            boost::shared_ptr<GSParams> gsparams);

        ~SBInterpolatedImageImpl();

        double xValue(const Position<double>& p) const;
        std::complex<double> kValue(const Position<double>& p) const;

        double maxK() const { return _maxk; }
        double stepK() const { return _stepk; }

        void calculateMaxK() const;
        void calculateStepK() const;

        void getXRange(double& xmin, double& xmax, std::vector<double>& ) const 
        { xmin = -_max_size; xmax = _max_size; }

        void getYRange(double& ymin, double& ymax, std::vector<double>& ) const 
        { ymin = -_max_size; ymax = _max_size; }

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

        double _max_size; ///< Calculated value: Ninitial+2*xInterp->xrange())*dx
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

            Pixel(double x_=0., double y_=0., double flux_=0.): 
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

#endif // SBINTERPOLATED_IMAGE_IMPL_H
