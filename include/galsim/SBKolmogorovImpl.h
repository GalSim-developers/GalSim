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

#ifndef SBKOLMOGOROV_IMPL_H
#define SBKOLMOGOROV_IMPL_H

#include "SBProfileImpl.h"
#include "SBKolmogorov.h"
#include "LRUCache.h"

namespace galsim {

    /**
     * @brief A class that caches the photon shooting objects and real-space
     *        lookup table, so they don't have to be set up again each time.
     */
    class KolmogorovInfo 
    {
    public:
        /** 
         * @brief Constructor
         */
        KolmogorovInfo(const GSParams* gsparams);

        /// @brief Destructor: deletes photon-shooting classes if necessary
        ~KolmogorovInfo() {}

        /** 
         * @brief Returns the real space value of the Kolmogorov function,
         * normalized to unit flux (see private attributes).
         * @param[in] r should be given in units of lam_over_r0  (i.e. r_true*r0)
         *
         * This is used to calculate the real xValue, but it comes back unnormalized.
         * The value needs to be multiplied by flux * r0^2.
         */
        double xValue(double r) const;

        /**
         * @brief Returns the k-space value of the Kolmogorov function.
         * @param[in] ksq_over_pisq should be given in units of lam_over_r0  
         * (i.e. k_true^2 / (pi^2 * r0^2))
         *
         * This is used to calculate the real kValue, but it comes back unnormalized.
         * The value at k=0 is Pi, so the value needs to be multiplied
         * by flux / Pi.
         */
        double kValue(double ksq_over_pisq) const;

        double stepK() const { return _stepk; }
        double maxK() const { return _maxk; }

        /**
         * @brief Shoot photons through unit-size, unnormalized profile
         * Kolmogorov profiles are sampled with a numerical method, using class
         * `OneDimensionalDeviate`.
         *
         * @param[in] N Total number of photons to produce.
         * @param[in] ud UniformDeviate that will be used to draw photons from distribution.
         * @returns PhotonArray containing all the photons' info.
         */
        boost::shared_ptr<PhotonArray> shoot(int N, UniformDeviate ud) const;

    private:
        KolmogorovInfo(const KolmogorovInfo& rhs); ///< Hides the copy constructor.
        void operator=(const KolmogorovInfo& rhs); ///<Hide assignment operator.

        double _stepk; ///< Sampling in k space necessary to avoid folding 
        double _maxk; ///< Maximum k value to use

        TableDD _radial;  ///< Lookup table for Fourier transform of MTF.

        ///< Class that can sample radial distribution
        boost::shared_ptr<OneDimensionalDeviate> _sampler; 
    };

    class SBKolmogorov::SBKolmogorovImpl : public SBProfileImpl 
    {
    public:
        SBKolmogorovImpl(double lam_over_r0, double flux,
                         boost::shared_ptr<GSParams> gsparams);
        ~SBKolmogorovImpl() {}

        double xValue(const Position<double>& p) const;
        std::complex<double> kValue(const Position<double>& k) const;

        bool isAxisymmetric() const { return true; } 
        bool hasHardEdges() const { return false; }
        bool isAnalyticX() const { return true; }
        bool isAnalyticK() const { return true; }

        double maxK() const;
        double stepK() const;

        Position<double> centroid() const 
        { return Position<double>(0., 0.); }

        double getFlux() const { return _flux; }
        double getLamOverR0() const { return _lam_over_r0; }

        /**
         * @brief Kolmogorov photon-shooting is done numerically with `OneDimensionalDeviate` class.
         *
         * @param[in] N Total number of photons to produce.
         * @param[in] ud UniformDeviate that will be used to draw photons from distribution.
         * @returns PhotonArray containing all the photons' info.
         */
        boost::shared_ptr<PhotonArray> shoot(int N, UniformDeviate ud) const;

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

    private:
        
        double _lam_over_r0; ///< lambda / r0
        double _k0; ///< Derived value so MTF is exp(-(k/k0)^5/3)
        double _k0sq;
        double _inv_k0;
        double _inv_k0sq;

        double _flux; ///< Flux.
        double _xnorm; ///< Calculated value for normalizing xValues returned from Info class.

        const boost::shared_ptr<KolmogorovInfo> _info;

        // Copy constructor and op= are undefined.
        SBKolmogorovImpl(const SBKolmogorovImpl& rhs);
        void operator=(const SBKolmogorovImpl& rhs);

        static LRUCache<const GSParams*, KolmogorovInfo> cache;
    };

}

#endif // SBKOLMOGOROV_IMPL_H

