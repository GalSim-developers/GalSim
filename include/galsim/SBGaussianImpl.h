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

#ifndef SBGAUSSIAN_IMPL_H
#define SBGAUSSIAN_IMPL_H

#include "SBProfileImpl.h"
#include "SBGaussian.h"

namespace galsim {

    class SBGaussian::SBGaussianImpl : public SBProfileImpl
    {
    public:
      SBGaussianImpl(double sigma, double flux);

        ~SBGaussianImpl() {}

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

        /**
         * @brief Shoot photons through this SBGaussian.
         *
         * SBGaussian shoots photons by analytic transformation of the unit disk.  Slightly more
         * than 2 uniform deviates are drawn per photon, with some analytic function calls (sqrt,
         * etc.)
         *
         * @param[in] N Total number of photons to produce.
         * @param[in] ud UniformDeviate that will be used to draw photons from distribution.
         * @returns PhotonArray containing all the photons' info.
         */
        boost::shared_ptr<PhotonArray> shoot(int N, UniformDeviate ud) const;

        double getSigma() const { return _sigma; }

    private:
        double _flux; ///< Flux of the Surface Brightness Profile.

        /// Characteristic size, surface brightness scales as `exp[-r^2 / (2. * sigma^2)]`.
        double _sigma;
        double _sigma_sq; ///< Calculated value: sigma*sigma
        double _ksq_min; ///< If ksq < _kq_min, then use faster taylor approximation for kvalue
        double _ksq_max; ///< If ksq > _kq_max, then use kvalue = 0
        double _norm; ///< flux / sigma^2 / 2pi

        // Copy constructor and op= are undefined.
        SBGaussianImpl(const SBGaussianImpl& rhs);
        void operator=(const SBGaussianImpl& rhs);
    };
}

#endif // SBGAUSSIAN_IMPL_H

