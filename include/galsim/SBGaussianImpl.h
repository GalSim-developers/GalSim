// -*- c++ -*-
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

