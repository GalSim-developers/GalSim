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

#ifndef SBSERSIC_IMPL_H
#define SBSERSIC_IMPL_H

#include "SBProfileImpl.h"
#include "SBSersic.h"
#include "LRUCache.h"

namespace galsim {

    /** 
     * @brief Subclass of `SBSersic` which provides the un-normalized radial function.
     *
     * Serves as interface to `OneDimensionalDeviate` used for sampling from this 
     * distribution.
     */
    class SersicRadialFunction: public FluxDensity 
    {
    public:
        /**
         * @brief Constructor
         *
         * @param[in] n  Sersic index
         * @param[in] b  Scale factor which makes radius argument enclose half the flux.
         */
        SersicRadialFunction(double n, double b): _invn(1./n), _b(b) {}
        /**
         * @brief The un-normalized Sersic function
         * @param[in] r radius, in units of half-light radius.
         * @returns Sersic function, normalized to unity at origin
         */
        double operator()(double r) const { return std::exp(-_b*std::pow(r,_invn)); } 
    private:
        double _invn; ///> 1/n
        double _b;  ///> radial normalization constant
    };

    /**
     * @brief A key for mapping Sersic cache, consisting of a triplet of values
     * `(n, maxRre, flux_untruncated)`.
     */
    struct SersicKey {
        /**
         * @param[in] n                 Sersic index
         * @param[in] maxRre            Maximum radius in units of half-light radius
         * @param[in] flux_untruncated  If true, flux is set to the untruncated Sersic with
         *                              index `n`.  Ignored if `maxRre = 0.`.
         */
        SersicKey(double _n, double _maxRre, bool _flux_untruncated) :
            n(_n), maxRre(_maxRre), flux_untruncated(_flux_untruncated) {}

        // less operator required for map::find()
        bool operator<(const SersicKey& rhs) const
        {
            return ( 
                n == rhs.n ? (
                    maxRre == rhs.maxRre ? (
                        flux_untruncated < rhs.flux_untruncated ) :
                    maxRre < rhs.maxRre ) :
                n < rhs.n );
        }

        double n;
        double maxRre;
        bool flux_untruncated;
    };

    /// @brief A private class that caches the needed parameters for each Sersic index `n`.
    class SersicInfo 
    {
    public:
        /// @brief Constructor takes SersicKey, which is really (n,maxRre,flux_untruncated)
        SersicInfo(const SersicKey& key, const GSParams* gsparams);

        /// @brief Destructor: deletes photon-shooting classes if necessary
        ~SersicInfo() {}

        /** 
         * @brief Returns the real space value of the Sersic function,
         * normalized to unit flux (see private attributes).
         * @param[in] xsq The *square* of the radius, in units of half-light radius.
         * Avoids taking sqrt in most user code.
         * @returns Value of Sersic function, normalized to unit flux.
         */
        double xValue(double xsq) const;

        /// @brief Looks up the k value for the SBProfile from a lookup table.
        double kValue(double ksq) const;

        double maxK() const { return _maxK; }
        double stepK() const { return _stepK; }

        double getKsqMax() const { return _ksq_max; }

        /// @brief Returns the maximum relevant R, in units of half-light radius `re`.
        double getMaxRRe() const { return _maxRre; }

        /// @brief Returns the ratio of the actual flux to the specified flux the object, which is
        /// not unity when `_truncated` and `_flux_untruncated` are both true.
        double getTrueFluxFraction() const { return _flux_fraction; }
        /// @brief Returns the ratio of the actual half-light radius `re` to the specified one,
        /// which is not unity when `_truncated` and `_flux_untruncated` are both true.
        double getTrueReFraction() const { return _re_fraction; }

        /**
         * @brief Shoot photons through unit-size, unnormalized profile
         * Sersic profiles are sampled with a numerical method, using class
         * `OneDimensionalDeviate`.
         *
         * @param[in] N  Total number of photons to produce.
         * @param[in] ud UniformDeviate that will be used to draw photons from distribution.
         * @returns PhotonArray containing all the photons' info.
         */
        boost::shared_ptr<PhotonArray> shoot(int N, UniformDeviate ud) const;

    private:

        SersicInfo(const SersicInfo& rhs); ///< Hides the copy constructor.
        void operator=(const SersicInfo& rhs); ///<Hide assignment operator.

        double _n; ///< Sersic index.
        double _maxRre; ///< Truncation radius `trunc` in units of half-light radius `re`.
        double _maxRre_sq;

        /** 
         * @brief Scaling in Sersic profile `exp(-b*pow(xsq,inv2n))`,
         * calculated from Sersic index `n`, half-light radius `re`, and truncation radius `trunc`.
         */
        double _b;

        double _inv2n;   ///< `1 / (2 * n)`
        double _maxK;    ///< Value of k beyond which aliasing can be neglected.
        double _stepK;   ///< Sampling in k space necessary to avoid folding 

        double _norm; ///< Amplitude scaling in Sersic profile `exp(-b*pow(xsq,inv2n))`.
        bool _flux_untruncated; ///< If true, flux is set to the untruncated Sersic with index `_n`.
        double _flux_fraction; ///< Ratio of true (truncated) flux to the specified flux.
        double _re_fraction; ///< Ratio of true `re` to the specified `re`.
        double _kderiv2; ///< Quadratic dependence near k=0.
        double _kderiv4; ///< Quartic dependence near k=0.
        Table<double,double> _ft;  ///< Lookup table for Fourier transform of Sersic.
        double _ksq_min; ///< Minimum ksq to use lookup table.
        double _ksq_max; ///< Maximum ksq to use lookup table.
        bool _truncated; ///< True if this Sersic profile is truncated.

        /// Function class used for photon shooting
        boost::shared_ptr<SersicRadialFunction> _radial;  

        /// Class that does numerical photon shooting
        boost::shared_ptr<OneDimensionalDeviate> _sampler;   

        double findMaxRre(double missing_flux_fraction, double gamma2n);
    };

    class SBSersic::SBSersicImpl : public SBProfileImpl
    {
    public:
        SBSersicImpl(double n, double re, double flux,
                     double trunc, bool flux_untruncated,
                     boost::shared_ptr<GSParams> gsparams);
        ~SBSersicImpl() {}

        double xValue(const Position<double>& p) const;
        std::complex<double> kValue(const Position<double>& k) const;

        double maxK() const;
        double stepK() const;

        void getXRange(double& xmin, double& xmax, std::vector<double>& splits) const 
        {
            splits.push_back(0.);
            if (!_truncated) { xmin = -integ::MOCK_INF; xmax = integ::MOCK_INF; }
            else { xmin = -_maxR; xmax = _maxR; }
        }

        void getYRange(double& ymin, double& ymax, std::vector<double>& splits) const 
        {
            splits.push_back(0.);
            if (!_truncated) { ymin = -integ::MOCK_INF; ymax = integ::MOCK_INF; }
            else { ymin = -_maxR; ymax = _maxR; }
        }

        void getYRangeX(double x, double& ymin, double& ymax, std::vector<double>& splits) const 
        {
            if (!_truncated) { ymin = -integ::MOCK_INF; ymax = integ::MOCK_INF; }
            else { ymax = sqrt(_maxR_sq - x*x);  ymin=-ymax; }
            if (std::abs(x/_re) < 1.e-2) splits.push_back(0.); 
        }

        bool isAxisymmetric() const { return true; }
        bool hasHardEdges() const { return _truncated; }
        bool isAnalyticX() const { return true; }
        bool isAnalyticK() const { return true; }  // 1d lookup table

        Position<double> centroid() const 
        { return Position<double>(0., 0.); }

        /// @brief Returns the true flux (may be different from the specified flux)
        double getFlux() const { return _actual_flux; }

        /// @brief Sersic photon shooting done by rescaling photons from appropriate `SersicInfo`
        boost::shared_ptr<PhotonArray> shoot(int N, UniformDeviate ud) const;

        double getN() const { return _n; }
        /// @brief Returns the true half-light radius (may be different from the specified value)
        double getHalfLightRadius() const { return _actual_re; }

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
        double _n; ///< Sersic index.
        double _flux; ///< Flux specified at the constructor.
        double _re;   ///< Half-light radius specified at the constructor.
        double _re_sq;
        double _inv_re;
        double _inv_re_sq;
        double _norm; ///< Calculated value: _flux/_re_sq
        double _trunc; ///< Truncation radius in same physical units as `_re` (0 if no truncation).
        bool _flux_untruncated; ///< If true, flux is set to the untruncated Sersic with index `_n`.
        double _actual_flux; ///< True flux of object.
        double _actual_re; ///< True half-light radius of object.
        double _maxRre; ///< Maximum (truncation) `r` in units of `_re`.
        double _maxRre_sq;
        double _maxR; ///< Maximum (truncation) radius `r`.
        double _maxR_sq;
        double _ksq_max; ///< The ksq_max value from info rescaled with this re value.
        bool _truncated; ///< Set true if `_trunc > 0`.

        const boost::shared_ptr<SersicInfo> _info; ///< Points to info structure for this n.

        // Copy constructor and op= are undefined.
        SBSersicImpl(const SBSersicImpl& rhs);
        void operator=(const SBSersicImpl& rhs);

        static LRUCache<std::pair<SersicKey, const GSParams*>, SersicInfo> cache;
    };
}

#endif // SBSERSIC_IMPL_H

