// -*- c++ -*-
#ifndef SBSERSIC_IMPL_H
#define SBSERSIC_IMPL_H

#include "SBProfileImpl.h"
#include "SBSersic.h"

namespace galsim {

    /** 
     * @brief Subclass of `SBSersic` which provides the un-normalized radial function.
     *
     * Serves as interface to `OneDimensionalDeviate` used for sampling from this 
     * distribution.
     */
    class SBSersic::SersicRadialFunction: public FluxDensity 
    {
    public:
        /**
         * @brief Constructor
         *
         * @param[in] n  Sersic index
         * @param[in] b  Factor which makes radius argument enclose half the flux.
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
        double _b;  /// radial normalization constant
    };

    /// @brief A private class that caches the needed parameters for each Sersic index `n`.
    class SBSersic::SersicInfo 
    {
    public:
        /** 
         * @brief Constructor
         * @param[in] n Sersic index
         */
        SersicInfo(double n); 

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

        /**
         * @brief Shoot photons through unit-size, unnormalized profile
         * Sersic profiles are sampled with a numerical method, using class
         * `OneDimensionalDeviate`.
         *
         * @param[in] N Total number of photons to produce.
         * @param[in] ud UniformDeviate that will be used to draw photons from distribution.
         * @returns PhotonArray containing all the photons' info.
         */
        boost::shared_ptr<PhotonArray> shoot(int N, UniformDeviate ud) const;

    private:
        SersicInfo(const SersicInfo& rhs); ///< Hides the copy constructor.
        void operator=(const SersicInfo& rhs); ///<Hide assignment operator.

        double _n; ///< Sersic index.

        /** 
         * @brief Scaling in Sersic profile `exp(-b*pow(xsq,inv2n))`,
         * calculated from Sersic index `n` and half-light radius `re`.
         */
        double _b; 

        double _inv2n;   ///< `1 / (2 * n)`
        double _maxK;    ///< Value of k beyond which aliasing can be neglected.
        double _stepK;   ///< Sampling in k space necessary to avoid folding 

        double _norm; ///< Amplitude scaling in Sersic profile `exp(-b*pow(xsq,inv2n))`.
        double _kderiv2; ///< Quadratic dependence near k=0.
        double _kderiv4; ///< Quartic dependence near k=0.
        Table<double,double> _ft;  ///< Lookup table for Fourier transform of Sersic.
        double _ksq_min; ///< Minimum ksq to use lookup table.
        double _ksq_max; ///< Maximum ksq to use lookup table.

        /// Function class used for photon shooting
        boost::shared_ptr<SersicRadialFunction> _radial;  

        /// Class that does numerical photon shooting
        boost::shared_ptr<OneDimensionalDeviate> _sampler;   

        double findMaxR(double missing_flux_fraction, double gamma2n);
    };

    /** 
     * @brief A map to hold one copy of the SersicInfo for each `n` ever used during the 
     * program run.  Make one static copy of this map.  
     * *Be careful of this when multithreading:*
     * Should build one `SBSersic` with each `n` value before dispatching multiple threads.
     */
    class SBSersic::InfoBarn : public std::map<double, boost::shared_ptr<SersicInfo> > 
    {
    public:

        /**
         * @brief Get the SersicInfo table for a specified `n`.
         *
         * @param[in] n Sersic index for which the information table is required.
         */
        const SersicInfo* get(double n) 
        {
            /** 
             * @brief The currently hardwired max number of Sersic `n` info tables that can be 
             * stored.  Should be plenty.
             */
            const int MAX_SERSIC_TABLES = 100; 

            MapIter it = _map.find(n);
            if (it == _map.end()) {
                boost::shared_ptr<SersicInfo> info(new SersicInfo(n));
                _map[n] = info;
                if (int(_map.size()) > MAX_SERSIC_TABLES)
                    throw SBError("Storing Sersic info for too many n values");
                return info.get();
            } else {
                return it->second.get();
            }
        }

    private:
        typedef std::map<double, boost::shared_ptr<SersicInfo> >::iterator MapIter;
        std::map<double, boost::shared_ptr<SersicInfo> > _map;
    };

    class SBSersic::SBSersicImpl : public SBProfileImpl
    {
    public:
        SBSersicImpl(double n, double re, double flux);

        ~SBSersicImpl() {}

        double xValue(const Position<double>& p) const;
        std::complex<double> kValue(const Position<double>& k) const;

        double maxK() const;
        double stepK() const;

        void getXRange(double& xmin, double& xmax, std::vector<double>& splits) const 
        { xmin = -integ::MOCK_INF; xmax = integ::MOCK_INF; splits.push_back(0.); }

        void getYRange(double& ymin, double& ymax, std::vector<double>& splits) const 
        { ymin = -integ::MOCK_INF; ymax = integ::MOCK_INF; splits.push_back(0.); }

        void getYRangeX(double x, double& ymin, double& ymax, std::vector<double>& splits) const 
        {
            ymin = -integ::MOCK_INF; ymax = integ::MOCK_INF; 
            if (std::abs(x/_re) < 1.e-2) splits.push_back(0.); 
        }

        bool isAxisymmetric() const { return true; }
        bool hasHardEdges() const { return false; }
        bool isAnalyticX() const { return true; }
        bool isAnalyticK() const { return true; }  // 1d lookup table

        Position<double> centroid() const 
        { return Position<double>(0., 0.); }

        double getFlux() const { return _flux; }

        /// @brief Sersic photon shooting done by rescaling photons from appropriate `SersicInfo`
        boost::shared_ptr<PhotonArray> shoot(int N, UniformDeviate ud) const;

        double getN() const { return _n; }
        double getHalfLightRadius() const { return _re; }

    private:
        double _n; ///< Sersic index.
        double _flux; ///< Flux.
        double _re;   ///< Half-light radius.
        double _re_sq; ///< Calculated value: _re*_re
        double _norm; ///< Calculated value: _flux/_re_sq
        double _ksq_max; ///< The ksq_max value from info rescaled with this re value.

        const SersicInfo* _info; ///< Points to info structure for this n.

        // Copy constructor and op= are undefined.
        SBSersicImpl(const SBSersicImpl& rhs);
        void operator=(const SBSersicImpl& rhs);
    };
}

#endif // SBSERSIC_IMPL_H

