// -*- c++ -*-
#ifndef SBKOLMOGOROV_IMPL_H
#define SBKOLMOGOROV_IMPL_H

#include "SBProfileImpl.h"
#include "SBKolmogorov.h"

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
        KolmogorovInfo();

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
        SBKolmogorovImpl(double lam_over_r0, double flux);

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

    private:
        
        double _lam_over_r0; ///< lambda / r0
        double _k0; ///< Derived value so MTF is exp(-(k/k0)^5/3)
        double _k0sq; ///< Calculated value: k0*k0
        double _inv_k0sq; ///< Calculated value: 1/k0^2

        double _flux; ///< Flux.
        double _xnorm; ///< Calculated value for normalizing xValues returned from Info class.

        /// One static `KolmogorovInfo` object for whole program.
        static KolmogorovInfo _info;

        // Copy constructor and op= are undefined.
        SBKolmogorovImpl(const SBKolmogorovImpl& rhs);
        void operator=(const SBKolmogorovImpl& rhs);
    };

}

#endif // SBKOLMOGOROV_IMPL_H

