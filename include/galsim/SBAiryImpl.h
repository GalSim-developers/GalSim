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

#ifndef SBAIRY_IMPL_H
#define SBAIRY_IMPL_H

#include "SBProfileImpl.h"
#include "SBAiry.h"
#include "LRUCache.h"

namespace galsim {

    /**
     * @brief A private class that caches the photon shooting objects for a given
     *         obscuration value, so they don't have to be set up again each time.
     * 
     * This is helpful if people use only 1 or a small number of obscuration values.
     */
    class AiryInfo
    {
    public:
        /** 
         * @brief Constructor
         */
        AiryInfo() {}

        /// @brief Destructor: deletes photon-shooting classes if necessary
        virtual ~AiryInfo() {}

        /** 
         * @brief Returns the real space value of the Airy function,
         * normalized to unit flux (see private attributes).
         * @param[in] r should be given in units of lam_over_D  (i.e. r_true*D)
         *
         * This is used to calculate the real xValue, but it comes back unnormalized.
         * The value needs to be multiplied by flux * D^2.
         */
        virtual double xValue(double r) const = 0;

        /**
         * @brief Returns the k-space value of the Airy function.
         * @param[in] ksq_over_pisq should be given in units of lam_over_D  
         * (i.e. k_true^2 / (pi^2 * D^2))
         *
         * This is used to calculate the real kValue, but it comes back unnormalized.
         * The value at k=0 is Pi * (1-obs^2), so the value needs to be multiplied
         * by flux / (Pi * (1-obs^2)).
         */
        virtual double kValue(double ksq_over_pisq) const = 0;

        double stepK() const { return _stepk; }

        /**
         * @brief Shoot photons through unit-size, unnormalized profile
         * Airy profiles are sampled with a numerical method, using class
         * `OneDimensionalDeviate`.
         *
         * @param[in] N Total number of photons to produce.
         * @param[in] ud UniformDeviate that will be used to draw photons from distribution.
         * @returns PhotonArray containing all the photons' info.
         */
        boost::shared_ptr<PhotonArray> shoot(int N, UniformDeviate ud) const;

    protected:
        double _stepk; ///< Sampling in k space necessary to avoid folding 

        virtual void checkSampler() const = 0;

        ///< Class that can sample radial distribution
        mutable boost::shared_ptr<OneDimensionalDeviate> _sampler; 

    private:
        AiryInfo(const AiryInfo& rhs); ///< Hides the copy constructor.
        void operator=(const AiryInfo& rhs); ///<Hide assignment operator.

    };

    // The definition for obs != 0
    class AiryInfoObs : public AiryInfo
    {
    public:
        AiryInfoObs(double obscuration, const GSParams* _gsparams); 
        ~AiryInfoObs() {}

        double xValue(double r) const;
        double kValue(double ksq_over_pisq) const;

    private:
        /**
         * @brief Subclass is a scale-free version of the Airy radial function.
         *
         * Serves as interface to numerical photon-shooting class `OneDimensionalDeviate`.
         *
         * Input radius is in units of lambda/D.  Output normalized
         * to integrate to unity over input units.
         */
        class RadialFunction : public FluxDensity 
        {
        public:
            /**
             * @brief Constructor
             * @param[in] obscuration Fractional linear size of central obscuration of pupil.
             * @param[in] obssq       Pre-computed obscuration^2 supplied as input for speed.
             */
            RadialFunction(double obscuration, double obssq, const GSParams* gsparams) : 
                _obscuration(obscuration), _obssq(obssq),
                _norm(M_PI / (1.-_obssq)), _gsparams(gsparams) {}

            /**
             * @brief Return the Airy function
             * @param[in] radius Radius in units of (lambda / D)
             * @returns Airy function, normalized to integrate to unity.
             */
            double operator()(double radius) const;

        private:
            double _obscuration; ///< Central obstruction size
            double _obssq; ///< _obscuration*_obscuration
            double _norm; ///< Calculated value M_PI / (1-obs^2)
            const GSParams* _gsparams;
        };

        double _obscuration; ///< Radius ratio of central obscuration.
        double _obssq; ///< _obscuration*_obscuration

        RadialFunction _radial;  ///< Class that embodies the radial Airy function.
        const GSParams* _gsparams;

        /// Circle chord length at `h < r`.
        double chord(double r, double h, double rsq, double hsq) const; 

        /// @brief Area inside intersection of 2 circles radii `r` & `s`, seperated by `t`.
        double circle_intersection(
            double r, double s, double rsq, double ssq, double tsq) const; 
        double circle_intersection(double r, double rsq, double tsq) const; 

        /// @brief Area of two intersecting identical annuli.
        double annuli_intersect(
            double r1, double r2, double r1sq, double r2sq, double tsq) const; 

        void checkSampler() const; ///< Check if `OneDimensionalDeviate` is configured.
    };

    // The definition for obs -= 0
    class AiryInfoNoObs : public AiryInfo
    {
    public:
        AiryInfoNoObs(const GSParams* gsparams);
        ~AiryInfoNoObs() {}

        double xValue(double r) const;
        double kValue(double ksq_over_pisq) const;

    private:
        class RadialFunction : public FluxDensity 
        {
        public:
            RadialFunction(const GSParams* gsparams) : _gsparams(gsparams) {}

            double operator()(double radius) const;

        private:
            const GSParams* _gsparams;
        };


        RadialFunction _radial;  ///< Class that embodies the radial Airy function.
        const GSParams* _gsparams;

        void checkSampler() const; ///< Check if `OneDimensionalDeviate` is configured.
    };

    class SBAiry::SBAiryImpl : public SBProfileImpl 
    {
    public:
        SBAiryImpl(double lam_over_D, double obs, double flux,
                   boost::shared_ptr<GSParams> gsparams);
        ~SBAiryImpl() {}

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
        double getLamOverD() const { return _lam_over_D; }
        double getObscuration() const { return _obscuration; }

        /**
         * @brief Airy photon-shooting is done numerically with `OneDimensionalDeviate` class.
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
        
        double _lam_over_D;  ///< inverse of _D (see below), harmonise inputs with other GSObjects
        /** 
         * `_D` = (telescope diam) / (lambda * focal length) if arg is focal plane position, 
         *  else `_D` = (telescope diam) / lambda if arg is in radians of field angle.
         */
        double _D;
        double _obscuration; ///< Radius ratio of central obscuration.
        double _flux; ///< Flux.

        double _Dsq; ///< Calculated value: D*D
        double _obssq; ///< Calculated value: _obscuration * _obscuration
        double _inv_D_pi; ///< Calculated value: 1/(D pi)
        double _inv_Dsq_pisq; ///< Calculated value: 1/(D^2 pi^2)
        double _xnorm; ///< Calculated value: flux * D^2
        double _knorm; ///< Calculated value: flux / (pi (1-obs^2))

        // Copy constructor and op= are undefined.
        SBAiryImpl(const SBAiryImpl& rhs);
        void operator=(const SBAiryImpl& rhs);

        /// Info object that stores things that are common to all Airy functions with this 
        /// obscuration value.
        const boost::shared_ptr<AiryInfo> _info; 

        /// One static map of all `AiryInfo` structures for whole program.
        static LRUCache<std::pair<double,const GSParams*>, AiryInfo> cache;
    };
}

#endif // SBAIRY_IMPL_H

