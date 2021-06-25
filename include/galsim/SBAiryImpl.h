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

#ifndef GalSim_SBAiryImpl_H
#define GalSim_SBAiryImpl_H

#include "SBProfileImpl.h"
#include "SBAiry.h"
#include "LRUCache.h"
#include "OneDimensionalDeviate.h"

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
         * @param[in] photons PhotonArray in which to write the photon information
         * @param[in] ud UniformDeviate that will be used to draw photons from distribution.
         */
        void shoot(PhotonArray& photons, UniformDeviate ud) const;

    protected:
        double _stepk; ///< Sampling in k space necessary to avoid folding

        virtual void checkSampler() const = 0;

        ///< Class that can sample radial distribution
        mutable shared_ptr<OneDimensionalDeviate> _sampler;

    private:
        AiryInfo(const AiryInfo& rhs); ///< Hides the copy constructor.
        void operator=(const AiryInfo& rhs); ///<Hide assignment operator.

    };

    // The definition for obs != 0
    class AiryInfoObs : public AiryInfo
    {
    public:
        AiryInfoObs(double obscuration, const GSParamsPtr& _gsparams);
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
            RadialFunction(double obscuration, double obssq, const GSParamsPtr& gsparams) :
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
            GSParamsPtr _gsparams;
        };

        double _obscuration; ///< Radius ratio of central obscuration.
        double _obssq; ///< _obscuration*_obscuration

        RadialFunction _radial;  ///< Class that embodies the radial Airy function.
        GSParamsPtr _gsparams;

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

    // The definition for obs == 0
    class AiryInfoNoObs : public AiryInfo
    {
    public:
        AiryInfoNoObs(const GSParamsPtr& gsparams);
        ~AiryInfoNoObs() {}

        double xValue(double r) const;
        double kValue(double ksq_over_pisq) const;

    private:
        class RadialFunction : public FluxDensity
        {
        public:
            RadialFunction(const GSParamsPtr& gsparams) : _gsparams(gsparams) {}

            double operator()(double radius) const;

        private:
            GSParamsPtr _gsparams;
        };


        RadialFunction _radial;  ///< Class that embodies the radial Airy function.
        GSParamsPtr _gsparams;

        void checkSampler() const; ///< Check if `OneDimensionalDeviate` is configured.
    };

    class SBAiry::SBAiryImpl : public SBProfileImpl
    {
    public:
        SBAiryImpl(double lam_over_D, double obs, double flux, const GSParams& gsparams);

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
        double maxSB() const { return _xnorm * _info->xValue(0.); }

        /**
         * @brief Airy photon-shooting is done numerically with `OneDimensionalDeviate` class.
         *
         * @param[in] photons PhotonArray in which to write the photon information
         * @param[in] ud UniformDeviate that will be used to draw photons from distribution.
         */
        void shoot(PhotonArray& photons, UniformDeviate ud) const;

        // Overrides for better efficiency
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

        void doFillXImage(ImageView<double> im,
                          double x0, double dx, int izero,
                          double y0, double dy, int jzero) const
        { fillXImage(im,x0,dx,izero,y0,dy,jzero); }
        void doFillXImage(ImageView<double> im,
                          double x0, double dx, double dxy,
                          double y0, double dy, double dyx) const
        { fillXImage(im,x0,dx,dxy,y0,dy,dyx); }
        void doFillXImage(ImageView<float> im,
                          double x0, double dx, int izero,
                          double y0, double dy, int jzero) const
        { fillXImage(im,x0,dx,izero,y0,dy,jzero); }
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
        SBAiryImpl(const SBAiryImpl& rhs);
        void operator=(const SBAiryImpl& rhs);

        /// Info object that stores things that are common to all Airy functions with this
        /// obscuration value.
        const shared_ptr<AiryInfo> _info;

        /// One static map of all `AiryInfo` structures for whole program.
        static LRUCache<Tuple<double, GSParamsPtr>, AiryInfo> cache;
    };
}

#endif
