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

#ifndef GalSim_SBKolmogorovImpl_H
#define GalSim_SBKolmogorovImpl_H

#include "SBProfileImpl.h"
#include "SBKolmogorov.h"
#include "LRUCache.h"
#include "OneDimensionalDeviate.h"
#include "Table.h"

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
        KolmogorovInfo(const GSParamsPtr& gsparams);

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
         * @param[in] photons PhotonArray in which to write the photon information
         * @param[in] ud UniformDeviate that will be used to draw photons from distribution.
         */
        void shoot(PhotonArray& photons, UniformDeviate ud) const;

    private:
        KolmogorovInfo(const KolmogorovInfo& rhs); ///< Hides the copy constructor.
        void operator=(const KolmogorovInfo& rhs); ///<Hide assignment operator.

        double _stepk; ///< Sampling in k space necessary to avoid folding
        double _maxk; ///< Maximum k value to use

        TableBuilder _radial;  ///< Lookup table for Fourier transform of MTF.

        ///< Class that can sample radial distribution
        shared_ptr<OneDimensionalDeviate> _sampler;
    };

    class SBKolmogorov::SBKolmogorovImpl : public SBProfileImpl
    {
    public:
        SBKolmogorovImpl(double lam_over_r0, double flux, const GSParams& gsparams);

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
        double maxSB() const;

        /**
         * @brief Kolmogorov photon-shooting is done numerically with `OneDimensionalDeviate` class.
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

        double _lam_over_r0; ///< lambda / r0
        double _k0; ///< Derived value so MTF is exp(-(k/k0)^5/3)
        double _k0sq;
        double _inv_k0;
        double _inv_k0sq;

        double _flux; ///< Flux.
        double _xnorm; ///< Calculated value for normalizing xValues returned from Info class.

        const shared_ptr<KolmogorovInfo> _info;

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
        SBKolmogorovImpl(const SBKolmogorovImpl& rhs);
        void operator=(const SBKolmogorovImpl& rhs);

        static LRUCache<GSParamsPtr, KolmogorovInfo> cache;
    };

}

#endif
