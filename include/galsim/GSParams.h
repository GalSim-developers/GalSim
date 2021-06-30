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

#ifndef GalSim_GSParams_H
#define GalSim_GSParams_H

#include <ostream>

#include "Std.h"

namespace galsim {

    struct PUBLIC_API GSParams
    {

        /**
         * @brief A set of numbers that govern how SBProfiles make various speed/accuracy
         * tradeoff decisions.
         *
         * These parameters can be broadly split into two groups: i) parameters that affect the
         * rendering of objects by Discrete Fourier Transform (DFT) and by real space convolution;
         * and ii) parameters that affect rendering by photon shooting.
         *
         * The DFT and real space convolution relevant params are:
         *
         * @param minimum_fft_size    Constant giving minimum FFT size we're willing to do.
         * @param maximum_fft_size    Constant giving maximum FFT size we're willing to do.
         * @param folding_threshold   A threshold parameter used for setting the stepK value for
         *                            FFTs.  The FFT's stepK is set so that at most a fraction
         *                            folding_threshold of the flux of any profile is folded.
         * @param stepk_minimum_hlr   In addition to the above constraint for aliasing, also set
         *                            stepk such that pi/stepk is at least stepk_minimum_hlr
         *                            times the profile's half-light radius (for profiles that
         *                            have a well-defined half-light radius).
         * @param maxk_threshold      A threshold parameter used for setting the maxK value for
         *                            FFTs.  The FFT's maxK is set so that the k-values that are
         *                            excluded off the edge of the image are less than
         *                            maxk_threshold.
         * @param kvalue_accuracy     Accuracy of values in k-space.
         *                            If a k-value is less than kvalue_accuracy, then it may be set
         *                            to zero.  Similarly, if an alternate calculation has errors
         *                            less than kvalue_accuracy, then it may be used instead of an
         *                            exact calculation.
         *                            Note: This does not necessarily imply that all kvalues are
         *                            this accurate.  There may be cases where other choices we
         *                            have made lead to errors greater than this.  But whenever we
         *                            do an explicit calculation about this, this is the value we
         *                            use.
         *                            This should typically be set to a lower, more stringent value
         *                            than maxk_threshold.
         * @param xvalue_accuracy     Accuracy of values in real space.
         *                            If a value in real space is less than xvalue_accuracy, then
         *                            it may be set to zero.  Similarly, if an alternate
         *                            calculation has errors less than xvalue_accuracy, then it may
         *                            be used instead of an exact calculation.
         * @param table_spacing       Several profiles use lookup tables for either the Hankel
         *                            transform (Sersic, truncated Moffat) or the real space
         *                            radial function (Kolmogorov).  We try to estimate a good
         *                            spacing between values in the lookup tables based on
         *                            either xvalue_accuracy or kvalue_accuracy as appropriate.
         *                            However, you may change the spacing with table_spacing.
         *                            Using table_spacing < 1 will use a spacing value that much
         *                            smaller than the default, which should produce more accurate
         *                            interpolations.
         * @param realspace_relerr    The target relative accuracy for real-space convolution.
         * @param realspace_abserr    The target absolute accuracy for real-space convolution.
         * @param integration_relerr  Target relative accuracy for integrals (other than real-space
         *                            convolution).
         * @param integration_abserr  Target absolute accuracy for integrals (other than real-space
         *                            convolution).
         *
         * The Photon Shooting relevant params are:
         *
         * @param shoot_accuracy              Accuracy of total flux for photon shooting.
         *                                    The photon shooting algorithm sometimes needs to
         *                                    sample the radial profile out to some value.  We
         *                                    choose the outer radius such that the integral
         *                                    encloses at least (1-shoot_accuracy) of the flux.
         */
        GSParams(int _minimum_fft_size,
                 int _maximum_fft_size,
                 double _folding_threshold,
                 double _stepk_minimum_hlr,
                 double _maxk_threshold,
                 double _kvalue_accuracy,
                 double _xvalue_accuracy,
                 double _table_spacing,
                 double _realspace_relerr,
                 double _realspace_abserr,
                 double _integration_relerr,
                 double _integration_abserr,
                 double _shoot_accuracy);

        /**
         * A reasonable set of default values
         */
        GSParams() :
            minimum_fft_size(128),
            maximum_fft_size(8192),
            folding_threshold(5.e-3),
            stepk_minimum_hlr(5.),
            maxk_threshold(1.e-3),

            kvalue_accuracy(1.e-5),
            xvalue_accuracy(1.e-5),
            table_spacing(1.),

            realspace_relerr(1.e-4),
            realspace_abserr(1.e-6),
            integration_relerr(1.e-6),
            integration_abserr(1.e-8),

            shoot_accuracy(1.e-5)
            {}

        bool operator==(const GSParams& rhs) const;
        bool operator<(const GSParams& rhs) const;

        // These are all public.  So you access them just as member values.
        int minimum_fft_size;
        int maximum_fft_size;

        double folding_threshold;
        double stepk_minimum_hlr;
        double maxk_threshold;

        double kvalue_accuracy;
        double xvalue_accuracy;
        double table_spacing;

        double realspace_relerr;
        double realspace_abserr;
        double integration_relerr;
        double integration_abserr;

        double shoot_accuracy;

    };

    PUBLIC_API std::ostream& operator<<(std::ostream& os, const GSParams& gsp);

    struct GSParamsPtr
    {
        /**
         * @brief Basically equivalent to shared_ptr<GSParams>, but adds op<, so
         * we can use it in stl containers.
         */
        GSParamsPtr(GSParams* p) : _p(p) {}
        GSParamsPtr(shared_ptr<GSParams> p) : _p(p) {}
        GSParamsPtr(const GSParams& gsp) : _p(new GSParams(gsp)) {}
        GSParamsPtr() {}
        GSParamsPtr(const GSParamsPtr& rhs) : _p(rhs._p) {}
        GSParamsPtr& operator=(const GSParamsPtr& rhs) { _p = rhs._p; return *this; }

        GSParams& operator*() { assert(_p); return *_p; }
        const GSParams& operator*() const { assert(_p); return *_p; }

        GSParams* operator->() { assert(_p); return _p.get(); }
        const GSParams* operator->() const { assert(_p); return _p.get(); }

        shared_ptr<GSParams> getSharedPtr() { return _p; }
        const shared_ptr<GSParams> getSharedPtr() const { return _p; }

        const GSParams* get() const { return _p.get(); }
        GSParams* get() { return _p.get(); }
        operator bool() const { return _p.get(); }
        const shared_ptr<GSParams> getP() const { return _p; }
        shared_ptr<GSParams> getP() { return _p; }

        GSParamsPtr duplicate() const { return GSParamsPtr(new GSParams(*_p)); }

        bool operator==(const GSParamsPtr& rhs) const { return *_p == *rhs; }
        bool operator<(const GSParamsPtr& rhs) const { return *_p < *rhs; }

    private :
        shared_ptr<GSParams> _p;
    };

}

#endif

