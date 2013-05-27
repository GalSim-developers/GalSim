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

#ifndef GSPARAMS_H
#define GSPARAMS_H

#include <boost/shared_ptr.hpp>

namespace galsim {

    struct GSParams {

        /**
         * @brief A set of numbers that govern how SBProfiles make various speed/accuracy
         * tradeoff decisions.
         *
         * These parameters can be broadly split into two groups: i) parameters that affect the
         * rendering of objects by Discrete Fourier Transform (DFT) and by real space convolution; 
         * and ii) parameters that affect rendering by Photon Shooting, invoked using the SBProfile
         * .draw() and .drawShoot() member functions, respectively.
         *
         * The DFT and real space convolution relevant params are:
         *
         * @param minimum_fft_size    Constant giving minimum FFT size we're willing to do.
         * @param maximum_fft_size    Constant giving maximum FFT size we're willing to do.
         * @param alias_threshold     A threshold parameter used for setting the stepK value for 
         *                            FFTs.  The FFT's stepK is set so that at most a fraction 
         *                            alias_threshold of the flux of any profile is aliased.
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
         * @param shoot_relerr                The target relative error allowed on any flux integral
         *                                    for photon shooting.
         * @param shoot_abserr                The target absolute error allowed on any flux integral
         *                                    for photon shooting.
         * @param allowed_flux_variation      Max range of allowed (abs value of) photon fluxes
         *                                    within an Interval before rejection sampling is
         *                                    invoked.
         * @param range_division_for_extrema  Range will be split into this many parts to bracket
         *                                    extrema.
         * @param small_fraction_of_flux      Intervals with less than this fraction of probability
         *                                    are ok to use dominant-sampling method.
         */
        GSParams(int _minimum_fft_size,
                 int _maximum_fft_size,
                 double _alias_threshold,
                 double _maxk_threshold,
                 double _kvalue_accuracy,
                 double _xvalue_accuracy,
                 double _realspace_relerr,
                 double _realspace_abserr,
                 double _integration_relerr,
                 double _integration_abserr,
                 double _shoot_accuracy,
                 double _shoot_relerr,
                 double _shoot_abserr,
                 double _allowed_flux_variation,
                 int _range_division_for_extrema,
                 double _small_fraction_of_flux) :
            minimum_fft_size(_minimum_fft_size),
            maximum_fft_size(_maximum_fft_size),
            alias_threshold(_alias_threshold),
            maxk_threshold(_maxk_threshold),
            kvalue_accuracy(_kvalue_accuracy),
            xvalue_accuracy(_xvalue_accuracy),
            realspace_relerr(_realspace_relerr),
            realspace_abserr(_realspace_abserr),
            integration_relerr(_integration_relerr),
            integration_abserr(_integration_abserr),
            shoot_accuracy(_shoot_accuracy),
            shoot_relerr(_shoot_relerr),
            shoot_abserr(_shoot_abserr),
            allowed_flux_variation(_allowed_flux_variation),
            range_division_for_extrema(_range_division_for_extrema),
            small_fraction_of_flux(_small_fraction_of_flux)
        {}

        /**
         * A reasonable set of default values
         */
        GSParams() :
            minimum_fft_size(128),
            maximum_fft_size(4096),
            alias_threshold(5.e-3),
            maxk_threshold(1.e-3),

            kvalue_accuracy(1.e-5),
            xvalue_accuracy(1.e-5),

            realspace_relerr(1.e-3),
            realspace_abserr(1.e-6),
            integration_relerr(1.e-5),
            integration_abserr(1.e-7),

            shoot_accuracy(1.e-5),
            shoot_relerr(1.e-6),
            shoot_abserr(1.e-8),
            allowed_flux_variation(0.81),
            range_division_for_extrema(32),
            small_fraction_of_flux(1.e-4)
            {}

        // These are all public.  So you access them just as member values.
        int minimum_fft_size;
        int maximum_fft_size;

        double alias_threshold;
        double maxk_threshold;

        double kvalue_accuracy;
        double xvalue_accuracy;

        double realspace_relerr;
        double realspace_abserr;
        double integration_relerr;
        double integration_abserr;

        double shoot_accuracy;
        double shoot_relerr;
        double shoot_abserr;
        double allowed_flux_variation;
        int range_division_for_extrema;
        double small_fraction_of_flux;

    };

    struct GSParamsPtr 
    {
        /**
         * @brief Basically equivalent to boost::shared_ptr<GSParams>, but adds op<, so 
         * we can use it in stl containers.
         */
        GSParamsPtr(GSParams* p) : _p(p) {}
        GSParamsPtr(const GSParamsPtr& rhs) : _p(rhs._p) {}

        GSParams& operator*() { return *_p; }
        const GSParams& operator*() const { return *_p; }

        GSParams* operator->() { return _p.get(); }
        const GSParams* operator->() const { return _p.get(); }

        boost::shared_ptr<GSParams> getSharedPtr() { return _p; }
        const boost::shared_ptr<GSParams> getSharedPtr() const { return _p; }

    private : 
        boost::shared_ptr<GSParams> _p;
    };

}

#endif // GSPARAMS_H

