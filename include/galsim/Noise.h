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

#ifndef NOISE_H
#define NOISE_H

/** 
 * @file Noise.h @brief Add noise to image using various noise models
 *
 */

#include <cmath>
#include "Std.h"
#include "Random.h"
#include "Image.h"

namespace galsim {

    /** 
     * @brief Base class for noise models.  
     *
     * The BaseNoise class defines the interface for classes that define noise models
     * for how to add noise to an image.
     */
    class BaseNoise
    {
    public:
        /**
         * @brief Construct a new noise model, using time of day as seed.
         *
         * @param[in] rng   The BaseDeviate to use for the random number generation.
         */
        BaseNoise(BaseDeviate& rng) : _rng(rng) {}

        /**
         * @brief Copy constructor shares the underlying rng.
         */
        BaseNoise(const BaseNoise& rhs) : _rng(rhs._rng) {}

        /**
         * @brief Destructor is virtual
         */
        virtual ~BaseNoise() {}
 
        /**
         * @brief Add noise to an Image.
         *
         * @param[in,out] data The Image to be noise-ified.
         */
        template <typename T>
        void applyTo(ImageView<T> data) 
        { 
            // This uses the standard workaround for the fact that you can't have a 
            // virtual template function.  The doApplyTo functions are virtual and 
            // are listed for each allowed value of T (
            doApplyTo(data);
        }

    protected:

        mutable BaseDeviate& _rng;

        // These need to be defined by the derived class.  They typically would in turn
        // immediately call their own templated applyTo function that defines the actual
        // application of the noise.
        virtual void doApplyTo(ImageView<double>& data) = 0;
        virtual void doApplyTo(ImageView<float>& data) = 0;
        virtual void doApplyTo(ImageView<int>& data) = 0;
        virtual void doApplyTo(ImageView<short>& data) = 0;
    };


    /** 
     * @brief Class implementing basic CCD noise model.  
     *
     * The CCDNoise class encapsulates the noise model of a normal CCD image.  The noise has two
     * components: first, Poisson noise corresponding to the number of electrons in each pixel
     * (including an optional extra sky level); second, Gaussian read noise.
     *
     * read_noise=0 will shut off the Gaussian noise.
     *
     * gain<=0 will shut off the Poisson noise, and Gaussian noise will have RMS = reasNoise.
     *
     * Note that if the image to which you are adding noise already has a sky level on it,
     * then you should not provide the sky level here as well.  The sky level here corresponds
     * to a level is taken to be already subtracted from the image, but which was present
     * for the Poisson noise.
     */
    class CCDNoise : public BaseNoise
    {
    public:
 
        /**
         * @brief Construct a new noise model, sharing the random number generator with dev.
         *
         * @param[in] rng        The BaseDeviate to use for the random number generation.
         * @param[in] sky_level  The sky level in electrons per pixel that was originally in
         *                       the input image, but which is taken to have already been 
         *                       subtracted off.
         * @param[in] gain       Electrons per ADU in the input Images, used for Poisson noise.
         * @param[in] read_noise RMS of Gaussian noise, in electrons (if gain>0.) or ADU (gain<=0.)
         */
        CCDNoise(BaseDeviate& rng, double sky_level=0., double gain=1., double read_noise=0.) :
            BaseNoise(rng),
            _sky_level(sky_level), _gain(gain), _read_noise(read_noise)
        {}

        /**
         * @brief Construct a copy that shares the RNG with rhs.
         *
         * Note: the default constructed op= function will do the same thing.
         */
        CCDNoise(const CCDNoise& rhs) : 
            BaseNoise(rhs),
            _sky_level(rhs._sky_level), _gain(rhs._gain), _read_noise(rhs._read_noise)
        {}
 

        /**
         * @brief Report current sky_level
         *
         * @return Sky level in electrons / pixel.
         */
        double getSkyLevel() const { return _sky_level; }

        /**
         * @brief Report current gain value
         *
         * @return Gain value (e-/ADU)
         */
        double getGain() const { return _gain; }

        /**
         * @brief Report current read noise
         *
         * @return Read noise value (e-, if gain>0, else in ADU)
         */
        double getReadNoise() const { return _read_noise; }

        /**
         * @brief Set sky level
         *
         * @param[in] sky_level Sky level in electrons / pixel.
         */
        void setSkyLevel(double sky_level) { _sky_level = sky_level; }

        /**
         * @brief Set gain value
         *
         * @param[in] gain Gain value (e-/ADU)
         */
        void setGain(double gain) { _gain = gain; }

        /**
         * @brief Set read noise
         *
         * @param[in] read_noise Read noise value (e-, if gain>0, else in ADU)
         */
        void setReadNoise(double read_noise) { _read_noise = read_noise; }

        /**
         * @brief Add noise to an Image.
         *
         * Poisson and/or Gaussian noise are added to each pixel of the image according
         * to standard CCD model.
         * @param[in,out] data The Image to be noise-ified.
         */
        template <typename T>
        void applyTo(ImageView<T> data) 
        {
            // Above this many e's, assume Poisson distribution =Gaussian 
            static const double MAX_POISSON=1.e5;
            // Typedef for image row iterable
            typedef typename ImageView<T>::iterator ImIter;

            data += T(_sky_level);

            // Add the Poisson noise first:
            if (_gain > 0.) {
                PoissonDeviate pd(_rng, 1.); // will reset the mean for each pixel below.
                GaussianDeviate gd(_rng, 0., 1.);
                for (int y = data.getYMin(); y <= data.getYMax(); y++) {  // iterate over y
                    ImIter ee = data.rowEnd(y);
                    for (ImIter it = data.rowBegin(y); it != ee; ++it) {
                        double electrons = *it * _gain;
                        if (electrons <= 0.) continue;
                        if (electrons < MAX_POISSON) {
                            pd.setMean(electrons);
                            *it = T(pd() / _gain);
                        } else {
                            // ??? This might be even slower than large-N Poisson...
                            gd.setSigma(sqrt(electrons)/_gain);
                            *it = T(*it + gd());
                        }
                    }
                }
            }

            // Next add the Gaussian noise:
            if (_read_noise > 0.) {
                GaussianDeviate gd(_rng, 0., _read_noise / (_gain > 0. ? _gain : 1.));
                for (int y = data.getYMin(); y <= data.getYMax(); y++) {  // iterate over y
                    ImIter ee = data.rowEnd(y);
                    for (ImIter it = data.rowBegin(y); it != ee; ++it) {
                        *it = T(*it + gd());
                    }
                }
            }

            data -= T(_sky_level);
        }

        /**
         * @brief Add noise to an Image and also report variance of each pixel.
         *
         * Adds noise as in applyTo(Image) signature, but second Image is filled with
         * variance of added noise.  Note: the variance image must be the same size as the 
         * data image.
         *
         * @param[in,out] data The Image to be noise-ified.
         * @param[in,out] var  The Image to fill with variance of applied noise.
         */
        template <class T>
        void applyToVar(ImageView<T> data, ImageView<T> var) 
        {
            // Typedef for image row iterable
            typedef typename ImageView<T>::iterator ImIter;
            assert(data.getBounds() == var.getBounds());
            // Fill with the (constant) Gaussian contribution to variance
            if (_read_noise > 0.) {
                double sigma = _read_noise / (_gain > 0. ? _gain : 1.);
                var.fill(sigma * sigma);
            }
            // Add the Poisson variance:
            if (_gain > 0.) {
                for (int y = data.getYMin(); y <= data.getYMax(); y++) {  // iterate over y
                    ImIter ee = data.rowEnd(y);
                    ImIter it2 = var.rowBegin(y);
                    for (ImIter it = data.rowBegin(y); it != ee; ++it, ++it2) {
                        if (*it > 0.) *it2 += (*it + _sky_level) / _gain;
                    }
                } 
            }
            // then call noise method to instantiate noise
            applyTo(data);
        }

    protected:
        void doApplyTo(ImageView<double>& data) { applyTo(data); }
        void doApplyTo(ImageView<float>& data) { applyTo(data); }
        void doApplyTo(ImageView<int>& data) { applyTo(data); }
        void doApplyTo(ImageView<short>& data) { applyTo(data); }

    private: 
        double _sky_level;   // sky level in e- per pixel
        double _gain;        // flux corresponding to one photon (e-/ADU)
        double _read_noise;  // std. dev. of uniform Gaussian noise (when divided by _gain).
    };


    /** 
     * @brief Add noise according to a given Deviate
     *
     * For each pixel, draw the amount of noise from the provided Deviate object.
     */
    class DeviateNoise : public BaseNoise
    {
    public:
 
        /**
         * @brief Construct a new noise model, sharing the random number generator with dev.
         *
         * @param[in] dev       The Deviate to use for the noise values
         */
        DeviateNoise(BaseDeviate& rng) : BaseNoise(rng) {}

        /**
         * @brief Construct a copy that shares the RNG with rhs.
         *
         * Note: the default constructed op= function will do the same thing.
         */
        DeviateNoise(const DeviateNoise& rhs) : BaseNoise(rhs) {}
 
        /**
         * @brief Add noise to an Image.
         *
         * @param[in,out] data The Image to be noise-ified.
         */
        template <typename T>
        void applyTo(ImageView<T> data) 
        {
            // Typedef for image row iterable
            typedef typename ImageView<T>::iterator ImIter;

            for (int y = data.getYMin(); y <= data.getYMax(); y++) {  // iterate over y
                ImIter ee = data.rowEnd(y);
                for (ImIter it = data.rowBegin(y); it != ee; ++it) { *it += T(_rng()); }
            }
        }

    protected:
        using BaseNoise::_rng;
        void doApplyTo(ImageView<double>& data) { applyTo(data); }
        void doApplyTo(ImageView<float>& data) { applyTo(data); }
        void doApplyTo(ImageView<int>& data) { applyTo(data); }
        void doApplyTo(ImageView<short>& data) { applyTo(data); }
    };

};  // namespace galsim

#endif
