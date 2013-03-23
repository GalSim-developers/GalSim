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
        BaseNoise(boost::shared_ptr<BaseDeviate> rng) : _rng(rng) 
        { if (!rng.get()) _rng.reset(new BaseDeviate()); }

        /**
         * @brief Copy constructor shares the underlying rng.
         */
        BaseNoise(const BaseNoise& rhs) : _rng(rhs._rng) {}

        /**
         * @brief Destructor is virtual
         */
        virtual ~BaseNoise() {}

        /**
         * @brief Get the BaseDeviate being used to generate random numbers for the noise model
         */
        boost::shared_ptr<BaseDeviate> getRNG() { return _rng; }

        /**
         * @brief Set the BaseDeviate that will be used to generate random numbers for the noise
         * model
         */
        void setRNG(boost::shared_ptr<BaseDeviate> rng) { _rng = rng; }

        /**
         * @brief Get the variance of the noise model
         */
        virtual double getVariance() const = 0;

        /**
         * @brief Set the variance of the noise model
         */
        virtual void setVariance(double variance) = 0;

        /**
         * @brief Set the variance of the noise model
         */
        virtual void scaleVariance(double variance_ratio) = 0;

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

        mutable boost::shared_ptr<BaseDeviate> _rng;

        // These need to be defined by the derived class.  They typically would in turn
        // immediately call their own templated applyTo function that defines the actual
        // application of the noise.
        virtual void doApplyTo(ImageView<double>& data) = 0;
        virtual void doApplyTo(ImageView<float>& data) = 0;
        virtual void doApplyTo(ImageView<int32_t>& data) = 0;
        virtual void doApplyTo(ImageView<int16_t>& data) = 0;
    };


    /** 
     * @brief Class implementing simple Gaussain noise.
     *
     * The GaussianNoise class implements a simple Gaussian noise with a given sigma.
     */
    class GaussianNoise : public BaseNoise
    {
    public:
 
        /**
         * @brief Construct a new noise model with a given sigma.
         *
         * @param[in] rng      The BaseDeviate to use for the random number generation.
         * @param[in] sigma    RMS of Gaussian noise
         */
        GaussianNoise(boost::shared_ptr<BaseDeviate> rng, double sigma) :
            BaseNoise(rng), _sigma(sigma)
        {}

        /**
         * @brief Construct a copy that shares the RNG with rhs.
         *
         * Note: the default constructed op= function will do the same thing.
         */
        GaussianNoise(const GaussianNoise& rhs) : 
            BaseNoise(rhs), _sigma(rhs._sigma)
        {}
 
        /**
         * @brief Report current sigma.
         */
        double getSigma() const { return _sigma; }

        /**
         * @brief Set sigma
         */
        void setSigma(double sigma) { _sigma = sigma; }
 
        /**
         * @brief Get the variance of the noise model
         */
        double getVariance() const { return _sigma*_sigma; }

        /**
         * @brief Set the variance of the noise model
         */
        void setVariance(double variance) 
        {
            if (!(variance >= 0.)) 
                throw std::runtime_error("Cannot setVariance to < 0");
            _sigma = sqrt(variance); 
        }

        /**
         * @brief Scale the variance of the noise model
         */
        void scaleVariance(double variance_ratio) 
        {
            if (!(variance_ratio >= 0.)) 
                throw std::runtime_error("Cannot scaleVariance to < 0");
            _sigma *= sqrt(variance_ratio); 
        }

        /**
         * @brief Add noise to an Image.
         */
        template <typename T>
        void applyTo(ImageView<T> data) 
        {
            // Typedef for image row iterable
            typedef typename ImageView<T>::iterator ImIter;

            GaussianDeviate gd(*_rng, 0., _sigma);
            for (int y = data.getYMin(); y <= data.getYMax(); y++) {  // iterate over y
                ImIter ee = data.rowEnd(y);
                for (ImIter it = data.rowBegin(y); it != ee; ++it) {
                    *it = T(*it + gd());
                }
            }
        }

    protected:
        using BaseNoise::_rng;
        void doApplyTo(ImageView<double>& data) { applyTo(data); }
        void doApplyTo(ImageView<float>& data) { applyTo(data); }
        void doApplyTo(ImageView<int32_t>& data) { applyTo(data); }
        void doApplyTo(ImageView<int16_t>& data) { applyTo(data); }

    private: 
        double _sigma;
    };

    /** 
     * @brief Class implementing simple Poisson noise.
     *
     * The PoissonNoise class encapsulates the noise model of Poisson noise corresponding 
     * to the current value in each pixel (including an optional extra sky level).
     *
     * It is equivalent to CCDNoise with gain=1, read_noise=0.
     */
    class PoissonNoise : public BaseNoise
    {
    public:
 
        /**
         * @brief Construct a new noise model, sharing the random number generator with rng.
         *
         * @param[in] rng        The BaseDeviate to use for the random number generation.
         * @param[in] sky_level  The sky level in counts per pixel that was originally in
         *                       the input image, but which is taken to have already been 
         *                       subtracted off.
         */
        PoissonNoise(boost::shared_ptr<BaseDeviate> rng, double sky_level=0.) :
            BaseNoise(rng), _sky_level(sky_level)
        {}

        /**
         * @brief Construct a copy that shares the RNG with rhs.
         *
         * Note: the default constructed op= function will do the same thing.
         */
        PoissonNoise(const PoissonNoise& rhs) : 
            BaseNoise(rhs), _sky_level(rhs._sky_level)
        {}
 

        /**
         * @brief Report current sky_level
         */
        double getSkyLevel() const { return _sky_level; }

        /**
         * @brief Set sky level
         */
        void setSkyLevel(double sky_level) { _sky_level = sky_level; }
 
        /**
         * @brief Get the variance of the noise model
         */
        double getVariance() const { return _sky_level; }

        /**
         * @brief Set the variance of the noise model
         */
        void setVariance(double variance) 
        { 
            if (!(variance >= 0.)) 
                throw std::runtime_error("Cannot setVariance to < 0");
            _sky_level = variance; 
        }

        /**
         * @brief Scale the variance of the noise model
         */
        void scaleVariance(double variance_ratio) 
        { 
            if (!(variance_ratio >= 0.)) 
                throw std::runtime_error("Cannot scaleVariance to < 0");
            _sky_level *= variance_ratio; 
        }

        /**
         * @brief Add noise to an Image.
         */
        template <typename T>
        void applyTo(ImageView<T> data) 
        {
            // Above this many e's, assume Poisson distribution == Gaussian 
            // The Gaussian deviate is about 20% faster than Poisson, and for high N
            // they are virtually identical.
            const double MAX_POISSON=1.e5;
            // Typedef for image row iterable
            typedef typename ImageView<T>::iterator ImIter;

            data += T(_sky_level);

            PoissonDeviate pd(*_rng, 1.); // will reset the mean for each pixel below.
            GaussianDeviate gd(*_rng, 0., 1.);
            for (int y = data.getYMin(); y <= data.getYMax(); y++) {  // iterate over y
                ImIter ee = data.rowEnd(y);
                for (ImIter it = data.rowBegin(y); it != ee; ++it) {
                    if (*it <= 0.) continue;
                    if (*it < MAX_POISSON) {
                        pd.setMean(*it);
                        *it = T(pd());
                    } else {
                        gd.setSigma(sqrt(*it));
                        *it = T(*it + gd());
                    }
                }
            }

            data -= T(_sky_level);
        }


    protected:
        using BaseNoise::_rng;
        void doApplyTo(ImageView<double>& data) { applyTo(data); }
        void doApplyTo(ImageView<float>& data) { applyTo(data); }
        void doApplyTo(ImageView<int32_t>& data) { applyTo(data); }
        void doApplyTo(ImageView<int16_t>& data) { applyTo(data); }

    private: 
        double _sky_level;
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
         * @brief Construct a new noise model, sharing the random number generator with rng.
         *
         * @param[in] rng        The BaseDeviate to use for the random number generation.
         * @param[in] sky_level  The sky level in ADU per pixel that was originally in
         *                       the input image, but which is taken to have already been 
         *                       subtracted off.
         * @param[in] gain       Electrons per ADU in the input Images, used for Poisson noise.
         * @param[in] read_noise RMS of Gaussian noise, in electrons (if gain>0.) or ADU (gain<=0.)
         */
        CCDNoise(boost::shared_ptr<BaseDeviate> rng,
                 double sky_level=0., double gain=1., double read_noise=0.) :
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
         */
        double getSkyLevel() const { return _sky_level; }

        /**
         * @brief Report current gain value
         */
        double getGain() const { return _gain; }

        /**
         * @brief Report current read noise
         */
        double getReadNoise() const { return _read_noise; }

        /**
         * @brief Set sky level
         */
        void setSkyLevel(double sky_level) { _sky_level = sky_level; }

        /**
         * @brief Set gain value
         */
        void setGain(double gain) { _gain = gain; }

        /**
         * @brief Set read noise
         */
        void setReadNoise(double read_noise) { _read_noise = read_noise; }
 
        /**
         * @brief Get the variance of the noise model
         */
        double getVariance() const 
        {
            if (_gain > 0) return (_sky_level + _read_noise*_read_noise) / _gain;
            else return _read_noise * _read_noise;
        }

        /**
         * @brief Set the variance of the noise model
         *
         * This keeps the same relative contribution of sky noise and read noise.
         */
        void setVariance(double variance)
        {
            if (!(variance >= 0.)) 
                throw std::runtime_error("Cannot setVariance to < 0");
            scaleVariance(variance / getVariance());
        }

        /**
         * @brief Scale the variance of the noise model
         *
         * This keeps the same relative contribution of sky noise and read noise.
         */
        void scaleVariance(double variance_ratio)
        {
            if (!(variance_ratio >= 0.)) 
                throw std::runtime_error("Cannot scaleVariance to < 0");
            _sky_level *= variance_ratio;
            _read_noise *= sqrt(variance_ratio);
        }

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
            // Above this many e's, assume Poisson distribution == Gaussian 
            // The Gaussian deviate is about 20% faster than Poisson, and for high N
            // they are virtually identical.
            const double MAX_POISSON=1.e5;
            // Typedef for image row iterable
            typedef typename ImageView<T>::iterator ImIter;

            data += T(_sky_level);

            // Add the Poisson noise first:
            if (_gain > 0.) {
                PoissonDeviate pd(*_rng, 1.); // will reset the mean for each pixel below.
                GaussianDeviate gd(*_rng, 0., 1.);
                for (int y = data.getYMin(); y <= data.getYMax(); y++) {  // iterate over y
                    ImIter ee = data.rowEnd(y);
                    for (ImIter it = data.rowBegin(y); it != ee; ++it) {
                        if (*it <= 0.) continue;
                        double electrons = *it * _gain;
                        if (electrons < MAX_POISSON) {
                            pd.setMean(electrons);
                            *it = T(pd() / _gain);
                        } else {
                            gd.setSigma(sqrt(electrons)/_gain);
                            *it = T(*it + gd());
                        }
                    }
                }
            }

            // Next add the Gaussian noise:
            if (_read_noise > 0.) {
                GaussianDeviate gd(*_rng, 0., _read_noise / (_gain > 0. ? _gain : 1.));
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
        using BaseNoise::_rng;
        void doApplyTo(ImageView<double>& data) { applyTo(data); }
        void doApplyTo(ImageView<float>& data) { applyTo(data); }
        void doApplyTo(ImageView<int32_t>& data) { applyTo(data); }
        void doApplyTo(ImageView<int16_t>& data) { applyTo(data); }

    private: 
        double _sky_level;
        double _gain;
        double _read_noise;
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
         * @brief Construct a new noise model using a given BaseDeviate object.
         *
         * @param[in] dev       The Deviate to use for the noise values
         */
        DeviateNoise(boost::shared_ptr<BaseDeviate> dev) : BaseNoise(dev) {}

        /**
         * @brief Construct a copy that shares the RNG with rhs.
         *
         * Note: the default constructed op= function will do the same thing.
         */
        DeviateNoise(const DeviateNoise& rhs) : BaseNoise(rhs) {}
  
        /**
         * @brief Get the variance of the noise model
         */
        double getVariance() const 
        {
            throw std::runtime_error("getVariance not implemented for DeviateNoise");
        }

        /**
         * @brief Set the variance of the noise model
         */
        void setVariance(double variance) 
        {
            if (!(variance >= 0.)) 
                throw std::runtime_error("Cannot setVariance to < 0");
            throw std::runtime_error("setVariance not implemented for DeviateNoise");
        }

        /**
         * @brief Set the variance of the noise model
         */
        void scaleVariance(double variance_ratio) 
        {
            if (!(variance_ratio >= 0.)) 
                throw std::runtime_error("Cannot scaleVariance to < 0");
            throw std::runtime_error("scaleVariance not implemented for DeviateNoise");
        }


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
                for (ImIter it = data.rowBegin(y); it != ee; ++it) { *it = T(*it + (*_rng)()); }
            }
        }

    protected:
        using BaseNoise::_rng;
        void doApplyTo(ImageView<double>& data) { applyTo(data); }
        void doApplyTo(ImageView<float>& data) { applyTo(data); }
        void doApplyTo(ImageView<int32_t>& data) { applyTo(data); }
        void doApplyTo(ImageView<int16_t>& data) { applyTo(data); }
    };

};  // namespace galsim

#endif
