/* -*- c++ -*-
 * Copyright (c) 2012-2017 by the GalSim developers team on GitHub
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

#ifndef GalSim_Noise_H
#define GalSim_Noise_H

/**
 * @file Noise.h @brief Add noise to image using various noise models
 *
 */

#include <cmath>
#include "Std.h"
#include "Random.h"
#include "Image.h"

namespace galsim {

    template <typename T>
    inline T SQR(T x) { return x*x; }

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
        { if (!_rng) _rng.reset(new BaseDeviate(0)); }

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
        void applyToView(ImageView<T> data)
        {
            // This uses the standard workaround for the fact that you can't have a
            // virtual template function.  The doApplyTo functions are virtual and
            // are listed for each allowed value of T.
            doApplyTo(data);
        }

    protected:

        mutable boost::shared_ptr<BaseDeviate> _rng;

        // These need to be defined by the derived class.  They typically would in turn
        // immediately call their own templated applyToView function that defines the actual
        // application of the noise.
        virtual void doApplyTo(ImageView<double>& data) = 0;
        virtual void doApplyTo(ImageView<float>& data) = 0;
        virtual void doApplyTo(ImageView<int32_t>& data) = 0;
        virtual void doApplyTo(ImageView<int16_t>& data) = 0;
        virtual void doApplyTo(ImageView<uint32_t>& data) = 0;
        virtual void doApplyTo(ImageView<uint16_t>& data) = 0;
    };


    /**
     * @brief Class implementing simple Gaussian noise.
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
        double getVariance() const { return SQR(_sigma); }

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

        template <typename T>
        class NoiseAdder
        {
        public:
            NoiseAdder(GaussianDeviate& gd) : _gd(gd) {}
            T operator()(const T& pix) { return pix + _gd(); }
        private:
            GaussianDeviate& _gd;
        };

        /**
         * @brief Add noise to an Image.
         */
        template <typename T>
        void applyToView(ImageView<T> data)
        {
            GaussianDeviate gd(*_rng, 0., _sigma);
            NoiseAdder<T> adder(gd);
            transform_pixel(data, adder);
        }

    protected:
        using BaseNoise::_rng;
        void doApplyTo(ImageView<double>& data) { applyToView(data); }
        void doApplyTo(ImageView<float>& data) { applyToView(data); }
        void doApplyTo(ImageView<int32_t>& data) { applyToView(data); }
        void doApplyTo(ImageView<int16_t>& data) { applyToView(data); }
        void doApplyTo(ImageView<uint32_t>& data) { applyToView(data); }
        void doApplyTo(ImageView<uint16_t>& data) { applyToView(data); }

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
        PoissonNoise(boost::shared_ptr<BaseDeviate> rng, double sky_level) :
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

        template <typename T>
        class NoiseAdder
        {
        public:
            NoiseAdder(PoissonDeviate& pd) : _pd(pd) {}
            T operator()(const T& pix) {
                if (pix <= 0.) return pix;
                _pd.setMean(pix);
                return T(_pd());
            }
        private:
            PoissonDeviate& _pd;
        };

        /**
         * @brief Add noise to an Image.
         */
        template <typename T>
        void applyToView(ImageView<T> data)
        {
            data += T(_sky_level);

            PoissonDeviate pd(*_rng, 1.); // will reset the mean for each pixel below.
            NoiseAdder<T> adder(pd);
            transform_pixel(data, adder);

            data -= T(_sky_level);
        }


    protected:
        using BaseNoise::_rng;
        void doApplyTo(ImageView<double>& data) { applyToView(data); }
        void doApplyTo(ImageView<float>& data) { applyToView(data); }
        void doApplyTo(ImageView<int32_t>& data) { applyToView(data); }
        void doApplyTo(ImageView<int16_t>& data) { applyToView(data); }
        void doApplyTo(ImageView<uint32_t>& data) { applyToView(data); }
        void doApplyTo(ImageView<uint16_t>& data) { applyToView(data); }

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
                 double sky_level, double gain, double read_noise) :
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
            if (_gain > 0) return _sky_level/_gain + SQR(_read_noise/_gain);
            else return SQR(_read_noise);
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
            double var = getVariance();
            if (var > 0.)
                scaleVariance(variance / getVariance());
            else
                _sky_level = variance;
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

        template <typename T>
        class SkyNoiseAdder
        {
        public:
            SkyNoiseAdder(PoissonDeviate& pd, double gain) :
                _pd(pd), _gain(gain) {}
            T operator()(const T& pix)
            {
                if (pix <= 0.) return pix;
                double elec = pix * _gain;
                _pd.setMean(elec);
                return T(_pd() / _gain);
            }
        private:
            PoissonDeviate& _pd;
            const double _gain;
        };

        template <typename T>
        class ReadNoiseAdder
        {
        public:
            ReadNoiseAdder(GaussianDeviate& gd) : _gd(gd) {}
            T operator()(const T& pix) { return pix + _gd(); }
        private:
            GaussianDeviate& _gd;
        };

        /**
         * @brief Add noise to an Image.
         *
         * Poisson and/or Gaussian noise are added to each pixel of the image according
         * to standard CCD model.
         * @param[in,out] data The Image to be noise-ified.
         */
        template <typename T>
        void applyToView(ImageView<T> data)
        {
            data += T(_sky_level);

            // Add the Poisson noise first:
            if (_gain > 0.) {
                PoissonDeviate pd(*_rng, 1.); // will reset the mean for each pixel below.
                SkyNoiseAdder<T> adder(pd, _gain);
                transform_pixel(data, adder);
            }

            // Next add the Gaussian noise:
            if (_read_noise > 0.) {
                GaussianDeviate gd(*_rng, 0., _read_noise / (_gain > 0. ? _gain : 1.));
                ReadNoiseAdder<T> adder(gd);
                transform_pixel(data, adder);
            }

            data -= T(_sky_level);
        }

    protected:
        using BaseNoise::_rng;
        void doApplyTo(ImageView<double>& data) { applyToView(data); }
        void doApplyTo(ImageView<float>& data) { applyToView(data); }
        void doApplyTo(ImageView<int32_t>& data) { applyToView(data); }
        void doApplyTo(ImageView<int16_t>& data) { applyToView(data); }
        void doApplyTo(ImageView<uint32_t>& data) { applyToView(data); }
        void doApplyTo(ImageView<uint16_t>& data) { applyToView(data); }

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

        template <typename T>
        class NoiseAdder
        {
        public:
            NoiseAdder(BaseDeviate& dev) : _dev(dev) {}
            T operator()(const T& pix) { return pix + _dev(); }
        private:
            BaseDeviate& _dev;
        };

        /**
         * @brief Add noise to an Image.
         *
         * @param[in,out] data The Image to be noise-ified.
         */
        template <typename T>
        void applyToView(ImageView<T> data)
        {
            // Typedef for image row iterable
            NoiseAdder<T> adder(*_rng);
            transform_pixel(data, adder);
        }

    protected:
        using BaseNoise::_rng;
        void doApplyTo(ImageView<double>& data) { applyToView(data); }
        void doApplyTo(ImageView<float>& data) { applyToView(data); }
        void doApplyTo(ImageView<int32_t>& data) { applyToView(data); }
        void doApplyTo(ImageView<int16_t>& data) { applyToView(data); }
        void doApplyTo(ImageView<uint32_t>& data) { applyToView(data); }
        void doApplyTo(ImageView<uint16_t>& data) { applyToView(data); }
    };

    /**
     * @brief Class implementing variable Gaussian noise.
     *
     * The VarGaussianNoise class implements Gaussian noise where each pixel may have
     * a different variance.
     */
    class VarGaussianNoise : public BaseNoise
    {
    public:

        /**
         * @brief Construct a new noise model with a given variance image.
         *
         * @param[in] rng         The BaseDeviate to use for the random number generation.
         * @param[in] var_image   Image with the variance values for the noise in each pixel.
         */
        VarGaussianNoise(boost::shared_ptr<BaseDeviate> rng,
                              const BaseImage<float>& var_image) :
            BaseNoise(rng), _var_image(var_image.view())
        {}

        /**
         * @brief Construct a copy that shares the RNG with rhs.
         *
         * Note: the default constructed op= function will do the same thing.
         */
        VarGaussianNoise(const VarGaussianNoise& rhs) :
            BaseNoise(rhs), _var_image(rhs._var_image)
        {}

        /**
         * @brief Report current variance image.
         */
        ConstImageView<float> getVarImage() const { return _var_image; }

        /**
         * @brief Get the variance of the noise model
         */
        double getVariance() const
        {
            throw std::runtime_error("No single variance value for VariableGaussianNoise");
            return 0.;
        }

        /**
         * @brief Set the variance of the noise model
         */
        void setVariance(double variance)
        {
            throw std::runtime_error(
                "Changing the variance is not allowed for VariableGaussianNoise");
        }

        /**
         * @brief Scale the variance of the noise model
         */
        void scaleVariance(double variance_ratio)
        {
            throw std::runtime_error(
                "Changing the variance is not allowed for VariableGaussianNoise");
        }

        template <typename T>
        class NoiseAdder
        {
        public:
            NoiseAdder(GaussianDeviate& gd) : _gd(gd) {}
            T operator()(const T& pix, const T& var)
            {
                if (!(var >= 0))
                    throw std::runtime_error("variance image has elements < 0.");
                _gd.setSigma(sqrt(var));
                return T(pix + _gd());
            }
        private:
            GaussianDeviate& _gd;
        };

        /**
         * @brief Add noise to an Image.
         */
        template <typename T>
        void applyToView(ImageView<T> data)
        {
            if ( (data.getYMax()-data.getYMin() != _var_image.getYMax()-_var_image.getYMin()) ||
                 (data.getXMax()-data.getXMin() != _var_image.getXMax()-_var_image.getXMin()) ) {
                throw std::runtime_error("The given image does not have the same shape as the "
                                         "variance image in VariableGaussianNoise object.");
            }

            GaussianDeviate gd(*_rng, 0., 1.);
            NoiseAdder<T> adder(gd);
            transform_pixel(data, _var_image, adder);
        }

    protected:
        using BaseNoise::_rng;
        void doApplyTo(ImageView<double>& data) { applyToView(data); }
        void doApplyTo(ImageView<float>& data) { applyToView(data); }
        void doApplyTo(ImageView<int32_t>& data) { applyToView(data); }
        void doApplyTo(ImageView<int16_t>& data) { applyToView(data); }
        void doApplyTo(ImageView<uint32_t>& data) { applyToView(data); }
        void doApplyTo(ImageView<uint16_t>& data) { applyToView(data); }

    private:
        ConstImageView<float> _var_image;

    };


};  // namespace galsim

#endif
