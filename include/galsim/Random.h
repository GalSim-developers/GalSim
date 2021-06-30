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

#ifndef GalSim_Random_H
#define GalSim_Random_H
/**
 * @file Random.h
 *
 * @brief Random-number-generator classes
 *
 * Pseudo-random-number generators with various parent distributions: uniform, Gaussian, binomial,
 * Poisson, Weibull (generalization of Rayleigh and Exponential), and Gamma, all living within the
 * galsim namespace.
 *
 * Wraps Boost.Random classes in a way that lets us swap Boost RNG's without affecting client code.
 */

#include <sstream>

#include "Image.h"

namespace galsim {

    // Function for applying deviates to an image... Used as a method for all Deviates below.
    template <typename D, typename T>
    static void ApplyDeviateToImage(D& dev, ImageView<T>& data)
    {
        // Typedef for image row iterable
        typedef typename ImageView<T>::iterator ImIter;

        for (int y = data.getYMin(); y <= data.getYMax(); y++) {  // iterate over y
            ImIter ee = data.rowEnd(y);
            for (ImIter it = data.rowBegin(y); it != ee; ++it) { *it += T(dev()); }
        }
    }

    /**
     * @brief Base class for all the various Deviates.
     *
     * This holds the essential random number generator that all the other classes use.
     *
     * All deviates have three constructors that define different ways of setting up
     * the random number generator.
     *
     * 1) Only the arguments particular to the derived class (e.g. mean and sigma for
     *    GaussianDeviate).  In this case, a new random number generator is created and
     *    it is seeded using the computer's microsecond counter.
     *
     * 2) Using a particular seed as the first argument to the constructor.
     *    This will also create a new random number generator, but seed it with the
     *    provided value.
     *
     * 3) Passing another BaseDeviate as the first arguemnt to the constructor.
     *    This will make the new Deviate share the same underlying random number generator
     *    with the other Deviate.  So you can make one Deviate (of any type), and seed
     *    it with a particular deterministic value.  Then if you pass that Deviate
     *    to any other one you make, they will all be using the same rng and have a
     *    particular deterministic series of values.  (It doesn't have to be the first
     *    one -- any one you've made later can also be used to seed a new one.)
     *
     * There is not much you can do with something that is only known to be a BaseDeviate
     * rather than one of the derived classes other than construct it and change the
     * seed, and use it as an argument to pass to other Deviate constructors.
     */
    class PUBLIC_API BaseDeviate
    {
    public:
        /**
         * @brief Construct and seed a new BaseDeviate, using the provided value as seed.
         *
         * If lseed == 0, this means to use a random seed from the system: either /dev/urandom
         * if possible, or the time of day otherwise.  Note that in the latter case, the
         * microsecond counter is the seed, so BaseDeviates constructed in rapid succession may
         * not be independent.
         *
         * @param[in] lseed A long-integer seed for the RNG.
         */
        explicit BaseDeviate(long lseed);

        /**
         * @brief Construct a new BaseDeviate, sharing the random number generator with rhs.
         */
        BaseDeviate(const BaseDeviate& rhs);

        /**
         * @brief Construct a new BaseDeviate from a serialization string
         */
        BaseDeviate(const char* str_c);

        /**
         * @brief Destructor
         *
         * Only deletes the underlying RNG if this is the last one using it.
         */
        virtual ~BaseDeviate() {}

        /// @brief return a serialization string for this BaseDeviate
        std::string serialize();

        /**
         * @brief Construct a duplicate of this BaseDeviate object.
         *
         * Both this and the returned duplicate will produce identical sequences of values.
         */
        BaseDeviate duplicate();

        /**
         * @brief Return a string that can act as the repr in python
         */
        std::string repr() { return make_repr(true); }

        /**
         * @brief Return a string that can act as the str in python
         *
         * For this we use the same thing as the repr, but omit the (verbose!) seed parameter.
         */
        std::string str() { return make_repr(false); }

        /**
         * @brief Re-seed the PRNG using specified seed
         *
         * If lseed == 0, this means to use a random seed from the system: either /dev/urandom
         * if possible, or the time of day otherwise.  Note that in the latter case, the
         * microsecond counter is the seed, so BaseDeviates constructed in rapid succession may
         * not be independent.
         *
         * @param[in] lseed A long-integer seed for the RNG.
         *
         * Note that this will reseed all Deviates currently sharing the RNG with this one.
         */
        virtual void seed(long lseed);

        /**
         * @brief Like seed(lseed), but severs the relationship between other Deviates.
         *
         * Other Deviates that had been using the same RNG will be unaffected, while this
         * Deviate will obtain a fresh RNG seed according to lseed.
         */
        void reset(long lseed);

        /**
         * @brief Make this object share its random number generator with another Deviate.
         *
         * It discards whatever rng it had been using and starts sharing the one held by dev.
         */
        void reset(const BaseDeviate& dev);

        /**
         * @brief Clear the internal cache of the rng object.
         *
         * Sometimes this is required to get two sequences synced up if the other one
         * is reseeded.  e.g. GaussianDeviate generates two deviates at a time for efficiency,
         * so if you don't do this, and there is still an internal cached value, you'll get
         * that rather than a new one generated with the new seed.
         *
         * As far as I know, GaussianDeviate is the only one to require this, but just in
         * case something changes about how boost implements any of these deviates, I overload
         * the virtual function for all of them and call the distribution's reset() method.
         */
        virtual void clearCache() {}

        /**
         * @brief Discard some number of values from the random number generator.
         */
        void discard(int n);

        /**
         * @brief Get a random value in its raw form as a long integer.
         */
        long raw();

        /**
         * @brief Draw a new random number from the distribution
         *
         * This is invalid for a BaseDeviate object that is not a derived class.
         * However, we don't make it pure virtual, since we want to be able to make
         * BaseDeviate objects as a direct way to define a common seed for other Deviates.
         */
        double operator()()
        { return generate1(); }

        // This is the virtual function that is overridden in subclasses.
        virtual double generate1()
        { throw std::runtime_error("Cannot draw random values from a pure BaseDeviate object."); }

        /**
         * @brief Draw N new random numbers from the distribution and save the values in
         * an array
         *
         * @param N     The number of values to draw
         * @param data  The array into which to write the values
         */
        void generate(int N, double* data);

        /**
         * @brief Draw N new random numbers from the distribution and add them to the values in
         * an array
         *
         * @param N     The number of values to draw
         * @param data  The array into which to add the values
         */
        void addGenerate(int N, double* data);

   protected:
        struct BaseDeviateImpl;
        shared_ptr<BaseDeviateImpl> _impl;

        /// Helper to make the repr with or without the (lengthy!) seed item.
        virtual std::string make_repr(bool incl_seed);

        /**
         * @brief Private routine to seed with microsecond counter from time-of-day structure.
         */
        void seedtime();

        /**
         * @brief Private routine to seed using /dev/random.  This will throw an exception
         * if this is not possible.
         */
        void seedurandom();

    private:
        BaseDeviate();  // Private no-action constructor used by duplicate().
    };

    /**
     * @brief Pseudo-random number generator with uniform distribution in interval [0.,1.).
     */
    class PUBLIC_API UniformDeviate : public BaseDeviate
    {
    public:
        /** @brief Construct and seed a new UniformDeviate, using the provided value as seed.
         *
         * If lseed == 0, this means to use a random seed from the system: either /dev/urandom
         * if possible, or the time of day otherwise.  Note that in the latter case, the
         * microsecond counter is the seed, so BaseDeviates constructed in rapid succession may
         * not be independent.
         *
         * @param[in] lseed A long-integer seed for the RNG.
         */
        UniformDeviate(long lseed);

        /// @brief Construct a new UniformDeviate, sharing the random number generator with rhs.
        UniformDeviate(const BaseDeviate& rhs);

        /// @brief Construct a copy that shares the RNG with rhs.
        UniformDeviate(const UniformDeviate& rhs);

        /// @brief Construct a new UniformDeviate from a serialization string
        UniformDeviate(const char* str_c);

        /**
         * @brief Construct a duplicate of this UniformDeviate object.
         *
         * Both this and the returned duplicate will produce identical sequences of values.
         */
        UniformDeviate duplicate()
        { return UniformDeviate(BaseDeviate::duplicate()); }

        /**
         * @brief Draw a new random number from the distribution
         *
         * @return A uniform deviate in the interval [0.,1.)
         */
        double generate1();

        /**
         * @brief Clear the internal cache
         */
        void clearCache();

    protected:
        std::string make_repr(bool incl_seed);

    private:
        struct UniformDeviateImpl;
        shared_ptr<UniformDeviateImpl> _devimpl;
    };

    /**
     * @brief Pseudo-random number generator with Gaussian distribution.
     */
    class PUBLIC_API GaussianDeviate : public BaseDeviate
    {
    public:

        /**
         * @brief Construct a new Gaussian-distributed RNG, using the provided value as seed.
         *
         * If lseed == 0, this means to use a random seed from the system: either /dev/urandom
         * if possible, or the time of day otherwise.  Note that in the latter case, the
         * microsecond counter is the seed, so BaseDeviates constructed in rapid succession may
         * not be independent.
         *
         * @param[in] lseed Seed to use.
         * @param[in] mean  Mean of the output distribution
         * @param[in] sigma Standard deviation of the distribution
         */
        GaussianDeviate(long lseed, double mean, double sigma);

        /**
         * @brief Construct a new Gaussian-distributed RNG, sharing the random number
         * generator with rhs.
         *
         * @param[in] rhs   Other deviate with which to share the RNG
         * @param[in] mean  Mean of the output distribution
         * @param[in] sigma Standard deviation of the distribution
         */
        GaussianDeviate(const BaseDeviate& rhs, double mean, double sigma);

        /**
         * @brief Construct a copy that shares the RNG with rhs.
         *
         * Note: the default constructed op= function will do the same thing.
         */
        GaussianDeviate(const GaussianDeviate& rhs);

        /// @brief Construct a new GaussianDeviate from a serialization string
        GaussianDeviate(const char* str_c, double mean, double sigma);

        /**
         * @brief Construct a duplicate of this GaussianDeviate object.
         *
         * Both this and the returned duplicate will produce identical sequences of values.
         */
        GaussianDeviate duplicate()
        { return GaussianDeviate(BaseDeviate::duplicate(), getMean(), getSigma()); }

        /**
         * @brief Draw a new random number from the distribution
         *
         * @return A Gaussian deviate with current mean and sigma
         */
        double generate1();

        /**
         * @brief Get current distribution mean
         *
         * @return Mean of distribution
         */
        double getMean();

        /**
         * @brief Get current distribution standard deviation
         *
         * @return Standard deviation of distribution
         */
        double getSigma();

        /**
         * @brief Set current distribution mean
         *
         * @param[in] mean New mean for distribution
         */
        void setMean(double mean);

        /**
         * @brief Set current distribution standard deviation
         *
         * @param[in] sigma New standard deviation for distribution.  Behavior for non-positive
         * value is undefined.
         */
        void setSigma(double sigma);

        /**
         * @brief Clear the internal cache
         *
         * This one is definitely required, since _normal generates two deviates at a time
         * and stores one for later.  So this clears that out when necessary.
         */
        void clearCache();

        /**
         * @brief Replace data with Gaussian draws using the existing data as the variances.
         *
         * @param N     The number of values to draw
         * @param data  The array with the given variances to replace with Gaussian draws.
         */
        void generateFromVariance(int N, double* data);

    protected:
        std::string make_repr(bool incl_seed);

    private:
        struct GaussianDeviateImpl;
        shared_ptr<GaussianDeviateImpl> _devimpl;
    };


    /**
     * @brief A Binomial deviate for N trials each of probability p.
     */
    class PUBLIC_API BinomialDeviate : public BaseDeviate
    {
    public:

        /**
         * @brief Construct a new binomial-distributed RNG, using the provided value as seed.
         *
         * If lseed == 0, this means to use a random seed from the system: either /dev/urandom
         * if possible, or the time of day otherwise.  Note that in the latter case, the
         * microsecond counter is the seed, so BaseDeviates constructed in rapid succession may
         * not be independent.
         *
         * @param[in] lseed Seed to use
         * @param[in] N Number of "coin flips" per trial
         * @param[in] p Probability of success per coin flip.
         */
        BinomialDeviate(long lseed, int N, double p);

        /**
         * @brief Construct a new binomial-distributed RNG, sharing the random number
         * generator with rhs.
         *
         * @param[in] rhs   Other deviate with which to share the RNG
         * @param[in] N Number of "coin flips" per trial
         * @param[in] p Probability of success per coin flip.
         */
        BinomialDeviate(const BaseDeviate& rhs, int N, double p);

        /**
         * @brief Construct a copy that shares the RNG with rhs.
         *
         * Note: the default constructed op= function will do the same thing.
         */
        BinomialDeviate(const BinomialDeviate& rhs);

        /// @brief Construct a new BinomialDeviate from a serialization string
        BinomialDeviate(const char* str_c, int N, double p);

        /**
         * @brief Construct a duplicate of this BinomialDeviate object.
         *
         * Both this and the returned duplicate will produce identical sequences of values.
         */
        BinomialDeviate duplicate()
        { return BinomialDeviate(BaseDeviate::duplicate(), getN(), getP()); }

        /**
         * @brief Draw a new random number from the distribution
         *
         * @return A binomial deviate with current N and p
         */
        double generate1();

        /**
         * @brief Report current value of N
         *
         * @return Current value of N
         */
        int getN();

        /**
         * @brief Report current value of p
         *
         * @return Current value of p
         */
        double getP();

        /**
         * @brief Reset value of N
         *
         * @param[in] N New value of N
         */
        void setN(int N);

        /**
         * @brief Reset value of p
         *
         * @param[in] p New value of p
         */
        void setP(double p);

        /**
         * @brief Clear the internal cache
         */
        void clearCache();

    protected:
        std::string make_repr(bool incl_seed);

    private:
        struct BinomialDeviateImpl;
        shared_ptr<BinomialDeviateImpl> _devimpl;
    };

    /**
     * @brief A Poisson deviate with specified mean.
     */
    class PUBLIC_API PoissonDeviate : public BaseDeviate
    {
    public:

        /**
         * @brief Construct a new Poisson-distributed RNG, using the provided value as seed.
         *
         * If lseed == 0, this means to use a random seed from the system: either /dev/urandom
         * if possible, or the time of day otherwise.  Note that in the latter case, the
         * microsecond counter is the seed, so BaseDeviates constructed in rapid succession may
         * not be independent.
         *
         * @param[in] lseed Seed to use
         * @param[in] mean  Mean of the output distribution
         */
        PoissonDeviate(long lseed, double mean);

        /**
         * @brief Construct a new Poisson-distributed RNG, sharing the random number
         * generator with rhs.
         *
         * @param[in] rhs   Other deviate with which to share the RNG
         * @param[in] mean  Mean of the output distribution
         */
        PoissonDeviate(const BaseDeviate& rhs, double mean);

        /**
         * @brief Construct a copy that shares the RNG with rhs.
         *
         * Note: the default constructed op= function will do the same thing.
         */
        PoissonDeviate(const PoissonDeviate& rhs);

        /// @brief Construct a new PoissonDeviate from a serialization string
        PoissonDeviate(const char* str_c, double mean);

        /**
         * @brief Draw a new random number from the distribution
         *
         * @return A Poisson deviate with current mean
         */
        double generate1();

        /**
         * @brief Construct a duplicate of this PoissonDeviate object.
         *
         * Both this and the returned duplicate will produce identical sequences of values.
         */
        PoissonDeviate duplicate()
        { return PoissonDeviate(BaseDeviate::duplicate(), getMean()); }

        /**
         * @brief Report current distribution mean
         *
         * @return Current mean value
         */
        double getMean();

        /**
         * @brief Reset distribution mean
         *
         * @param[in] mean New mean value
         */
        void setMean(double mean);

        /**
         * @brief Clear the internal cache
         */
        void clearCache();

        /**
         * @brief Replace data with Poisson draws using the existing data as the expectation
         * value.
         *
         * @param N     The number of values to draw
         * @param data  The array with the given data to replace with Poisson draws.
         */
        void generateFromExpectation(int N, double* data);


    protected:
        std::string make_repr(bool incl_seed);

    private:
        struct PoissonDeviateImpl;
        shared_ptr<PoissonDeviateImpl> _devimpl;
    };

    /**
     * @brief A Weibull-distributed deviate with shape parameter a and scale parameter b.
     *
     * The Weibull distribution is related to a number of other probability distributions; in
     * particular, it interpolates between the exponential distribution (a=1) and the Rayleigh
     * distribution (a=2). See http://en.wikipedia.org/wiki/Weibull_distribution (a=k and b=lambda
     * in the notation adopted in the Wikipedia article).  The Weibull distribution is a real valued
     * distribution producing deviates >= 0.
     */
    class PUBLIC_API WeibullDeviate : public BaseDeviate
    {
    public:

        /**
         * @brief Construct a new Weibull-distributed RNG, using the provided value as seed.
         *
         * If lseed == 0, this means to use a random seed from the system: either /dev/urandom
         * if possible, or the time of day otherwise.  Note that in the latter case, the
         * microsecond counter is the seed, so BaseDeviates constructed in rapid succession may
         * not be independent.
         *
         * @param[in] lseed Seed to use
         * @param[in] a    Shape parameter of the output distribution, must be > 0.
         * @param[in] b    Scale parameter of the distribution, must be > 0.
         */
        WeibullDeviate(long lseed, double a, double b);

        /**
         * @brief Construct a new Weibull-distributed RNG, sharing the random number
         * generator with rhs.
         *
         * @param[in] rhs   Other deviate with which to share the RNG
         * @param[in] a    Shape parameter of the output distribution, must be > 0.
         * @param[in] b    Scale parameter of the distribution, must be > 0.
         */
        WeibullDeviate(const BaseDeviate& rhs, double a, double b);

        /**
         * @brief Construct a copy that shares the RNG with rhs.
         *
         * Note: the default constructed op= function will do the same thing.
         */
        WeibullDeviate(const WeibullDeviate& rhs);

        /// @brief Construct a new WeibullDeviate from a serialization string
        WeibullDeviate(const char* str_c, double a, double b);

        /**
         * @brief Construct a duplicate of this WeibullDeviate object.
         *
         * Both this and the returned duplicate will produce identical sequences of values.
         */
        WeibullDeviate duplicate()
        { return WeibullDeviate(BaseDeviate::duplicate(), getA(), getB()); }

        /**
         * @brief Draw a new random number from the distribution.
         *
         * @return A Weibull deviate with current shape k and scale lam.
         */
        double generate1();

        /**
         * @brief Get current distribution shape parameter a.
         *
         * @return Shape parameter a of distribution.
         */
        double getA();

        /**
         * @brief Get current distribution scale parameter b.
         *
         * @return Scale parameter b of distribution.
         */
        double getB();

        /**
         * @brief Set current distribution shape parameter a.
         *
         * @param[in] a  New shape parameter for distribution. Behaviour for non-positive value
         * is undefined.
         */
        void setA(double a);

        /**
         * @brief Set current distribution scale parameter b.
         *
         * @param[in] b  New scale parameter for distribution.  Behavior for non-positive
         * value is undefined.
         */
        void setB(double b);

        /**
         * @brief Clear the internal cache
         */
        void clearCache();

    protected:
        std::string make_repr(bool incl_seed);

    private:
        struct WeibullDeviateImpl;
        shared_ptr<WeibullDeviateImpl> _devimpl;
    };

    /**
     * @brief A Gamma-distributed deviate with shape parameter k and scale parameter theta.
     *
     * See http://en.wikipedia.org/wiki/Gamma_distribution.
     * (Note: we use the k, theta notation.  If you prefer alpha, beta, use k=alpha, theta=1/beta.)
     * The Gamma distribution is a real valued distribution producing deviates >= 0.
     */
    class PUBLIC_API GammaDeviate : public BaseDeviate
    {
    public:

        /**
         * @brief Construct a new Gamma-distributed RNG, using the provided value as seed.
         *
         * If lseed == 0, this means to use a random seed from the system: either /dev/urandom
         * if possible, or the time of day otherwise.  Note that in the latter case, the
         * microsecond counter is the seed, so BaseDeviates constructed in rapid succession may
         * not be independent.
         *
         * @param[in] lseed  Seed to use.
         * @param[in] k      Shape parameter of the output distribution, must be > 0.
         * @param[in] theta  Scale parameter of the distribution, must be > 0.
         */
        GammaDeviate(long lseed, double k, double theta);

        /**
         * @brief Construct a new Gamma-distributed RNG, sharing the random number
         * generator with rhs.
         *
         * @param[in] rhs    Other deviate with which to share the RNG
         * @param[in] k      Shape parameter of the output distribution, must be > 0.
         * @param[in] theta  Scale parameter of the distribution, must be > 0.
         */
        GammaDeviate(const BaseDeviate& rhs, double k, double theta);

        /**
         * @brief Construct a copy that shares the RNG with rhs.
         *
         * Note: the default constructed op= function will do the same thing.
         */
        GammaDeviate(const GammaDeviate& rhs);

        /// @brief Construct a new GammaDeviate from a serialization string
        GammaDeviate(const char* str_c, double k, double theta);

        /**
         * @brief Construct a duplicate of this GammaDeviate object.
         *
         * Both this and the returned duplicate will produce identical sequences of values.
         */
        GammaDeviate duplicate()
        { return GammaDeviate(BaseDeviate::duplicate(), getK(), getTheta()); }

        /**
         * @brief Draw a new random number from the distribution.
         *
         * @return A Gamma deviate with current shape k and scale theta.
         */
        double generate1();

        /**
         * @brief Get current distribution shape parameter k.
         *
         * @return Shape parameter k of distribution.
         */
        double getK();

        /**
         * @brief Get current distribution scale parameter theta.
         *
         * @return Scale parameter theta of distribution.
         */
        double getTheta();

        /**
         * @brief Set current distribution shape parameter k.
         *
         * @param[in] k  New shape parameter for distribution. Behaviour for non-positive value
         *               is undefined.
         */
        void setK(double k);

        /**
         * @brief Set current distribution scale parameter theta.
         *
         * @param[in] theta  New scale parameter for distribution.  Behavior for non-positive
         *                   value is undefined.
         */
        void setTheta(double theta);

        /**
         * @brief Clear the internal cache
         */
        void clearCache();

    protected:
        std::string make_repr(bool incl_seed);

    private:
        // Note: confusingly, boost calls the internal values alpha and beta, even though they
        // don't conform to the normal beta=1/theta.  Rather, they have beta=theta.
        struct GammaDeviateImpl;
        shared_ptr<GammaDeviateImpl> _devimpl;
    };

    /**
     * @brief A Chi^2-distributed deviate with degrees-of-freedom parameter n.
     *
     * See http://en.wikipedia.org/wiki/Chi-squared_distribution (although note that in the Boost
     * random routine this class calls the notation is k=n for the number of degrees of freedom).
     * The Chi^2 distribution is a real valued distribution producing deviates >= 0.
     */
    class PUBLIC_API Chi2Deviate : public BaseDeviate
    {
    public:

        /**
         * @brief Construct a new Chi^2-distributed RNG, using the provided value as seed.
         *
         * If lseed == 0, this means to use a random seed from the system: either /dev/urandom
         * if possible, or the time of day otherwise.  Note that in the latter case, the
         * microsecond counter is the seed, so BaseDeviates constructed in rapid succession may
         * not be independent.
         *
         * @param[in] lseed Seed to use
         * @param[in] n     Number of degrees of freedom for the output distribution, must be > 0.
         */
        Chi2Deviate(long lseed, double n);

        /**
         * @brief Construct a new Chi^2-distributed RNG, sharing the random number
         * generator with rhs.
         *
         * @param[in] rhs   Other deviate with which to share the RNG
         * @param[in] n     Number of degrees of freedom for the output distribution, must be > 0.
         */
        Chi2Deviate(const BaseDeviate& rhs, double n);

        /**
         * @brief Construct a copy that shares the RNG with rhs.
         *
         * Note: the default constructed op= function will do the same thing.
         */
        Chi2Deviate(const Chi2Deviate& rhs);

        /// @brief Construct a new Chi2Deviate from a serialization string
        Chi2Deviate(const char* str_c, double n);

        /**
         * @brief Construct a duplicate of this Chi2Deviate object.
         *
         * Both this and the returned duplicate will produce identical sequences of values.
         */
        Chi2Deviate duplicate()
        { return Chi2Deviate(BaseDeviate::duplicate(), getN()); }

        /**
         * @brief Draw a new random number from the distribution.
         *
         * @return A Chi^2 deviate with current degrees-of-freedom parameter n.
         */
        double generate1();

        /**
         * @brief Get current distribution degrees-of-freedom parameter n.
         *
         * @return Degrees-of-freedom parameter n of distribution.
         */
        double getN();

        /**
         * @brief Set current distribution degrees-of-freedom parameter n.
         *
         * @param[in] n  New degrees-of-freedom parameter n for distribution. Behaviour for
         *               non-positive value is undefined.
         */
        void setN(double n);

        /**
         * @brief Clear the internal cache
         */
        void clearCache();

    protected:
        std::string make_repr(bool incl_seed);

    private:

        struct Chi2DeviateImpl;
        shared_ptr<Chi2DeviateImpl> _devimpl;
    };

}  // namespace galsim

#endif
