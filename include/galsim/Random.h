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

#include "galsim/IgnoreWarnings.h"

// Variable defined to use a private copy of Boost.Random, modified
// to avoid any reference to Boost.Random elements that might be on
// the local machine.
// Undefine this to use Boost.Random from the local distribution.
#define DIVERT_BOOST_RANDOM

#ifdef DIVERT_BOOST_RANDOM
#include "galsim/boost1_48_0/random/mersenne_twister.hpp"
#include "galsim/boost1_48_0/random/normal_distribution.hpp"
#include "galsim/boost1_48_0/random/binomial_distribution.hpp"
#include "galsim/boost1_48_0/random/poisson_distribution.hpp"
#include "galsim/boost1_48_0/random/uniform_real_distribution.hpp"
#include "galsim/boost1_48_0/random/weibull_distribution.hpp"
#include "galsim/boost1_48_0/random/gamma_distribution.hpp"
#include "galsim/boost1_48_0/random/chi_squared_distribution.hpp"
#else
#include "boost/random/mersenne_twister.hpp"
#include "boost/random/normal_distribution.hpp"
#include "boost/random/binomial_distribution.hpp"
#include "boost/random/poisson_distribution.hpp"
#include "boost/random/uniform_real_distribution.hpp"
#include "boost/random/weibull_distribution.hpp"
#include "boost/random/gamma_distribution.hpp"
#include "boost/random/chi_squared_distribution.hpp"
#endif
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
    class BaseDeviate
    {
        // Note that this class could be templated with the type of Boost.Random generator that
        // you want to use instead of mt19937
        typedef boost::mt19937 rng_type;

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
        explicit BaseDeviate(long lseed) : _rng(new rng_type()) { seed(lseed); }

        /**
         * @brief Construct a new BaseDeviate, sharing the random number generator with rhs.
         */
        BaseDeviate(const BaseDeviate& rhs) : _rng(rhs._rng) {}

        /**
         * @brief Construct a new BaseDeviate from a serialization string
         */
        BaseDeviate(const std::string& str) : _rng(new rng_type())
        {
            std::istringstream iss(str);
            iss >> *_rng;
        }

        /**
         * @brief Destructor
         *
         * Only deletes the underlying RNG if this is the last one using it.
         */
        virtual ~BaseDeviate() {}

        /// @brief return a serialization string for this BaseDeviate
        std::string serialize()
        {
            // When serializing, we need to make sure there is no cache being stored
            // by the derived class.
            clearCache();
            std::ostringstream oss;
            oss << *_rng;
            return oss.str();
        }

        /**
         * @brief Construct a duplicate of this BaseDeviate object.
         *
         * Both this and the returned duplicate will produce identical sequences of values.
         */
        BaseDeviate duplicate()
        { return BaseDeviate(serialize()); }

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
        void reset(long lseed) { _rng.reset(new rng_type()); seed(lseed); }

        /**
         * @brief Make this object share its random number generator with another Deviate.
         *
         * It discards whatever rng it had been using and starts sharing the one held by dev.
         */
        void reset(const BaseDeviate& dev) { _rng = dev._rng; clearCache(); }

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
        void discard(int n) { _rng->discard(n); }

        /**
         * @brief Get a random value in its raw form as a long integer.
         */
        long raw() { return (*_rng)(); }

        /**
         * @brief Draw a new random number from the distribution
         *
         * This is invalid for a BaseDeviate object that is not a derived class.
         * However, we don't make it pure virtual, since we want to be able to make
         * BaseDeviate objects as a direct way to define a common seed for other Deviates.
         */
        double operator()() { return _val(); }

        /**
         * @brief Draw N new random numbers from the distribution and save the values in
         * an array
         *
         * @param N     The number of values to draw
         * @param data  The array into which to write the values
         */
        void generate(int N, double* data);

   protected:

        boost::shared_ptr<rng_type> _rng;

        // This is the virtual function that is actually overridden.
        virtual double _val()
        { throw std::runtime_error("Cannot draw random values from a pure BaseDeviate object."); }

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
    };

    /**
     * @brief Pseudo-random number generator with uniform distribution in interval [0.,1.).
     */
    class UniformDeviate : public BaseDeviate
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
        UniformDeviate(long lseed) : BaseDeviate(lseed), _urd(0.,1.) {}

        /// @brief Construct a new UniformDeviate, sharing the random number generator with rhs.
        UniformDeviate(const BaseDeviate& rhs) : BaseDeviate(rhs), _urd(0.,1.) {}

        /// @brief Construct a copy that shares the RNG with rhs.
        UniformDeviate(const UniformDeviate& rhs) : BaseDeviate(rhs), _urd(0.,1.) {}

        /// @brief Construct a new UniformDeviate from a serialization string
        UniformDeviate(const std::string& str) : BaseDeviate(str), _urd(0.,1.) {}

        /**
         * @brief Construct a duplicate of this UniformDeviate object.
         *
         * Both this and the returned duplicate will produce identical sequences of values.
         */
        UniformDeviate duplicate()
        { return UniformDeviate(serialize()); }

        /**
         * @brief Clear the internal cache
         */
        void clearCache() { _urd.reset(); }

    protected:
        double _val() { return _urd(*this->_rng); }
        std::string make_repr(bool incl_seed);

    private:
        boost::random::uniform_real_distribution<> _urd;
    };

    /**
     * @brief Pseudo-random number generator with Gaussian distribution.
     */
    class GaussianDeviate : public BaseDeviate
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
        GaussianDeviate(long lseed, double mean, double sigma) :
            BaseDeviate(lseed), _normal(mean,sigma) {}

        /**
         * @brief Construct a new Gaussian-distributed RNG, sharing the random number
         * generator with rhs.
         *
         * @param[in] rhs   Other deviate with which to share the RNG
         * @param[in] mean  Mean of the output distribution
         * @param[in] sigma Standard deviation of the distribution
         */
        GaussianDeviate(const BaseDeviate& rhs, double mean, double sigma) :
            BaseDeviate(rhs), _normal(mean,sigma) {}

        /**
         * @brief Construct a copy that shares the RNG with rhs.
         *
         * Note: the default constructed op= function will do the same thing.
         */
        GaussianDeviate(const GaussianDeviate& rhs) : BaseDeviate(rhs), _normal(rhs._normal) {}

        /// @brief Construct a new GaussianDeviate from a serialization string
        GaussianDeviate(const std::string& str, double mean, double sigma) :
            BaseDeviate(str), _normal(mean,sigma) {}

        /**
         * @brief Construct a duplicate of this GaussianDeviate object.
         *
         * Both this and the returned duplicate will produce identical sequences of values.
         */
        GaussianDeviate duplicate()
        { return GaussianDeviate(serialize(),getMean(),getSigma()); }

        /**
         * @brief Get current distribution mean
         *
         * @return Mean of distribution
         */
        double getMean() { return _normal.mean(); }

        /**
         * @brief Get current distribution standard deviation
         *
         * @return Standard deviation of distribution
         */
        double getSigma() { return _normal.sigma(); }

        /**
         * @brief Set current distribution mean
         *
         * @param[in] mean New mean for distribution
         */
        void setMean(double mean)
        { _normal.param(boost::random::normal_distribution<>::param_type(mean,_normal.sigma())); }

        /**
         * @brief Set current distribution standard deviation
         *
         * @param[in] sigma New standard deviation for distribution.  Behavior for non-positive
         * value is undefined.
         */
        void setSigma(double sigma)
        { _normal.param(boost::random::normal_distribution<>::param_type(_normal.mean(),sigma)); }

        /**
         * @brief Clear the internal cache
         *
         * This one is definitely required, since _normal generates two deviates at a time
         * and stores one for later.  So this clears that out when necessary.
         */
        void clearCache() { _normal.reset(); }

    protected:
        double _val() { return _normal(*this->_rng); }
        std::string make_repr(bool incl_seed);

    private:
        boost::random::normal_distribution<> _normal;
    };


    /**
     * @brief A Binomial deviate for N trials each of probability p.
     */
    class BinomialDeviate : public BaseDeviate
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
        BinomialDeviate(long lseed, int N, double p) : BaseDeviate(lseed), _bd(N,p) {}

        /**
         * @brief Construct a new binomial-distributed RNG, sharing the random number
         * generator with rhs.
         *
         * @param[in] rhs   Other deviate with which to share the RNG
         * @param[in] N Number of "coin flips" per trial
         * @param[in] p Probability of success per coin flip.
         */
        BinomialDeviate(const BaseDeviate& rhs, int N, double p) : BaseDeviate(rhs), _bd(N,p) {}

        /**
         * @brief Construct a copy that shares the RNG with rhs.
         *
         * Note: the default constructed op= function will do the same thing.
         */
        BinomialDeviate(const BinomialDeviate& rhs) : BaseDeviate(rhs), _bd(rhs._bd) {}

        /// @brief Construct a new BinomialDeviate from a serialization string
        BinomialDeviate(const std::string& str, int N, double p) : BaseDeviate(str), _bd(N,p) {}

        /**
         * @brief Construct a duplicate of this BinomialDeviate object.
         *
         * Both this and the returned duplicate will produce identical sequences of values.
         */
        BinomialDeviate duplicate()
        { return BinomialDeviate(serialize(),getN(),getP()); }

        /**
         * @brief Report current value of N
         *
         * @return Current value of N
         */
        int getN() { return _bd.t(); }

        /**
         * @brief Report current value of p
         *
         * @return Current value of p
         */
        double getP() { return _bd.p(); }

        /**
         * @brief Reset value of N
         *
         * @param[in] N New value of N
         */
        void setN(int N) {
            _bd.param(boost::random::binomial_distribution<>::param_type(N,_bd.p()));
        }

        /**
         * @brief Reset value of p
         *
         * @param[in] p New value of p
         */
        void setP(double p) {
            _bd.param(boost::random::binomial_distribution<>::param_type(_bd.t(),p));
        }

        /**
         * @brief Clear the internal cache
         */
        void clearCache() { _bd.reset(); }

    protected:
        double _val() { return _bd(*this->_rng); }
        std::string make_repr(bool incl_seed);

    private:
        boost::random::binomial_distribution<> _bd;
    };

    /**
     * @brief A Poisson deviate with specified mean.
     */
    class PoissonDeviate : public BaseDeviate
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
        PoissonDeviate(long lseed, double mean) :
            BaseDeviate(lseed), _getValue(&PoissonDeviate::getPDValue)
        { setMean(mean); }

        /**
         * @brief Construct a new Poisson-distributed RNG, sharing the random number
         * generator with rhs.
         *
         * @param[in] rhs   Other deviate with which to share the RNG
         * @param[in] mean  Mean of the output distribution
         */
        PoissonDeviate(const BaseDeviate& rhs, double mean) :
            BaseDeviate(rhs), _getValue(&PoissonDeviate::getPDValue)
        { setMean(mean); }

        /**
         * @brief Construct a copy that shares the RNG with rhs.
         *
         * Note: the default constructed op= function will do the same thing.
         */
        PoissonDeviate(const PoissonDeviate& rhs) :
            BaseDeviate(rhs), _getValue(rhs._getValue), _pd(rhs._pd), _gd(rhs._gd) {}

        /// @brief Construct a new PoissonDeviate from a serialization string
        PoissonDeviate(const std::string& str, double mean) :
            BaseDeviate(str), _getValue(&PoissonDeviate::getPDValue)
        { setMean(mean); }

        /**
         * @brief Construct a duplicate of this PoissonDeviate object.
         *
         * Both this and the returned duplicate will produce identical sequences of values.
         */
        PoissonDeviate duplicate()
        { return PoissonDeviate(serialize(),getMean()); }

        /**
         * @brief Report current distribution mean
         *
         * @return Current mean value
         */
        double getMean() { return _pd.mean(); }

        /**
         * @brief Reset distribution mean
         *
         * @param[in] mean New mean value
         */
        void setMean(double mean);

        /**
         * @brief Clear the internal cache
         */
        void clearCache() { _pd.reset(); if (_gd) _gd->reset(); }

    protected:
        double _val();

        double (PoissonDeviate::*_getValue)(); // A variable equal to either getPDValue (normal)
                                               // or getGDValue (if mean > 2^30)

        std::string make_repr(bool incl_seed);

        double getPDValue();
        double getGDValue();


    private:
        boost::random::poisson_distribution<> _pd;
        boost::shared_ptr<boost::random::normal_distribution<> > _gd;
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
    class WeibullDeviate : public BaseDeviate
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
        WeibullDeviate(long lseed, double a, double b) :
            BaseDeviate(lseed), _weibull(a,b) {}

        /**
         * @brief Construct a new Weibull-distributed RNG, sharing the random number
         * generator with rhs.
         *
         * @param[in] rhs   Other deviate with which to share the RNG
         * @param[in] a    Shape parameter of the output distribution, must be > 0.
         * @param[in] b    Scale parameter of the distribution, must be > 0.
         */
        WeibullDeviate(const BaseDeviate& rhs, double a, double b) :
            BaseDeviate(rhs), _weibull(a,b) {}

        /**
         * @brief Construct a copy that shares the RNG with rhs.
         *
         * Note: the default constructed op= function will do the same thing.
         */
        WeibullDeviate(const WeibullDeviate& rhs) : BaseDeviate(rhs), _weibull(rhs._weibull) {}

        /// @brief Construct a new WeibullDeviate from a serialization string
        WeibullDeviate(const std::string& str, double a, double b) :
            BaseDeviate(str), _weibull(a,b) {}

        /**
         * @brief Construct a duplicate of this WeibullDeviate object.
         *
         * Both this and the returned duplicate will produce identical sequences of values.
         */
        WeibullDeviate duplicate()
        { return WeibullDeviate(serialize(),getA(),getB()); }

        /**
         * @brief Get current distribution shape parameter a.
         *
         * @return Shape parameter a of distribution.
         */
        double getA() { return _weibull.a(); }

        /**
         * @brief Get current distribution scale parameter b.
         *
         * @return Scale parameter b of distribution.
         */
        double getB() { return _weibull.b(); }

        /**
         * @brief Set current distribution shape parameter a.
         *
         * @param[in] a  New shape parameter for distribution. Behaviour for non-positive value
         * is undefined.
         */
        void setA(double a) {
            _weibull.param(boost::random::weibull_distribution<>::param_type(a, _weibull.b()));
        }

        /**
         * @brief Set current distribution scale parameter b.
         *
         * @param[in] b  New scale parameter for distribution.  Behavior for non-positive
         * value is undefined.
         */
        void setB(double b) {
            _weibull.param(boost::random::weibull_distribution<>::param_type(_weibull.a(), b));
        }

        /**
         * @brief Clear the internal cache
         */
        void clearCache() { _weibull.reset(); }

    protected:
        double _val() { return _weibull(*this->_rng); }
        std::string make_repr(bool incl_seed);

    private:
        boost::random::weibull_distribution<> _weibull;
    };

    /**
     * @brief A Gamma-distributed deviate with shape parameter k and scale parameter theta.
     *
     * See http://en.wikipedia.org/wiki/Gamma_distribution.
     * (Note: we use the k, theta notation.  If you prefer alpha, beta, use k=alpha, theta=1/beta.)
     * The Gamma distribution is a real valued distribution producing deviates >= 0.
     */
    class GammaDeviate : public BaseDeviate
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
        GammaDeviate(long lseed, double k, double theta) :
            BaseDeviate(lseed), _gamma(k,theta) {}

        /**
         * @brief Construct a new Gamma-distributed RNG, sharing the random number
         * generator with rhs.
         *
         * @param[in] rhs    Other deviate with which to share the RNG
         * @param[in] k      Shape parameter of the output distribution, must be > 0.
         * @param[in] theta  Scale parameter of the distribution, must be > 0.
         */
        GammaDeviate(const BaseDeviate& rhs, double k, double theta) :
            BaseDeviate(rhs), _gamma(k,theta) {}

        /**
         * @brief Construct a copy that shares the RNG with rhs.
         *
         * Note: the default constructed op= function will do the same thing.
         */
        GammaDeviate(const GammaDeviate& rhs) : BaseDeviate(rhs), _gamma(rhs._gamma) {}

        /// @brief Construct a new GammaDeviate from a serialization string
        GammaDeviate(const std::string& str, double k, double theta) :
            BaseDeviate(str), _gamma(k,theta) {}

        /**
         * @brief Construct a duplicate of this GammaDeviate object.
         *
         * Both this and the returned duplicate will produce identical sequences of values.
         */
        GammaDeviate duplicate()
        { return GammaDeviate(serialize(),getK(),getTheta()); }

        /**
         * @brief Get current distribution shape parameter k.
         *
         * @return Shape parameter k of distribution.
         */
        double getK() { return _gamma.alpha(); }

        /**
         * @brief Get current distribution scale parameter theta.
         *
         * @return Scale parameter theta of distribution.
         */
        double getTheta() { return _gamma.beta(); }

        /**
         * @brief Set current distribution shape parameter k.
         *
         * @param[in] k  New shape parameter for distribution. Behaviour for non-positive value
         *               is undefined.
         */
        void setK(double k) {
            _gamma.param(boost::random::gamma_distribution<>::param_type(k, _gamma.beta()));
        }

        /**
         * @brief Set current distribution scale parameter theta.
         *
         * @param[in] theta  New scale parameter for distribution.  Behavior for non-positive
         *                   value is undefined.
         */
        void setTheta(double theta) {
            _gamma.param(boost::random::gamma_distribution<>::param_type(_gamma.alpha(), theta));
        }

        /**
         * @brief Clear the internal cache
         */
        void clearCache() { _gamma.reset(); }

    protected:
        double _val() { return _gamma(*this->_rng); }
        std::string make_repr(bool incl_seed);

    private:
        // Note: confusingly, boost calls the internal values alpha and beta, even though they
        // don't conform to the normal beta=1/theta.  Rather, they have beta=theta.
        boost::random::gamma_distribution<> _gamma;
    };

    /**
     * @brief A Chi^2-distributed deviate with degrees-of-freedom parameter n.
     *
     * See http://en.wikipedia.org/wiki/Chi-squared_distribution (although note that in the Boost
     * random routine this class calls the notation is k=n for the number of degrees of freedom).
     * The Chi^2 distribution is a real valued distribution producing deviates >= 0.
     */
    class Chi2Deviate : public BaseDeviate
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
        Chi2Deviate(long lseed, double n) : BaseDeviate(lseed), _chi_squared(n) {}

        /**
         * @brief Construct a new Chi^2-distributed RNG, sharing the random number
         * generator with rhs.
         *
         * @param[in] rhs   Other deviate with which to share the RNG
         * @param[in] n     Number of degrees of freedom for the output distribution, must be > 0.
         */
        Chi2Deviate(const BaseDeviate& rhs, double n) : BaseDeviate(rhs), _chi_squared(n) {}

        /**
         * @brief Construct a copy that shares the RNG with rhs.
         *
         * Note: the default constructed op= function will do the same thing.
         */
        Chi2Deviate(const Chi2Deviate& rhs) : BaseDeviate(rhs), _chi_squared(rhs._chi_squared) {}

        /// @brief Construct a new Chi2Deviate from a serialization string
        Chi2Deviate(const std::string& str, double n) : BaseDeviate(str), _chi_squared(n) {}

        /**
         * @brief Construct a duplicate of this Chi2Deviate object.
         *
         * Both this and the returned duplicate will produce identical sequences of values.
         */
        Chi2Deviate duplicate()
        { return Chi2Deviate(serialize(),getN()); }

        /**
         * @brief Get current distribution degrees-of-freedom parameter n.
         *
         * @return Degrees-of-freedom parameter n of distribution.
         */
        double getN() { return _chi_squared.n(); }

        /**
         * @brief Set current distribution degrees-of-freedom parameter n.
         *
         * @param[in] n  New degrees-of-freedom parameter n for distribution. Behaviour for
         *               non-positive value is undefined.
         */
        void setN(double n) {
            _chi_squared.param(boost::random::chi_squared_distribution<>::param_type(n));
        }

        /**
         * @brief Clear the internal cache
         */
        void clearCache() { _chi_squared.reset(); }

    protected:
        double _val() { return _chi_squared(*this->_rng); }
        std::string make_repr(bool incl_seed);

    private:
        boost::random::chi_squared_distribution<> _chi_squared;
    };

}  // namespace galsim

#endif
