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

#ifndef RANDOM_H
#define RANDOM_H
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

// icpc pretends to be GNUC, since it thinks it's compliant, but it's not.
// It doesn't understand "pragma GCC"
#ifndef __INTEL_COMPILER

// There are some uninitialized values in boost.random stuff, which aren't a problem but
// sometimes confuse the compiler sufficiently that it emits a warning.
#if defined(__GNUC__) && __GNUC__ >= 4 && (__GNUC__ >= 5 || __GNUC_MINOR__ >= 2)
#pragma GCC diagnostic ignored "-Wuninitialized"
#endif

#endif


// Variable defined to use a private copy of Boost.Random, modified
// to avoid any reference to Boost.Random elements that might be on
// the local machine.
// Undefine this to use Boost.Random from the local distribution.
#define DIVERT_BOOST_RANDOM

#include "Image.h"
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
         * If lseed == 0, this means to use the time of day as the seed.  Note that microsecond 
         * counter is the seed, so BaseDeviates constructed in rapid succession may not be 
         * independent. 
         *
         * @param[in] lseed A long-integer seed for the RNG. (default=0)
         */
        BaseDeviate(long lseed=0) : _rng(new rng_type()) { seed(lseed); }

        /**
         * @brief Construct a new BaseDeviate, sharing the random number generator with rhs.
         */
        BaseDeviate(const BaseDeviate& rhs) : _rng(rhs._rng) {}

        /**
         * @brief Destructor
         *
         * Only deletes the underlying RNG if this is the last one using it.
         */
        virtual ~BaseDeviate() {}

        /**
         * @brief Re-seed the PRNG using specified seed
         *
         * If lseed == 0, this means to use the time of day as the seed.  Note that microsecond 
         * counter is the seed, so BaseDeviates constructed in rapid succession may not be 
         * independent. 
         *
         * @param[in] lseed A long-integer seed for the RNG. (default=0)
         *
         * Note that this will reseed all Deviates currently sharing the RNG with this one.
         */
        virtual void seed(long lseed);

        /**
         * @brief Like seed(lseed), but severs the relationship between other Deviates.
         *
         * Other Deviates that had been using the same RNG will be unaffected, while this 
         * Deviate will obtain a fresh RNG seeding by the current time.
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
         * @brief Draw a new random number from the distribution
         *
         * This is invalid for a BaseDeviate object that is not a derived class.
         * However, we don't make it pure virtual, since we want to be able to make
         * BaseDeviate objects a direct way to define a common seed for other Deviates.
         */
        double operator()() { return val(); }

   protected:

        boost::shared_ptr<rng_type> _rng;

        // This is the virtual function that is actually overridden.  This is because 
        // some derived classes prefer to return an int.  (e.g. Binom, Poisson)
        // So this provides the interface that returns a double.
        virtual double val() 
        { throw std::runtime_error("Cannot draw random values from a pure BaseDeviate object."); }

        /**
         * @brief Private routine to seed with microsecond counter from time-of-day structure.
         */
        void seedtime();
    };

    /**
     * @brief Pseudo-random number generator with uniform distribution in interval [0.,1.).
     */
    class UniformDeviate : public BaseDeviate
    {
    public:
        /** @brief Construct and seed a new UniformDeviate, using the provided value as seed.
         *
         * If lseed == 0, this means to use the time of day as the seed.  Note that microsecond 
         * counter is the seed, so BaseDeviates constructed in rapid succession may not be 
         * independent. 
         *
         * @param[in] lseed A long-integer seed for the RNG. (default=0)
         */
        UniformDeviate(long lseed=0) : BaseDeviate(lseed), _urd(0.,1.) {} 

        /// @brief Construct a new UniformDeviate, sharing the random number generator with rhs.
        UniformDeviate(const BaseDeviate& rhs) : BaseDeviate(rhs), _urd(0.,1.) {}

        /// @brief Construct a copy that shares the RNG with rhs.
        UniformDeviate(const UniformDeviate& rhs) : BaseDeviate(rhs), _urd(0.,1.) {}

        /**
         * @brief Draw a new random number from the distribution
         *
         * @return A uniform deviate in the interval [0.,1.)
         */
        double operator()() { return _urd(*this->_rng); }

        /**
         * @brief Clear the internal cache
         */
        void clearCache() { _urd.reset(); }

    protected:
        double val() { return operator()(); }

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
         * If lseed == 0, this means to use the time of day as the seed.  Note that microsecond 
         * counter is the seed, so BaseDeviates constructed in rapid succession may not be 
         * independent. 
         *
         * @param[in] lseed Seed to use. (default = 0)
         * @param[in] mean  Mean of the output distribution (default = 0)
         * @param[in] sigma Standard deviation of the distribution (default = 1)
         */
        GaussianDeviate(long lseed=0, double mean=0., double sigma=1.) : 
            BaseDeviate(lseed), _normal(mean,sigma) {}

        /**
         * @brief Construct a new Gaussian-distributed RNG, sharing the random number 
         * generator with rhs.
         *
         * @param[in] rhs   Other deviate with which to share the RNG
         * @param[in] mean  Mean of the output distribution
         * @param[in] sigma Standard deviation of the distribution
         */
        GaussianDeviate(const BaseDeviate& rhs, double mean=0., double sigma=1.) :
            BaseDeviate(rhs), _normal(mean,sigma) {}

        /**
         * @brief Construct a copy that shares the RNG with rhs.
         *
         * Note: the default constructed op= function will do the same thing.
         */
        GaussianDeviate(const GaussianDeviate& rhs) : BaseDeviate(rhs), _normal(rhs._normal) {}
 
        /**
         * @brief Draw a new random number from the distribution
         *
         * @return A Gaussian deviate with current mean and sigma
         */
        double operator()() { return _normal(*this->_rng); }

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
        double val() { return operator()(); }

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
         * If lseed == 0, this means to use the time of day as the seed.  Note that microsecond 
         * counter is the seed, so BaseDeviates constructed in rapid succession may not be 
         * independent. 
         *
         * @param[in] lseed Seed to use (default = 0)
         * @param[in] N Number of "coin flips" per trial (default = 1)
         * @param[in] p Probability of success per coin flip. (default = 0.5)
         */
        BinomialDeviate(long lseed=0, int N=1, double p=0.5) :
            BaseDeviate(lseed), _bd(N,p) {}

        /**
         * @brief Construct a new binomial-distributed RNG, sharing the random number 
         * generator with rhs.
         *
         * @param[in] rhs   Other deviate with which to share the RNG
         * @param[in] N Number of "coin flips" per trial
         * @param[in] p Probability of success per coin flip.
         */
        BinomialDeviate(const BaseDeviate& rhs, int N=1, double p=0.5) :
            BaseDeviate(rhs), _bd(N,p) {}

        /**
         * @brief Construct a copy that shares the RNG with rhs.
         *
         * Note: the default constructed op= function will do the same thing.
         */
        BinomialDeviate(const BinomialDeviate& rhs) : BaseDeviate(rhs), _bd(rhs._bd) {}

        /**
         * @brief Draw a new random number from the distribution
         *
         * @return A binomial deviate with current N and p
         */
        int operator()() { return _bd(*this->_rng); }

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
        double val() { return double(operator()()); }

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
         * If lseed == 0, this means to use the time of day as the seed.  Note that microsecond 
         * counter is the seed, so BaseDeviates constructed in rapid succession may not be 
         * independent. 
         *
         * @param[in] lseed Seed to use (default = 0)
         * @param[in] mean  Mean of the output distribution (default = 1)
         */
        PoissonDeviate(long lseed=0, double mean=1.) : BaseDeviate(lseed), _pd(mean) {}

        /**
         * @brief Construct a new Poisson-distributed RNG, sharing the random number 
         * generator with rhs.
         *
         * @param[in] rhs   Other deviate with which to share the RNG
         * @param[in] mean  Mean of the output distribution
         */
        PoissonDeviate(const BaseDeviate& rhs, double mean=1.) : BaseDeviate(rhs), _pd(mean) {}

        /**
         * @brief Construct a copy that shares the RNG with rhs.
         *
         * Note: the default constructed op= function will do the same thing.
         */
        PoissonDeviate(const PoissonDeviate& rhs) : BaseDeviate(rhs), _pd(rhs._pd) {}
 
        /**
         * @brief Draw a new random number from the distribution
         *
         * @return A Poisson deviate with current mean
         */
        int operator()() { return _pd(*this->_rng); }

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
        void setMean(double mean) {
            _pd.param(boost::random::poisson_distribution<>::param_type(mean));
        }

        /**
         * @brief Clear the internal cache
         */
        void clearCache() { _pd.reset(); }

    protected:
        double val() { return double(operator()()); }

    private:
        boost::random::poisson_distribution<> _pd;
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
         * If lseed == 0, this means to use the time of day as the seed.  Note that microsecond 
         * counter is the seed, so BaseDeviates constructed in rapid succession may not be 
         * independent. 
         *
         * @param[in] lseed Seed to use (default = 0)
         * @param[in] a    Shape parameter of the output distribution, must be > 0. (default = 1)
         * @param[in] b    Scale parameter of the distribution, must be > 0. (default = 1)
         */
        WeibullDeviate(long lseed=0, double a=1., double b=1.) : 
            BaseDeviate(lseed), _weibull(a,b) {}

        /**
         * @brief Construct a new Weibull-distributed RNG, sharing the random number 
         * generator with rhs.
         *
         * @param[in] rhs   Other deviate with which to share the RNG
         * @param[in] a    Shape parameter of the output distribution, must be > 0.
         * @param[in] b    Scale parameter of the distribution, must be > 0.
         */
        WeibullDeviate(const BaseDeviate& rhs, double a=1., double b=1.) :
            BaseDeviate(rhs), _weibull(a,b) {}

        /**
         * @brief Construct a copy that shares the RNG with rhs.
         *
         * Note: the default constructed op= function will do the same thing.
         */
        WeibullDeviate(const WeibullDeviate& rhs) : BaseDeviate(rhs), _weibull(rhs._weibull) {}

        /**
         * @brief Draw a new random number from the distribution.
         *
         * @return A Weibull deviate with current shape k and scale lam.
         */
        double operator()() { return _weibull(*this->_rng); }

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
        double val() { return operator()(); }

    private:
        boost::random::weibull_distribution<> _weibull;
    };

    /**
     * @brief A Gamma-distributed deviate with shape parameter k and scale parameter beta.
     *
     * See http://en.wikipedia.org/wiki/Gamma_distribution.  
     * (Note: we use the k, theta notation. If you prefer alpha, beta, use k=alpha, theta=1/beta.)
     * The Gamma distribution is a real valued distribution producing deviates >= 0.
     */
    class GammaDeviate : public BaseDeviate
    {
    public:
    
        /**
         * @brief Construct a new Gamma-distributed RNG, using the provided value as seed.
         *
         * If lseed == 0, this means to use the time of day as the seed.  Note that microsecond 
         * counter is the seed, so BaseDeviates constructed in rapid succession may not be 
         * independent. 
         *
         * @param[in] lseed  Seed to use (default = 0)
         * @param[in] k      Shape parameter of the output distribution, must be > 0. (default = 0)
         * @param[in] beta   Scale paramter of the distribution, must be > 0. (default = 1)
         */
        GammaDeviate(long lseed=0, double k=0., double beta=1.) : 
            BaseDeviate(lseed), _gamma(k,beta) {}

        /**
         * @brief Construct a new Gamma-distributed RNG, sharing the random number 
         * generator with rhs.
         *
         * @param[in] rhs    Other deviate with which to share the RNG
         * @param[in] k      Shape parameter of the output distribution, must be > 0.
         * @param[in] theta  Scale parameter of the distribution, must be > 0.
         */
        GammaDeviate(const BaseDeviate& rhs, double k=0., double theta=1.) :
            BaseDeviate(rhs), _gamma(k,theta) {}

        /**
         * @brief Construct a copy that shares the RNG with rhs.
         *
         * Note: the default constructed op= function will do the same thing.
         */
        GammaDeviate(const GammaDeviate& rhs) : BaseDeviate(rhs), _gamma(rhs._gamma) {}

        /**
         * @brief Draw a new random number from the distribution.
         *
         * @return A Gamma deviate with current shape k and scale theta.
         */
        double operator()() { return _gamma(*this->_rng); }

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
        double val() { return operator()(); }

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
         * If lseed == 0, this means to use the time of day as the seed.  Note that microsecond 
         * counter is the seed, so BaseDeviates constructed in rapid succession may not be 
         * independent. 
         *
         * @param[in] lseed Seed to use (default = 0)
         * @param[in] n     Number of degrees of freedom for the output distribution, must be > 0.
         *                  (default = 1)
         */
        Chi2Deviate(long lseed=0, double n=1.) : BaseDeviate(lseed), _chi_squared(n) {}

        /**
         * @brief Construct a new Chi^2-distributed RNG, sharing the random number 
         * generator with rhs.
         *
         * @param[in] rhs   Other deviate with which to share the RNG
         * @param[in] n     Number of degrees of freedom for the output distribution, must be > 0.
         */
        Chi2Deviate(const BaseDeviate& rhs, double n=1.) : BaseDeviate(rhs), _chi_squared(n) {}

        /**
         * @brief Construct a copy that shares the RNG with rhs.
         *
         * Note: the default constructed op= function will do the same thing.
         */
        Chi2Deviate(const Chi2Deviate& rhs) : BaseDeviate(rhs), _chi_squared(rhs._chi_squared) {}
 
        /**
         * @brief Draw a new random number from the distribution.
         *
         * @return A Chi^2 deviate with current degrees-of-freedom parameter n.
         */
        double operator()() { return _chi_squared(*this->_rng); }

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
        double val() { return operator()(); }

    private:
        boost::random::chi_squared_distribution<> _chi_squared;
    };

}  // namespace galsim

#endif
