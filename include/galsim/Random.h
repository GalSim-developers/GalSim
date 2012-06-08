// -*- c++ -*-
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

// Variable defined to use a private copy of Boost.Random, modified
// to avoid any reference to Boost.Random elements that might be on
// the local machine.
// Undefine this to use Boost.Random from the local distribution.
#define DIVERT_BOOST_RANDOM

#include <sys/time.h>
#include "Image.h"
#ifdef DIVERT_BOOST_RANDOM
#include "galsim/boost1_48_0.random/mersenne_twister.hpp"
#include "galsim/boost1_48_0.random/normal_distribution.hpp"
#include "galsim/boost1_48_0.random/binomial_distribution.hpp"
#include "galsim/boost1_48_0.random/poisson_distribution.hpp"
#include "galsim/boost1_48_0.random/uniform_real_distribution.hpp"
#include "galsim/boost1_48_0.random/weibull_distribution.hpp"
#include "galsim/boost1_48_0.random/gamma_distribution.hpp"
#include "galsim/boost1_48_0.random/chi_squared_distribution.hpp"
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
            for (ImIter it = data.rowBegin(y); it != ee; ++it) { *it += dev(); }
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
         * @brief Construct and seed a new BaseDeviate, using time of day as seed
         *
         * Note that microsecond counter is the seed, so BaseDeviates constructed in rapid
         * succession will not be independent. 
         */
        BaseDeviate() : _rng(new rng_type()) { seedtime(); } 

        /**
         * @brief Construct and seed a new BaseDeviate, using the provided value as seed.
         *
         * @param[in] lseed A long-integer seed for the RNG.
         */
        BaseDeviate(long lseed) : _rng(new rng_type(lseed)) {} 

        /**
         * @brief Construct a new BaseDeviate, sharing the random number generator with rhs.
         */
        BaseDeviate(const BaseDeviate& rhs) : _rng(rhs._rng) {} 

        /**
         * @brief Destructor
         *
         * Only deletes the underlying RNG if this is the last one using it.
         */
        ~BaseDeviate() {}

        /**
         * @brief Re-seed the PRNG using current time
         *
         * Note that this will reseed all Deviates currently sharing the RNG with this one.
         */
        void seed() { seedtime(); }

        /**
         * @brief Re-seed the PRNG using specified seed
         *
         * @param[in] lseed A long-integer seed for the RNG.
         *
         * Note that this will reseed all Deviates currently sharing the RNG with this one.
         */
        void seed(long lseed) { _rng->seed(lseed); }

        /**
         * @brief Like seed(), but severs the relationship between other Deviates.
         *
         * Other Deviates that had been using the same RNG will be unaffected, while this 
         * Deviate will obtain a fresh RNG seeding by the current time.
         */
        void reset() { _rng.reset(new rng_type()); seedtime(); }

        /**
         * @brief Like seed(lseed), but severs the relationship between other Deviates.
         *
         * Other Deviates that had been using the same RNG will be unaffected, while this 
         * Deviate will obtain a fresh RNG seeding by the current time.
         */
        void reset(long lseed) { _rng.reset(new rng_type(lseed)); }

        /**
         * @brief Make this object share its random number generator with another Deviate.
         *
         * It discards whatever rng it had been using and starts sharing the one held by dev.
         */
        void reset(const BaseDeviate& dev) { _rng = dev._rng; }

   protected:

        boost::shared_ptr<rng_type> _rng;

        /**
         * @brief Private routine to seed with microsecond counter from time-of-day structure.
         */
        void seedtime() 
        {
            struct timeval tp;
            gettimeofday(&tp,NULL);
            _rng->seed(tp.tv_usec);
        }
    };

    /**
     * @brief Pseudo-random number generator with uniform distribution in interval [0.,1.).
     */
    class UniformDeviate : public BaseDeviate
    {
    public:
        /// @brief Construct and seed a new UniformDeviate, using time of day as seed.
        UniformDeviate() : urd(0.,1.) {}

        /// @brief Construct and seed a new UniformDeviate, using the provided value as seed.
        UniformDeviate(long lseed) : BaseDeviate(lseed), urd(0.,1.) {} 

        /// @brief Construct a new UniformDeviate, sharing the random number generator with rhs.
        UniformDeviate(const BaseDeviate& rhs) : BaseDeviate(rhs), urd(0.,1.) {}

        /// @brief Construct a copy that shares the RNG with rhs.
        UniformDeviate(const UniformDeviate& rhs) : BaseDeviate(rhs), urd(0.,1.) {}

        /**
         * @brief Draw a new random number from the distribution
         *
         * @return A uniform deviate in the interval [0.,1.)
         */
        double operator() () { return urd(*this->_rng); }

        /**
         * @brief Add Uniform pseudo-random deviates to every element in a supplied Image.
         *
         * @param[in,out] data The Image to be noise-ified.
         */
        template <typename T>
        void applyTo(ImageView<T> data) { ApplyDeviateToImage(*this, data); }

    private:
        boost::random::uniform_real_distribution<> urd;
    };

    /**
     * @brief Pseudo-random number generator with Gaussian distribution.
     */
    class GaussianDeviate : public BaseDeviate
    {
    public:

        /**
         * @brief Construct a new Gaussian-distributed RNG, using time of day as seed.
         *
         * @param[in] mean  Mean of the output distribution
         * @param[in] sigma Standard deviation of the distribution
         */
        GaussianDeviate(double mean=0., double sigma=1.) : normal(mean,sigma) {}

        /**
         * @brief Construct a new Gaussian-distributed RNG, using the provided value as seed.
         *
         * @param[in] lseed Seed to use
         * @param[in] mean  Mean of the output distribution
         * @param[in] sigma Standard deviation of the distribution
         */
        GaussianDeviate(long lseed, double mean=0., double sigma=1.) : 
            BaseDeviate(lseed), normal(mean,sigma) {}

        /**
         * @brief Construct a new Gaussian-distributed RNG, sharing the random number 
         * generator with rhs.
         *
         * @param[in] rhs   Other deviate with which to share the RNG
         * @param[in] mean  Mean of the output distribution
         * @param[in] sigma Standard deviation of the distribution
         */
        GaussianDeviate(const BaseDeviate& rhs, double mean=0., double sigma=1.) :
            BaseDeviate(rhs), normal(mean,sigma) {}

        /**
         * @brief Construct a copy that shares the RNG with rhs.
         *
         * Note: the default constructed op= function will do the same thing.
         */
        GaussianDeviate(const GaussianDeviate& rhs) : BaseDeviate(rhs), normal(rhs.normal) {}
 
        /**
         * @brief Draw a new random number from the distribution
         *
         * @return A Gaussian deviate with current mean and sigma
         */
        double operator() () { return normal(*this->_rng); }

        /**
         * @brief Get current distribution mean
         *
         * @return Mean of distribution
         */
        double getMean() { return normal.mean(); }

        /**
         * @brief Get current distribution standard deviation
         *
         * @return Standard deviation of distribution
         */
        double getSigma() { return normal.sigma(); }

        /**
         * @brief Set current distribution mean
         *
         * @param[in] mean New mean for distribution
         */
        void setMean(double mean) 
        { normal.param(boost::random::normal_distribution<>::param_type(mean,normal.sigma())); }

        /**
         * @brief Set current distribution standard deviation
         *
         * @param[in] sigma New standard deviation for distribution.  Behavior for non-positive
         * value is undefined. 
         */
        void setSigma(double sigma) 
        { normal.param(boost::random::normal_distribution<>::param_type(normal.mean(),sigma)); }

        /**
         * @brief Add Gaussian pseudo-random deviates to every element in a supplied Image.
         *
         * @param[in,out] data The Image to be noise-ified.
         */
        template <typename T>
        void applyTo(ImageView<T> data) { ApplyDeviateToImage(*this, data); }

    private:
        boost::random::normal_distribution<> normal;
    };


    /**
     * @brief A Binomial deviate for N trials each of probability p.
     */
    class BinomialDeviate : public BaseDeviate
    {
    public:

        /**
         * @brief Construct a new binomial-distributed RNG, using time of day as seed.
         *
         * @param[in] N Number of "coin flips" per trial
         * @param[in] p Probability of success per coin flip.
         */
        BinomialDeviate(int N=1, double p=0.5) : bd(N,p) {}

        /**
         * @brief Construct a new binomial-distributed RNG, using the provided value as seed.
         *
         * @param[in] lseed Seed to use
         * @param[in] N Number of "coin flips" per trial
         * @param[in] p Probability of success per coin flip.
         */
        BinomialDeviate(long lseed, int N=1, double p=0.5) :
            BaseDeviate(lseed), bd(N,p) {}

        /**
         * @brief Construct a new binomial-distributed RNG, sharing the random number 
         * generator with rhs.
         *
         * @param[in] rhs   Other deviate with which to share the RNG
         * @param[in] N Number of "coin flips" per trial
         * @param[in] p Probability of success per coin flip.
         */
        BinomialDeviate(const BaseDeviate& rhs, int N=1, double p=0.5) :
            BaseDeviate(rhs), bd(N,p) {}

        /**
         * @brief Construct a copy that shares the RNG with rhs.
         *
         * Note: the default constructed op= function will do the same thing.
         */
        BinomialDeviate(const BinomialDeviate& rhs) : BaseDeviate(rhs), bd(rhs.bd) {}

        /**
         * @brief Draw a new random number from the distribution
         *
         * @return A binomial deviate with current N and p
         */
        int operator()() { return bd(*this->_rng); }

        /**
         * @brief Report current value of N
         *
         * @return Current value of N
         */
        int getN() { return bd.t(); }

        /**
         * @brief Report current value of p
         *
         * @return Current value of p
         */
        double getP() { return bd.p(); }

        /**
         * @brief Reset value of N
         *
         * @param[in] N New value of N
         */
        void setN(int N) {
            bd.param(boost::random::binomial_distribution<>::param_type(N,bd.p()));
        }

        /**
         * @brief Reset value of p
         *
         * @param[in] p New value of p
         */
        void setP(double p) {
            bd.param(boost::random::binomial_distribution<>::param_type(bd.t(),p));
        }

        /**
         * @brief Add Binomial pseudo-random deviates to every element in a supplied Image.
         *
         * @param[in,out] data The Image to be noise-ified.
         */
         template <typename T>
         void applyTo(ImageView<T> data) { ApplyDeviateToImage(*this, data); }


    private:
        boost::random::binomial_distribution<> bd;
    };

    /**
     * @brief A Poisson deviate with specified mean.
     */
    class PoissonDeviate : public BaseDeviate
    {
    public:

        /**
         * @brief Construct a new Poisson-distributed RNG, using time of day as seed.
         *
         * @param[in] mean  Mean of the output distribution
         */
        PoissonDeviate(double mean=1.) : pd(mean) {}

        /**
         * @brief Construct a new Poisson-distributed RNG, using the provided value as seed.
         *
         * @param[in] lseed Seed to use
         * @param[in] mean  Mean of the output distribution
         */
        PoissonDeviate(long lseed, double mean=1.) : BaseDeviate(lseed), pd(mean) {}

        /**
         * @brief Construct a new Poisson-distributed RNG, sharing the random number 
         * generator with rhs.
         *
         * @param[in] rhs   Other deviate with which to share the RNG
         * @param[in] mean  Mean of the output distribution
         */
        PoissonDeviate(const BaseDeviate& rhs, double mean=1.) : BaseDeviate(rhs), pd(mean) {}

        /**
         * @brief Construct a copy that shares the RNG with rhs.
         *
         * Note: the default constructed op= function will do the same thing.
         */
        PoissonDeviate(const PoissonDeviate& rhs) : BaseDeviate(rhs), pd(rhs.pd) {}
 
        /**
         * @brief Draw a new random number from the distribution
         *
         * @return A Poisson deviate with current mean
         */
        int operator()() { return pd(*this->_rng); }

        /**
         * @brief Report current distribution mean
         * 
         * @return Current mean value
         */
        double getMean() { return pd.mean(); }

        /**
         * @brief Reset distribution mean
         *
         * @param[in] mean New mean value
         */
        void setMean(double mean) {
            pd.param(boost::random::poisson_distribution<>::param_type(mean));
        }

        /**
         * @brief Add Poisson pseudo-random deviates to every element in a supplied Image.
         *
         * @param[in,out] data The Image to be noise-ified.
         */
        template <typename T>
        void applyTo(ImageView<T> data) { ApplyDeviateToImage(*this, data); }

    private:
        boost::random::poisson_distribution<> pd;
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
         * @brief Construct a new Weibull-distributed RNG, using time of day as seed.
         *
         * @param[in] a    Shape parameter of the output distribution, must be > 0.
         * @param[in] b    Scale parameter of the distribution, must be > 0.
         */
        WeibullDeviate(double a=1., double b=1.) : weibull(a,b) {}

        /**
         * @brief Construct a new Weibull-distributed RNG, using the provided value as seed.
         *
         * @param[in] lseed Seed to use
         * @param[in] a    Shape parameter of the output distribution, must be > 0.
         * @param[in] b    Scale parameter of the distribution, must be > 0.
         */
        WeibullDeviate(long lseed, double a=1., double b=1.) : 
            BaseDeviate(lseed), weibull(a,b) {}

        /**
         * @brief Construct a new Weibull-distributed RNG, sharing the random number 
         * generator with rhs.
         *
         * @param[in] rhs   Other deviate with which to share the RNG
         * @param[in] a    Shape parameter of the output distribution, must be > 0.
         * @param[in] b    Scale parameter of the distribution, must be > 0.
         */
        WeibullDeviate(const BaseDeviate& rhs, double a=1., double b=1.) :
            BaseDeviate(rhs), weibull(a,b) {}

        /**
         * @brief Construct a copy that shares the RNG with rhs.
         *
         * Note: the default constructed op= function will do the same thing.
         */
        WeibullDeviate(const WeibullDeviate& rhs) : BaseDeviate(rhs), weibull(rhs.weibull) {}

        /**
         * @brief Draw a new random number from the distribution.
         *
         * @return A Weibull deviate with current shape k and scale lam.
         */
        double operator() () { return weibull(*this->_rng); }

        /**
         * @brief Get current distribution shape parameter a.
         *
         * @return Shape parameter a of distribution.
         */
        double getA() { return weibull.a(); }

        /**
         * @brief Get current distribution scale parameter b.
         *
         * @return Scale parameter b of distribution.
         */
        double getB() { return weibull.b(); }

        /**
         * @brief Set current distribution shape parameter a.
         *
         * @param[in] a  New shape parameter for distribution. Behaviour for non-positive value
         * is undefined.
         */
        void setA(double a) {
            weibull.param(boost::random::weibull_distribution<>::param_type(a, weibull.b()));
        }

        /**
         * @brief Set current distribution scale parameter b.
         *
         * @param[in] b  New scale parameter for distribution.  Behavior for non-positive
         * value is undefined. 
         */
        void setB(double b) {
            weibull.param(boost::random::weibull_distribution<>::param_type(weibull.a(), b));
        }

        /**
         * @brief Add Weibull pseudo-random deviates to every element in a supplied Image.
         *
         * @param[in,out] data  The Image.
         */
        template <typename T>
        void applyTo(ImageView<T> data) { ApplyDeviateToImage(*this, data); }


    private:
        boost::random::weibull_distribution<> weibull;
    };

    /**
     * @brief A Gamma-distributed deviate with shape parameter alpha and scale parameter beta.
     *
     * See http://en.wikipedia.org/wiki/Gamma_distribution (although note that in the Boost random
     * routine this class calls the notation is alpha=k and beta=theta).  The Gamma distribution is
     * a real valued distribution producing deviates >= 0.
     */
    class GammaDeviate : public BaseDeviate
    {
    public:
    
        /**
         * @brief Construct a new Gamma-distributed RNG, using time of day as seed.
         *
         * @param[in] alpha  Shape parameter of the output distribution, must be > 0.
         * @param[in] beta   Scale parameter of the distribution, must be > 0.
         */
        GammaDeviate(double alpha=0., double beta=1.) : gamma(alpha,beta) {}

        /**
         * @brief Construct a new Gamma-distributed RNG, using the provided value as seed.
         *
         * @param[in] lseed  Seed to use
         * @param[in] alpha  Shape parameter of the output distribution, must be > 0.
         * @param[in] beta   Scale parameter of the distribution, must be > 0.
         */
        GammaDeviate(long lseed, double alpha=0., double beta=1.) : 
            BaseDeviate(lseed), gamma(alpha,beta) {}

        /**
         * @brief Construct a new Gamma-distributed RNG, sharing the random number 
         * generator with rhs.
         *
         * @param[in] rhs    Other deviate with which to share the RNG
         * @param[in] alpha  Shape parameter of the output distribution, must be > 0.
         * @param[in] beta   Scale parameter of the distribution, must be > 0.
         */
        GammaDeviate(const BaseDeviate& rhs, double alpha=0., double beta=1.) :
            BaseDeviate(rhs), gamma(alpha,beta) {}

        /**
         * @brief Construct a copy that shares the RNG with rhs.
         *
         * Note: the default constructed op= function will do the same thing.
         */
        GammaDeviate(const GammaDeviate& rhs) : BaseDeviate(rhs), gamma(rhs.gamma) {}

        /**
         * @brief Draw a new random number from the distribution.
         *
         * @return A Gamma deviate with current shape alpha and scale beta.
         */
        double operator() () { return gamma(*this->_rng); }

        /**
         * @brief Get current distribution shape parameter alpha.
         *
         * @return Shape parameter alpha of distribution.
         */
        double getAlpha() { return gamma.alpha(); }

        /**
         * @brief Get current distribution scale parameter beta.
         *
         * @return Scale parameter beta of distribution.
         */
        double getBeta() { return gamma.beta(); }

        /**
         * @brief Set current distribution shape parameter alpha.
         *
         * @param[in] alpha  New shape parameter for distribution. Behaviour for non-positive value
         *                   is undefined.
         */
        void setAlpha(double alpha) {
            gamma.param(boost::random::gamma_distribution<>::param_type(alpha, gamma.beta()));
        }

        /**
         * @brief Set current distribution scale parameter beta.
         *
         * @param[in] beta  New scale parameter for distribution.  Behavior for non-positive
         *                  value is undefined. 
         */
        void setBeta(double beta) {
            gamma.param(boost::random::gamma_distribution<>::param_type(gamma.alpha(), beta));
        }

        /**
         * @brief Add Gamma pseudo-random deviates to every element in a supplied Image.
         *
         * @param[in,out] data  The Image.
         */
        template <typename T>
        void applyTo(ImageView<T> data) { ApplyDeviateToImage(*this, data); }


    private:
        boost::random::gamma_distribution<> gamma;
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
         * @brief Construct a new Chi^2-distributed RNG, using time of day as seed.
         *
         * @param[in] n    Number of degrees of freedom for the output distribution, must be > 0.
         */
        Chi2Deviate(double n=1.) : chi_squared(n) {}

        /**
         * @brief Construct a new Chi^2-distributed RNG, using the provided value as seed.
         *
         * @param[in] lseed Seed to use
         * @param[in] n     Number of degrees of freedom for the output distribution, must be > 0.
         */
        Chi2Deviate(long lseed, double n=1.) : BaseDeviate(lseed), chi_squared(n) {}

        /**
         * @brief Construct a new Chi^2-distributed RNG, sharing the random number 
         * generator with rhs.
         *
         * @param[in] rhs   Other deviate with which to share the RNG
         * @param[in] n     Number of degrees of freedom for the output distribution, must be > 0.
         */
        Chi2Deviate(const BaseDeviate& rhs, double n=1.) : BaseDeviate(rhs), chi_squared(n) {}

        /**
         * @brief Construct a copy that shares the RNG with rhs.
         *
         * Note: the default constructed op= function will do the same thing.
         */
        Chi2Deviate(const Chi2Deviate& rhs) : BaseDeviate(rhs), chi_squared(rhs.chi_squared) {}
 
        /**
         * @brief Draw a new random number from the distribution.
         *
         * @return A Chi^2 deviate with current degrees-of-freedom parameter n.
         */
        double operator() () { return chi_squared(*this->_rng); }

        /**
         * @brief Get current distribution degrees-of-freedom parameter n.
         *
         * @return Degrees-of-freedom parameter n of distribution.
         */
        double getN() { return chi_squared.n(); }

        /**
         * @brief Set current distribution degrees-of-freedom parameter n.
         *
         * @param[in] n  New degrees-of-freedom parameter n for distribution. Behaviour for 
         *               non-positive value is undefined.
         */
        void setN(double n) {
            chi_squared.param(boost::random::chi_squared_distribution<>::param_type(n));
        }

        /**
         * @brief Add Chi^2 pseudo-random deviates to every element in a supplied Image.
         *
         * @param[in,out] data  The Image.
         */
        template <typename T>
        void applyTo(ImageView<T> data) { ApplyDeviateToImage(*this, data); }


    private:
        boost::random::chi_squared_distribution<> chi_squared;
    };

}  // namespace galsim

#endif
