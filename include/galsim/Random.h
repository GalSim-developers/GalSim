/** 
 * @file Random.h @brief Random-number-generator classes.
 * Pseudo-random-number generators with various parent distributions: uniform, Gaussian,
 * binomial, and Poisson.  
 *
 * Wraps Boost.Random classes in a way that lets us swap Boost RNG's without affecting
 * client code.
 */

#ifndef RANDOM_H
#define RANDOM_H

// Variable defined to use a private copy of Boost.Random, modified to avoid any reference to 
// Boost.Random elements that might be on the local machine.
// Undefine this to use Boost.Random from the local distribution.
#define DIVERT_BOOST_RANDOM ///< Undefine this to use Boost.Random from the local distribution.

#include <sys/time.h>
#ifdef DIVERT_BOOST_RANDOM
#include "galsim/boost1_48_0.random/mersenne_twister.hpp"
#include "galsim/boost1_48_0.random/normal_distribution.hpp"
#include "galsim/boost1_48_0.random/binomial_distribution.hpp"
#include "galsim/boost1_48_0.random/poisson_distribution.hpp"
#include "galsim/boost1_48_0.random/uniform_real_distribution.hpp"
#else
#include "boost/random/mersenne_twister.hpp"
#include "boost/random/normal_distribution.hpp"
#include "boost/random/binomial_distribution.hpp"
#include "boost/random/poisson_distribution.hpp"
#include "boost/random/uniform_real_distribution.hpp"
#endif
namespace galsim {

    /**
     * @brief Pseudo-random number generator with uniform distribution in interval [0.,1.).
     *
     * UniformDeviate is foundation of the Random.h classes: other distributions take a
     * UniformDeviate as construction argument and execute some transformation of the distribution. 
     * Can be seeded with a long int, or by default will be seeded by the system microsecond 
     * counter.
     * Copy constructor and assignment operator are kept private since you probably do not want two
     * "random" number generators producing the same sequence of numbers in your code!
     */
    class UniformDeviate 
    // Note that this class could be templated with the type of Boost.Random generator that
    // you want to use instead of mt19937
    {
    public:
        /** 
         * @brief Construct and seed a new UniformDeviate, using time of day as seed.
         *
         * Note that microsecond counter is the seed, so UniformDeviates constructed in rapid
         * succession will not be independent.
         */
        UniformDeviate(): urd(0.,1.) { seedtime(); } // seed with time

        /** 
         * @brief Construct and seed a new UniformDeviate, seeded with a specific number.
         *
         * @param[in] lseed A long-integer seed for the RNG.
         */
        UniformDeviate(const long lseed): urng(lseed), urd(0.,1.) {} //seed with specific number

        /** 
         * Draw a new random number from the distribution.
         *
         * @returns A uniform deviate in the interval [0.,1.).
         */
        double operator() () { return urd(urng); }

        /** 
         * @brief Cast to double draws a new random number from the distribution.
         *
	 * Cast operator allows you to simply use your UniformDeviate instance in arithmetic 
	 * assignments and every appearance will be replaced with a new deviate.
         * @returns A uniform deviate in the interval [0.,1.)
         */
        operator double() { return urd(urng); }

	/// @brief Re-seed the PRNG using current time.
        void seed() { seedtime(); }

	/** 
         * @brief Re-seed the PRNG using specified seed.
         *
         * @param lseed A long-integer seed for the RNG.
         */
        void seed(const long lseed) { urng.seed(lseed); }

    private:
	boost::mt19937 urng;
        boost::random::uniform_real_distribution<> urd;
	/// @brief Private routine to seed with microsecond counter from time-of-day structure.
        void seedtime() 
        {
            struct timeval tp;
            gettimeofday(&tp,NULL);
            urng.seed(tp.tv_usec);
        }

	/// @brief Hide copy and assignment so users do not create duplicate (correlated!) RNG's.
	UniformDeviate(const UniformDeviate& rhs) {}

	/// @brief Hide copy and assignment so users do not create duplicate (correlated!) RNG's.
	void operator=(const UniformDeviate& rhs) {}

        // make friends able to see the RNG without the distribution wrapper:
        friend class GaussianDeviate;
        friend class PoissonDeviate;
        friend class BinomialDeviate;

    };

    /** 
     * \@brief Pseudo-random number generator with Gaussian distribution.
     *
     * GaussianDeviate is constructed with reference to a UniformDeviate that will actually generate
     * the randoms, which are then transformed to Gaussian distribution with chosen mean and
     * standard deviation.
     * Copy constructor and assignment operator are kept private since you probably do not want two
     * "random" number generators producing the same sequence of numbers in your code!
     *
     *  Wraps the Boost.Random normal_distribution so that the parent UniformDeviate is given once 
     *  at construction, and copy/assignment are hidden.
     */
    class GaussianDeviate 
    {
    public:

        /** 
         * @brief Construct a new Gaussian-distributed RNG. 
         *
         * Constructor requires reference to a UniformDeviate that generates the randoms, which
         * are then transformed to a Gaussian distribution.
         * @param[in,out] u_ UniformDeviate that will be called to generate all randoms.
         * @param[in] mean   Mean of the output distribution (default `mean = 0.`).
         * @param[in] sigma  Standard deviation of the distribution (default `sigma = 1.`).
         */
        GaussianDeviate(UniformDeviate& u_, double mean=0., double sigma=1.) : 
	    u(u_), normal(mean,sigma) {}

        /** 
         * @brief Draw a new random number from the distribution.
         *
         * @returns A Gaussian deviate with current mean and sigma.
         */
        double operator() () { return normal(u.urng); }

        /** 
         * @brief Cast to double draws a new random number from the distribution.
         *
	 * Cast operator allows you to simply use your GaussianDeviate instance in arithmetic 
	 * assignments and every appearance will be replaced with a new deviate.
         * @returns A Gaussian deviate with current mean and sigma.
         */
        operator double() { return normal(u.urng); }

	/** 
         * @brief Get current distribution mean.
	 *
	 * @returns Mean of distribution.
	 */
	double getMean() {return normal.mean();}

	/** 
         * @brief Get current distribution standard deviation.
	 *
	 * @returns Standard deviation of distribution.
	 */
	double getSigma() {return normal.sigma();}

	/** 
         * @brief Set current distribution mean.
	 *
	 * @param[in]  mean New mean for distribution
	 */
	void setMean(double mean) {
	  normal.param(boost::random::normal_distribution<>::param_type(mean,
								normal.sigma()));
	}

	/** 
         * @brief Set current distribution standard deviation
	 *
	 * @param[in] sigma New standard deviation for distribution.  Behavior for non-positive 
	 * value is undefined.
	 */
	void setSigma(double sigma) {
	  normal.param(boost::random::normal_distribution<>::param_type(normal.mean(),
								sigma));
	}
    private:
        UniformDeviate& u;
        boost::random::normal_distribution<> normal;

	/// @brief Hide copy and assignment so users do not create duplicate (correlated!) RNG's.
        GaussianDeviate(const GaussianDeviate& rhs): u(rhs.u) {}

	/// @brief Hide copy and assignment so users do not create duplicate (correlated!) RNG's.
        void operator=(const GaussianDeviate& rhs) {}
    };


    /** 
     * @brief A Binomial deviate for N trials each of probability p.
     *
     * BinomialDeviate is constructed with reference to a UniformDeviate that will actually generate
     * the randoms, which are then transformed to Binomial distribution.  N is number of "coin 
     * flips," p is probability of "heads," and each call returns integer 0<=value<=N giving number
     * of heads.  Copy constructor and assignment operator are kept private since you probably do 
     * not want two "random" number generators producing the same sequence of numbers in your code!
     */
    class BinomialDeviate 
    {
    public:

        /** 
         * @brief Construct a new binomial-distributed RNG 
         *
         * Constructor requires reference to a UniformDeviate that generates the randoms, which
         * are then transformed to Binomial distribution.
         * @param u_  UniformDeviate that will be called to generate all randoms.
         * @param N   Number of "coin flips" per trial (default `N = 1`).
         * @param p   Probability of success per coin flip (default `p = 0.5`).
         */
        BinomialDeviate(UniformDeviate& u_, const int N=1, const double p=0.5): 
	  u(u_), bd(N,p) {}

        /** 
         * @brief Draw a new random number from the distribution.
         *
         * @returns A binomial deviate with current N and p.
         */
        int operator()() { return bd(u.urng); }

        /** 
         * @brief Cast to int draws a new random number from the distribution.
         *
	 * Cast operator allows you to simply use your BinomialDeviate instance in arithmetic 
	 * assignments and every appearance will be replaced with a new deviate.
         * @returns A binomial deviate with current N and p.
         */
        operator int() { return bd(u.urng); }

	/** 
         * @brief Report current value of N.
	 *
	 * @returns Current value of N.
	 */
	int getN() {return bd.t();}

	/**
         * @brief Report current value of p.
	 *
	 * @returns Current value of p.
	 */
	double getP() {return bd.p();}

	/**
         * @brief Reset value of N.
	 *
	 * @param[in] N New value of N.
	 */
	void setN(int N) {
	    bd.param(boost::random::binomial_distribution<>::param_type(N,
								        bd.p()));
	}
	
        /** 
         * @brief Reset value of p.
	 *
	 * @param[in] p New value of p.
	 */
	void setP(double p) {
	    bd.param(boost::random::binomial_distribution<>::param_type(bd.t(),
									p));
	}
    private:
        UniformDeviate& u;
        boost::random::binomial_distribution<> bd;

	/// @brief Hide copy and assignment so users do not create duplicate (correlated!) RNG's.
        BinomialDeviate(const BinomialDeviate& rhs): u(rhs.u) {}

	/// @brief Hide copy and assignment so users do not create duplicate (correlated!) RNG's.
        void operator=(const BinomialDeviate& rhs) {}
    };

    /** 
     * @brief A Poisson deviate with specified mean.
     *
     * PoissonDeviate is constructed with reference to a UniformDeviate that will actually generate
     * the randoms, which are then transformed to Poisson distribution.  
     * Copy constructor and assignment operator are kept private since you probably do not want two
     * "random" number generators producing the same sequence of numbers in your code!
     */
    class PoissonDeviate 
    {
    public:

        /**
         * @brief Construct a new Poisson-distributed RNG. 
         *
         * Constructor requires reference to a UniformDeviate that generates the randoms, which
         * are then transformed to Poisson distribution.
         * @param[in] u_    UniformDeviate that will be called to generate all randoms.
         * @param[in] mean  Mean of the distribution (default `mean = 1.`).
         */
        PoissonDeviate(UniformDeviate& u_, const double mean=1.): u(u_), pd(mean)  {}

        /** 
         * @brief Draw a new random number from the distribution.
         *
         * @returns A Poisson deviate with current mean.
         */
        int operator()() { return pd(u.urng); }

        /** 
         * @brief Cast to int draws a new random number from the distribution.
         *
	 * Cast operator allows you to simply use your PoissonDeviate instance in arithmetic 
	 * assignments and every appearance will be replaced with a new deviate.
         * @returns A binomial deviate with current mean.
         */
        operator int() { return pd(u.urng); }

	/** 
         * @brief Report current distribution mean.
	 *
	 * @returns Current mean value.
	 */
	double getMean() {return pd.mean();}

        /** 
         * @brief Reset distribution mean.
	 *
	 * @param[in] mean New mean value.
	 */
	void setMean(double mean) {
	  pd.param(boost::random::poisson_distribution<>::param_type(mean));
	}
    private:
        UniformDeviate& u;
        boost::random::poisson_distribution<> pd;

	/// @brief Hide copy and assignment so users do not create duplicate (correlated!) RNG's.
        PoissonDeviate(const PoissonDeviate& rhs): u(rhs.u) {}

	/// @brief Hide copy and assignment so users do not create duplicate (correlated!) RNG's.
        void operator=(const PoissonDeviate& rhs) {}
    };

}  // namespace galsim

#endif
