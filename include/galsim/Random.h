/**
 * @file Random.h 
 * 
 * @brief Random-number-generator classes
 *
 * Pseudo-random-number generators with various parent distributions: uniform, Gaussian, binomial,
 * and Poisson, all living within the galsim namespace. 
 * 
 * Wraps Boost.Random classes in a way that lets us swap Boost RNG's without affecting client code.
 */

#ifndef RANDOM_H
#define RANDOM_H

// Variable defined to use a private copy of Boost.Random, modified
// to avoid any reference to Boost.Random elements that might be on
// the local machine.
// Undefine this to use Boost.Random from the local distribution.
#define DIVERT_BOOST_RANDOM

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
     * UniformDeviate is the foundation of the Random.h classes: other distributions take a
     * UniformDeviate as construction argument and execute some transformation of the
     * distribution. Can be seeded with a long int, or by default will be seeded by the system
     * microsecond counter. Copy constructor and assignment operator are kept private since you
     * probably do not want two "random" number generators producing the same sequence of numbers in
     * your code! 
     */
    class UniformDeviate 
    // Note that this class could be templated with the type of Boost.Random generator that
    // you want to use instead of mt19937
    {
    public:
        /**
         * @brief Construct and seed a new UniformDeviate, using time of day as seed
         *
         * Note that microsecond counter is the seed, so UniformDeviates constructed in rapid
         * succession will not be independent. 
         */
        UniformDeviate(): urd(0.,1.) { seedtime(); } // seed with time

        /**
         * @brief Construct and seed a new UniformDeviate, using time of day as seed
         *
         * @param[in] lseed A long-integer seed for the RNG.
         */
        UniformDeviate(const long lseed): urng(lseed), urd(0.,1.) {} //seed with specific number

        /**
         * @brief Draw a new random number from the distribution
         *
         * @return A uniform deviate in the interval [0.,1.)
         */
        double operator() () { return urd(urng); }

        /**
         * @brief Re-seed the PRNG using current time
         */
        void seed() { seedtime(); }
        
        /**
         * @brief Re-seed the PRNG using specified seed
         *
         * @param[in] lseed A long-integer seed for the RNG.
         */
        void seed(const long lseed) { urng.seed(lseed); }

    private:
        boost::mt19937 urng;
        boost::random::uniform_real_distribution<> urd;

        /**
         * @brief Private routine to seed with microsecond counter from time-of-day structure.
         */
        void seedtime() 
        {
            struct timeval tp;
            gettimeofday(&tp,NULL);
            urng.seed(tp.tv_usec);
        }

        /**
         * @brief Hide copy and assignment so users do not create duplicate (correlated!) RNG's 
         */
        UniformDeviate(const UniformDeviate& rhs);

        /**
         * @brief Hide copy and assignment so users do not create duplicate (correlated!) RNG's
         */
        void operator=(const UniformDeviate& rhs);

        // make friends able to see the RNG without the distribution wrapper:
        friend class GaussianDeviate;
        friend class PoissonDeviate;
        friend class BinomialDeviate;

    };

    /**
     * @brief Pseudo-random number generator with Gaussian distribution.
     *
     * GaussianDeviate is constructed with reference to a UniformDeviate that will actually generate
     * the randoms, which are then transformed to Gaussian distribution with chosen mean and
     * standard deviation. Copy constructor and assignment operator are kept private since you
     * probably do not want two "random" number generators producing the same sequence of numbers in
     * your code!
     */
    //  Wraps the Boost.Random normal_distribution so that
    // the parent UniformDeviate is given once at construction, and copy/assignment are hidden.
    class GaussianDeviate 
    {
    public:

        /**
         * @brief Construct a new Gaussian-distributed RNG.
         *
         * Constructor requires reference to a UniformDeviate that generates the randoms, which are
         * then transformed to Gaussian distribution. 
         * @param[in] u_ UniformDeviate that will be called to generate all randoms
         * @param[in] mean  Mean of the output distribution
         * @param[in] sigma Standard deviation of the distribution
         */
        GaussianDeviate(UniformDeviate& u_, double mean=0., double sigma=1.) : 
	    u(u_), normal(mean,sigma) {}

        /**
         * @brief Draw a new random number from the distribution
         *
         * @return A Gaussian deviate with current mean and sigma
         */
        double operator() () { return normal(u.urng); }

        /**
         * @brief Get current distribution mean
         *
         * @return Mean of distribution
         */
        double getMean() {return normal.mean();}
        
        /**
         * @brief Get current distribution standard deviation
         *
         * @return Standard deviation of distribution
         */
        double getSigma() {return normal.sigma();}

        /**
         * @brief Set current distribution mean
         *
         * @param[in] mean New mean for distribution
         */
        void setMean(double mean) {
            normal.param(boost::random::normal_distribution<>::param_type(mean,normal.sigma()));
        }

        /**
         * @brief Set current distribution standard deviation
         *
         * @param[in] sigma New standard deviation for distribution.  Behavior for non-positive
         * value is undefined. 
         */
        void setSigma(double sigma) {
            normal.param(boost::random::normal_distribution<>::param_type(normal.mean(),sigma));
        }
        
    private:

        UniformDeviate& u;
        boost::random::normal_distribution<> normal;

        /**
         * @brief Hide copy and assignment so users do not create duplicate (correlated!) RNG's:
         */
        GaussianDeviate(const GaussianDeviate& rhs);
	/// Hide copy and assignment so users do not create duplicate (correlated!) RNG's:
        void operator=(const GaussianDeviate& rhs);
    };

    
    /**
     * @brief A Binomial deviate for N trials each of probability p.
     *
     * BinomialDeviate is constructed with reference to a UniformDeviate that will actually generate
     * the randoms, which are then transformed to Binomial distribution.  N is number of "coin
     * flips," p is probability of "heads," and each call returns integer 0<=value<=N giving number
     * of heads. Copy constructor and assignment operator are kept private since you probably do not
     * want two "random" number generators producing the same sequence of numbers in your code!
     */
    class BinomialDeviate 
    {
    public:

        /**
         * @brief Construct a new binomial-distributed RNG
         *
         * Constructor requires reference to a UniformDeviate that generates the randoms, which are
         * then transformed to Binomial distribution. 
         * @param[in] u_ UniformDeviate that will be called to generate all randoms
         * @param[in] N Number of "coin flips" per trial
         * @param[in] p Probability of success per coin flip.
         */
        BinomialDeviate(UniformDeviate& u_, const int N=1, const double p=0.5): 
        u(u_), bd(N,p) {}

        /**
         * @brief Draw a new random number from the distribution
         *
         * @return A binomial deviate with current N and p
         */
        int operator()() { return bd(u.urng); }

        /**
         * @brief Report current value of N
         *
         * @return Current value of N
         */
        int getN() {return bd.t();}

        /**
         * @brief Report current value of p
         *
         * @return Current value of p
         */
        double getP() {return bd.p();}

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

    private:
        UniformDeviate& u;
        boost::random::binomial_distribution<> bd;
        /// Hide copy and assignment so users do not create duplicate (correlated!) RNG's:
        BinomialDeviate(const BinomialDeviate& rhs);
        /// Hide copy and assignment so users do not create duplicate (correlated!) RNG's:
        void operator=(const BinomialDeviate& rhs);
    };

    /**
     * @brief A Poisson deviate with specified mean.
     *
     * PoissonDeviate is constructed with reference to a UniformDeviate that will actually generate
     * the randoms, which are then transformed to Poisson distribution.  Copy constructor and
     * assignment operator are kept private since you probably do not want two "random" number
     * generators producing the same sequence of numbers in your code!  
     */
    class PoissonDeviate 
    {
    public:

        /**
         * @brief Construct a new Poisson-distributed RNG 
         *
         * Constructor requires reference to a UniformDeviate that generates the randoms, which are
         * then transformed to Poisson distribution. 
         * @param[in] u_ UniformDeviate that will be called to generate all randoms
         * @param[in] mean Mean of the distribution
         */
        PoissonDeviate(UniformDeviate& u_, const double mean=1.): u(u_), pd(mean)  {}

        /**
         * @brief Draw a new random number from the distribution
         *
         * @return A Poisson deviate with current mean
         */
        int operator()() { return pd(u.urng); }

        /**
         * @brief Report current distribution mean
         * 
         * @return Current mean value
         */
        double getMean() {return pd.mean();}

        /**
         * @brief Reset distribution mean
         *
         * @param[in] mean New mean value
         */
        void setMean(double mean) {
            pd.param(boost::random::poisson_distribution<>::param_type(mean));
        }

    private:
        UniformDeviate& u;
        boost::random::poisson_distribution<> pd;
        /// Hide copy and assignment so users do not create duplicate (correlated!) RNG's:
        PoissonDeviate(const PoissonDeviate& rhs);
        /// Hide copy and assignment so users do not create duplicate (correlated!) RNG's:
        void operator=(const PoissonDeviate& rhs);
    };

}  // namespace galsim

#endif
