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
#else
#include "boost/random/mersenne_twister.hpp"
#include "boost/random/normal_distribution.hpp"
#include "boost/random/binomial_distribution.hpp"
#include "boost/random/poisson_distribution.hpp"
#include "boost/random/uniform_real_distribution.hpp"
#include "boost/random/weibull_distribution.hpp"
#include "boost/random/gamma_distribution.hpp"
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

        /**
         * @brief Add Uniform pseudo-random deviates to every element in a supplied Image.
         *
         * @param[in,out] data The Image to be noise-ified.
         */
        template <typename T>
        void applyTo(ImageView<T> data) {
            // typedef for image row iterable
            typedef typename ImageView<T>::iterator ImIter;

            for (int y = data.getYMin(); y <= data.getYMax(); y++) {  // iterate over y
                ImIter ee = data.rowEnd(y);
                for (ImIter it = data.rowBegin(y); it != ee; ++it) { *it += (*this)(); }
            }
        }
            

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
        friend class WeibullDeviate;
        friend class GammaDeviate;

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

        /**
         * @brief Add Gaussian pseudo-random deviates to every element in a supplied Image.
         *
         * @param[in,out] data The Image to be noise-ified.
         */
        template <typename T>
        void applyTo(ImageView<T> data) {
            // Typedef for image row iterable
            typedef typename ImageView<T>::iterator ImIter;

            for (int y = data.getYMin(); y <= data.getYMax(); y++) {  // iterate over y
                ImIter ee = data.rowEnd(y);
                for (ImIter it = data.rowBegin(y); it != ee; ++it) { *it += (*this)(); }
            }
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

        /**
         * @brief Add Binomial pseudo-random deviates to every element in a supplied Image.
         *
         * @param[in,out] data The Image to be noise-ified.
         */
        template <typename T>
        void applyTo(ImageView<T> data) {
            // Typedef for image row iterable
            typedef typename ImageView<T>::iterator ImIter;

            for (int y = data.getYMin(); y <= data.getYMax(); y++) {  // iterate over y
                ImIter ee = data.rowEnd(y);
                for (ImIter it = data.rowBegin(y); it != ee; ++it) { *it += (*this)(); }
            }
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

        /**
         * @brief Add Poisson pseudo-random deviates to every element in a supplied Image.
         *
         * @param[in,out] data The Image to be noise-ified.
         */
        template <typename T>
        void applyTo(ImageView<T> data) {
            // Typedef for image row iterable
            typedef typename ImageView<T>::iterator ImIter;

            for (int y = data.getYMin(); y <= data.getYMax(); y++) {  // iterate over y
                ImIter ee = data.rowEnd(y);
                for (ImIter it = data.rowBegin(y); it != ee; ++it) { *it += (*this)(); }
            }
        }

    private:
        UniformDeviate& u;
        boost::random::poisson_distribution<> pd;
        /// Hide copy and assignment so users do not create duplicate (correlated!) RNG's:
        PoissonDeviate(const PoissonDeviate& rhs);
        /// Hide copy and assignment so users do not create duplicate (correlated!) RNG's:
        void operator=(const PoissonDeviate& rhs);
    };

    /**
     * @brief A Weibull-distributed deviate with shape parameter a and scale parameter b.
     *
     * The Weibull distribution is related to a number of other probability distributions; in 
     * particular, it interpolates between the exponential distribution (a=1) and the Rayleigh 
     * distribution (a=2). See http://en.wikipedia.org/wiki/Weibull_distribution (a=k and b=lambda
     * in the notation adopted in the Wikipedia article).  The Weibull distribution is a real valued
     * distribution producing deviates >= 0.
     *
     * WeibullDeviate is constructed with reference to a UniformDeviate that will actually generate
     * the randoms, which are then transformed to Weibull distribution.  Copy constructor and
     * assignment operator are kept private since you probably do not want two "random" number
     * generators producing the same sequence of numbers in your code!  
     *
     */
    class WeibullDeviate 
    {
    public:
        /**
         * @brief Construct a new Weibull-distributed RNG.
         *
         * Constructor requires reference to a UniformDeviate that generates the randoms, which are
         * then transformed to Weibull distribution. 
         * @param[in] u_   UniformDeviate that will be called to generate all randoms
         * @param[in] a    Shape parameter of the output distribution, must be > 0.
         * @param[in] b    Scale parameter of the distribution, must be > 0.
         */
        WeibullDeviate(UniformDeviate& u_, double a=1., double b=1.) : 
            u(u_), weibull(a, b) {}

        /**
         * @brief Draw a new random number from the distribution.
         *
         * @return A Weibull deviate with current shape k and scale lam.
         */
        double operator() () { return weibull(u.urng); }

        /**
         * @brief Get current distribution shape parameter a.
         *
         * @return Shape parameter a of distribution.
         */
        double getA() {return weibull.a();}

        /**
         * @brief Get current distribution scale parameter b.
         *
         * @return Scale parameter b of distribution.
         */
        double getB() {return weibull.b();}

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
        void applyTo(ImageView<T>& data) {
            // Typedef for image row iterable
            typedef typename ImageView<T>::iterator ImIter;

            for (int y = data.getYMin(); y <= data.getYMax(); y++) {  // iterate over y
                ImIter ee = data.rowEnd(y);
                for (ImIter it = data.rowBegin(y); it != ee; ++it) { *it += (*this)(); }
            }
        }

    private:

        UniformDeviate& u;
        boost::random::weibull_distribution<> weibull;

        /**
         * @brief Hide copy and assignment so users do not create duplicate (correlated!) RNG's:
         */
        WeibullDeviate(const WeibullDeviate& rhs);
        /// Hide copy and assignment so users do not create duplicate (correlated!) RNG's:
        void operator=(const WeibullDeviate& rhs);
    };

    /**
     * @brief A Gamma-distributed deviate with shape parameter alpha and scale parameter beta.
     *
     * See http://en.wikipedia.org/wiki/Gamma_distribution (although note that in the Boost random
     * routine this class calls the notation is such that alpha=k and beta=theta).  The Gamma 
     * distribution is a real valued distribution producing deviates >= 0.
     *
     * GammaDeviate is constructed with reference to a UniformDeviate that will actually generate
     * the randoms, which are then transformed to Gamma distribution.  Copy constructor and
     * assignment operator are kept private since you probably do not want two "random" number
     * generators producing the same sequence of numbers in your code!  
     *
     */
    class GammaDeviate 
    {
    public:
        /**
         * @brief Construct a new Gamma-distributed RNG.
         *
         * Constructor requires reference to a UniformDeviate that generates the randoms, which are
         * then transformed to Gamma distribution. 
         * @param[in] u_     UniformDeviate that will be called to generate all randoms
         * @param[in] alpha  Shape parameter of the output distribution, must be > 0.
         * @param[in] beta   Scale parameter of the distribution, must be > 0.
         */
        GammaDeviate(UniformDeviate& u_, double alpha=1., double beta=1.) : 
            u(u_), gamma(alpha, beta) {}

        /**
         * @brief Draw a new random number from the distribution.
         *
         * @return A Gamma deviate with current shape alpha and scale beta.
         */
        double operator() () { return gamma(u.urng); }

        /**
         * @brief Get current distribution shape parameter alpha.
         *
         * @return Shape parameter alpha of distribution.
         */
        double getAlpha() {return gamma.alpha();}

        /**
         * @brief Get current distribution scale parameter beta.
         *
         * @return Scale parameter beta of distribution.
         */
        double getBeta() {return gamma.beta();}

        /**
         * @brief Set current distribution shape parameter alpha.
         *
         * @param[in] alpha  New shape parameter for distribution. Behaviour for non-positive value
         *                   is undefined.
         */
        void setA(double alpha) {
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
        void applyTo(ImageView<T>& data) {
            // Typedef for image row iterable
            typedef typename ImageView<T>::iterator ImIter;

            for (int y = data.getYMin(); y <= data.getYMax(); y++) {  // iterate over y
                ImIter ee = data.rowEnd(y);
                for (ImIter it = data.rowBegin(y); it != ee; ++it) { *it += (*this)(); }
            }
        }

    private:

        UniformDeviate& u;
        boost::random::gamma_distribution<> gamma;

        /**
         * @brief Hide copy and assignment so users do not create duplicate (correlated!) RNG's:
         */
        GammaDeviate(const GammaDeviate& rhs);
        /// Hide copy and assignment so users do not create duplicate (correlated!) RNG's:
        void operator=(const GammaDeviate& rhs);
    };

}  // namespace galsim

#endif
