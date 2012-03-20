// Random-number classes
// Will wrap Boost.Random classes in a way that lets us swap Boost RNG's without affecting
// client code.

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

    class UniformDeviate 
    // Note that this class could be templated with the type of Boost.Random generator that
    // you want to use instead of mt19937
    // Uniform wrapper is given arguments to map RNG's to [0.,1.) interval.
    {
    public:
        UniformDeviate(): urd(0.,1.) { seedtime(); } // seed with time
        UniformDeviate(const long lseed): urng(lseed), urd(0.,1.) {} //seed with specific number
        double operator() () { return urd(urng); }
        operator double() { return urd(urng); }
        void seed() { seedtime(); }
        void seed(const long lseed) { urng.seed(lseed); }

    private:
	boost::mt19937 urng;
        boost::random::uniform_real_distribution<> urd;
        void seedtime() 
        {
            struct timeval tp;
            gettimeofday(&tp,NULL);
            urng.seed(tp.tv_usec);
        }
	// Hide copy and assignment so users do not create duplicate (correlated!) RNG's:
	UniformDeviate(const UniformDeviate& rhs) {}
	void operator=(const UniformDeviate& rhs) {}

        // make friends able to see the RNG without the distribution wrapper:
        friend class GaussianDeviate;
        friend class PoissonDeviate;
        friend class BinomialDeviate;

    };

    // A Gaussian deviate (unit variance, zero-mean by default).
    //  Wraps the Boost.Random normal_distribution so that
    // the parent UniformDeviate is given once at construction, and copy/assignment are hidden.
    class GaussianDeviate 
    {
    public:
        GaussianDeviate(UniformDeviate& u_, double mean=0., double sigma=1.) : 
	    u(u_), normal(mean,sigma) {}
        double operator() () { return normal(u.urng); }
        operator double() { return normal(u.urng); }
	double getMean() {return normal.mean();}
	double getSigma() {return normal.sigma();}
	void setMean(double mean) {
	  normal.param(boost::random::normal_distribution<>::param_type(mean,
								normal.sigma()));
	}
	void setSigma(double sigma) {
	  normal.param(boost::random::normal_distribution<>::param_type(normal.mean(),
								sigma));
	}
    private:
        UniformDeviate& u;
        boost::random::normal_distribution<> normal;
	// Hide:
        GaussianDeviate(const GaussianDeviate& rhs): u(rhs.u) {}
        void operator=(const GaussianDeviate& rhs) {}
    };


    // A Binomial deviate for N trials each of probability p
    // Again, use Num. Recipes bnldev()
    class BinomialDeviate 
    {
    public:
        BinomialDeviate(UniformDeviate& u_, const int N=1, const double p=0.5): 
	  u(u_), bd(N,p) {}
        int operator()() { return bd(u.urng); }
        operator int() { return bd(u.urng); }
	int getN() {return bd.t();}
	double getP() {return bd.p();}
    private:
        UniformDeviate& u;
        boost::random::binomial_distribution<> bd;
        // Hide:
        BinomialDeviate(const BinomialDeviate& rhs): u(rhs.u) {}
        void operator=(const BinomialDeviate& rhs) {}
    };

    // Poisson deviate
    class PoissonDeviate 
    {
    public:
        //seed with time:
        PoissonDeviate(UniformDeviate& u_, const double mean=1.): u(u_), pd(mean)  {}
        // use supplied uniform deviate
        int operator()() { return pd(u.urng); }
        operator int() { return pd(u.urng); }
	double getMean() {return pd.mean();}
    private:
        UniformDeviate& u;
        boost::random::poisson_distribution<> pd;
        PoissonDeviate(const PoissonDeviate& rhs): u(rhs.u) {}
        void operator=(const PoissonDeviate& rhs) {}
    };

}  // namespace galsim

#endif
