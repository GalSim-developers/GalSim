// Random-number classes
// This is a version of Random.h that works with Boost releases <=1.47.  Is not being used right now
// But I am putting it into repository just in case something comes up later.

#ifndef RANDOM_H
#define RANDOM_H

#include <sys/time.h>

#include "boost/random/mersenne_twister.hpp"
#include "boost/random/normal_distribution.hpp"
#include "boost/random/binomial_distribution.hpp"
#include "boost/random/poisson_distribution.hpp"
#include "boost/random/uniform_real.hpp"
#include "boost/random/variate_generator.hpp"

namespace galsim {

    typedef boost::mt19937 OurURNG;

    class UniformDeviate 
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
        OurURNG urng;
        boost::uniform_real<> urd;
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
        typedef boost::variate_generator<OurURNG&, 
                boost::normal_distribution<> > GaussGenerator;
    public:
        GaussianDeviate(UniformDeviate& u_, double mean=0., double sigma=1.) : 
            u(u_), 
            normal(new boost::normal_distribution<>(mean,sigma)),
            gen(new GaussGenerator(u.urng,*normal))     {}
        ~GaussianDeviate() {if (gen) delete gen; if (normal) delete normal;}
        double operator() () { return (*gen)(); }
        operator double() { return (*gen)();}
        double getMean() {return normal->mean();}
        double getSigma() {return normal->sigma();}
        void setMean(double mean) {
            boost::normal_distribution<>* normal2 = 
                new boost::normal_distribution<>(mean,normal->sigma());
            delete normal;
            delete gen;
            normal = normal2;
            gen = new GaussGenerator(u.urng,*normal);
        }
        void setSigma(double sigma) {
            boost::normal_distribution<>* normal2 = 
                new boost::normal_distribution<>(normal->mean(),sigma);
            delete normal;
            delete gen;
            normal = normal2;
            gen = new GaussGenerator(u.urng,*normal);
        }
    private:
        UniformDeviate& u;
        boost::normal_distribution<>* normal;
        GaussGenerator* gen;
        // Hide:
        GaussianDeviate(const GaussianDeviate& rhs): u(rhs.u), gen(0), normal(0) {}
        void operator=(const GaussianDeviate& rhs) {}
    };

    // A Binomial deviate for N trials each of probability p
    class BinomialDeviate 
    {
        typedef boost::variate_generator<OurURNG&, 
                boost::binomial_distribution<> > BinomialGenerator;
    public:
        BinomialDeviate(UniformDeviate& u_, const int N=1, const double p=0.5): 
            u(u_), 
            binomial(new boost::binomial_distribution<>(N,p)),
            gen(new BinomialGenerator(u.urng,*binomial))     {}
        ~BinomialDeviate() {if (gen) delete gen; if (binomial) delete binomial;}
        int operator() () { return (*gen)(); }
        operator int() { return (*gen)();}
        int getN() {return binomial->t();}
        double getP() {return binomial->p();}
        void setN(int N) {
            boost::binomial_distribution<>* binomial2 = 
                new boost::binomial_distribution<>(N,binomial->p());
            delete binomial;
            delete gen;
            binomial = binomial2;
            gen = new BinomialGenerator(u.urng,*binomial);
        }
        void setP(double p) {
            boost::binomial_distribution<>* binomial2 = 
                new boost::binomial_distribution<>(binomial->t(),p);
            delete binomial;
            delete gen;
            binomial = binomial2;
            gen = new BinomialGenerator(u.urng,*binomial);
        }
    private:
        UniformDeviate& u;
        boost::binomial_distribution<>* binomial;
        BinomialGenerator* gen;
        // Hide:
        BinomialDeviate(const BinomialDeviate& rhs): u(rhs.u), gen(0), binomial(0) {}
        void operator=(const BinomialDeviate& rhs) {}
    };

    // Poisson deviate
    class PoissonDeviate 
    {
        typedef boost::variate_generator<OurURNG&, 
                boost::poisson_distribution<> > PoissonGenerator;
    public:
        PoissonDeviate(UniformDeviate& u_, const double mean=1.): 
            u(u_), 
            poisson(new boost::poisson_distribution<>(mean)),
            gen(new PoissonGenerator(u.urng,*poisson))     {}
        ~PoissonDeviate() {if (gen) delete gen; if (poisson) delete poisson;}
        int operator() () { return (*gen)(); }
        operator int() { return (*gen)();}
        int getMean() {return poisson->mean();}
        void setMean(double mean) {
            boost::poisson_distribution<>* poisson2 = 
                new boost::poisson_distribution<>(mean);
            delete poisson;
            delete gen;
            poisson = poisson2;
            gen = new PoissonGenerator(u.urng,*poisson);
        }
    private:
        UniformDeviate& u;
        boost::poisson_distribution<>* poisson;
        PoissonGenerator* gen;
        // Hide:
        PoissonDeviate(const PoissonDeviate& rhs): u(rhs.u), gen(0), poisson(0) {}
        void operator=(const PoissonDeviate& rhs) {}
    };

}  // namespace galsim

#endif
