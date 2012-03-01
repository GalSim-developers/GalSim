
// Random-number classes

#ifndef RANDOM_H
#define RANDOM_H

#ifndef PI
#define PI 3.141592654
#endif

#include <sys/time.h>
#include <complex>

#include "Std.h"

namespace galsim {

    class UniformDeviate 
    {
    public:
        UniformDeviate() { seedtime(); } // seed with time
        UniformDeviate(const long lseed) { seed(lseed); } //seed with specific number
        float operator() () { return newvalue(); }
        operator float() { return newvalue(); }
        void Seed() { seedtime(); }
        void Seed(const long lseed) { seed(lseed); }

    private:
        void seedtime() 
        {
            struct timeval tp;
            gettimeofday(&tp,NULL);
            seed(tp.tv_usec);
        }

        // The random number seed and generator are Numerical Recipes ran1.c
#define IA 16807
#define IM 2147483647
#define AM (1.0/IM)
#define IQ 127773
#define IR 2836
#define NTAB 32
#define NDIV (1+(IM-1)/NTAB)
#define EPS 1.2e-7
#define RNMX (1.0-EPS)
        long idum;
        long iy;
        long iv[NTAB];

        void seed(long lseed) 
        {
            long k;
            idum = lseed;
            if (idum <= 0) idum=1;
            for (int j=NTAB+7;j>=0;j--) {
                k=idum/IQ;
                idum=IA*(idum-k*IQ)-IR*k;
                if (idum < 0) idum += IM;
                if (j < NTAB) iv[j] = idum;
            }
            iy=iv[0];
        }

        float newvalue() 
        {
            int j;
            long k;
            float temp;

            k=idum/IQ;
            idum=IA*(idum-k*IQ)-IR*k;
            if (idum < 0) idum += IM;
            j=iy/NDIV;
            iy=iv[j];
            iv[j] = idum;
            if ((temp=AM*iy) > RNMX) return RNMX;
            else return temp;
        }
#undef IA
#undef IM
#undef AM
#undef IQ
#undef IR
#undef NTAB
#undef NDIV
#undef EPS
#undef RNMX
    };

    // A unit-Gaussian deviate
    class GaussianDeviate 
    {
    public:
        //seed with time:
        GaussianDeviate() : ownedU(new UniformDeviate), u(*ownedU), iset(0) {}; 
        //seed with specific number:
        GaussianDeviate(const long lseed) : 
            ownedU(new UniformDeviate(lseed)), u(*ownedU), iset(0) {} 
        // Use a supplied uniform deviate:
        GaussianDeviate(UniformDeviate& u_) : ownedU(0), u(u_), iset(0) {}
        ~GaussianDeviate() { if (ownedU) delete ownedU; ownedU=0; }

        float operator() () { return newvalue(); }
        operator float() { return newvalue(); }
        void Seed() { u.Seed(); iset=0; }
        void Seed(const long lseed) { u.Seed(lseed); iset=0; }

    private:
        UniformDeviate* ownedU;
        UniformDeviate& u;
        int iset;
        float gset;

        // Gaussian deviate from uniform deviate, gasdev.c in Numerical Recipes
        float newvalue() 
        {
            float fac,rsq,v1,v2;
            if  (iset == 0) {
                do {
                    v1=2.0*u-1.0;
                    v2=2.0*u-1.0;
                    rsq=v1*v1+v2*v2;
                } while (rsq >= 1.0 || rsq == 0.0);
                fac=sqrt(-2.0*log(rsq)/rsq);
                gset=v1*fac;
                iset=1;
                return v2*fac;
            } else {
                iset=0;
                return gset;
            }
        }
    };

    //Complex Gaussian deviate, unit dispersion each real & imaginary
    class CGaussianDeviate 
    {
    public:
        CGaussianDeviate() : g() {}; //seed with time
        CGaussianDeviate(const long lseed) : g(lseed) {} //seed with specific number
        CGaussianDeviate(UniformDeviate& u_) : g(u_) {} //use supplied uniform deviate

        std::complex<double> operator() () { return newvalue(); }
        operator std::complex<double>() { return newvalue(); }
        void Seed() { g.Seed(); }
        void Seed(const long lseed) { g.Seed(lseed); }

    private:
        GaussianDeviate g;
        std::complex<double> newvalue() { return std::complex<double>(g(), g()); }
    };

    // A unit-mean exponential deviate
    class ExponentialDeviate 
    {
    public:
        //seed with time:
        ExponentialDeviate() : ownedU(new UniformDeviate), u(*ownedU) {}; 
        //seed with specific number:
        ExponentialDeviate(const long lseed) : ownedU(new UniformDeviate(lseed)), u(*ownedU) {} 
        // Use a supplied uniform deviate:
        ExponentialDeviate(UniformDeviate& u_) : ownedU(0), u(u_) {}
        ~ExponentialDeviate() { if (ownedU) delete ownedU; ownedU=0; }

        float operator() () { return newvalue(); }
        operator float() { return newvalue(); }
        void Seed() { u.Seed(); }
        void Seed(const long lseed) { u.Seed(lseed); }
    private:
        UniformDeviate* ownedU;
        UniformDeviate& u;

        // exp deviate from uniform - expdev.c in Num. Recipes
        float newvalue() 
        {
            float dum;
            do
                dum = u();
            while (dum==0.);
            return -log(dum);
        }
    };


    // NR's log-of-gamma function
    class Gammln 
    {
    public:
        float operator()(float xx) 
        {
            double x,y,tmp,ser;
            static double cof[6]={ 
                76.18009172947146,-86.50532032941677,
                24.01409824083091,-1.231739572450155,
                0.1208650973866179e-2,-0.5395239384953e-5 };
            int j;

            y=x=xx;
            tmp=x+5.5;
            tmp -= (x+0.5)*log(tmp);
            ser=1.000000000190015;
            for (j=0;j<=5;j++) ser += cof[j]/++y;
            return -tmp+log(2.5066282746310005*ser/x);
        }
    };

    // A Binomial deviate for N trials each of probability p
    // Again, use Num. Recipes bnldev()
    class BinomialDeviate 
    {
    public:
        //seed with time:
        BinomialDeviate(const int _N, const double _p) : 
            ownedU(new UniformDeviate), u(*ownedU), N(-1) 
        { init(_N,_p); }

        //seed with specific number
        BinomialDeviate(const int _N, const double _p, const long lseed) : 
            ownedU(new UniformDeviate(lseed)), u(*ownedU), N(-1) 
        { init(_N,_p); } 

        //Use supplied uniform deviate:
        BinomialDeviate(const int _N, const double _p, UniformDeviate& u_) : 
            ownedU(0), u(u_), N(-1) 
        { init(_N,_p); } 

        ~BinomialDeviate() { if (ownedU) { delete ownedU; ownedU=0; } }

        //Reset the parameters
        void Reset(const int _N, const double _p) { init(_N,_p); }
        int operator()() { return newvalue(); }
        operator int() { return newvalue(); }
        void Seed() { u.Seed(); }
        void Seed(const long lseed) { u.Seed(lseed); }

    private:
        Gammln gammln;
        UniformDeviate* ownedU;
        UniformDeviate& u;
        int N; //# of trials
        float p; //prob per trial
        bool flip; // true if we inverted p -> 1-p
        float pc, plog, pclog, en, oldg; //constants used

        void init(const int _N, const double _p) 
        {
            if (N==_N && p==_p) return;
            N = _N;
            p = _p;

            if (p>0.5) {
                p = 1.-p;
                flip = true;
            } else {
                flip = false;
            }
            en=N;
            oldg=gammln(en+1.0);
            pc=1.0-p;
            plog=log(p);
            pclog=log(pc);
        }

        // binomial deviate NR's bnldev()
        int newvalue() 
        {
            int j, bnl;
            float am,em,g,angle,sq,t,y;

            am=N*p;
            if (N < 25) {
                // just do the trials directly:
                bnl=0;
                for (j=1;j<=N;j++)
                    if (u() < p) ++bnl;
            } else if (am < 1.0) {
                // Use Poisson form for small # events
                g=exp(-am);
                t=1.0;
                for (j=0;j<=N;j++) {
                    t *= u();
                    if (t < g) break;
                }
                bnl=(j <= N ? j : N);
            } else {
                sq=sqrt(2.0*am*pc);
                do {
                    do {
                        angle=PI*u();
                        y=tan(angle);
                        em=sq*y+am;
                    } while (em < 0.0 || em >= (en+1.0));
                    em=floor(em);
                    t=1.2*sq*(1.0+y*y)*exp(oldg-gammln(em+1.0)
                                           -gammln(en-em+1.0)+em*plog+(en-em)*pclog);
                } while (u() > t);
                bnl= static_cast<int> (em);
            }
            if (flip) bnl=N-bnl;
            return bnl;
        }
    };

    // Poisson deviate
    // Again, use Num. Recipes poidev() as modified to use C++ objects
    class PoissonDeviate 
    {
    public:
        //seed with time:
        PoissonDeviate(const double _mean) : ownedU(new UniformDeviate), u(*ownedU) 
        { init(_mean); }

        //seed with specific number
        PoissonDeviate(const double _mean, const long lseed) : 
            ownedU(new UniformDeviate(lseed)), u(*ownedU) 
        { init(_mean); }  

        // use supplied uniform deviate
        PoissonDeviate(const double _mean, UniformDeviate& u_) : ownedU(0), u(u_) 
        { init(_mean); }

        ~PoissonDeviate() { if (ownedU) delete ownedU; ownedU=0; }

        //Reset the parameters
        void Reset(const double _mean) { init(_mean); } 
        int operator()() { return newvalue(); }
        operator int() { return newvalue(); }
        void Seed() { u.Seed(); }
        void Seed(const long lseed) { u.Seed(lseed); }

    private:
        UniformDeviate* ownedU;
        UniformDeviate& u;
        Gammln gammln;
        float mean; //prob per trial
        float sq,alxm,g; //constants used/stored

        void init(const double _mean) 
        {
            mean = _mean;
            if (mean < 12.0) {
                g=exp(-mean);
            } else {
                sq=sqrt(2.0*mean);
                alxm=log(mean);
                g=mean*alxm-gammln(mean+1.0);
            }
        }

        int newvalue() 
        {
            if (mean < 12.0) {
                int em = -1;
                double t=1.0;
                do {
                    ++em;
                    t *= u;
                } while (t > g);
                return em;
            } else {
                double em, y, t;
                do {
                    do {
                        y=tan(PI*u);
                        em=sq*y+mean;
                    } while (em < 0.0);
                    em=floor(em);
                    t=0.9*(1.0+y*y)*exp(em*alxm-gammln(em+1.0)-g);
                } while (u > t);
                return static_cast<int> (em);
            }
        }
    };

}  // namespace ran

#endif
