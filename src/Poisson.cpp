
#include <cmath>
#include <limits>
#include "Poisson.h"
#include "Solve.h"

namespace galsim {

    // From Numerical Recipes:
    template <class T>
    T gammp(const T a, const T x);

    template <class T>
    T gammq(const T a, const T x);

    template <class T>
    void gser(T& gammser, const T a, const T x, T& gln);

    template <class T>
    void gcf(T& gammser, const T a, const T x, T& gln);

    template <class T>
    T Poisson<T>::operator()(const int N) const 
    {
        if (N<0) throw PoissonError("Negative counts");
        if (N==0) return exp(-mean);
        return exp( N*log(mean) - lgamma(static_cast<double> (N)) - mean);
    }

    template <class T>
    T Poisson<T>::cumulative(int N) const 
    {
        if (N<0) throw PoissonError("Negative counts");
        if (N==0) return exp(-mean);
        return gammq(static_cast<T> (N+1.), mean);
    }

    // Functor class for the percentile solver:
    template <class T>
    class percentileFunction 
    {
    private:
        T N;
        T pct;
    public:
        percentileFunction(int N_, T pct_): N(static_cast<T> (N_)), pct(pct_) {}
        T operator()(const T mean) const 
        {
            assert(N>=0.);
            if (N==0.) return exp(-mean) - pct;
            if (mean==0.) return 1.-pct; // Know N is positive
            //      return gammq(N+1, mean) - pct;
            return gammq(N, mean) - pct;
        }
    };

    template <class T>
    T Poisson<T>::percentileMean(int N, T pctile) 
    {
        if (N<0) throw PoissonError("Negative counts");
        percentileFunction<T> f(N, pctile);
        Solve<percentileFunction<T> > 
            solver(f, 0., std::max(20.,N+10.*sqrt(1.*N))); // Upper limit is not foolproof!
        return solver.root();
    }

    ///////////////////////////////////////////////////////////
    // Following adapted from Numerical Recipes
    ///////////////////////////////////////////////////////////

    template <class T>
    T gammp(const T a, const T x) 
    {
        T gamser,gammcf,gln;
        if (x < 0.0 || a <= 0.0) 
            throw PoissonError("Invalid arguments in routine gammp");
        if (x < (a+1.0)) {
            gser(gamser,a,x,gln);
            return gamser;
        } else {
            gcf(gammcf,a,x,gln);
            return 1.0-gammcf;
        }
    }
    template <class T>
    T gammq(const T a, const T x) 
    {
        T gamser,gammcf,gln;
        if (x < 0.0 || a <= 0.0) 
            throw PoissonError("Invalid arguments in routine gammp");
        if (x < (a+1.0)) {
            gser(gamser,a,x,gln);
            return 1.0 - gamser;
        } else {
            gcf(gammcf,a,x,gln);
            return gammcf;
        }
    }

    template <class T>
    void gser(T& gamser, const T a, const T x, T& gln) 
    {
        const int ITMAX=200;
        const T EPS=std::numeric_limits<T>::epsilon();

        gln=lgamma(a);
        if (x <= 0.0) {
            if (x < 0.0) throw PoissonError("x less than 0 in routine gser");
            gamser=0.0;
            return;
        } else {
            T ap=a;
            T sum=1.0/a;
            T del = sum;
            for (int n=1;n<=ITMAX;n++) {
                ++ap;
                del *= x/ap;
                sum += del;
                if (abs(del) < abs(sum)*EPS) {
                    gamser=sum*exp(-x+a*log(x)-gln);
                    return;
                }
            }
            throw PoissonError("a too large, ITMAX too small in routine gser");
            return;
        }
    }

    template <class T>
    void gcf(T& gammcf, const T a, const T x, T& gln) 
    {
        const int ITMAX=100;
        const T EPS = std::numeric_limits<T>::epsilon();
        const T FPMIN = std::numeric_limits<T>::min()/EPS;

        gln=lgamma(a);
        T b=x+1.0-a;
        T c=1.0/FPMIN;
        T d=1.0/b;
        T h=d;
        for (int i=1;i<=ITMAX;i++) {
            T an = -i*(i-a);
            b += 2.0;
            d=an*d+b;
            if (fabs(d) < FPMIN) d=FPMIN;
            c=b+an/c;
            if (fabs(c) < FPMIN) c=FPMIN;
            d=1.0/d;
            T del=d*c;
            h *= del;
            if (fabs(del-1.0) < EPS) break;
            if (i==ITMAX) throw PoissonError("a too large, ITMAX too small in gcf");
        }
        gammcf=exp(-x+a*log(x)-gln)*h;
    }

    template class Poisson<float>;
    template class Poisson<double>;

} // namespace poisson


