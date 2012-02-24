
// Binomial coefficients and factorials
// These compute the value the first time for a given i or (i,j), and then store
// it for future use.  So if you are doing a lot of these, they become effectively
// constant time functions rather than linear in the value of i.

#ifndef BinomFactH
#define BinomFactH

namespace galsim {

    double fact(int i);
    double sqrtfact(int i);
    double binom(int i,int j);
    double sqrtn(int i);

}

#endif
