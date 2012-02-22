// $Id: Poisson.h,v 1.1.1.1 2009/10/30 21:20:52 garyb Exp $
// Routines useful for Poisson distributions
#ifndef POISSON_H
#define POISSON_H

#include <stdexcept>
#include "Std.h"

namespace poisson {

  class PoissonError: public std::runtime_error {
  public:
    PoissonError(const string m): std::runtime_error("Poisson error: "+m) {}
  };

  template <class T=double>
  class Poisson {
  private:
    T mean;
  public:
    Poisson(T mean_): mean(mean_) {}
    T operator()(const int N) const;	//returns probability
    T getMean() const {return mean;}
    void setMean(const T mean_) {mean=mean_;}
    T cumulative(int N) const;		//probability of <=N
    //value of mean for which cumulative(N)=pctile
    static T percentileMean(int N, T pctile);	
  };
} // namespace poisson
#endif
