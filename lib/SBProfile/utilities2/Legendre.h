// $Id: Legendre.h,v 1.1.1.1 2009/10/30 21:20:52 garyb Exp $
// Class to calculate Legendre polynomials
#ifndef LEGENDRE_H
#define LEGENDRE_H

#include "UseTMV.h"

namespace legendre {

  class Legendre {
  public:
    Legendre(double xmin=-1., double xmax=+1, int order_=2):
      sub((xmin+xmax)*0.5), scale(2./(xmax-xmin)), order(order_) {}
      DVector operator()(double x) const {
      x = scale*(x-sub);
      DVector out(order+1);
      if (order>=0) out[0] = 1.;
      if (order>=1) out[1] = x;
      for (int i=2; i<=order; i++)
	out[i] = ( (2*i-1)*x*out[i-1] - (i-1)*out[i-2]) / i;
      return out;
    }
    void setOrder(int o) {order=o;}
    int getOrder() const {return order;}
  private:
    double sub;
    double scale;
    int order;
  };

} // namespace legendre

#endif // LEGENDRE_H
