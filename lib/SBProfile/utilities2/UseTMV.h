// $Id: UseTMV.h,v 1.1.1.1 2009/10/30 21:20:52 garyb Exp $
// Short header file that includes TMV and sets up some typedefs to use
#ifndef USETMV_H
#define USETMV_H

#include "TMV.h"

typedef tmv::Vector<double> DVector;
typedef tmv::Matrix<double> DMatrix; 
typedef tmv::Vector<int>    IVector; 
typedef tmv::Matrix<int>    IMatrix; 
typedef tmv::Vector<std::complex<double> > CVector;
typedef tmv::Matrix<std::complex<double> > CMatrix;

#endif
