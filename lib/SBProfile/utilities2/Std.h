//---------------------------------------------------------------------------
#ifndef StdH
#define StdH
//---------------------------------------------------------------------------
// 	$Id: Std.h,v 1.3 2009/11/02 22:48:53 garyb Exp $

// Things to include in every program

#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <string>
#include <cstdlib>
#include <cmath>
#include <complex>
#include <exception>
#include <stdexcept>

using namespace std;

//************** Useful templates:
template <class T>
void SWAP(T& a,T& b) {T temp=a; a=b; b=temp;}

template <class T>
T SQR(const T& x) {return x*x;}

template <class T>
const T& MAX(const T& a, const T& b) {return a>b ? a : b;}

template <class T>
const T& MIN(const T& a, const T& b) {return a<b ? a : b;}

//************* typedefs:
typedef unsigned int uint;
typedef unsigned short ushort;
typedef unsigned long ulong;
typedef std::complex<double> DComplex;

//************** constants:
#ifndef PI
#ifdef M_PI
#define PI M_PI
#else
#define PI 3.14159265358979323846
#endif
#endif

//********** Debugging classes:

#ifdef DEBUGLOGGING
extern ostream* dbgout;
#define dbg if(dbgout) (*dbgout)
#else
#define dbg if(false) (cerr)
#endif

#ifdef ASSERT
  #define Assert(x) \
    { if(!(x)) { \
      cerr << "Error - Assert " #x " failed"<<endl; \
      cerr << "on line "<<__LINE__<<" in file "<<__FILE__<<endl; \
      exit(1);} }
#else
  #define Assert(x)
#endif

// Convenience feature for std::exception catching
inline void
quit(const std::exception& s, const int exit_code=1) throw () {
#ifdef DEBUGLOGGING
  dbg << s.what() << endl;
#endif
  cerr << s.what() << endl;
  exit(exit_code);
}

template <class E>
class FormatAndThrow {
public:
  FormatAndThrow() {}
  template <class T>
  FormatAndThrow& operator<<(const T& t) {oss << t; return *this;}
  ~FormatAndThrow() {throw E(oss.str());}
private:
  ostringstream oss;
};

// For limited backward compatibility with old Std.h class:
typedef std::runtime_error MyException;

#endif
