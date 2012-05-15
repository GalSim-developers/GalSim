
// A few generically useful utilities.

#ifndef StdH
#define StdH

#include <cmath>
#define _USE_MATH_DEFINES  // To make sure M_PI is defined.
// It should be in math.h, but not necessarily in cmath.
#include <math.h>
#include <iostream>
#include <sstream>
#include <cstdlib>
#include <string>
#include <cassert>

// A nice memory checker if you need to track down some memory problem.
#ifdef MEM_TEST
#include "mmgr.h"
#endif

// Just in case the above bit for M_PI didn't work...
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Convenient debugging.
// Use as a normal C++ stream:
// dbg << "Here x = "<<x<<" and y = "<<y<<std::endl;
// If DEBUGLOGGING is not enabled, the compiler optimizes it away, so it 
// doesn't take any CPU cycles.
//
// You need to define dbgout and verbose_level in the .cpp file with main().
// And if you are using OpenMP, you can get debugging output from each thread into
// a separate file by calling SetupThreadDebug(name).
// Then each thread other than the main thread will actually write to a file 
// name_threadnum and not clobber each other.  (The main thread will write to name.)

//#define DEBUGLOGGING
#ifdef DEBUGLOGGING
extern std::ostream* dbgout;
extern int verbose_level;
#define dbg if(dbgout && verbose_level >= 1) (*dbgout)
#define xdbg if(dbgout && verbose_level >= 2) (*dbgout)
#define xxdbg if(dbgout && verbose_level >= 3) (*dbgout)
#else
#define dbg if(false) (std::cerr)
#define xdbg if(false) (std::cerr)
#define xxdbg if(false) (std::cerr)
#endif

// A nice way to throw exceptions that take a string argument and have that string
// include double or int information as well.
// e.g. FormatAndThrow<std::runtime_error>() << "x = "<<x<<" is invalid.";
template <class E>
class FormatAndThrow 
{
public:
    FormatAndThrow() {}

    template <class T>
    FormatAndThrow& operator<<(const T& t) 
    { oss << t; return *this; }

    ~FormatAndThrow() { throw E(oss.str()); }
private:
    std::ostringstream oss;
};


#endif
