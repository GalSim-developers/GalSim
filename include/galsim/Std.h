/* -*- c++ -*-
 * Copyright 2012-2014 The GalSim developers:
 * https://github.com/GalSim-developers
 *
 * This file is part of GalSim: The modular galaxy image simulation toolkit.
 *
 * GalSim is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * GalSim is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with GalSim.  If not, see <http://www.gnu.org/licenses/>
 */

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
#include <stdexcept>

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
#define xassert(x) assert(x)
#else
#define dbg if(false) (std::cerr)
#define xdbg if(false) (std::cerr)
#define xxdbg if(false) (std::cerr)
#define xassert(x)
#endif

// A nice way to throw exceptions that take a string argument and have that string
// include double or int information as well.
// e.g. FormatAndThrow<std::runtime_error>() << "x = "<<x<<" is invalid.";
template <class E=std::runtime_error>
class FormatAndThrow 
{
public:
    // OK, this is a bit weird, but mostly innocuous.  Mac's default gcc compiler for OSX >= 10.6
    // is apparently compiled with something called "fully dynamic strings".  If you combine
    // this with libraries that don't use fully dynamic strings, then you can have problems with
    // zero-length strings, such as the one in the default constructor for ostringstream.
    // It manifests with crashes, saying "pointer being freed was not allocated".
    // Here are some web sites that discuss the problem:
    //     http://newartisans.com/2009/10/a-c-gotcha-on-snow-leopard/
    //     http://gcc.gnu.org/bugzilla/show_bug.cgi?id=53838
    //     https://trac.macports.org/ticket/35070
    //     https://code.google.com/p/googletest/issues/detail?id=189
    // Anyway, my workaround is to initialize the string with a space and a backspace, which 
    // should print as nothing, so it should have no apparent result, and it avoids the 
    // attempted deallocation of the global empty string.
    
    FormatAndThrow() : oss(" ") {}

    template <class T>
    FormatAndThrow& operator<<(const T& t) 
    { oss << t; return *this; }

    ~FormatAndThrow() { throw E(oss.str()); }
private:
    std::ostringstream oss;
};


#endif
