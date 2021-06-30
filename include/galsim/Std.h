/* -*- c++ -*-
 * Copyright (c) 2012-2021 by the GalSim developers team on GitHub
 * https://github.com/GalSim-developers
 *
 * This file is part of GalSim: The modular galaxy image simulation toolkit.
 * https://github.com/GalSim-developers/GalSim
 *
 * GalSim is free software: redistribution and use in source and binary forms,
 * with or without modification, are permitted provided that the following
 * conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions, and the disclaimer given in the accompanying LICENSE
 *    file.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions, and the disclaimer given in the documentation
 *    and/or other materials provided with the distribution.
 */

// A few generically useful utilities.


// The regular assert is turned off by NDEBUG, which Python always sets.
// It's easier to just make our own than to try to undo the NDEBUG definition.
// Plus by making our own we can have it raise an exception, rather than abort, which
// is nicer behavior anyway.
// Note: do this outside of include guard since cassert doesn't have one, so we might need
// to revert this multiple times depending on order of #include lines in various files.
#ifdef assert
#undef assert
#endif
#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)
#define assert(x) do { if (!(x)) { dbg<<"Failed Assert: "<<#x<<std::endl; throw std::runtime_error("Failed Assert: " #x " at " __FILE__ ":" TOSTRING(__LINE__)); } } while (false)

#ifndef GalSim_Std_H
#define GalSim_Std_H

#include <cmath>
#include <iostream>
#include <sstream>
#include <cstdlib>
#include <string>
#include <stdexcept>
#include <cstddef>

#ifdef _WIN32
#include <Windows.h>
#else
#include <sys/time.h>
#endif

#if defined(__GNUC__)
#define PUBLIC_API __attribute__ ((visibility ("default")))
#else
#define PUBLIC_API
#endif

// A nice memory checker if you need to track down some memory problem.
#ifdef MEM_TEST
#include "mmgr.h"
#endif

// Just in case the above bit for M_PI didn't work...
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Bring either std::tr1::shared_ptr or std::shared_ptr into local namespace
// cf. https://stackoverflow.com/questions/18831151/is-c-0x-tr1-safe-to-use-when-portability-matters
#ifdef _LIBCPP_VERSION
// using libc++

#include <memory>
using std::shared_ptr;

#elif __cplusplus >= 201103L
// Also if using a real C++11 compiler, this should work.

#include <memory>
using std::shared_ptr;

#else  // !_LIBCPP_VERSION
// not using libc++

#include <tr1/memory>
using std::tr1::shared_ptr;

#endif  // _LIBCPP_VERSION

// Check if ptr is aligned on 128 bit boundary
inline bool IsAligned(const void* p) { return (reinterpret_cast<size_t>(p) & 0xf) == 0; }

// Convenient debugging.
// Use as a normal C++ stream:
// dbg << "Here x = "<<x<<" and y = "<<y<<std::endl;
// If DEBUGLOGGING is not enabled, the compiler optimizes it away, so it
// doesn't take any CPU cycles.
//

#ifdef DEBUGLOGGING

class Debugger // Use a Singleton model so it can be included multiple times.
{
public:
    std::ostream& get_dbgout() { return *dbgout; }
    void set_dbgout(std::ostream* new_dbgout) { dbgout = new_dbgout; }
    void set_verbose(int level) { verbose_level = level; }
    bool do_level(int level) { return verbose_level >= level; }
    int get_level() { return verbose_level; }

    static Debugger& instance()
    {
        static Debugger _instance;
        return _instance;
    }

private:
    std::ostream* dbgout;
    int verbose_level;

    Debugger() : dbgout(&std::cout), verbose_level(1) {}
    Debugger(const Debugger&);
    void operator=(const Debugger&);
};

#define dbg if(Debugger::instance().do_level(1)) Debugger::instance().get_dbgout()
#define xdbg if(Debugger::instance().do_level(2)) Debugger::instance().get_dbgout()
#define xxdbg if(Debugger::instance().do_level(3)) Debugger::instance().get_dbgout()
#define set_dbgout(dbgout) Debugger::instance().set_dbgout(dbgout)
#define set_verbose(level) Debugger::instance().set_verbose(level)
#define verbose_level Debugger::instance().get_level()
#define xassert(x) assert(x)
#else
#define dbg if(false) (std::cerr)
#define xdbg if(false) (std::cerr)
#define xxdbg if(false) (std::cerr)
#define set_dbgout(dbgout)
#define set_verbose(level)
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

    ~FormatAndThrow()
#if __cplusplus >= 201103L
        noexcept(false)
#endif
    { throw E(oss.str()); }

private:
    std::ostringstream oss;
};

/*
 *  A simple timer class to see how long a piece of code takes.
 *  Usage:
 *
 *  {
 *      static Timer timer("name");
 *
 *      ...
 *
 *      timer.start()
 *      [ The code you want timed ]
 *      timer.stop()
 *
 *      ...
 *  }
 *
 *  At the end of execution, you will get output:
 *
 *  Time for name: xxx seconds
 */
class Timer
{
public:
    Timer(std::string name, bool start_running=false) : _name(name), _accum(0), _running(false)
    {
        if (start_running) start();
    }
    ~Timer() { stop(); report(); }

    void start() {
        if (!_running) {
            _start_time = GetTimeMicroseconds();
            _running = true;
        }
    }
    void stop() {
        if (_running) {
            unsigned long long stop_time = GetTimeMicroseconds();
            _accum += stop_time - _start_time;
            _running = false;
        }
    }
    void report() { std::cout<<"Time for "<<_name<<": " << _accum / 1.e6 << " seconds\n"; }
private:
    // cf. http://stackoverflow.com/questions/1861294/how-to-calculate-execution-time-of-a-code-snippet-in-c
    unsigned long long GetTimeMicroseconds()
    {
#ifdef _WIN32
        /* Windows */
        FILETIME ft;
        LARGE_INTEGER li;

        /* Get the amount of 100 nano seconds intervals elapsed since January 1, 1601 (UTC) and copy it
         *   * to a LARGE_INTEGER structure. */
        GetSystemTimeAsFileTime(&ft);
        li.LowPart = ft.dwLowDateTime;
        li.HighPart = ft.dwHighDateTime;

        unsigned long long ret = li.QuadPart;
        ret -= 116444736000000000LL; /* Convert from file time to UNIX epoch time. */
        ret /= 10; /* From 100 nano seconds (10^-7) to 1 microsecond (10^-6) intervals */
#else
        /* Linux */
        struct timeval tv;

        gettimeofday(&tv, NULL);

        unsigned long long ret = tv.tv_usec;
        /* Adds the seconds (10^0) after converting them to microseconds (10^-6) */
        ret += (tv.tv_sec * 1000000);
#endif
        return ret;
    }
    std::string _name;
    long long _accum;
    unsigned long long _start_time;
    bool _running;
};


#endif
