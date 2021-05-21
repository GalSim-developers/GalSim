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

// The boost header files tend to emit lots of warnings, so pretty much any GalSim
// code that imports boost should import this first to suppress the warnings.

// icpc pretends to be GNUC, since it thinks it's compliant, but it's not.
// It doesn't understand #pragma GCC
// Rather, it uses #pragma warning(disable:nn)
#ifdef __INTEL_COMPILER

// Disable "overloaded virtual function ... is only partially overridden"
#pragma warning(disable:654)

#else

// The boost unit tests have some unused variables, so suppress the warnings about that.
// I think pragma GCC was introduced in gcc 4.2, so guard for >= that version
#if defined(__GNUC__) && __GNUC__ >= 4 && (__GNUC__ >= 5 || __GNUC_MINOR__ >= 2)
#pragma GCC diagnostic ignored "-Wunused-variable"
#endif

// Not sure when this was added.  Currently check for it for versions >= 4.3
#if defined(__GNUC__) && __GNUC__ >= 4 && (__GNUC__ >= 5 || __GNUC_MINOR__ >= 3)
#pragma GCC diagnostic ignored "-Warray-bounds"
#endif

#if defined(__GNUC__) && __GNUC__ >= 4 && (__GNUC__ >= 5 || __GNUC_MINOR__ >= 8)
#pragma GCC diagnostic ignored "-Wunused-local-typedefs"
#endif

// For 32-bit machines, g++ -O2 optimization in some TMV calculations uses an optimization
// that is technically not known to not overflow 32 bit integers.  In fact, it is totally
// fine to use, but we need to remove a warning about it in this file for gcc >= 4.5
#if defined(__GNUC__) && __GNUC__ >= 4 && (__GNUC__ >= 5 || __GNUC_MINOR__ >= 5)
#pragma GCC diagnostic ignored "-Wstrict-overflow"
#endif

// This is a bug in boost that was apparently fixed in 1.62.  But I don't think we much
// care about this, so we just disable it regardless of which boost is being used.
#if defined(__GNUC__) && __GNUC__ >= 6
#pragma GCC diagnostic ignored "-Wplacement-new"
#endif

// Something about mangled names changing in C++17.
#if defined(__GNUC__) && __GNUC__ >= 7
#pragma GCC diagnostic ignored "-Wnoexcept-type"
#endif

// This looks like a bug in the boost/python extract function.  Not sure which gcc version it
// started showing up.  4.8 doesn't warn, but 7.1 does.
#if defined(__GNUC__) && __GNUC__ >= 5
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif

#ifdef __clang__
// register is deprecated in g++-17
#if defined(__GNUC__) && __GNUC__ >= 5
#pragma GCC diagnostic ignored "-Wdeprecated-register"
#endif

// Only clang seems to have this
#if __has_warning("-Wlogical-op-parentheses")
#pragma GCC diagnostic ignored "-Wlogical-op-parentheses"
#endif

#if __has_warning("-Wshift-count-overflow")
#pragma GCC diagnostic ignored "-Wshift-count-overflow"
#endif

// And clang might need this even if it claims to be GNUC before 4.8.
#if __has_warning("-Wunused-local-typedefs")
#pragma GCC diagnostic ignored "-Wunused-local-typedefs"
#endif

#if __has_warning("-Wparentheses-equality")
#pragma GCC diagnostic ignored "-Wparentheses-equality"
#endif

#if __has_warning("-Wunused-value")
#pragma GCC diagnostic ignored "-Wunused-value"
#endif

#endif // clang

#endif // intel
