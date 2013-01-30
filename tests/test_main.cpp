// -*- c++ -*-
/*
 * Copyright 2012, 2013 The GalSim developers:
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
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE Main

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

// Only clang seems to have this
#ifdef __clang__
#if __has_warning("-Wlogical-op-parentheses")
#pragma GCC diagnostic ignored "-Wlogical-op-parentheses"
#endif
#endif

#endif // !INTEL

#include <boost/test/included/unit_test.hpp>

//JAZ Nothing needs to go here - the test module definitions above create a main function.
