/* -*- c++ -*-
 * Copyright (c) 2012-2018 by the GalSim developers team on GitHub
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

// Some helper functions for doing C++-level testing.
// We don't do very many C++ tests, preferring to use python-level tests when possible.
// So we don't bother with a large complicated unit test framework (like Boost or Google Tests).
// These simple helper functions are sufficient for our needs.

#include <cmath>
#include <string>
#include <iostream>
#include <fstream>

#ifndef _GALSIM_TEST_H
#define _GALSIM_TEST_H

inline std::ostream& getLogger()
{
    static std::ofstream logger("tests.log");
    return logger;
}

inline void Log(const std::string& msg)
{
    std::ostream& log = getLogger();
    if (msg != "") {
        log<<msg<<std::endl;
    }
}

template <typename T>
inline void _AssertTrue(T test_value, const std::string& msg="")
{
    Log(msg);
    if (!(test_value)) {
        std::cerr<<"Failed truth check: "<<msg<<std::endl;
        std::cerr<<test_value<<" != true\n";
        throw std::runtime_error("Error in "+msg);
    }
}

#define AssertTrue(a) \
    _AssertTrue(a, "Testing " #a " is true")

template <typename T1, typename T2>
inline void _AssertEqual(T1 test_value, T2 comparison, const std::string& msg="")
{
    Log(msg);
    if (!(test_value == comparison)) {
        std::cerr<<"Failed equality check: "<<msg<<std::endl;
        std::cerr<<test_value<<" != "<<comparison<<std::endl;
        throw std::runtime_error("Error in "+msg);
    }
}

#define AssertEqual(a, b) \
    _AssertEqual(a, b, "Testing " #a " equals " #b)

template <typename T1, typename T2>
inline void _AssertNotEqual(T1 test_value, T2 comparison, const std::string& msg="")
{
    Log(msg);
    if (!(test_value != comparison)) {
        std::cerr<<"Failed inequality check: "<<msg<<std::endl;
        std::cerr<<test_value<<" == "<<comparison<<std::endl;
        throw std::runtime_error("Error in "+msg);
    }
}

#define AssertNotEqual(a, b) \
    _AssertNotEqual(a, b, "Testing " #a " is not equal to " #b)

template <typename T1, typename T2>
inline void _AssertClose(T1 test_value, T2 comparison,
                         double rtol=1.e-8, double atol=1.e-15,
                         const std::string& msg="")
{
    Log(msg);
    double absdiff = std::abs(test_value - comparison);
    if (!(absdiff < rtol * std::abs(comparison)) && !(absdiff < atol)) {
        std::cerr<<"Failed close check: "<<msg<<std::endl;
        std::cerr<<test_value<<" !~= "<<comparison<<std::endl;
        std::cerr<<"|diff| = "<<absdiff<<" > "<<rtol<<" * "<<std::abs(comparison)<<std::endl;
        std::cerr<<"and      "<<absdiff<<" > "<<atol<<std::endl;
        throw std::runtime_error("Error in "+msg);
    }
}

#define AssertClose(a, b, args...) \
    __AssertClose(a, b, "Testing " #a " is close to " #b, ##args)

// Another helper that just changes the order of arguments (and has default rtol, atol values)
template <typename T1, typename T2>
inline void __AssertClose(T1 test_value, T2 comparison,
                         const std::string& msg,
                         double rtol=1.e-8, double atol=1.e-15)
{
    _AssertClose(test_value, comparison, rtol, atol, msg);
}

#endif
