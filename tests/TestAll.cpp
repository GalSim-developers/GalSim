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

// Without a snazzy unit test framework, we need to register all the tests to be run here
// by hand.

#include <stdexcept>
#include "Test.h"
#include <iostream>

extern void TestImage();
extern void TestInteg();
extern void TestVersion();

int main()
{
    try {
        std::cout<<"Start C++ tests.\n";
        // Run them all here:
        TestImage();
        std::cout<<"TestImage passed all tests.\n";
        TestInteg();
        std::cout<<"TestInteg passed all tests.\n";
        TestVersion();
        std::cout<<"TestVersion passed all tests.\n";

    } catch (std::exception& e) {
        std::cerr<<e.what()<<std::endl;
        return 1;
    }
    return 0;
}
