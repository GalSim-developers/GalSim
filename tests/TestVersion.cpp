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

#include <cstdio>
#include "GalSim.h"
#include "Test.h"

void TestVersion()
{
    Log("Start tests of C++ version macros/functions");

    // Here we mostly just check that the different ways to get the version are all consistent.

    // First, these are #define values we get from GalSim.h.
    int major = GALSIM_MAJOR;
    int minor = GALSIM_MINOR;
    int rev = GALSIM_REVISION;

    // Next, there are function versions of these that should return the same things.
    AssertEqual(galsim::major_version(), major);
    AssertEqual(galsim::minor_version(), minor);
    AssertEqual(galsim::revision(), rev);

    // Finally, there is a function that returns the version as a string.
    char str[80];
    std::sprintf(str, "%d.%d.%d", major, minor, rev);
    AssertEqual(galsim::version(), str);

    // And run the inline check as well.
    AssertTrue(galsim::check_version());
}

