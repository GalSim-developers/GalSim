/* -*- c++ -*-
 * Copyright (c) 2012-2026 by the GalSim developers team on GitHub
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

#ifndef GalSim_Stopwatch_H
#define GalSim_Stopwatch_H

#include <chrono>

namespace galsim {

class Stopwatch
{
private:
    typedef std::chrono::steady_clock clock_type;
    double seconds;
    clock_type::time_point tpStart;
    bool running;
public:
    Stopwatch() : seconds(0.), running(false) {}

    void start() { tpStart = clock_type::now(); running = true; }

    void stop()
    {
        if (!running) return;
        auto tp = clock_type::now();
        std::chrono::duration<double> dt = tp - tpStart;
        seconds += dt.count();
        running = false;
    }
    void reset() { seconds = 0.; running = false; }
    operator double() const { return seconds; }
};

}

#endif
