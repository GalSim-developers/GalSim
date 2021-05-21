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

#ifndef GalSim_Stopwatch_H
#define GalSim_Stopwatch_H

#include <sys/time.h>

namespace galsim {

class Stopwatch
{
private:
    double seconds;
    struct timeval tpStart;
    bool running;
public:
    Stopwatch() : seconds(0.), running(false) {}

    void start() { gettimeofday(&tpStart, NULL); running=true; }

    void stop()
    {
        if (!running) return;
        struct timeval tp;
        gettimeofday(&tp, NULL);
        seconds += (tp.tv_sec - tpStart.tv_sec)
            + 1e-6*(tp.tv_usec - tpStart.tv_usec);
        running = false;
    }
    void reset() { seconds=0.; running=false; }
    operator double() const { return seconds; }
};

}

#endif
