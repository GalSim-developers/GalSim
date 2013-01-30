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

#ifndef STOPWATCH_H
#define STOPWATCH_H

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
