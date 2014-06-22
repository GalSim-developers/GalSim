/* -*- c++ -*-
 * Copyright (c) 2012-2014 by the GalSim developers team on GitHub
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

#include <sys/time.h>
#include "Random.h"

namespace galsim {

    void BaseDeviate::seedtime() 
    {
        struct timeval tp;
        gettimeofday(&tp,NULL);
        _rng->seed(tp.tv_usec);
    }

    void BaseDeviate::seed(long lseed)
    {
        if (lseed == 0) seedtime();
        else {
            // We often use sequential seeds for our RNG's (so we can be sure that runs on multiple
            // processors are deterministic).  The Boost Mersenne Twister is supposed to work with
            // this kind of seeding, having been updated in April 2005 to address an issue with
            // precisely this sort of re-seeding.
            // (See http://www.boost.org/doc/libs/1_51_0/boost/random/mersenne_twister.hpp).
            // The issue itself is described briefly here:
            // http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/MT2002/emt19937ar.html,
            // and in more detail for an algorithm tt880 that is apparently a 'little cousin' to 
            // the Mersenne Twister: http://random.mat.sbg.ac.at/news/seedingTT800.html
            //
            // The worry is that updates to the methods claim improvements to the behaviour of 
            // close (in a bitwise sense) patterns, but we have not found ready quantified data.
            //
            // So just to be sure, we send the initial seed through a _different_ random number
            // generator for 2 iterations before using it to seed the RNG we will actually use.
            // This may not be necessary, but it's not much of a performance hit (only occurring on
            // the initial seed of each rng), it can't hurt, and it makes Barney and Mike somewhat
            // less disquieted.  :)

            boost::random::mt11213b alt_rng(lseed);
            alt_rng.discard(2);
            _rng->seed(alt_rng());
        }
        clearCache();
    }
}
