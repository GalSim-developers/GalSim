/* -*- c++ -*-
 * Copyright (c) 2012-2019 by the GalSim developers team on GitHub
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

// Binomial coefficients and factorials
// These compute the value the first time for a given i or (i,j), and then store
// it for future use.  So if you are doing a lot of these, they become effectively
// constant time functions rather than linear in the value of i.

#ifndef GalSim_BinomFactH
#define GalSim_BinomFactH

namespace galsim {

    double fact(int i);
    double sqrtfact(int i);
    double binom(int i,int j);
    double sqrtn(int i);

}

#endif
