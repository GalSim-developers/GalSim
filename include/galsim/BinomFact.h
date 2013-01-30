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

// Binomial coefficients and factorials
// These compute the value the first time for a given i or (i,j), and then store
// it for future use.  So if you are doing a lot of these, they become effectively
// constant time functions rather than linear in the value of i.

#ifndef BinomFactH
#define BinomFactH

namespace galsim {

    double fact(int i);
    double sqrtfact(int i);
    double binom(int i,int j);
    double sqrtn(int i);

}

#endif
