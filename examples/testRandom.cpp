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
// Test random-number classes
#include "galsim/Random.h"
#include <iostream>

using namespace std;

int
main(int argc,
     char *argv[])
{
    if (argc<2 || argc>3) {
        cerr << "test the Random.h random number generators by drawing from\n"
             << "uniform, Gaussian, binomial, and poisson deviates.\n"
             << "Usage: test_random <nTrials> [seed]\n"
             << " nTrials is number of trials to output\n"
             << " seed is optional long-int seed.  Time of day is used as seed if none given.\n"
             << " output: 1 line per trial, giving trial number, uniform deviate,\n"
             << "   Gaussian deviate (mean=0, sigma=1),\n"
             << "   binomial deviate (N=10, p=0.4), and Poisson deviate (mean=3.5)."
             << endl;
        exit(1);
    }
    galsim::UniformDeviate u;
    int nTrials = atoi(argv[1]);
    if (argc>2) u.seed(atoi(argv[2]));
    galsim::GaussianDeviate gd(u);
    galsim::BinomialDeviate bd(u,10,0.4);
    galsim::PoissonDeviate pd(u, 3.5);

    for (int i=0; i<nTrials; i++)
        cout << i << " " << u() << " " << gd() << " " << bd() << " " << pd() << endl;
}
