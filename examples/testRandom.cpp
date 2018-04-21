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
    galsim::UniformDeviate u(0);
    int nTrials = atoi(argv[1]);
    if (argc>2) u.seed(atoi(argv[2]));
    galsim::GaussianDeviate gd(u,0.,1.);
    galsim::BinomialDeviate bd(u,10,0.4);
    galsim::PoissonDeviate pd(u, 3.5);

    for (int i=0; i<nTrials; i++)
        cout << i << " " << u() << " " << gd() << " " << bd() << " " << pd() << endl;
}
