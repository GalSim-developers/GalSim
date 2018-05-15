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
#ifndef GalSim_WCS_H
#define GalSim_WCS_H

namespace galsim {

    void ApplyCD(int n, double* x, double* y, const double* cd);
    void ApplyPV(int n, int m, double* u, double* v, const double* pv);
    void InvertPV(double& u, double& v, const double* pv);
    void InvertAB(int m, double& x, double& y, const double* ab, const double* abp);

}

#endif
