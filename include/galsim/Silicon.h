/* -*- c++ -*-
 * Copyright (c) 2012-2016 by the GalSim developers team on GitHub
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

/*
 * ------------------------------------------------------------------------------
 * Author: Craig Lage, UC Davis
 * Date: Mar 14, 2016
 * Routines for integrating the CCD simulations into GalSim
 */

//****************** Silicon.h **************************

#ifndef SILICON_H
#define SILICON_H

#include "Polygon.h"
#include "Image.h"
#include "PhotonArray.h"

namespace galsim
{

    class Silicon
    {
    public:
        Polygon** polylist;
        Point* testpoint;
        double DiffStep, collXmin, collXwidth, collYmin, collYwidth;
        int Nx, Ny, Nv, NumVertices, NumElec;
        Silicon() {};
        Silicon(std::string); // Constructor
        ~Silicon();  // Destructor
        double random_gaussian(void);

        template <typename T>
        bool InsidePixel(int ix, int iy, double x, double y, double zconv,
                         ImageView<T> target) const;

        template <typename T>
        double accumulate(const PhotonArray& photons, UniformDeviate ud,
                          ImageView<T> target) const;
    };
}

#endif
