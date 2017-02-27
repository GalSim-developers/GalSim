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
        Silicon(int NumVertices, int NumElec, int Nx, int Ny, int QDist, int Nrecalc,
                double DiffStep, double PixelSize, double* vertex_data);

        template <typename T>
        bool insidePixel(int ix, int iy, double x, double y, double zconv,
                         ImageView<T> target) const;

        template <typename T>
        void updatePixelDistortions(ImageView<T> target);

        //double AbsLength(double lambda);

        template <typename T>
        double accumulate(const PhotonArray& photons, UniformDeviate ud, ImageView<T> target);

    private:
        std::vector<Polygon> _distortions;
        std::vector<Polygon> _emptypoly;
        std::vector<Polygon> _imagepolys;
        int _numVertices, _numElect, _nx, _ny, _nv, _nrecalc;
        double _qDist, _diffStep, _pixelSize;
    };
}

#endif
