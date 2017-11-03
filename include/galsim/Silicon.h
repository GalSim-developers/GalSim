/* -*- c++ -*-
 * Copyright (c) 2012-2017 by the GalSim developers team on GitHub
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
#include "Table.h"

namespace galsim
{

    class Silicon
    {
    public:
        Silicon(int numVertices, double numElec, int nx, int ny, int qDist, double nrecalc,
                double diffStep, double pixelSize, double sensorThickness, double* vertex_data,
                const Table<double, double>& tr_radial_table,
                Position<double> treeRingCenter,
                const Table<double, double>& abs_length_table);

        template <typename T>
        bool insidePixel(int ix, int iy, double x, double y, double zconv,
                         ImageView<T> target) const;

        void calculateConversionDepth(const PhotonArray& photons,
                                      std::vector<double>& depth, UniformDeviate ud) const;

        template <typename T>
        void updatePixelDistortions(ImageView<T> target);

        template <typename T>
        void addTreeRingDistortions(ImageView<T> target, Position<int> orig_center);

        template <typename T>
        double accumulate(const PhotonArray& photons, UniformDeviate ud, ImageView<T> target,
                          Position<int> orig_center);

    private:
        Polygon _emptypoly;
        mutable Polygon _testpoly;
        std::vector<Polygon> _distortions;
        std::vector<Polygon> _imagepolys;
        int _numVertices, _nx, _ny, _nv, _nrecalc;
        double _qDist, _diffStep, _pixelSize, _sensorThickness;
        Table<double, double> _tr_radial_table;
        Position<double> _treeRingCenter;
        Table<double, double> _abs_length_table;
    };
}

#endif
