/* -*- c++ -*-
 * Copyright (c) 2012-2021 by the GalSim developers team on GitHub
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

    class PUBLIC_API Silicon
    {
    public:
        Silicon(int numVertices, double numElec, int nx, int ny, int qDist, double nrecalc,
                double diffStep, double pixelSize, double sensorThickness, double* vertex_data,
                const Table& tr_radial_table, Position<double> treeRingCenter,
                const Table& abs_length_table, bool transpose, bool useNewBoundaries = false);

        // TO BE REMOVED (replaced by insidePixelNew)
        template <typename T>
        bool insidePixel(int ix, int iy, double x, double y, double zconv,
                         ImageView<T> target, bool* off_edge=0) const;

        void scaleBoundsToPoly(int i, int j, int nx, int ny,
                               const Polygon& emptypoly, Polygon& result,
                               double factor) const;

        template <typename T>
        bool insidePixelNew(int ix, int iy, double x, double y, double zconv,
                            ImageView<T> target, bool* off_edge=0) const;

        double calculateConversionDepth(const PhotonArray& photons, int i, double randomNumber) const;

        void applyPixelDistortion(int i, int j, int disti, int distj, int nx,
                                  int ny, double charge, bool lhs, bool rhs, bool bottom);

        template <typename T>
        void updatePixelDistortions(ImageView<T> target);

        void calculateTreeRingDistortion(int i, int j, Position<int> orig_center,
                                         Polygon& poly) const;
        void calculateTreeRingDistortion(int i, int j, Position<int> orig_center,
                                         Polygon& poly, int nx, int ny, int i1,
                                         int j1);

        template <typename T>
        void addTreeRingDistortions(ImageView<T> target, Position<int> orig_center);

        template <typename T>
        double accumulate(const PhotonArray& photons, BaseDeviate rng, ImageView<T> target,
                          Position<int> orig_center, bool resume);

        double pixelArea(int i, int j, int nx, int ny) const;

        template <typename T>
        void fillWithPixelAreas(ImageView<T> target, Position<int> orig_center, bool use_flux);

    private:
        // Convenience inline methods for access to linear boundary arrays.
        int horizontalPixelStride() const {
            return _numVertices + 2;
        }

        int verticalPixelStride() const {
            return _numVertices + 2;
        }

        int horizontalRowStride(int nx) const {
            return (_numVertices + 2) * nx;
        }

        int verticalColumnStride(int ny) const {
            return (_numVertices + 2) * ny;
        }

        int horizontalPixelIndex(int x, int y, int nx) const {
            return (y * horizontalRowStride(nx)) + (x * horizontalPixelStride());
        }

        int verticalPixelIndex(int x, int y, int ny) const {
            return (x * verticalColumnStride(ny)) + (((ny-1)-y) * verticalPixelStride());
        }

        // Returns the indices of the corner vertices within each pixel polygon
        // These are the indices for the horizontal stretch.  When converting two/from pixel
        // polygons, there are two corner points, which may or may not have the same position,
        // depending on the context in which it is being used.  That is, the last vertical point
        // before the corner and the first one after a corner may be identical to the position
        // from the horizontal corner.  Or it may be slightly offset, depending on which side
        // of a "double T" corner we are on.
        int cornerIndexBottomLeft() const {
            return _numVertices / 2 + 1;
        }

        int cornerIndexBottomRight() const {
            return 3 * (_numVertices / 2) + 2;
        }

        int cornerIndexTopRight() const {
            return 5 * (_numVertices / 2) + 5;
        }

        int cornerIndexTopLeft() const {
            return 7 * (_numVertices / 2) + 6;
        }

        // Converts pixel co-ordinates (x, y) and index n within pixel boundary
        // polygon into an index within the new horizontal or vertical boundary
        // arrays. horizontal is set to true if the point is in the horizontal
        // array, false if vertical.
        // If nx and ny are not given, _nx and _ny are used.
        int getBoundaryIndex(int x, int y, int n, bool* horizontal, int nx = -1,
                             int ny = -1) const {
            *horizontal = false;
            int idx;

            if (nx < 0) {
                nx = _nx;
                ny = _ny;
            }

            if (n < cornerIndexBottomLeft()) {
                // left hand side, lower
                idx = n + cornerIndexBottomLeft();
            }
            else if (n <= cornerIndexBottomRight()) {
                // bottom row including corners
                *horizontal = true;
                idx = n - cornerIndexBottomLeft();
            }
            else if (n < cornerIndexTopRight()) {
                // right hand side
                idx = (cornerIndexTopRight() - n - 1) + verticalColumnStride(ny);
            }
            else if (n <= cornerIndexTopLeft()) {
                // top row including corners
                *horizontal = true;
                idx = (cornerIndexTopLeft() - n) + horizontalRowStride(nx);
            }
            else {
                // left hand side, upper
                idx = n - cornerIndexTopLeft() - 1;
            }

            if (*horizontal) {
                return horizontalPixelIndex(x, y, nx) + idx;
            }
            return verticalPixelIndex(x, y, ny) + idx;
        }

        // Iterates over all the points in the given pixel's boundary and calls a
        // callback for each one. callback should take an index and a point
        // reference, then two bools (RHS point and top point)
        template<typename T>
        void iteratePixelBoundary(int i, int j, int nx, int ny, T callback)
        {
            int n;
            int idx;
            // LHS lower half
            for (n = 0; n < cornerIndexBottomLeft(); n++) {
                idx = verticalPixelIndex(i, j, ny) + n + cornerIndexBottomLeft();
                callback(n, _verticalBoundaryPoints[idx], false, false);
            }
            // bottom row including corners
            for (; n <= cornerIndexBottomRight(); n++) {
                idx = horizontalPixelIndex(i, j, nx) + (n - cornerIndexBottomLeft());
                callback(n, _horizontalBoundaryPoints[idx], false, false);
            }
            // RHS
            for (; n < cornerIndexTopRight(); n++) {
                idx = verticalPixelIndex(i + 1, j, ny) + (cornerIndexTopRight() - n - 1);
                callback(n, _verticalBoundaryPoints[idx], true, false);
            }
            // top row including corners
            for (; n <= cornerIndexTopLeft(); n++) {
                idx = horizontalPixelIndex(i, j + 1, nx) + (cornerIndexTopLeft() - n);
                callback(n, _horizontalBoundaryPoints[idx], false, true);
            }
            // LHS upper half
            for (; n < _nv; n++) {
                idx = verticalPixelIndex(i, j, ny) + (n - cornerIndexTopLeft() - 1);
                callback(n, _verticalBoundaryPoints[idx], false, false);
            }
        }

        // Const version of method above.
        template<typename T>
        void iteratePixelBoundary(int i, int j, int nx, int ny, T callback) const
        {
            int n;
            int idx;
            // LHS lower half
            for (n = 0; n < cornerIndexBottomLeft(); n++) {
                idx = verticalPixelIndex(i, j, ny) + n + cornerIndexBottomLeft();
                callback(n, _verticalBoundaryPoints[idx], false, false);
            }
            // bottom row including corners
            for (; n <= cornerIndexBottomRight(); n++) {
                idx = horizontalPixelIndex(i, j, nx) + (n - cornerIndexBottomLeft());
                callback(n, _horizontalBoundaryPoints[idx], false, false);
            }
            // RHS
            for (; n < cornerIndexTopRight(); n++) {
                idx = verticalPixelIndex(i + 1, j, ny) + (cornerIndexTopRight() - n - 1);
                callback(n, _verticalBoundaryPoints[idx], true, false);
            }
            // top row including corners
            for (; n <= cornerIndexTopLeft(); n++) {
                idx = horizontalPixelIndex(i, j + 1, nx) + (cornerIndexTopLeft() - n);
                callback(n, _horizontalBoundaryPoints[idx], false, true);
            }
            // LHS upper half
            for (; n < _nv; n++) {
                idx = verticalPixelIndex(i, j, ny) + (n - cornerIndexTopLeft() - 1);
                callback(n, _verticalBoundaryPoints[idx], false, false);
            }
        }

        void initializeBoundaryPoints(int nx, int ny);

        void updatePixelBounds(int nx, int ny, size_t k);

        // TO BE REMOVED
        bool checkPixel(int i, int j, int nx, int ny);

        // TO BE REMOVED
        void makeDistortionsConsistent();
        // TO BE REMOVED
        void addHalo();

        bool _useNewBoundaries;
        Polygon _emptypoly;
        mutable std::vector<Polygon> _testpoly;
        std::vector<Polygon> _distortions;
        std::vector<Polygon> _imagepolys;
        std::vector<Point> _horizontalBoundaryPoints;
        std::vector<Point> _verticalBoundaryPoints;
        std::vector<Bounds<double> > _pixelInnerBounds;
        std::vector<Bounds<double> > _pixelOuterBounds;
        std::vector<Point> _horizontalDistortions;
        std::vector<Point> _verticalDistortions;
        int _numVertices, _nx, _ny, _nv, _qDist;
        double _nrecalc, _diffStep, _pixelSize, _sensorThickness;
        Table _tr_radial_table;
        Position<double> _treeRingCenter;
        Table _abs_length_table;
        bool _transpose;
        double _resume_next_recalc;
        ImageAlloc<double> _delta;
    };

    PUBLIC_API int SetOMPThreads(int num_threads);
    PUBLIC_API int GetOMPThreads();

}

#endif
