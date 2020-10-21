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
                const Table& abs_length_table, bool transpose);

        template <typename T>
        bool insidePixel(int ix, int iy, double x, double y, double zconv,
                         ImageView<T> target, bool* off_edge=0) const;

        double calculateConversionDepth(const PhotonArray& photons, int i, double randomNumber) const;

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

        template <typename T>
        void fillWithPixelAreas(ImageView<T> target, Position<int> orig_center, bool use_flux);

    private:
	int horizontalPixelStride() {
	    return _numVertices + 1;
	}

	int verticalPixelStride() {
	    return _numVertices;
	}

	int horizontalRowStride(int nx) {
	    return ((_numVertices + 1) * nx) + 1;
	}

	int verticalColumnStride(int ny) {
	    return _numVertices * ny;
	}

	// Given x and y as co-ordinates of a boundary point, and n as the
	// corresponding index within the polygon, adjusts the point so that
	// it is zero based in both dimensions, as required for the new linear
	// boundary point arrays
	void adjustBase(double& x, double& y, int n) {
	    /*int nv2 = _numVertices / 2;
	    if ((n >= ((3*nv2)+1)) && (n <= ((5*nv2)+2))) x -= 1.0;
	    if ((n <= ((5*nv2)+2)) && (n <= ((7*nv2)+3))) y -= 1.0;*/	    
	    if (x >= 0.9999999999) x -= 1.0;
	    if (y >= 0.9999999999) y -= 1.0;
	}
	
	// Converts pixel co-ordinates (x, y) and index n within pixel boundary
	// polygon into an index within the new horizontal or vertical boundary
	// arrays. horizontal is set to true if the point is in the horizontal
	// array, false if vertical
	// NOTE: will only work for distortion array due to using _nx and _ny!
	int getBoundaryIndex(int x, int y, int n, bool& horizontal) {
	    int nv2 = _numVertices / 2;
	    horizontal = false;
	    int idx;
	    
	    if (n < nv2) {
		// left hand side, lower
		idx = n + nv2;
	    }
	    else if (n <= ((nv2*3)+1)) {
		// bottom row including corners
		horizontal = true;
		idx = n - nv2;
	    }
	    else if (n <= ((nv2*5)+1)) {
		// right hand side
		idx = (_numVertices - 1) - (n - ((nv2*3)+2)) + verticalColumnStride(_ny);
	    }
	    else if (n <= ((nv2*7)+3)) {
		// top row including corners
		horizontal = true;
		idx = (_numVertices + 1) - (n - ((nv2*5)+2)) + horizontalRowStride(_nx);
	    }
	    else {
		// left hand side, upper
		idx = n - ((nv2*7)+4);
	    }

	    if (horizontal) {
		return (y * horizontalRowStride(_nx)) + (x * horizontalPixelStride()) + idx;
	    }
	    return (x * verticalColumnStride(_ny)) + (y * verticalPixelStride()) + idx;
	}

	void initializeBoundaryPoints(int nx, int ny);

	void updatePixelBounds(int nx, int ny, size_t k);
	
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
