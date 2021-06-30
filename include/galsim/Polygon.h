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
 * Date: Jan 13, 2016
 * Polygon utilities in support of Silicon class
 */

#ifndef POLYGON_H
#define POLYGON_H

#include <vector>
#include "Bounds.h"

namespace galsim {

    // For these purposes, we use Point as an alias for Position<double>
    typedef Position<double> Point;

    class PUBLIC_API Polygon
    {
    public:
        Polygon() : _sorted(false), _area(0.0), _npoints(0) {}
        Polygon(const Polygon& p) :
            _sorted(p._sorted), _area(p._area), _points(p._points), _npoints(p._npoints),
            _inner(p._inner), _outer(p._outer) {}
        ~Polygon() {}

        // Add a point to a Polygon
        // Note: all points need to be added before doing area or contains.  If more points are
        // added after either of those calls, an exception will be thrown.
        void add(const Point& point);

        // Sort the points. The user is responsible for calling this after adding all the points
        // or after making any modification that might change the order of the points around
        // the origin.
        void sort();

        // Get the area of the Polygon.  The result is saved, so if you make modifications to
        // the polyon and want the saved value to be reset, you can call updateBounds() to reset it.
        double area() const;

        // Return whether the Polygon contains a given point
        bool contains(const Point& point) const;

        // Two functions that check whether the point is trivially inside or outside.
        inline bool triviallyContains(const Point& point) const
        { return _inner.includes(point); }

        inline bool mightContain(const Point& point) const
        { return _outer.includes(point); }

        // Some methods that let Polygon act (in some ways) like a vector<Point>
        size_t size() const { return _points.size(); }
        void clear() { _points.clear(); }
        void reserve(int n) { _points.reserve(n); }
        Point& operator[](int i) { return _points[i]; }
        const Point& operator[](int i) const { return _points[i]; }

        // Make the Polygon a scaled version of a reference one (relative to an empty Polygon).
        void scale(const Polygon& refpoly, const Polygon& emptypoly, double factor);

        // Distort positions by a scaled version of a reference polgon
        void distort(const Polygon& refpoly, double factor);

        // Update the inner and outer bounds approximations of the polygon.  Need to make
        // sure you do this any time you update the positions of the points.
        void updateBounds();

        const Bounds<double>& getInnerBounds() const { return _inner; }
        const Bounds<double>& getOuterBounds() const { return _outer; }

    private:

        bool _sorted;
        mutable double _area;
        std::vector<Point> _points;
        int _npoints;  // Always equivalent to _points.size(), but convenient to have it as an int.
        Bounds<double> _inner;
        Bounds<double> _outer;
    };

}

#endif
