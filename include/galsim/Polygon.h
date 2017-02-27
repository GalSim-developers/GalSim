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

    class Polygon
    {
    public:
        Polygon() : _sorted(false), _area(0.0) {}
        Polygon(const Polygon& p) : _sorted(p._sorted), _area(p._area), _points(p._points) {}
        ~Polygon() {}

        // Add a point to a Polygon
        // Note: all points need to be added before doing area or containts.  If more points are
        // added after either oth those calls, an exception will be thrown.
        void add(const Point& point);

        // Get the area of the Polygon
        double area();

        // Return whether the Polygon contains a given point
        bool contains(const Point& point);

        // Sort the points.  Done automatically by area and contains, but can be done manually.
        void sort();

        // Some methods that let Polygon act (in some ways) like a vector<Point>
        size_t size() const { return _points.size(); }
        void clear() { _points.clear(); }
        void reserve(int n) { _points.reserve(n); }
        Point& operator[](int i) { return _points[i]; }
        const Point& operator[](int i) const { return _points[i]; }

        // Make a new polygon that scales up the existing one by zfactor (relative to an empty
        // Polygon).
        Polygon scale(const Polygon& emptypoly, double zfactor) const;

    private:

        bool _sorted;
        double _area;
        std::vector<Point> _points;
    };

}

#endif
