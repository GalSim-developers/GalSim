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

#ifndef GalSim_Angle_H
#define GalSim_Angle_H

/**
 *  @file Angle.h
 *
 *  @brief Defines Angle class for dealing cleanly with angle values and a unit.
 *
 *  Based on the LSST Angle class from
 *  http://dev.lsstcorp.org/cgit/LSST/DMS/afw.git/tree/include/lsst/afw/geom/Angle.h
 *
 *  Modified significantly by MJ:
 *  - Remove implicit conversion to/from double (which seems to me to negate much
 *    of the point of having an Angle class).
 *  - Include scalar = Angle / AngleUnit
 *  - Removed non-sensical Angle = Angle * Angle
 *  - Remove templates.  Let compiler convert types to double as needed.
 *  - Can wrap values to [-pi,pi) if desired by calling theta.wrap().
 *  - Changed names of arcminutes -> arcmin and arcseconds -> arcsec.
 *  - Removed some extra functions that I don't think we need.
 *  - Switch to galsim namespace.
 *  - Removed dependancy on boost.
 *  - Streamlined the code a lot so all methods are now inline in the class.
 *    (Since all methods are one-liners, no need for obtuse macros.)
 */

#include <limits>
#include <iostream>

#include "Std.h"

namespace galsim {

    class Angle;

    /**
     *  @brief A class defining angle units
     *
     *  You probably won't ever have to use this directly.
     *  Instead you will you the pre-defined constants that are AngleUnits:
     *    radians
     *    degrees
     *    hours
     *    arcmin
     *    arcsec
     */
    class AngleUnit
    {
        friend class Angle;
    public:
        /**
         *  @brief The value is how many radians 1 AngleUnit corresponds to.
         *
         *  e.g. AngleUnit degrees(PI/180)
         */
        explicit AngleUnit(double val) : _val(val) {}

        //@{
        /// Comparisons of different units:
        bool operator==(AngleUnit rhs) const { return (_val == rhs._val); }
        bool operator!=(AngleUnit rhs) const { return (_val != rhs._val); }
        //@}

        double getValue() const { return _val; }

    private:
        double _val;
    };

    const AngleUnit radians(1.0); ///< constant with units of radians
    const AngleUnit degrees(M_PI/180.); ///< constant with units of degrees
    const AngleUnit hours(M_PI*15./180.); ///< constant with units of hours
    const AngleUnit arcmin(M_PI/60./180.); ///< constant with units of arcminutes
    const AngleUnit arcsec(M_PI/3600./180.); ///< constant with units of arcseconds

    // Defined below after we define the Angle class
    inline Angle operator*(double val, AngleUnit unit);

    /**
     *  @brief A class representing an Angle
     *
     *  Angles are a value with an AngleUnit.
     *
     *  You typically create an Angle by multiplying a number by an AngleUnit.
     *  e.g.
     *  @code
     *    Angle pixel = 0.27 * arcsec;
     *    Angle ra = 13.4 * hours;
     *    Angle dec = -32 * degrees;
     *    Angle theta = PI/2 * radians;
     *  @endcode
     *
     *  You can also use the constructor explicitly, which takes a value and a unit:
     *  @code
     *    Angle theta(90, degrees);
     *  @endcode
     *
     *  Since extracting the value in radians is extremely common, we have an accessor
     *  to do this quickly:
     *  @code
     *    x = theta.rad();
     *  @endcode
     *  It is equivalent to the more verbose:
     *  @code
     *    x = theta / radians;
     *  @endcode
     *  but without actually requiring the FLOP of dividing by 1.
     *
     *  Arithmetic with Angles include the following:
     *  (In the list below, x is a double, unit is an AngleUnit, and theta is an Angle.)
     *  @code
     *    theta = x * unit
     *    x = theta / unit
     *    theta3 = theta1 + theta2
     *    theta3 = theta1 - theta2
     *    theta2 = theta1 * x
     *    theta2 = x * theta1
     *    theta2 = theta1 / x
     *    theta2 += theta1
     *    theta2 -= theta1
     *    theta *= x
     *    theta /= x
     *    x = unit1/unit2
     *  @endcode
     *
     *  I/O:
     *    @code os << theta @endcode just outputs the value in radians.
     */
    class Angle {
    public:
        /** Construct an Angle with the specified value (interpreted in the given units) */
        explicit Angle(double val, AngleUnit unit) : _val(val*unit._val) {}
        /** Default constructor is 0 radians */
        Angle() : _val(0) {}
        /** Copy constructor. */
        Angle(const Angle& rhs) : _val(rhs._val) {}

        //@{
        /// Define conversion to a pure value
        double operator/(AngleUnit unit) const { return _val / unit._val; }
        double rad() const { return _val; }
        //@}

        //@{
        /// Define arithmetic for scaling an Angle
        Angle& operator*=(double scale) { _val *= scale; return *this; }
        Angle& operator/=(double scale) { _val /= scale; return *this; }
        Angle operator*(double scale) const { Angle theta = *this; theta *= scale; return theta; }
        friend Angle operator*(double scale, Angle theta) { return theta * scale; }
        Angle operator/(double scale) const { Angle theta = *this; theta /= scale; return theta; }
        //@}

        //@{
        /// Define arithmetic for adding/subtracting two Angles
        Angle& operator+=(Angle rhs) { _val += rhs._val; return *this; }
        Angle& operator-=(Angle rhs) { _val -= rhs._val; return *this; }
        Angle operator+(Angle rhs) const { Angle theta = *this; theta += rhs; return theta; }
        Angle operator-(Angle rhs) const { Angle theta = *this; theta -= rhs; return theta; }
        //@}

        //@{
        /// Define comparisons of two Angles
        bool operator==(Angle rhs) const { return _val == rhs._val; }
        bool operator!=(Angle rhs) const { return _val != rhs._val; }
        bool operator<=(Angle rhs) const { return _val <= rhs._val; }
        bool operator<(Angle rhs) const { return _val < rhs._val; }
        bool operator>=(Angle rhs) const { return _val >= rhs._val; }
        bool operator>(Angle rhs) const { return _val > rhs._val; }
        //@}

        /// Output operator for an Angle
        friend std::ostream& operator<<(std::ostream& os, Angle theta)
        { os << theta._val; return os; }

        /// Wraps this angle to the range (c-pi, c+pi]
        Angle wrap(Angle center=0.*radians)
        {
            const double TWOPI = 2.*M_PI;
            double new_val = std::fmod(_val-center.rad(), TWOPI); // now in range (-TWOPI, TWOPI)
            if (new_val <= -M_PI) new_val += TWOPI;
            if (new_val > M_PI) new_val -= TWOPI;
            return Angle(new_val+center.rad(), radians);
        }

        double sin() const { return std::sin(_val); }
        double cos() const { return std::cos(_val); }
        double tan() const { return std::tan(_val); }

        void sincos(double& sint, double& cost) const {
#ifdef _GLIBCXX_HAVE_SINCOS
            ::sincos(_val,&sint,&cost);
#else
            cost = std::cos(_val);
            sint = std::sin(_val);
#endif
        }

    private:
        double _val;
    };

    /// Define conversion from value to an Angle
    inline Angle operator*(double val, AngleUnit unit) { return Angle(val,unit); }

    /// A convenience function.  unit1/unit2 is equivalent to (1 * unit1) / unit2.
    inline double operator/(AngleUnit unit1, AngleUnit unit2)
    { return unit1.getValue() / unit2.getValue(); }

}
#endif
