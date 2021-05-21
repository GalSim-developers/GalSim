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

#ifndef GalSim_Ellipse_H
#define GalSim_Ellipse_H
/**
 * @file CppEllipse.h Contains a class definition for CppEllipse.
 *
 * CppEllipse includes shear, translation and magnification.
 * The class names include "Cpp" in front because the C++ definition of Ellipse is more restricted 
 * than the python version, so we use Cpp as a way of distinguishing easily between the 
 * definitions.
*/

#include <cmath>
#include "TMV.h"

#include "Std.h"
#include "Bounds.h"
#include "Angle.h"
#include "CppShear.h"

namespace galsim {

    /**
     * @brief A base class representing transformation from an ellipse to the unit circle.
     *
     * The purpose of this C++ class is to represent transformation from an ellipse with center x0,
     * size exp(mu), and shape s to the unit circle.  The map from source plane to image plane is
     * defined as E(x) = T(D(S(x))), where S=shear, D=dilation, T=translation.  Conventions for
     * order of compounding, etc., are same as for CppShear.
     */
    class CppEllipse 
    {
    public:
        explicit CppEllipse(const CppShear& _s = CppShear(), double _mu = 0., 
                         const Position<double> _p = Position<double>()) :
            s(_s), mu(_mu), x0(_p) 
        { expmu=std::exp(mu); }

        CppEllipse(const CppEllipse& rhs) : 
            s(rhs.s), mu(rhs.mu), x0(rhs.x0) 
        { expmu=std::exp(mu); }

        const CppEllipse& operator=(const CppEllipse& rhs) 
        {
            if (&rhs==this) return *this;
            s = rhs.s; mu = rhs.mu; expmu=rhs.expmu; x0=rhs.x0; return *this;
        }

        CppEllipse operator+(const CppEllipse& rhs) const; //composition
        CppEllipse operator-() const; //negation
        CppEllipse& operator+=(const CppEllipse& rhs); //composition
        CppEllipse& operator-=(const CppEllipse& rhs); 
        CppEllipse operator-(const CppEllipse& rhs) const; //composition

        bool operator==(const CppEllipse& rhs) const 
        { return (mu==rhs.mu && x0==rhs.x0 && s == rhs.s); }

        bool operator!=(const CppEllipse& rhs) const 
        { return (mu!=rhs.mu || x0!=rhs.x0 || s != rhs.s); }

        void reset(const CppShear& _s, double _mu, const Position<double> _p) 
        { s=_s; mu=_mu; expmu=std::exp(mu); x0=_p; }

        Position<double> fwd(const Position<double> x) const 
        { return (s.fwd(x)*expmu + x0); }

        Position<double> inv(const Position<double> x) const 
        { return s.inv((x-x0)/expmu); }

        CppEllipse& setS(const CppShear& _s) { s=_s; return *this; }
        CppEllipse& setMu(const double& _m) { mu=_m; expmu=std::exp(mu); return *this; }
        CppEllipse& setX0(const Position<double>& _x) { x0=_x; return *this; }

        CppShear getS() const { return s; }
        double getMu() const { return mu; }
        Position<double> getX0() const { return x0; }

        // Calculate major, minor axes & PA of ellipse
        // resulting from source-plane circle:
        double getMajor() const { return std::exp(mu+s.getEta()/2); }
        double getMinor() const { return std::exp(mu-s.getEta()/2); }
        Angle getBeta() const { return s.getBeta(); }

        // Return a rectangle that circumscribes this ellipse (times nSigma)
        Bounds<double> range(double nSigma=1.) const;

        // Return the 2x2 matrix that implements the forward transformation
        // of this ellipse (apart from the translation)
        tmv::Matrix<double> getMatrix() const;

        // Utility to return the CppEllipse that corresponds to an arbitrary 2x2
        // matrix.  One version returns the rotation that must precede the
        // CppEllipse in the transform, if matrix is asymmetric.
        static CppEllipse fromMatrix(
            const tmv::Matrix<double>& m, Angle& rotation, bool& parity);

        static CppEllipse fromMatrix(const tmv::Matrix<double>& m) 
        {
            Angle junk; 
            bool p;
            return fromMatrix(m, junk, p);
        }

        void write(std::ostream& fout) const;
        void read(std::istream& fin);

    private:
        CppShear s;
        double mu;
        Position<double> x0;
        mutable double expmu; //exp(mu).
    };

    inline std::ostream& operator<<(std::ostream& os, const CppEllipse& e) 
    { e.write(os); return os; }

    inline std::istream& operator>>(std::istream& is, CppEllipse& e) 
    { e.read(is); return is; }

}

#endif // ELLIPSE_H
