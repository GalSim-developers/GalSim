// -*- c++ -*-
/*
 * Copyright 2012, 2013 The GalSim developers:
 * https://github.com/GalSim-developers
 *
 * This file is part of GalSim: The modular galaxy image simulation toolkit.
 *
 * GalSim is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * GalSim is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with GalSim.  If not, see <http://www.gnu.org/licenses/>
 */
// Functions for the CppShear class

#include <limits>
#include <algorithm>

//#define DEBUGLOGGING

#include "deprecated/CppEllipse.h"

#ifdef DEBUGLOGGING
#include <fstream>
std::ostream* dbgout = new std::ofstream("debug.out");
int verbose_level = 2;
#endif

namespace galsim {

    tmv::Matrix<double> CppEllipse::getMatrix() const 
    {
        double a, b, c;
        double scale=std::exp(mu);
        s.getMatrix(a,b,c);
        tmv::Matrix<double> m(2,2);
        m(0,0) = a*scale;
        m(1,1) = b*scale;
        m(0,1) = c*scale;
        m(1,0) = c*scale;
        return m;
    }

    CppEllipse CppEllipse::fromMatrix(
        const tmv::Matrix<double>& m, Angle& rotation, bool& parityFlip) 
    {
        dbg<<"ellipse from matrix "<<m<<'\n';
        assert(m.nrows()==2 && m.ncols()==2);
        double det = m(0,0)*m(1,1) - m(0,1)*m(1,0);
        parityFlip = false;
        double scale;
        if (det < 0) {
            parityFlip = true;
            scale = -det;
        } else if (det==0.) {
            // Degenerate transformation.  Return some junk
            return CppEllipse(CppShear(0.0, 0.0), -std::numeric_limits<double>::max());
        } else {
            scale = det;
        }
        // Determine and remove the dilation
        double mu = 0.5*std::log(scale);

        // Now make m m^T matrix, which is symmetric
        // a & b are diagonal elements here
        double a = m(0,1)*m(0,1) + m(0,0)*m(0,0);
        double b = m(1,1)*m(1,1) + m(1,0)*m(1,0);
        double c = m(1,1)*m(0,1) + m(1,0)*m(0,0);

        double eta = acosh(std::max(1.,0.5*(a+b)/scale));
        Angle beta = 0.5*std::atan2(2.*c, a-b) * radians;
        CppShear s;
        s.setEtaBeta(eta,beta);
        s.getMatrix(a,b,c);

        // Now look for the rotation
        rotation = std::atan2(-c*m(0,0)+a*m(1,0), b*m(0,0)-c*m(1,0)) * radians;
        return CppEllipse(s,mu, Position<double>(0.,0.));
    }

    // CppEllipses share the ordering conventions:  e1 + e2 is transform
    // e1 followed by transform e2.  Transform objects, not coords.
    CppEllipse CppEllipse::operator-() const 
    {
        Position<double> x3(-x0);
        x3 /= expmu;
        return CppEllipse(-s, -mu, s.inv(x3));
    }

    CppEllipse& CppEllipse::operator+=(const CppEllipse& e2) 
    {
        dbg<<"ellipse +=\n";
        Position<double> x3 = fwd(e2.getX0());
        x0 = x3;
        s += e2.getS();
        mu += e2.getMu();
        expmu = std::exp(mu);
        return *this;
    }

    CppEllipse& CppEllipse::operator-=(const CppEllipse& e2) 
    { return operator+=(-e2); }

    CppEllipse CppEllipse::operator+(const CppEllipse& rhs) const 
    {
        CppEllipse out(*this);
        out += rhs;
        return out;
    }

    CppEllipse CppEllipse::operator-(const CppEllipse& rhs) const 
    {
        CppEllipse out(*this);
        out -= rhs;
        return out;
    }

    void CppEllipse::write(std::ostream& fout) const 
    { s.write(fout); fout << " " << mu << " " ; x0.write(fout); }

    void CppEllipse::read(std::istream& fin) 
    { s.read(fin); fin >> mu; x0.read(fin); }

    Bounds<double> CppEllipse::range(double sig) const 
    {
        double a,b,c;
        s.getMatrix(a,b,c);
        // ??? note that below depends on s matrix being inverse and
        // with unit determinant
        double xmax=std::sqrt(a*a+c*c);
        double ymax=std::sqrt(b*b+c*c);
        return Bounds<double>( 
            x0.x - xmax*expmu*sig, x0.x + xmax*expmu*sig,
            x0.y - ymax*expmu*sig, x0.y + ymax*expmu*sig);
    }

}

