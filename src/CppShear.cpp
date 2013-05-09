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

#include "CppShear.h"

#ifdef DEBUGLOGGING
#include <fstream>
std::ostream* dbgout = new std::ofstream("debug.out");
int verbose_level = 2;
#endif

// Convention for CppShear addition is that s1 + s2 is shear by s1
// followed by shear of s2.  Note that this differs from the
// notation in the methods paper.

namespace galsim {

    CppShear& CppShear::setE1E2(double e1in, double e2in) 
    {
        dbg<<"CppShear setE1E2 "<<e1in<<','<<e2in<<'\n';
        hasMatrix = false;
        e1 = e1in;
        e2 = e2in;
        return *this;
    }

    CppShear& CppShear::setEBeta(double ein, Angle betain) 
    {
        dbg<<"CppShear setEBeta "<<ein<<','<<betain<<'\n';
        hasMatrix = false;
        e1 = ein*std::cos(2.*betain.rad());
        e2 = ein*std::sin(2.*betain.rad());
        return *this;
    }

    CppShear& CppShear::setEta1Eta2(double eta1in, double eta2in) 
    {
        dbg<<"CppShear setEta1Eta2 "<<eta1in<<','<<eta2in<<'\n';
        double scale;
        hasMatrix = false;
        // get ratio of e amplitude to eta amplitude:
        scale = std::sqrt(eta1in*eta1in + eta2in*eta2in);
        if (scale>0.001) scale = std::tanh(scale)/scale;
        else scale=1.;
        e1 = eta1in*scale;
        e2 = eta2in*scale;
        return *this;
    }

    CppShear& CppShear::setEtaBeta(double etain, Angle betain) 
    {
        dbg<<"CppShear setEtaBeta "<<etain<<','<<betain<<'\n';
        double e;
        hasMatrix = false;
        e = std::tanh(etain);
        e1 = e * std::cos(2.*betain.rad());
        e2 = e * std::sin(2.*betain.rad());
        return *this;
    }

    void CppShear::getEta1Eta2(double& eta1, double& eta2) const 
    {
        double scale;
        // get ratio of eta amplitude to e amplitude:
        scale = std::sqrt(e1*e1 + e2*e2);
        if (scale>0.001) scale = atanh(scale)/scale;
        else scale=1.;
        eta1 = e1*scale;
        eta2 = e2*scale;
    }

    void CppShear::getG1G2(double& g1, double& g2) const 
    {
        // get ratio of eta amplitude to e amplitude:
        double esq = getESq();
        double scale = (esq>1.e-6) ? (1.-std::sqrt(1.-esq))/esq : 0.5;
        g1 = e1*scale;
        g2 = e2*scale;
    }

    CppShear& CppShear::setG1G2(double g1, double g2) 
    {
        dbg<<"CppShear setG1G2 "<<g1<<','<<g2<<'\n';
        // get ratio of eta amplitude to e amplitude:
        double scale = 2./(1.+g1*g1+g2*g2);
        e1 = g1*scale;
        e2 = g2*scale;
        return *this;
    }

    CppShear& CppShear::operator+=(const CppShear& s2) 
    {
        dbg<<"CppShear op+=\n";
        double s1sq = e1*e1+e2*e2;
        if (s1sq==0.) { (*this)=s2; return *this;}

        hasMatrix = false;

#ifndef NDEBUG
        double s2sq = s2.e1*s2.e1+s2.e2*s2.e2;
#endif
        assert(s1sq<=1. && s2sq<=1.); //addition requires a realizable shear.

        double denom = 1. + e1*s2.e1 + e2*s2.e2;
        if (denom==0.) { e1=e2=0.; return *this; }

        double temp = 1.-std::sqrt(1.-s1sq);
        double e1new = e1 + s2.e1 + temp*(e1 * s2.e2 - e2 * s2.e1)*e2/s1sq;
        e2 = e2 + s2.e2 + temp*(e2 * s2.e1 - e1 * s2.e2)*e1/s1sq;
        e1 = e1new/denom;
        e2 /= denom;

        return *this;
    }

    CppShear& CppShear::operator-=(const CppShear& s2) 
    {
        dbg<<"CppShear op-=\n";
        //NOTE that s1 -= s2 will produce s1 + (-s2) according to 
        // the local convention.
        return CppShear::operator+=(-s2);
    }

    CppShear CppShear::operator+(const CppShear& s2) const 
    {
        //returns s1 + s2 
        CppShear out=*this;
        out += s2;
        return out;
    }

    CppShear CppShear::operator-(const CppShear& s2) const 
    {
        //returns s1 - s2 
        CppShear out=*this;
        out += -s2;
        return out;
    }

    Angle CppShear::rotationWith(const CppShear& rhs) const 
    {
        double a, b, c;
        double ra, rb, rc;
        double tot11, tot21;
        getMatrix(a,b,c);
        rhs.getMatrix(ra, rb, rc);
        tot11 = a*ra + c*rc;
        tot21 = c*ra + b*rc;
        CppShear sum = -(*this + rhs);
        sum.getMatrix(ra, rb, rc);
        double cc = ra * tot11 + rc * tot21;
        double ss = rc * tot11 + rb * tot21;
        return std::atan2(ss, cc) * radians;
    }

    void CppShear::write(std::ostream& fout) const 
    { fout << "(" << e1 << "," << e2 << ")" ; }

    std::ostream& operator<<(std::ostream& os, const CppShear& s) 
    { s.write(os); return os; }

    void CppShear::read(std::istream& fin) 
    {
        char ch;
        hasMatrix = false;
        fin >> ch >> e1 >> ch >> e2 >> ch ;
    }

    std::istream& operator<<(std::istream& is, CppShear& s) 
    { s.read(is); return is; }

    void CppShear::calcMatrix() const 
    {
        if (hasMatrix) return;
        dbg<<"CppShear calcMatrix\n";
        dbg<<"e1,e2 = "<<e1<<','<<e2<<'\n';

        //  Matrix is defined here to be for forward point map, source plane
        // to image plane for a circular source that acquires this shape.
        // +eta in xx posn.
        double esq= e1*e1+e2*e2;
        dbg<<"esq = "<<esq<<'\n';
        assert (esq < 1.); //Must be realizable

        if (esq < 1.e-4) {
            //Small-e approximation ok to part in 10^-6.
            matrixA = 1.+0.5*e1+0.125*esq;
            matrixB = 1.-0.5*e1+0.125*esq;
            matrixC = +0.5*e2;
        } else {
            double temp = std::sqrt(1.-esq);
            double cc=std::sqrt(0.5*(1.+1./temp));
            temp = (1.-temp)/esq;
            matrixA = cc*(1.+temp*e1);
            matrixB = cc*(1.-temp*e1);
            matrixC = +cc*temp*e2;
        }
        dbg<<"matrix = "<<matrixA<<','<<matrixB<<','<<matrixC<<'\n';
        hasMatrix = true;
    }

}

