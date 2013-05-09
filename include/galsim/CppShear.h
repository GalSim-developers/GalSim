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

#ifndef SHEAR_H
#define SHEAR_H
/**
 * @file CppShear.h Contains a class definition for CppShear.
 *
 * CppShear is used to represent shape distortions.
 * The class names include "Cpp" in front because the C++ definition of Shear 
 * is more restricted than the python version, so we use Cpp as a way of distinguishing
 * easily between the definitions.
*/

#include <cmath>
#include "TMV.h"

#include "Std.h"
#include "Bounds.h"
#include "Angle.h"

// Shear is represented internally by e1 and e2, which are the second-moment
// definitions: ellipse with axes a & b has e=(a^2-b^2)/(a^2+b^2).
// But can get/set the ellipticity by two other measures:
// g is "reduced shear" such that g=(a-b)/(a+b)
// eta is "conformal shear" such that a/b = exp(eta).
// Beta is always the position angle of major axis.

namespace galsim {

    /**
     * @brief A base class representing shears.
     *
     * The purpose of this C++ class is to represent both deviations from roundness due to galaxy
     * intrinsic shapes, and due to lensing shears.  Given semi-major and semi-minor axis lengths
     * "a" and "b", there are numerous ways to represent shears:
     *
     * eta = "conformal shear", a/b = exp(|eta|)
     *
     * g   = "reduced shear", |g| = (a-b)/(a+b)
     *
     * e   = "distortion", |e| = (a^2-b^2)/(a^2+b^2)
     *
     * q   = "axis ratio", q=b/a
     * 
     * To specify both components, we have a value Beta that is the real-space position angle of 
     * the major axis, i.e. e1 = |e| cos(2*Beta) and e2 = |e| sin(2*Beta).
     *
     * Shears are represented internally by e1 and e2, which relate to second moments via
     * e1 = (Mxx - Myy) / (Mxx + Myy),
     * e2 = 2 Mxy / (Mxx + Myy).
     *
     * However, given that lensing specialists most commonly think in terms of reduced shear, the
     * constructor that takes two numbers expects (g1, g2).  A user who wishes to specify another
     * type of shape should use methods, i.e.
     *     s = CppShear();
     *     s.setE1E2(my_e1, my_e2);
     */

    class CppShear 
    {
        friend CppShear operator*(const double, const CppShear& );

    public:
        /** 
         * @brief Construct without variance / without initializing transformation matrix.
         *
         * @param[in] g1 first component (reduced shear definition).
         * @param[in] g2 second shear component (reduced shear definition).
         */
        explicit CppShear(double g1=0., double g2=0.) :
            hasMatrix(false), matrixA(0), matrixB(0), matrixC(0)
                { setG1G2(g1, g2); }

        /// @brief Copy constructor.
        CppShear(const CppShear& rhs) :
            e1(rhs.e1), e2(rhs.e2), hasMatrix(rhs.hasMatrix),
            matrixA(rhs.matrixA), matrixB(rhs.matrixB), matrixC(rhs.matrixC) 
        {}

        /// @brief Copy assignment.
        const CppShear& operator=(const CppShear& s)
        {
            e1 = s.e1; e2=s.e2;
            matrixA=s.matrixA; matrixB=s.matrixB; matrixC=s.matrixC;
            hasMatrix=s.hasMatrix;
            return *this;
        }

        /// @brief Set (e1, e2) using distortion definition.
        CppShear& setE1E2(double e1in=0., double e2in=0.);

        /** 
         * @brief Set (|e|, beta) polar ellipticity representation using distortion definition.
         * beta must be an Angle.
         */
        CppShear& setEBeta(double etain=0., Angle betain=Angle());

        /// @brief Set (eta1, eta2) using conformal shear definition.
        CppShear& setEta1Eta2(double eta1in=0., double eta2in=0.);

        /// @brief Set (|eta|, beta) using conformal shear definition. beta must be an Angle.
        CppShear& setEtaBeta(double =0., Angle betain=Angle());

        /// @brief set (g1, g2) using reduced shear |g| = (a-b)/(a+b) definition.
        CppShear& setG1G2(double g1in=0., double g2in=0.);

        /// @brief Get e1 using distortion definition.
        double getE1() const { return e1; }

        /// @brief Get e2 using distortion definition.
        double getE2() const { return e2; }

        /// @brief Get |e| using distortion definition.
        double getE() const { return std::sqrt(e1*e1+e2*e2); }

        /// @brief Get |e|^2 using distortion definition.
        double getESq() const { return e1*e1+e2*e2; }

        /// @brief Get polar angle beta (returns an Angle class object).
        Angle getBeta() const { return std::atan2(e2,e1)*0.5 * radians; }

        /// @brief Get |eta| using conformal shear definition.
        double getEta() const { return atanh(std::sqrt(e1*e1+e2*e2)); } //error checking?

        /// @brief Get |g| using reduced shear |g| = (a-b)/(a+b) definition.
        double getG() const 
        {
            double e=getE();  
            return e>0. ? (1-std::sqrt(1-e*e))/e : 0.;
        }

        /// @brief Get g1 using reduced shear |g| = (a-b)/(a+b) definition.
        double getG1() const
        {
            double esq = getESq();
            double scale = (esq>1.e-6) ? (1.-std::sqrt(1.-esq))/esq : 0.5;
            return e1*scale;
        }

        /// @brief Get g1 using reduced shear |g| = (a-b)/(a+b) definition.
        double getG2() const
        {
            double esq = getESq();
            double scale = (esq>1.e-6) ? (1.-std::sqrt(1.-esq))/esq : 0.5;
            return e2*scale;
        }

        /**
         * @brief Get (eta1, eta2) using conformal shear definition.
         *
         * @param[in,out] eta1 Reference to eta1 variable.
         * @param[in,out] eta2 Reference to eta2 variable.
         *
         */
        void getEta1Eta2(double& eta1, double& eta2) const;
        
        /**
         * @brief Get (g1, g2) using reduced shear |g| = (a-b)/(a+b) definition.
         *
         * @param[in,out] g1 Reference to g1 variable.
         * @param[in,out] g2 Reference to g2 variable.
         *
         */
        void getG1G2(double& g1, double& g2) const;

        /// @brief Unary negation (both components negated).
        CppShear operator-() const 
        {
            double esq = getESq();
            double scale = (esq>1.e-6) ? (1.-std::sqrt(1.-esq))/esq : 0.5;
            return esq>0. ? CppShear(-e1*scale, -e2*scale) : CppShear(0.0, 0.0);
        }

        /**
         * @brief Composition operation.
         *
         * Note that this 'addition' is ***not commutative***!
         *
         * @returns Ellipticity of circle that is sheared first by RHS and then by
         * LHS CppShear.  
         */
        CppShear operator+(const CppShear& ) const;

        /**
         * @brief Composition (with RHS negation) operation.
         *
         * Note that this 'subtraction' is ***not commutative***!
         *
         * @returns Ellipticity of circle that is sheared first by the negative RHS and then by
         * LHS CppShear.
         */ 
        CppShear operator-(const CppShear& ) const;
        
        // In the += and -= operations, this is LHS
        // and the operand is RHS of + or - .
        
        /**
         * @brief Inplace composition operation. 
         *
         * Note that this 'addition' is ***not commutative***!
         *
         * In the += operation, this is LHS and the operand is RHS of +.
         *
         * @returns Ellipticity of circle that is sheared first by RHS and then by
         * LHS CppShear. 
         */
        CppShear& operator+=(const CppShear& );
        
        /**
         * @brief Inplace composition (with RHS negation) operation.
         *
         * Note that this 'addition' is ***not commutative***!
         *
         * In the -= operation, this is LHS and the operand is RHS of -.
         *
         * @returns Ellipticity of circle that is sheared first by the negative RHS and then by
         * LHS CppShear.
         */
        CppShear& operator-=(const CppShear& );

        /** @brief Give the rotation angle for this+rhs.
         *
         * Detail on the above: s1 + s2 operation on points in
         * the plane induces a rotation as well as a shear.
         * Above method tells you what the rotation was for LHS+RHS.
         */
        Angle rotationWith(const CppShear& rhs) const; 

        ///@brief Test equivalence by comparing e1 and e2.
        bool operator==(const CppShear& rhs) const 
        { return e1==rhs.e1 && e2==rhs.e2; }

        ///@brief Test non-equivalence by comparing e1 and e2.
        bool operator!=(const CppShear& rhs) const 
        { return e1!=rhs.e1 || e2!=rhs.e2; }

        // Classes that treat shear as a point-set map:
        /**
         * @brief Forward transformation from image to source plane coordinates under shear.
         *
         * @param[in] p  2D vector Position in image plane.
         *
         * @returns      2D vector Position in source plane.
         */
        template <class T>
        Position<T> fwd(Position<T> p) const 
        {
            // Fwd is map from image to source plane coordinates.
            calcMatrix();
            Position<T> out(matrixA*p.x+matrixC*p.y, matrixC*p.x+matrixB*p.y);
            return out;
        }

        /**
         * @brief Inverse transformation from source to image plane coordinates under shear.
         *
         * @param[in] p  2D vector Position in source plane.
         *
         * @returns      2D vector Position in image plane.
         */
        template <class T>
        Position<T> inv(Position<T> p) const 
        {
            calcMatrix();
            // Note that we define shear to have unit determinant
            Position<T> out(matrixB*p.x-matrixC*p.y, -matrixC*p.x+matrixA*p.y);
            return out;
        }

        /**
         * @brief Get matrix representation of the forward transformation.
         *
         * Matrix is   ... and in limit of small shear:
         *  ( a   c     =    ( 1+g1  g2
         *    c   b )            g2  1-g1 )
         */
        void getMatrix(double& a, double& b, double& c) const 
        { calcMatrix(); a=matrixA; b=matrixB; c=matrixC; }

        void write(std::ostream& fout) const;
        friend std::ostream& operator<<(std::ostream& os, const CppShear& s);

        void read(std::istream& fin);
        friend std::istream& operator>>(std::istream& is, CppShear& s);

    private:

        void calcMatrix() const;
        double e1, e2;
        // Matrix elements for forward/inverse x/y mapping
        mutable bool hasMatrix;
        mutable double matrixA, matrixB, matrixC;
    };

    std::ostream& operator<<(std::ostream& os, const CppShear& s);
    std::istream& operator>>(std::istream& is, CppShear& s);

}

#endif // SHEAR_H
