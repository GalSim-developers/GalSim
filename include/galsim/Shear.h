/**
 * @file Shear.h Contains a class definition for Shear and Ellipse.
 *
 * Shear is used to represent shape distortions; Ellipse includes shear, translation and
 * magnification.
*/

#ifndef SHEAR_H
#define SHEAR_H
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
     * eta = "conformal shear", a/b = exp(eta)
     * g = "reduced shear", g=(a-b)/(a+b)
     * e = "distortion", e=(a^2-b^2)/(a^2+b^2)
     * q = "axis ratio", q=b/a
     * To specify both components, we have a value Beta that is the position angle of the major
     * axis.
     *
     * Shears are represented internally by e1 and e2, which relate to second moments via
     * e1 = (Mxx - Myy) / (Mxx + Myy)
     * e2 = 2 Mxy / (Mxx + Myy)
     * however, given that lensing specialists most commonly think in terms of reduced shear, the
     * constructor that takes two numbers expects (g1, g2).  A user who wishes to specify another
     * type of shape should use methods, i.e.
     *     s = Shear();
     *     s.setE1E2(my_e1, my_e2);
     */

    class Shear 
    {
        friend Shear operator*(const double, const Shear& );

    public:
        // Construct w/o variance
        explicit Shear(double g1=0., double g2=0.) :
            hasMatrix(false), matrixA(0), matrixB(0), matrixC(0)
                { setG1G2(g1, g2); }

        Shear(const Shear& rhs) :
            e1(rhs.e1), e2(rhs.e2), hasMatrix(rhs.hasMatrix),
            matrixA(rhs.matrixA), matrixB(rhs.matrixB), matrixC(rhs.matrixC) 
        {}

        const Shear& operator=(const Shear& s)
        {
            e1 = s.e1; e2=s.e2;
            matrixA=s.matrixA; matrixB=s.matrixB; matrixC=s.matrixC;
            hasMatrix=s.hasMatrix;
            return *this;
        }

        Shear& setE1E2(double e1in=0., double e2in=0.);
        Shear& setEBeta(double etain=0., Angle betain=Angle());
        Shear& setEta1Eta2(double eta1in=0., double eta2in=0.);
        Shear& setEtaBeta(double etain=0., Angle betain=Angle());
        Shear& setG1G2(double g1in=0., double g2in=0.);

        double getE1() const { return e1; }
        double getE2() const { return e2; }
        double getE() const { return std::sqrt(e1*e1+e2*e2); }
        double getESq() const { return e1*e1+e2*e2; }
        Angle getBeta() const { return std::atan2(e2,e1)*0.5 * radians; }
        double getEta() const { return atanh(std::sqrt(e1*e1+e2*e2)); } //error checking?

        // g = gamma / (1-kappa)
        double getG() const 
        {
            double e=getE();  
            return e>0. ? (1-std::sqrt(1-e*e))/e : 0.;
        }
        double getG1() const
        {
            double esq = getESq();
            double scale = (esq>1.e-6) ? (1.-std::sqrt(1.-esq))/esq : 0.5;
            return e1*scale;
        }
        double getG2() const
        {
            double esq = getESq();
            double scale = (esq>1.e-6) ? (1.-std::sqrt(1.-esq))/esq : 0.5;
            return e2*scale;
        }

        void getEta1Eta2(double& eta1, double& eta2) const;
        void getG1G2(double& g1, double& g2) const;


        //negation
        Shear operator-() const 
        {
            double esq = getESq();
            double scale = (esq>1.e-6) ? (1.-std::sqrt(1.-esq))/esq : 0.5;
            return esq>0. ? Shear(-e1*scale, -e2*scale) : Shear(0.0, 0.0);
        }

        // Composition operation: returns ellipticity of
        // circle that is sheared first by RHS and then by
        // LHS Shear.  Note that this addition is
        // ***not commutative***!
        // In the += and -= operations, this is LHS
        // and the operand is RHS of + or - .
        Shear operator+(const Shear& ) const;
        Shear operator-(const Shear& ) const;
        Shear& operator+=(const Shear& );
        Shear& operator-=(const Shear& );

        // Give the rotation angle for this+rhs:
        Angle rotationWith(const Shear& rhs) const;
        // Detail on the above: s1 + s2 operation on points in
        // the plane induces a rotation as well as a shear.
        // Above method tells you what the rotation was for LHS+RHS

        bool operator==(const Shear& rhs) const 
        { return e1==rhs.e1 && e2==rhs.e2; }

        bool operator!=(const Shear& rhs) const 
        { return e1!=rhs.e1 || e2!=rhs.e2; }

        // Classes that treat shear as a point-set map:
        template <class T>
        Position<T> fwd(Position<T> p) const 
        {
            // Fwd is map from image to source plane coordinates.
            calcMatrix();
            Position<T> out(matrixA*p.x+matrixC*p.y, matrixC*p.x+matrixB*p.y);
            return out;
        }

        template <class T>
        Position<T> inv(Position<T> p) const 
        {
            calcMatrix();
            // Note that we define shear to have unit determinant
            Position<T> out(matrixB*p.x-matrixC*p.y, -matrixC*p.x+matrixA*p.y);
            return out;
        }

        // Get matrix representation of the forward transformation.
        // Matrix is   ... and in limit of small shear:
        // (  a   c     =    (  1+g1  g2
        //    c   b )            g2  1-g1 )
        void getMatrix(double& a, double& b, double& c) const 
        { calcMatrix(); a=matrixA; b=matrixB; c=matrixC; }

        void write(std::ostream& fout) const;
        friend std::ostream& operator<<(std::ostream& os, const Shear& s);

        void read(std::istream& fin);
        friend std::istream& operator>>(std::istream& is, Shear& s);

    private:

        void calcMatrix() const;
        double e1, e2;
        // Matrix elements for forward/inverse x/y mapping
        mutable bool hasMatrix;
        mutable double matrixA, matrixB, matrixC;
    };

    std::ostream& operator<<(std::ostream& os, const Shear& s);
    std::istream& operator>>(std::istream& is, Shear& s);

    /**
     * @brief A base class representing transformation from an ellipse to the unit circle.
     *
     * The purpose of this C++ class is to represent transformation from an ellipse with center x0,
     * size exp(mu), and shape s to the unit circle.  The map from source plane to image plane is
     * defined as E(x) = T(D(S(x))), where S=shear, D=dilation, T=translation.  Conventions for
     * order of compounding, etc., are same as for Shear.
     */
    class Ellipse 
    {
    public:
        explicit Ellipse(const Shear& _s = Shear(), double _mu = 0., 
                         const Position<double> _p = Position<double>()) :
            s(_s), mu(_mu), x0(_p) 
        { expmu=std::exp(mu); }

        Ellipse(const Ellipse& rhs) : 
            s(rhs.s), mu(rhs.mu), x0(rhs.x0) 
        { expmu=std::exp(mu); }

        const Ellipse& operator=(const Ellipse& rhs) 
        {
            if (&rhs==this) return *this;
            s = rhs.s; mu = rhs.mu; expmu=rhs.expmu; x0=rhs.x0; return *this;
        }

        Ellipse operator+(const Ellipse& rhs) const; //composition
        Ellipse operator-() const; //negation
        Ellipse& operator+=(const Ellipse& rhs); //composition
        Ellipse& operator-=(const Ellipse& rhs); 
        Ellipse operator-(const Ellipse& rhs) const; //composition

        bool operator==(const Ellipse& rhs) const 
        { return (mu==rhs.mu && x0==rhs.x0 && s == rhs.s); }

        bool operator!=(const Ellipse& rhs) const 
        { return (mu!=rhs.mu || x0!=rhs.x0 || s != rhs.s); }

        void reset(const Shear& _s, double _mu, const Position<double> _p) 
        { s=_s; mu=_mu; expmu=std::exp(mu); x0=_p; }

        Position<double> fwd(const Position<double> x) const 
        { return (s.fwd(x)*expmu + x0); }

        Position<double> inv(const Position<double> x) const 
        { return s.inv((x-x0)/expmu); }

        Ellipse& setS(const Shear& _s) { s=_s; return *this; }
        Ellipse& setMu(const double& _m) { mu=_m; expmu=std::exp(mu); return *this; }
        Ellipse& setX0(const Position<double>& _x) { x0=_x; return *this; }

        Shear getS() const { return s; }
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

        // Utility to return the Ellipse that corresponds to an arbitrary 2x2
        // matrix.  One version returns the rotation that must precede the
        // Ellipse in the transform, if matrix is asymmetric.
        static Ellipse fromMatrix(
            const tmv::Matrix<double>& m, Angle& rotation, bool& parity);

        static Ellipse fromMatrix(const tmv::Matrix<double>& m) 
        {
            Angle junk; 
            bool p;
            return fromMatrix(m, junk, p);
        }

        void write(std::ostream& fout) const;
        void read(std::istream& fin);

    private:
        Shear s;
        double mu;
        Position<double> x0;
        mutable double expmu; //exp(mu).
    };

    inline std::ostream& operator<<(std::ostream& os, const Ellipse& e) 
    { e.write(os); return os; }

    inline std::istream& operator>>(std::istream& is, Ellipse& e) 
    { e.read(is); return is; }

}

#endif // SHEAR_H
