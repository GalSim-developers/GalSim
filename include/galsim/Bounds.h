/// @file Bounds.h @brief Classes defining 2d positions and rectangles.

#ifndef Bounds_H
#define Bounds_H

#include <vector>
#include <iostream>

#include "Std.h"

namespace galsim {

    template <class T>
    /// @brief Class for storing 2d position vectors in an (x, y) format.
    class Position 
    {
    public:
        /// @brief Publicly visible x & y attributes of the position.
        T x,y;

        ///@brief Constructor.
        Position(const T xin=0, const T yin=0) : x(xin), y(yin) {}

        ///@brief Assignment.
        Position& operator=(const Position rhs) 
        {
            if (&rhs == this) return *this;
            else { x=rhs.x; y=rhs.y; return *this; }
        }

        /// @brief Overloaded += operator, following standard vector algebra rules.
        Position& operator+=(const Position rhs) { x+=rhs.x; y+=rhs.y; return *this; }

        /// @brief Overloaded -= operator, following standard vector algebra rules.
        Position& operator-=(const Position rhs) { x-=rhs.x; y-=rhs.y; return *this; }

        /// @brief Overloaded *= operator for scalar multiplication.
        Position& operator*=(const T rhs) { x*=rhs; y*=rhs; return *this; }

        /// @brief Overloaded /= operator for scalar division.
        Position& operator/=(const T rhs) { x/=rhs; y/=rhs; return *this; }

        /// @brief Overloaded * vector multiplication operator for scalar on rhs.
        Position operator*(const T rhs) const { return Position(x*rhs, y*rhs); }

        /// @brief Overloaded / vector division operator for scalar on rhs.
        Position operator/(const T rhs) const { return Position(x/rhs, y/rhs); }

        /// @brief Unary negation (x, y) -> (-x, -y).
        Position operator-() const { return Position(-x,-y); }

        /// @brief Overloaded vector + addition operator with a Position on the rhs.
        Position operator+(Position<T> rhs) const { return Position(x+rhs.x,y+rhs.y); }

        /// @brief Overloaded vector - subtraction operator with a Position on the rhs.
        Position operator-(const Position<T> rhs) const { return Position(x-rhs.x,y-rhs.y); }

        /// @brief Overloaded == relational equality operator.
        bool operator==(const Position& rhs) const { return (x==rhs.x && y==rhs.y); }
        
        /// @brief Overloaded != relational non-equality operator.
        bool operator!=(const Position& rhs) const { return (x!=rhs.x || y!=rhs.y); }

        /// @brief Write (x, y) position to output stream.
        void write(std::ostream& fout) const { fout << "(" << x << "," << y << ")"; }

        /// @brief Read (x, y) position from input istream.
        void read(std::istream& fin) { char ch; fin >> ch >> x >> ch >> y >> ch; }

    }; // Position

    /// @brief Overloaded << operator which uses write() method of Position class.
    template <class T>
    inline std::ostream& operator<<(std::ostream& os, const Position<T> p) 
    { p.write(os); return os; }

    /// @brief Overloaded >> operator which uses read() method of Position class.
    template <class T>
    inline std::istream& operator>>(std::istream& is, Position<T>& p) 
    { p.read(is); return is; }

    /** 
     * @brief Class for storing image bounds, essentially the vertices of a rectangle.  
     *
     * This is used to keep track of the bounds of catalogs and fields.  You can set values, 
     * but generally you just keep including positions of each galaxy or the bounds of each 
     * catalog respectively using the += operators
     *
     * Rectangle is undefined if min>max in either direction.
     */
    template <class T>
    class Bounds 
    {
        /** 
         * @brief Class for storing image bounds, essentially the vertices of a rectangle.  
         *
         * This is used to keep track of the bounds of catalogs and fields.  You can set values, 
         * but generally you just keep including positions of each galaxy or the bounds of each 
         * catalog respectively using the += operators
         *
         * Rectangle is undefined if min>max in either direction.
         */
    public:
        Bounds(const T x1, const T x2, const T y1, const T y2) :
            defined(x1<=x2 && y1<=y2),xmin(x1),xmax(x2),ymin(y1),ymax(y2) {}
        Bounds(const Position<T>& pos) :
            defined(1),xmin(pos.x),xmax(pos.x),ymin(pos.y),ymax(pos.y) {}
        Bounds(const Position<T>& pos1, const Position<T>& pos2) :
            defined(1),xmin(std::min(pos1.x,pos2.x)),xmax(std::max(pos1.x,pos2.x)),
            ymin(std::min(pos1.y,pos2.y)),ymax(std::max(pos1.y,pos2.y)) {}
        Bounds() : defined(0),xmin(0),xmax(0),ymin(0),ymax(0) {}
        ~Bounds() {}
        void setXMin(const T x) { xmin = x; defined= xmin<=xmax && ymin<=ymax; }
        void setXMax(const T x) { xmax = x; defined= xmin<=xmax && ymin<=ymax; }
        void setYMin(const T y) { ymin = y; defined= xmin<=xmax && ymin<=ymax; }
        void setYMax(const T y) { ymax = y; defined= xmin<=xmax && ymin<=ymax; }
        T getXMin() const { return xmin; }
        T getXMax() const { return xmax; }
        T getYMin() const { return ymin; }
        T getYMax() const { return ymax; }
        bool isDefined() const { return defined; }

        Position<T> center() const;
        void operator+=(const Position<T>& pos); //expand to include point
        void operator+=(const Bounds<T>& rec); //bounds of union
        void addBorder(const T d);
        void operator+=(const T d) { addBorder(d); }
        void expand(const double m); // expand by a multiple m, about bounds center
        const Bounds<T> operator&(const Bounds<T>& rhs) const; // Finds intersection
        void shift(const T dx, const T dy) { xmin+=dx; xmax+=dx; ymin+=dy; ymax+=dy; }
        void shift(Position<T> dx) { shift(dx.x,dx.y); }
        bool includes(const Position<T>& pos) const
        { return (defined && pos.x<=xmax && pos.x>=xmin && pos.y<=ymax && pos.y>=ymin); }
        bool includes(const T x, const T y) const
        { return (defined && x<=xmax && x>=xmin && y<=ymax && y>=ymin); }
        bool includes(const Bounds<T>& rhs) const
        { 
            return (defined && rhs.defined && 
                    rhs.xmin>=xmin && rhs.xmax<=xmax &&
                    rhs.ymin>=ymin && rhs.ymax<=ymax);
        }
        bool operator==(const Bounds<T>& rhs) const 
        {
            return defined && rhs.defined && (xmin==rhs.xmin) &&
                (ymin==rhs.ymin) && (xmax==rhs.xmax) && (ymax==rhs.ymax);
        }
        bool operator!=(const Bounds<T>& rhs) const 
        {
            return !defined || !rhs.defined || (xmin!=rhs.xmin) ||
                (ymin!=rhs.ymin) || (xmax!=rhs.xmax) || (ymax!=rhs.ymax);
        }

        T area() const { return defined ? (xmax-xmin)*(ymax-ymin) : 0.; }
        typename std::vector<Bounds<T> > divide(int nx, int ny) const;
        void write(std::ostream& fout) const
        { 
            if (defined) fout << xmin << ' ' << xmax << ' ' << ymin << ' ' << ymax << ' ';
            else fout << "Undefined ";
        }
        void read(std::istream& fin)
        { fin >> xmin >> xmax >> ymin >> ymax; defined = xmin<=xmax && ymin<=ymax; }

    private:
        bool defined;
        T xmin,xmax,ymin,ymax;

    };

    template <class T>
    inline std::ostream& operator<<(std::ostream& fout, const Bounds<T>& b)
    { b.write(fout); return fout; }

    template <class T>
    inline std::istream& operator>>(std::istream& fin, Bounds<T>& b)
    { b.read(fin); return fin; }

    ///////////////////////////////////////////////////////////////////////
    //  Following are the implementations:
    ///////////////////////////////////////////////////////////////////////

    template <class T>
    void Bounds<T>::operator+=(const Position<T>& pos)
    {
        if (defined) {
            if(pos.x < xmin) xmin = pos.x;
            else if (pos.x > xmax) xmax = pos.x;
            if(pos.y < ymin) ymin = pos.y;
            else if (pos.y > ymax) ymax = pos.y;
        } else {
            xmin = xmax = pos.x;
            ymin = ymax = pos.y;
            defined = 1;
        }
    }

    template <class T>
    void Bounds<T>::operator+=(const Bounds<T>& rec)
    {
        if (!rec.isDefined()) return;
        if (defined) {
            if(rec.getXMin() < xmin) xmin = rec.getXMin();
            if(rec.getXMax() > xmax) xmax = rec.getXMax();
            if(rec.getYMin() < ymin) ymin = rec.getYMin();
            if(rec.getYMax() > ymax) ymax = rec.getYMax();
        } else {
            *this = rec;
            defined = 1;
        }
    }

    template <class T>
    Position<T> Bounds<T>::center() const
    { return Position<T>((xmin + xmax)/2.,(ymin + ymax)/2.); }

    // & operator finds intersection, if any
    template <class T>
    const Bounds<T> Bounds<T>::operator&(const Bounds<T>& rhs) const
    {
        if (!defined || !rhs.defined) return Bounds<T>();
        Bounds<T> temp(
            xmin<rhs.xmin ? rhs.xmin : xmin,
            xmax>rhs.xmax ? rhs.xmax : xmax,
            ymin<rhs.ymin ? rhs.ymin : ymin,
            ymax>rhs.ymax ? rhs.ymax : ymax);
        if (temp.xmin>temp.xmax || temp.ymin>temp.ymax) return Bounds<T>();
        else return temp;
    }

    template <class T>
    void Bounds<T>::addBorder(T d)
    { if(defined) { xmax += d; xmin -= d; ymax += d; ymin -= d; } }

    template <class T>
    std::vector< Bounds<T> > Bounds<T>::divide(int nx, int ny) const
    {
        if (!defined) return std::vector< Bounds<T> >(nx,ny);
        typename std::vector< Bounds<T> > temp(nx*ny);
        std::vector<double> x(nx+1);
        std::vector<double> y(ny+1);
        x[0] = xmin;  x[nx] = xmax;
        y[0] = ymin;  y[ny] = ymax;
        double xstep = (xmax-xmin)/nx;
        double ystep = (ymax-ymin)/ny;
        for(int i=1;i<nx;i++) x[i] = x[0]+i*xstep;
        for(int j=1;j<ny;j++) y[j] = y[0]+j*ystep;
        typename std::vector< Bounds<T> >::iterator ii=temp.begin();
        for(int i=0;i<nx;i++) for(int j=0;j<ny;j++, ++i)
            *ii = Bounds<T>(x[i],x[i+1],y[j],y[j+1]);
        return temp;
    }

    template <class T>
    void Bounds<T>::expand(const double m) 
    {
        T dx = xmax-xmin;
        T dy = ymax-ymin;
        dx = dx*0.5*(m-1); 
        dy = dy*0.5*(m-1);
        xmax += dx;  xmin -= dx;
        ymax += dy;  ymin -= dy;
    }

}

#endif
